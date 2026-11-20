"""Model artifact loading and inference engine."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import (
    MODELS_DIR, FEATURES_DIR, PROCESSED_DIR,
    FORECAST_HORIZONS, EXPECTED_STATIONS, STATION_COORDS,
    WARNING_THRESHOLDS
)
from src.models import (
    PersistenceBaseline, RandomForestModel, XGBoostModel,
    LSTMModel, CNNLSTMModel, STCNNGRUModel, GAKELMModel, HybridForecaster
)
from src.warnings_system.early_warning import EarlyWarningSystem, EarlyWarningEvent


class InferenceEngine:
    """Singleton inference manager providing fast in-memory predictions."""
    _instance = None

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.test_df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.target_cols: List[str] = []
        self.warning_system = EarlyWarningSystem()
        self.load_test_data()
        self.preload_available_models()

    @classmethod
    def get_instance(cls) -> "InferenceEngine":
        if cls._instance is None:
            cls._instance = InferenceEngine()
        return cls._instance

    def load_test_data(self):
        feat_path = FEATURES_DIR / "test_features.parquet"
        if feat_path.exists():
            self.test_df = pd.read_parquet(feat_path)
            self.target_cols = [f"target_h{h}" for h in FORECAST_HORIZONS]
            exclude = ["datetime", "station"] + self.target_cols
            self.feature_cols = [c for c in self.test_df.columns if c not in exclude]
            logger.info(f"Loaded test dataset into InferenceEngine ({len(self.test_df)} rows, {len(self.feature_cols)} features)")
        else:
            logger.warning(f"Feature dataset not found at {feat_path}")

    def preload_available_models(self):
        model_loaders = [
            ("persistence", MODELS_DIR / "persistence_model.pkl", PersistenceBaseline.load),
            ("random_forest", MODELS_DIR / "random_forest_model.pkl", RandomForestModel.load),
            ("xgboost", MODELS_DIR / "xgboost_model.pkl", XGBoostModel.load),
            ("lstm", MODELS_DIR / "lstm_model.pt", LSTMModel.load),
            ("cnn_lstm", MODELS_DIR / "cnn_lstm_model.pt", CNNLSTMModel.load),
            ("stcnn_gru", MODELS_DIR / "stcnn_gru_model.pt", STCNNGRUModel.load),
            ("ga_kelm", MODELS_DIR / "ga_kelm_model.pkl", GAKELMModel.load),
        ]
        for name, path, loader in model_loaders:
            if path.exists():
                try:
                    self.models[name] = loader(path)
                    logger.info(f"Loaded model artifact: {name} from {path.name}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")

        # Set up hybrid forecaster if both primary (stcnn_gru or cnn_lstm or lstm) and fallback (ga_kelm) exist
        primary = self.models.get("stcnn_gru") or self.models.get("cnn_lstm") or self.models.get("lstm")
        fallback = self.models.get("ga_kelm")
        if primary and fallback:
            self.models["hybrid_forecaster"] = HybridForecaster(
                primary_model=primary,
                fallback_model=fallback,
                benchmark_mae=38.0,
                fallback_multiplier=1.3
            )
            logger.info("Loaded HybridForecaster into InferenceEngine (primary + GA-KELM fallback)")

    def get_active_model_name(self, requested: Optional[str] = None) -> str:
        if requested and requested in self.models:
            return requested
        for candidate in ["hybrid_forecaster", "stcnn_gru", "cnn_lstm", "lstm", "xgboost", "random_forest", "ga_kelm", "persistence"]:
            if candidate in self.models:
                return candidate
        return "none"

    def predict_for_station(self, station: str,
                            model_name: Optional[str] = None,
                            offset: int = 0) -> Dict[str, Any]:
        """
        Produce a multi-horizon forecast (1, 6, 12, 24, 48, 72h) for a given station.
        Uses historical test replay sequence.
        """
        if self.test_df is None or len(self.test_df) == 0:
            raise RuntimeError("Test feature data is not loaded.")

        st_df = self.test_df[self.test_df["station"] == station].sort_values("datetime")
        if len(st_df) == 0:
            raise ValueError(f"Station '{station}' not found in dataset.")

        # Pick recent row index based on offset
        idx = max(0, min(len(st_df) - 1, len(st_df) - 1 - offset))
        row = st_df.iloc[idx]
        ref_time = str(row["datetime"])

        # Determine current PM2.5 (from lag1h or previous value)
        current_pm25 = float(row.get("PM2.5_lag1h", row.get("PM2.5", 50.0)))

        active_model_key = self.get_active_model_name(model_name)
        model = self.models.get(active_model_key)

        predictions_dict = {}
        if model is None:
            # Fallback naive projection if model file isn't generated yet
            for h in FORECAST_HORIZONS:
                predictions_dict[h] = current_pm25
        else:
            is_hybrid = isinstance(model, HybridForecaster)
            is_pt = hasattr(model, "network")

            # Prepare 24h sequence chunk for deep learning architectures
            seq_len = 24
            start_i = max(0, idx - seq_len + 1)
            seq_chunk = st_df.iloc[start_i:idx + 1][self.feature_cols].fillna(0).values.astype(np.float32)
            if len(seq_chunk) < seq_len:
                padding = np.zeros((seq_len - len(seq_chunk), len(self.feature_cols)), dtype=np.float32)
                seq_chunk = np.vstack([padding, seq_chunk])
            with_batch = np.expand_dims(seq_chunk, axis=0)

            # Prepare tabular row for tree / kernel architectures
            X_feat = row[self.feature_cols].to_frame().T.fillna(0)

            if is_hybrid:
                choice = model._select_model(station)
                if choice == "fallback":
                    preds = model.fallback_model.predict(X_feat)
                    preds_arr = preds[0] if preds.ndim > 1 else preds
                else:
                    import torch
                    t = torch.tensor(with_batch)
                    model.primary_model.network.eval()
                    with torch.no_grad():
                        preds_arr = model.primary_model.network(t).cpu().numpy()[0]
                for i, h in enumerate(FORECAST_HORIZONS):
                    predictions_dict[h] = max(0.0, float(preds_arr[i]))
            elif is_pt:
                import torch
                t = torch.tensor(with_batch)
                model.network.eval()
                with torch.no_grad():
                    pred_arr = model.network(t).cpu().numpy()[0]
                for i, h in enumerate(FORECAST_HORIZONS):
                    predictions_dict[h] = max(0.0, float(pred_arr[i]))
            else:
                preds = model.predict(X_feat)
                preds_arr = preds[0] if preds.ndim > 1 else preds
                for i, h in enumerate(FORECAST_HORIZONS):
                    predictions_dict[h] = max(0.0, float(preds_arr[i]))

        # Generate warnings
        warning_events = self.warning_system.generate_station_warnings(
            station=station,
            current_pm25=current_pm25,
            forecasts_by_horizon=predictions_dict,
            timestamp=ref_time
        )

        return {
            "station": station,
            "reference_datetime": ref_time,
            "model_name": active_model_key,
            "current_pm25": round(current_pm25, 1),
            "forecasts": predictions_dict,
            "warnings": warning_events,
        }
