"""Delhi-only inference engine for AeroSense AI."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

# ── Path constants (relative to backend/) ───────────────────────────────────
_BACKEND = Path(__file__).parent.parent.parent.resolve()
_DATA_CSV = _BACKEND / "data" / "delhi_synthetic_air_quality.csv"
_MODELS_DIR = _BACKEND / "artifacts" / "models"
_SCALERS_DIR = _BACKEND / "artifacts" / "scalers"
_EXPERIMENTS_DIR = _BACKEND / "artifacts" / "experiments"

_SCALER_PATH = _SCALERS_DIR / "delhi_feature_scaler.pkl"
_FEATURE_COLS_PATH = _EXPERIMENTS_DIR / "feature_columns.json"

HORIZONS = [1, 6, 12, 24, 48, 72]
HORIZON_LABELS = ["+1h", "+6h", "+12h", "+24h", "+48h", "+72h"]
SEQ_LEN = 24  # lookback window for DL models

# ── CPCB AQI category helper ─────────────────────────────────────────────────
def pm25_to_category(pm25: float) -> str:
    if pm25 < 30:
        return "Good"
    elif pm25 < 60:
        return "Satisfactory"
    elif pm25 < 90:
        return "Moderate"
    elif pm25 < 120:
        return "Poor"
    elif pm25 < 250:
        return "Very Poor"
    else:
        return "Severe"


# ── Feature builder from raw CSV rows ────────────────────────────────────────
def _build_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Given the raw CSV dataframe (sorted chronologically), compute all lag/roll
    features matching feature_cols. Returns a row-aligned feature DataFrame.
    Missing columns are zero-filled.
    """
    out = pd.DataFrame(index=df.index)

    # ---- Lag features ----
    lag_sources = {
        "PM2.5": "PM2.5",
        "PM10": "PM10",
        "NO2": "NO2",
        "Temperature": "Temperature",
        "Wind_Speed": "Wind_Speed",
    }
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72]
    for col, src in lag_sources.items():
        if src in df.columns:
            for h in lag_hours:
                out[f"{col}_lag_{h}h"] = df[src].shift(h)

    # ---- Rolling features ----
    if "PM2.5" in df.columns:
        for w in [6, 24, 72]:
            out[f"PM2.5_roll_mean_{w}h"] = df["PM2.5"].shift(1).rolling(w, min_periods=1).mean()
            out[f"PM2.5_roll_std_{w}h"] = df["PM2.5"].shift(1).rolling(w, min_periods=1).std().fillna(0)

    # ---- Interaction features ----
    if "PM2.5" in df.columns and "PM10" in df.columns:
        pm10_safe = df["PM10"].replace(0, np.nan)
        out["pm25_to_pm10_ratio"] = df["PM2.5"] / pm10_safe
    if "NO2" in df.columns and "NOx" in df.columns:
        nox_safe = df["NOx"].replace(0, np.nan)
        out["no2_to_nox_ratio"] = df["NO2"] / nox_safe
    if "Temperature" in df.columns and "Wind_Speed" in df.columns:
        out["temp_wind_interaction"] = df["Temperature"] * df["Wind_Speed"]
    if "Humidity" in df.columns and "Wind_Speed" in df.columns:
        wind_safe = df["Wind_Speed"].replace(0, np.nan)
        out["humidity_wind_stagnation"] = df["Humidity"] / wind_safe

    # ---- Cyclical time features ----
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        out["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
        out["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
        out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # Align to exactly the requested feature columns (zero-fill unknowns)
    result = pd.DataFrame(index=out.index)
    for fc in feature_cols:
        if fc in out.columns:
            result[fc] = out[fc]
        else:
            result[fc] = 0.0

    result = result.fillna(0.0)
    return result


# ── LSTM / GRU / CNN-LSTM inference networks ─────────────────────────────────
# We re-define lightweight network classes here so inference.py is self-contained
# (avoids importing from models/* which may carry side-effects).

class _LSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, n_horizons=6):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, n_horizons),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class _GRUNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, n_horizons=6):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, n_horizons),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class _CNNLSTMNet(torch.nn.Module):
    def __init__(self, input_size, cnn_channels=32, lstm_hidden=64, lstm_layers=1, dropout=0.2, n_horizons=6):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(cnn_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(cnn_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.lstm = torch.nn.LSTM(cnn_channels * 2, lstm_hidden, lstm_layers,
                                  batch_first=True,
                                  dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden, lstm_hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(lstm_hidden // 2, n_horizons),
        )
    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_perm).permute(0, 2, 1)
        out, _ = self.lstm(cnn_out)
        return self.fc(self.dropout(out[:, -1, :]))


# ── Main Inference Engine ────────────────────────────────────────────────────

class DelhiInferenceEngine:
    """
    Lazy-loading inference engine for all Delhi models.
    Thread-safe for read operations (models loaded once per process).
    """

    def __init__(self):
        self._model_cache: Dict[str, object] = {}
        self._scaler = None
        self._feature_cols: Optional[list] = None
        self._df: Optional[pd.DataFrame] = None

    # ── Internal loaders ──────────────────────────────────────────────────

    def _get_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(_DATA_CSV)
        return self._df

    def _get_scaler(self):
        if self._scaler is None:
            self._scaler = joblib.load(_SCALER_PATH)
        return self._scaler

    def _get_feature_cols(self) -> list:
        if self._feature_cols is None:
            with open(_FEATURE_COLS_PATH) as f:
                self._feature_cols = json.load(f)
        return self._feature_cols

    def _load_sklearn(self, path: Path):
        return joblib.load(path)

    def _load_dl(self, path: Path, model_type: str, input_size: int):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        horizons = cfg.get("horizons", HORIZONS)
        n_horizons = len(horizons)

        if model_type == "lstm":
            net = _LSTMNet(
                input_size=input_size,
                hidden_size=cfg.get("hidden_size", 128),
                num_layers=cfg.get("num_layers", 2),
                n_horizons=n_horizons,
            )
        elif model_type == "gru":
            net = _GRUNet(
                input_size=input_size,
                hidden_size=cfg.get("hidden_size", 128),
                num_layers=cfg.get("num_layers", 2),
                n_horizons=n_horizons,
            )
        elif model_type == "cnn_lstm":
            net = _CNNLSTMNet(
                input_size=input_size,
                cnn_channels=cfg.get("cnn_channels", 64),
                lstm_hidden=cfg.get("lstm_hidden", 128),
                lstm_layers=cfg.get("lstm_layers", 2),
                n_horizons=n_horizons,
            )
        else:
            raise ValueError(f"Unknown DL type: {model_type}")

        net.load_state_dict(checkpoint["network_state"])
        net.eval()
        return net, horizons

    def _get_model(self, model_name: str):
        """Return cached model or load from disk. Raises FileNotFoundError if missing."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        model_map = {
            "random_forest": (_MODELS_DIR / "random_forest_delhi.pkl", "sklearn"),
            "xgboost": (_MODELS_DIR / "xgboost_delhi.pkl", "sklearn"),
            "lstm": (_MODELS_DIR / "lstm_delhi.pt", "lstm"),
            "gru": (_MODELS_DIR / "gru_delhi.pt", "gru"),
            "cnn_lstm": (_MODELS_DIR / "cnn_lstm_delhi.pt", "cnn_lstm"),
        }

        if model_name not in model_map:
            raise KeyError(f"Unknown model: {model_name}")

        path, kind = model_map[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info(f"Loading model '{model_name}' from {path}")
        if kind == "sklearn":
            m = self._load_sklearn(path)
        else:
            feature_cols = self._get_feature_cols()
            m = self._load_dl(path, kind, len(feature_cols))

        self._model_cache[model_name] = m
        return m

    # ── Feature matrix for the last N rows ───────────────────────────────

    def _prepare_features(self, offset: int = 0) -> np.ndarray:
        """Return scaled feature array from recent CSV rows, shape (N, n_features)."""
        feature_cols = self._get_feature_cols()
        df = self._get_df()

        # Build features over the full df, then take the tail we need
        min_rows = 100  # need enough history for lags
        feat_df = _build_features(df, feature_cols)

        # Drop the last `offset` rows (future simulation)
        if offset > 0:
            feat_df = feat_df.iloc[:-offset]

        # Take last (SEQ_LEN + 1) rows for inference
        window = feat_df.iloc[-(SEQ_LEN + 1):]
        X = window.values.astype(np.float32)

        scaler = self._get_scaler()
        X_scaled = scaler.transform(X)
        return X_scaled

    # ── Baseline predictors ──────────────────────────────────────────────

    def _baseline_persistence(self, df_tail: pd.DataFrame) -> Dict[str, float]:
        last_pm25 = float(df_tail["PM2.5"].iloc[-1])
        return {lbl: last_pm25 for lbl in HORIZON_LABELS}

    def _baseline_seasonal_persistence(self, df: pd.DataFrame, offset: int) -> Dict[str, float]:
        """Same hour, day-of-week from last week."""
        if offset > 0:
            df = df.iloc[:-offset]
        last_idx = len(df) - 1
        result = {}
        for h, lbl in zip(HORIZONS, HORIZON_LABELS):
            # Look back 7 days (168h) + horizon h
            lookback = 168 + h
            idx = last_idx - lookback
            if idx >= 0:
                result[lbl] = max(0.0, float(df["PM2.5"].iloc[idx]))
            else:
                result[lbl] = max(0.0, float(df["PM2.5"].iloc[0]))
        return result

    def _baseline_moving_average(self, df_tail: pd.DataFrame) -> Dict[str, float]:
        avg = float(df_tail["PM2.5"].tail(24).mean())
        return {lbl: max(0.0, avg) for lbl in HORIZON_LABELS}

    # ── Public API ───────────────────────────────────────────────────────

    def get_current_data(self) -> dict:
        """Return the last row of the CSV as a dict."""
        df = self._get_df()
        row = df.iloc[-1].to_dict()
        return row

    def predict(self, model_name: str, offset: int = 0) -> Dict[str, float]:
        """
        Return PM2.5 forecasts for all 6 horizons.
        Returns {"+1h": float, "+6h": float, "+12h": float, "+24h": float, "+48h": float, "+72h": float}
        """
        df = self._get_df()
        df_window = df.iloc[:-offset] if offset > 0 else df
        df_tail = df_window.tail(24 + SEQ_LEN)

        # --- Baselines (no model loading) ---
        if model_name == "persistence":
            return self._baseline_persistence(df_window.tail(1))
        elif model_name == "seasonal_persistence":
            return self._baseline_seasonal_persistence(df, offset)
        elif model_name == "moving_average":
            return self._baseline_moving_average(df_window)

        # linear_regression / ridge / decision_tree → not in artifacts, return error signal
        UNSUPPORTED_SKLEARN = {"linear_regression", "ridge", "decision_tree"}
        if model_name in UNSUPPORTED_SKLEARN:
            raise FileNotFoundError(f"Model '{model_name}' not trained for Delhi")

        # Load real model
        model_obj = self._get_model(model_name)
        X_scaled = self._prepare_features(offset)
        feature_cols = self._get_feature_cols()

        # --- sklearn (RF, XGB): use last row features ---
        if model_name in ("random_forest", "xgboost"):
            x_last = X_scaled[-1:, :]  # shape (1, n_features)
            # These models store MultiOutputRegressor; predict returns (1, 6)
            raw = model_obj.predict(x_last)
            if hasattr(raw, "tolist"):
                raw = raw[0]
            vals = [max(0.0, float(v)) for v in raw]
            return dict(zip(HORIZON_LABELS, vals))

        # --- DL models ---
        net, horizons = model_obj
        if len(X_scaled) < SEQ_LEN:
            raise ValueError("Not enough history to build sequence")
        seq = X_scaled[-SEQ_LEN:]  # (SEQ_LEN, n_features)
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, SEQ_LEN, n_features)
        with torch.no_grad():
            raw = net(x_tensor).cpu().numpy()[0]  # (n_horizons,)
        h_labels = [f"+{h}h" for h in horizons]
        return {lbl: max(0.0, float(v)) for lbl, v in zip(h_labels, raw)}

    def get_active_model_name(self) -> str:
        """Return the name of the default active model."""
        return "random_forest"

    def get_available_models(self) -> list:
        """Return all supported model names."""
        return ["random_forest", "xgboost", "lstm", "gru", "cnn_lstm",
                "persistence", "seasonal_persistence", "moving_average"]

    # ── Singleton accessor ─────────────────────────────────────────────────

    _instance: Optional["DelhiInferenceEngine"] = None

    @classmethod
    def get_instance(cls) -> "DelhiInferenceEngine":
        """Return the process-level singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance