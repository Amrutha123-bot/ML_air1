"""Training orchestrator and experiment tracking for all models."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from loguru import logger
import sys

from src.config.settings import (
    MODELS_DIR, EXPERIMENTS_DIR, FORECAST_HORIZONS, RANDOM_SEEDS
)


def load_feature_data(features_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test feature DataFrames."""
    train = pd.read_parquet(features_dir / "train_features.parquet")
    val   = pd.read_parquet(features_dir / "val_features.parquet")
    test  = pd.read_parquet(features_dir / "test_features.parquet")
    logger.info(f"Loaded datasets: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def get_feature_target_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (feature_cols, target_cols) from DataFrame."""
    target_cols = [f"target_h{h}" for h in FORECAST_HORIZONS]
    exclude = ["datetime", "station"] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols, target_cols


def train_sklearn_model(model, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        model_name: str, seed: int = 42) -> Dict:
    """Train a scikit-learn / XGBoost compatible multi-output model."""
    feature_cols, target_cols = get_feature_target_cols(train_df)
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_cols]
    X_val   = val_df[feature_cols].fillna(0)
    y_val   = val_df[target_cols]

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    preds_val = model.predict(X_val)
    val_mae = float(np.mean(np.abs(preds_val - y_val.values)))
    val_rmse = float(np.sqrt(np.mean((preds_val - y_val.values) ** 2)))

    result = {
        "model_name": model_name,
        "seed": seed,
        "train_rows": len(train_df),
        "val_mae": round(val_mae, 4),
        "val_rmse": round(val_rmse, 4),
        "train_time_s": round(elapsed, 2),
    }
    logger.info(f"[{model_name} | Seed {seed}] val_MAE={val_mae:.4f} val_RMSE={val_rmse:.4f} ({elapsed:.1f}s)")
    return result


def train_pytorch_model(model, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        model_name: str, seed: int = 42,
                        checkpoint_path: Optional[Path] = None) -> Dict:
    """Train a PyTorch sequence model (LSTM, CNN-LSTM, ST-CNN-GRU)."""
    feature_cols, target_cols = get_feature_target_cols(train_df)

    train_sorted = train_df.sort_values(["station", "datetime"])
    val_sorted   = val_df.sort_values(["station", "datetime"])

    X_train = train_sorted[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_sorted[target_cols].fillna(0).values.astype(np.float32)
    X_val   = val_sorted[feature_cols].fillna(0).values.astype(np.float32)
    y_val   = val_sorted[target_cols].fillna(0).values.astype(np.float32)

    start = time.time()
    model.fit(X_train, y_train, X_val, y_val, checkpoint_path=checkpoint_path)
    elapsed = time.time() - start

    preds_val = model.predict(X_val)
    n_valid = len(preds_val)
    y_val_aligned = y_val[-n_valid:] if n_valid < len(y_val) else y_val[:n_valid]
    val_mae = float(np.mean(np.abs(preds_val - y_val_aligned)))
    val_rmse = float(np.sqrt(np.mean((preds_val - y_val_aligned) ** 2)))

    result = {
        "model_name": model_name,
        "seed": seed,
        "best_val_loss": round(float(model.best_val_loss), 6),
        "best_epoch": int(model.best_epoch),
        "val_mae": round(val_mae, 4),
        "val_rmse": round(val_rmse, 4),
        "train_time_s": round(elapsed, 2),
    }
    logger.info(f"[{model_name} | Seed {seed}] best_epoch={model.best_epoch} val_MAE={val_mae:.4f} ({elapsed:.1f}s)")
    return result


class ExperimentTracker:
    """Record all experiment configurations and validation metrics to JSON."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (EXPERIMENTS_DIR / "results.json")
        self.results: List[Dict] = []
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self.results = json.load(f)
            except Exception:
                self.results = []

    def add(self, result: Dict) -> None:
        self.results.append(result)
        self.save()

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def summary_table(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                "model": r.get("model_name"),
                "seed": r.get("seed"),
                "val_mae": r.get("val_mae"),
                "val_rmse": r.get("val_rmse"),
                "train_time_s": r.get("train_time_s"),
            })
        return pd.DataFrame(rows)
