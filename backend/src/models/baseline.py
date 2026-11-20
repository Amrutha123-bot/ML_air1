"""Persistence baseline model for PM2.5 forecasting."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import joblib
from loguru import logger
from src.config.settings import MODELS_DIR


class PersistenceBaseline:
    """
    Persistence (naive) baseline: predicts future PM2.5 as the most recent known PM2.5.
    For horizon h in [1, 6, 12, 24, 48, 72], prediction = current PM2.5 (or PM2.5_lag1h).
    This provides the mandatory benchmark against which all ML/DL models are compared.
    """

    def __init__(self, horizons: List[int] = [1, 6, 12, 24, 48, 72]):
        self.horizons = horizons
        self.fitted = False
        self.name = "persistence"

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "PersistenceBaseline":
        self.fitted = True
        logger.info("PersistenceBaseline fitted (uses most recent observed PM2.5)")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if "PM2.5_lag1h" in X.columns:
                last_pm25 = X["PM2.5_lag1h"].values
            elif "PM2.5" in X.columns:
                last_pm25 = X["PM2.5"].values
            else:
                last_pm25 = X.iloc[:, 0].values
        elif isinstance(X, np.ndarray):
            last_pm25 = X[:, 0]
        else:
            raise ValueError("Unsupported input format for PersistenceBaseline")

        preds = np.column_stack([last_pm25] * len(self.horizons))
        return np.maximum(preds, 0.0)

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or (MODELS_DIR / "persistence_model.pkl")
        joblib.dump(self, save_path)
        logger.info(f"PersistenceBaseline saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PersistenceBaseline":
        load_path = path or (MODELS_DIR / "persistence_model.pkl")
        return joblib.load(load_path)
