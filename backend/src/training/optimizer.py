"""Hyperparameter optimization with Optuna."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import optuna
from optuna.samplers import TPESampler
from loguru import logger
import sys

from src.config.settings import EXPERIMENTS_DIR, FORECAST_HORIZONS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_target_cols(df: pd.DataFrame):
    target_cols = [f"target_h{h}" for h in FORECAST_HORIZONS]
    exclude = ["datetime", "station"] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols, target_cols


def optimize_random_forest(train_df: pd.DataFrame, val_df: pd.DataFrame,
                           n_trials: int = 15, seed: int = 42) -> Dict:
    """Optuna HPO for Random Forest."""
    from src.models.random_forest import RandomForestModel
    feature_cols, target_cols = get_feature_target_cols(train_df)
    n_hpo = min(30000, len(train_df))
    train_sub = train_df.sample(n=n_hpo, random_state=seed)
    X_tr = train_sub[feature_cols].fillna(0)
    y_tr = train_sub[target_cols]
    X_va = val_df[feature_cols].fillna(0)
    y_va = val_df[target_cols]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 10, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "n_jobs": -1,
            "random_state": seed,
        }
        model = RandomForestModel(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        return float(np.mean(np.abs(preds - y_va.values)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = seed
    logger.info(f"RF best HPO params: {best} (val_MAE={study.best_value:.4f})")
    result = {"model": "random_forest", "best_params": best, "best_val_mae": study.best_value}
    path = EXPERIMENTS_DIR / "rf_hpo.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def optimize_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     n_trials: int = 15, seed: int = 42) -> Dict:
    """Optuna HPO for XGBoost."""
    from src.models.xgboost_model import XGBoostModel
    feature_cols, target_cols = get_feature_target_cols(train_df)
    n_hpo = min(30000, len(train_df))
    train_sub = train_df.sample(n=n_hpo, random_state=seed)
    X_tr = train_sub[feature_cols].fillna(0)
    y_tr = train_sub[target_cols]
    X_va = val_df[feature_cols].fillna(0)
    y_va = val_df[target_cols]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "n_jobs": -1,
            "random_state": seed,
        }
        model = XGBoostModel(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        return float(np.mean(np.abs(preds - y_va.values)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = seed
    logger.info(f"XGB best HPO params: {best} (val_MAE={study.best_value:.4f})")
    result = {"model": "xgboost", "best_params": best, "best_val_mae": study.best_value}
    path = EXPERIMENTS_DIR / "xgb_hpo.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def optimize_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  n_trials: int = 5, seed: int = 42) -> Dict:
    """Optuna HPO for LSTM."""
    from src.models.lstm import LSTMModel
    feature_cols, target_cols = get_feature_target_cols(train_df)
    train_sorted = train_df.sort_values(["station", "datetime"])
    val_sorted   = val_df.sort_values(["station", "datetime"])
    n_hpo = min(20000, len(train_sorted))
    X_tr = train_sorted[feature_cols].fillna(0).values[:n_hpo].astype(np.float32)
    y_tr = train_sorted[target_cols].fillna(0).values[:n_hpo].astype(np.float32)
    X_va = val_sorted[feature_cols].fillna(0).values.astype(np.float32)
    y_va = val_sorted[target_cols].fillna(0).values.astype(np.float32)
    input_size = X_tr.shape[1]

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        seq_len = 24

        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=lr,
            seq_len=seq_len,
            max_epochs=10,
            patience=3,
            random_state=seed,
        )
        model.fit(X_tr, y_tr, X_va, y_va)
        preds = model.predict(X_va)
        n_v = len(preds)
        y_aligned = y_va[-n_v:] if n_v < len(y_va) else y_va[:n_v]
        return float(np.mean(np.abs(preds - y_aligned)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    logger.info(f"LSTM best HPO params: {best} (val_MAE={study.best_value:.4f})")
    result = {"model": "lstm", "best_params": best, "best_val_mae": study.best_value}
    path = EXPERIMENTS_DIR / "lstm_hpo.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_best_params(model_name: str) -> Dict:
    """Load saved HPO params or return defaults."""
    hpo_file = EXPERIMENTS_DIR / f"{model_name}_hpo.json"
    if hpo_file.exists():
        with open(hpo_file, "r") as f:
            data = json.load(f)
        return data.get("best_params", {})
    return {}
