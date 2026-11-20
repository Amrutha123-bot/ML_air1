#!/usr/bin/env python
"""
XGBoost Hyperparameter Optimization & Retraining Pipeline for AeroSense AI V2.

Runs Optuna Bayesian Optimization over key XGBoost tree hyperparameters to
minimize validation MAE across all 6 forecast horizons (1h, 6h, 12h, 24h, 48h, 72h).
Evaluates the optimized model on the held-out test split and saves artifacts.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
import xgboost as xgb

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config.settings import MODELS_DIR, EXPERIMENTS_DIR, FEATURES_DIR, FORECAST_HORIZONS
from src.training.trainer import load_feature_data, get_feature_target_cols
from src.models.xgboost_model import XGBoostModel


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain XGBoost with Optuna HPO")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna HPO trials")
    parser.add_argument("--max-train-samples", type=int, default=60000, help="Max rows for HPO search speed")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    return parser.parse_args()


def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("=" * 65)
    logger.info("AEROSENSE AI V2 — XGBOOST HPO & RETRAINING")
    logger.info(f"Target Horizons: {FORECAST_HORIZONS} hours | Seed: {args.seed}")
    logger.info("=" * 65)

    # 1. Load feature data
    train_df, val_df, test_df = load_feature_data(FEATURES_DIR)
    feature_cols, target_cols = get_feature_target_cols(train_df)

    # Subsample for HPO search efficiency if dataset is large
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_sub = train_df.tail(args.max_train_samples).copy()
    else:
        train_sub = train_df.copy()

    val_sub = val_df.head(20000).copy() if len(val_df) > 20000 else val_df.copy()

    X_tr = train_sub[feature_cols].fillna(0.0)
    y_tr = train_sub[target_cols]
    X_va = val_sub[feature_cols].fillna(0.0)
    y_va = val_sub[target_cols]

    logger.info(f"HPO search on {len(X_tr)} train rows, {len(X_va)} val rows, {len(feature_cols)} features")

    # 2. Define Optuna objective
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        model = XGBoostModel(**params, random_state=args.seed, horizons=FORECAST_HORIZONS)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        mae = float(np.mean(np.abs(preds - y_va.values)))
        return mae

    # 3. Run Optimization
    logger.info(f"Launching Optuna Study ({args.trials} trials)...")
    start_hpo = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    hpo_duration = time.time() - start_hpo

    best_params = study.best_params
    best_val_mae = study.best_value
    logger.info(f"HPO Complete in {hpo_duration:.1f}s | Best Val MAE: {best_val_mae:.4f}")
    logger.info(f"Best Parameters: {best_params}")

    # 4. Save best parameters
    hpo_record = {
        "best_params": best_params,
        "best_val_mae": round(best_val_mae, 4),
        "n_trials": args.trials,
        "search_time_seconds": round(hpo_duration, 2),
        "target_horizons": FORECAST_HORIZONS,
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_DIR / "xgb_hpo_v2.json", "w") as f:
        json.dump(hpo_record, f, indent=2)

    # 5. Retrain on full training set with best hyperparameters
    logger.info("Retraining final XGBoost on full training dataset with best parameters...")
    X_train_full = train_df[feature_cols].fillna(0.0)
    y_train_full = train_df[target_cols]

    final_model = XGBoostModel(**best_params, random_state=args.seed, horizons=FORECAST_HORIZONS)
    t0 = time.time()
    final_model.fit(X_train_full, y_train_full)
    fit_time = time.time() - t0
    logger.info(f"Final model fit completed in {fit_time:.1f}s")

    # 6. Evaluate on held-out test split
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df[target_cols].values

    test_preds = final_model.predict(X_test)
    test_mae = float(np.mean(np.abs(test_preds - y_test)))
    test_rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))
    ss_res = np.sum((y_test - test_preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
    test_r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))

    logger.info("=" * 65)
    logger.info("FINAL TEST EVALUATION RESULTS (XGBoost V2)")
    logger.info(f"Overall MAE:  {test_mae:.4f} µg/m³")
    logger.info(f"Overall RMSE: {test_rmse:.4f} µg/m³")
    logger.info(f"Overall R²:   {test_r2:.4f}")
    logger.info("=" * 65)

    # Per-horizon breakdown
    for i, h in enumerate(FORECAST_HORIZONS):
        h_mae = float(np.mean(np.abs(test_preds[:, i] - y_test[:, i])))
        h_rmse = float(np.sqrt(np.mean((test_preds[:, i] - y_test[:, i]) ** 2)))
        logger.info(f"  Horizon +{h:02d}h -> MAE: {h_mae:6.2f} | RMSE: {h_rmse:6.2f}")

    # 7. Save model checkpoint
    save_path = MODELS_DIR / "xgboost_model_v2.pkl"
    final_model.save(save_path)
    logger.info(f"Saved optimized XGBoost model to {save_path}")


if __name__ == "__main__":
    main()
