#!/usr/bin/env python
"""
Residual Analysis & Error Distribution Diagnostics for AeroSense AI V2.

Computes prediction residuals (e = y_pred - y_true) on the held-out test dataset
across all candidate forecasting architectures to quantify systematic bias,
variance, heteroskedasticity, and horizon degradation.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config.settings import MODELS_DIR, EXPERIMENTS_DIR, FEATURES_DIR, FORECAST_HORIZONS
from src.training.trainer import get_feature_target_cols
from src.models import (
    PersistenceBaseline, RandomForestModel, XGBoostModel,
    LSTMModel, CNNLSTMModel, STCNNGRUModel, GAKELMModel
)


def main():
    logger.info("=" * 65)
    logger.info("AEROSENSE AI V2 — RESIDUAL DIAGNOSTICS & ERROR AUDIT")
    logger.info("=" * 65)

    test_path = FEATURES_DIR / "test_features.parquet"
    if not test_path.exists():
        logger.error(f"Test feature file not found at {test_path}")
        return

    test_df = pd.read_parquet(test_path)
    feature_cols, target_cols = get_feature_target_cols(test_df)
    logger.info(f"Loaded test dataset: {len(test_df)} rows, {len(feature_cols)} features")

    # Sample if needed for sequence evaluation speed
    test_eval_df = test_df.copy()
    X_tab = test_eval_df[feature_cols].fillna(0.0)
    y_true = test_eval_df[target_cols].values

    models_to_evaluate = [
        ("xgboost", MODELS_DIR / "xgboost_model.pkl", XGBoostModel.load, "tabular"),
        ("random_forest", MODELS_DIR / "random_forest_model.pkl", RandomForestModel.load, "tabular"),
        ("persistence", MODELS_DIR / "persistence_model.pkl", PersistenceBaseline.load, "tabular"),
        ("ga_kelm", MODELS_DIR / "ga_kelm_model.pkl", GAKELMModel.load, "tabular"),
        ("lstm", MODELS_DIR / "lstm_model.pt", LSTMModel.load, "pytorch"),
        ("cnn_lstm", MODELS_DIR / "cnn_lstm_model.pt", CNNLSTMModel.load, "pytorch"),
        ("stcnn_gru", MODELS_DIR / "stcnn_gru_model.pt", STCNNGRUModel.load, "pytorch"),
    ]

    diagnostics = {}
    summary_rows = []

    for name, path, loader, mtype in models_to_evaluate:
        if not path.exists():
            logger.warning(f"Skipping {name}: artifact not found at {path}")
            continue

        logger.info(f"Computing residuals for {name} ({path.name})...")
        try:
            model = loader(path)
            if mtype == "tabular":
                preds = model.predict(X_tab)
            elif name == "stcnn_gru":
                # ST-CNN-GRU predicts (B, 12, 6) across the 12 stations
                raw_preds = model.predict(test_eval_df)
                if raw_preds.ndim == 3 and raw_preds.shape[1] == 12:
                    # Flatten (B, 12, 6) into (B*12, 6)
                    preds = raw_preds.reshape(-1, len(FORECAST_HORIZONS))
                else:
                    preds = raw_preds
            else:
                # LSTM / CNN-LSTM require numeric arrays without Timestamp
                X_num = test_eval_df[feature_cols].fillna(0.0).values.astype(np.float32)
                preds = model.predict(X_num)

            preds = np.asarray(preds, dtype=np.float32)
            n_eval = min(len(preds), len(y_true))
            p_slice = preds[:n_eval]
            y_slice = y_true[:n_eval]

            # Residuals: (y_pred - y_true)
            res = p_slice - y_slice

            # Metrics for 1h and 24h horizons
            h1_idx = FORECAST_HORIZONS.index(1) if 1 in FORECAST_HORIZONS else 0
            h24_idx = FORECAST_HORIZONS.index(24) if 24 in FORECAST_HORIZONS else min(3, len(FORECAST_HORIZONS)-1)

            res_h1 = res[:, h1_idx]
            res_h24 = res[:, h24_idx]

            bias_h1 = float(np.mean(res_h1))
            std_h1 = float(np.std(res_h1))
            mae_h1 = float(np.mean(np.abs(res_h1)))
            p5_h1 = float(np.percentile(res_h1, 5))
            p95_h1 = float(np.percentile(res_h1, 95))

            bias_h24 = float(np.mean(res_h24))
            std_h24 = float(np.std(res_h24))
            mae_h24 = float(np.mean(np.abs(res_h24)))

            diagnostics[name] = {
                "sample_count": n_eval,
                "horizon_1h": {
                    "bias_mean": round(bias_h1, 4),
                    "std_dev": round(std_h1, 4),
                    "mae": round(mae_h1, 4),
                    "p05": round(p5_h1, 4),
                    "p95": round(p95_h1, 4),
                },
                "horizon_24h": {
                    "bias_mean": round(bias_h24, 4),
                    "std_dev": round(std_h24, 4),
                    "mae": round(mae_h24, 4),
                }
            }

            summary_rows.append({
                "model": name,
                "MAE_1h": round(mae_h1, 2),
                "Bias_1h": round(bias_h1, 2),
                "Std_1h": round(std_h1, 2),
                "P05_1h": round(p5_h1, 2),
                "P95_1h": round(p95_h1, 2),
                "MAE_24h": round(mae_h24, 2),
                "Bias_24h": round(bias_h24, 2),
                "Std_24h": round(std_h24, 2),
            })

            # Save sample residuals for XGBoost and RF to CSV
            if name in ("xgboost", "random_forest", "ga_kelm"):
                res_df = pd.DataFrame({
                    "datetime": test_eval_df["datetime"].iloc[:n_eval].values,
                    "station": test_eval_df["station"].iloc[:n_eval].values,
                    "y_true_1h": y_slice[:, h1_idx],
                    "y_pred_1h": p_slice[:, h1_idx],
                    "residual_1h": res_h1,
                    "y_true_24h": y_slice[:, h24_idx],
                    "y_pred_24h": p_slice[:, h24_idx],
                    "residual_24h": res_h24,
                })
                res_path = EXPERIMENTS_DIR / f"residuals_{name}.csv"
                res_df.head(5000).to_csv(res_path, index=False)
                logger.info(f"Saved residual sample to {res_path.name}")

        except Exception as e:
            logger.error(f"Error computing residuals for {name}: {e}")

    # Output summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = EXPERIMENTS_DIR / "residuals_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\nSaved summary CSV to {summary_csv}")
    print("\n" + summary_df.to_string(index=False) + "\n")

    diag_json = EXPERIMENTS_DIR / "residual_analysis.json"
    with open(diag_json, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info(f"Saved diagnostic details to {diag_json}")


if __name__ == "__main__":
    main()
