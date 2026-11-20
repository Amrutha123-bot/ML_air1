"""Comprehensive scientific model evaluation on untouched test set."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger
import sys

from src.config.settings import (
    MODELS_DIR, EXPERIMENTS_DIR, FEATURES_DIR, FORECAST_HORIZONS, EXPECTED_STATIONS
)
from src.evaluation.metrics import (
    compute_all_metrics, mae, rmse, r2, mape,
    warning_classification_metrics, forecast_skill_score
)


def load_test_features(features_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    feat_dir = features_dir or FEATURES_DIR
    test_df = pd.read_parquet(feat_dir / "test_features.parquet")
    target_cols = [f"target_h{h}" for h in FORECAST_HORIZONS]
    exclude = ["datetime", "station"] + target_cols
    feature_cols = [c for c in test_df.columns if c not in exclude]
    return test_df, feature_cols, target_cols


class ModelEvaluator:
    """Evaluates multiple trained forecasting models against the test benchmark."""

    def __init__(self, test_df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]):
        self.test_df = test_df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.horizons = FORECAST_HORIZONS
        self.stations = sorted(test_df["station"].unique())

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str, baseline_rmse_by_h: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Evaluate (n_samples, n_horizons) predictions.
        Returns:
        - overall_metrics: average across all horizons
        - horizon_breakdown: metrics for each h in [1, 6, 12, 24, 48, 72]
        - residual_stats: residual distribution characteristics
        """
        horizon_results = {}
        maes, rmses, r2s, f1s = [], [], [], []

        for idx, h in enumerate(self.horizons):
            yt_h = y_true[:, idx]
            yp_h = y_pred[:, idx]

            base_rmse = baseline_rmse_by_h.get(h) if baseline_rmse_by_h else None
            m = compute_all_metrics(yt_h, yp_h, baseline_rmse=base_rmse)
            horizon_results[f"h{h}"] = m

            maes.append(m["MAE"])
            rmses.append(m["RMSE"])
            r2s.append(m["R2"])
            f1s.append(m["warning_f1"])

        # Residuals
        residuals = y_pred - y_true
        residual_stats = {
            "mean_residual": round(float(np.mean(residuals)), 4),
            "std_residual": round(float(np.std(residuals)), 4),
            "median_residual": round(float(np.median(residuals)), 4),
            "q25_residual": round(float(np.percentile(residuals, 25)), 4),
            "q75_residual": round(float(np.percentile(residuals, 75)), 4),
        }

        return {
            "model_name": model_name,
            "overall_MAE": round(float(np.mean(maes)), 4),
            "overall_RMSE": round(float(np.mean(rmses)), 4),
            "overall_R2": round(float(np.mean(r2s)), 4),
            "overall_warning_f1": round(float(np.mean(f1s)), 4),
            "horizons": horizon_results,
            "residuals": residual_stats,
        }

    def evaluate_station_breakdown(self, model, model_name: str,
                                   is_pytorch: bool = False) -> Dict[str, Dict[str, float]]:
        """Evaluate performance station-by-station across all 12 stations."""
        station_metrics = {}
        for station in self.stations:
            st_df = self.test_df[self.test_df["station"] == station].sort_values("datetime")
            if len(st_df) == 0:
                continue

            X_st = st_df[self.feature_cols].fillna(0)
            y_st = st_df[self.target_cols].values

            if is_pytorch:
                preds = model.predict(X_st.values.astype(np.float32))
                n_valid = len(preds)
                y_st_aligned = y_st[-n_valid:] if n_valid < len(y_st) else y_st[:n_valid]
            else:
                preds = model.predict(X_st)
                y_st_aligned = y_st

            st_mae = mae(y_st_aligned, preds)
            st_rmse = rmse(y_st_aligned, preds)
            st_r2 = r2(y_st_aligned, preds)
            warn_m = warning_classification_metrics(y_st_aligned[:, 0], preds[:, 0], threshold=75.0)

            station_metrics[station] = {
                "MAE": round(st_mae, 4),
                "RMSE": round(st_rmse, 4),
                "R2": round(st_r2, 4),
                "warning_precision": warn_m["precision"],
                "warning_recall": warn_m["recall"],
                "warning_f1": warn_m["f1_score"],
            }
        return station_metrics


def run_full_test_evaluation(models_dict: Dict[str, Any],
                             test_df: pd.DataFrame,
                             output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run complete benchmark on test set:
    1. Evaluates all models (Persistence baseline first).
    2. Calculates relative skill score improvements over baseline.
    3. Computes horizon-specific breakdown (1h, 6h, 12h, 24h, 48h, 72h).
    4. Computes station-specific breakdown across all 12 stations.
    5. Saves full evaluation report to JSON & CSV.
    """
    out_dir = output_dir or EXPERIMENTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    target_cols = [f"target_h{h}" for h in FORECAST_HORIZONS]
    exclude = ["datetime", "station"] + target_cols
    feature_cols = [c for c in test_df.columns if c not in exclude]

    evaluator = ModelEvaluator(test_df, feature_cols, target_cols)

    # First evaluate Persistence baseline to extract baseline RMSE per horizon
    base_model = models_dict.get("persistence")
    baseline_rmse_by_h = {}
    if base_model:
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_cols].values
        base_preds = base_model.predict(X_test)
        for idx, h in enumerate(FORECAST_HORIZONS):
            baseline_rmse_by_h[h] = rmse(y_test[:, idx], base_preds[:, idx])

    full_report = {}
    station_reports = {}

    for name, model in models_dict.items():
        is_pt = hasattr(model, "network") or hasattr(getattr(model, "primary_model", None), "network")
        logger.info(f"Evaluating {name} on untouched test set ({len(test_df)} records)...")

        if is_pt:
            X_test_arr = test_df[feature_cols].fillna(0).values.astype(np.float32)
            preds = model.predict(X_test_arr)
            y_test_arr = test_df[target_cols].values
            n_valid = len(preds)
            y_test_aligned = y_test_arr[-n_valid:] if n_valid < len(y_test_arr) else y_test_arr[:n_valid]
        else:
            X_test_df = test_df[feature_cols].fillna(0)
            preds = model.predict(X_test_df)
            y_test_aligned = test_df[target_cols].values

        eval_res = evaluator.evaluate_predictions(y_test_aligned, preds, name, baseline_rmse_by_h)
        st_res = evaluator.evaluate_station_breakdown(model, name, is_pytorch=is_pt)

        full_report[name] = eval_res
        station_reports[name] = st_res

    # Save to JSON
    report_file = out_dir / "test_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump({"models": full_report, "stations": station_reports}, f, indent=2)

    # Save summary table to CSV
    summary_rows = []
    for name, res in full_report.items():
        row = {
            "model": name,
            "MAE": res["overall_MAE"],
            "RMSE": res["overall_RMSE"],
            "R2": res["overall_R2"],
            "Warning_F1": res["overall_warning_f1"],
        }
        for h in FORECAST_HORIZONS:
            h_metrics = res["horizons"].get(f"h{h}", {})
            row[f"MAE_{h}h"] = h_metrics.get("MAE")
            row[f"RMSE_{h}h"] = h_metrics.get("RMSE")
            row[f"Skill_{h}h"] = h_metrics.get("skill_score", 0.0)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values("MAE")
    csv_file = out_dir / "test_metrics_summary.csv"
    df_summary.to_csv(csv_file, index=False)
    logger.info(f"Evaluation report saved to {report_file} and {csv_file}")

    return {"report_file": report_file, "summary_file": csv_file, "summary": df_summary}
