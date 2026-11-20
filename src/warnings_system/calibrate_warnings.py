"""
Warning Threshold Calibration.

Calibrates alert decision boundaries on validation data to optimize
the trade-off between Recall (Detection Rate) and False Alarm Rate (FAR).
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import EXPERIMENTS_DIR, FORECAST_HORIZONS
from src.evaluation.metrics import warning_classification_metrics


def calibrate_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_event_threshold: float = 75.0,  # e.g., Unhealthy for Sensitive Groups (75 ug/m^3)
    candidate_offsets: List[float] = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
) -> Dict:
    """
    Search over candidate decision thresholds around target_event_threshold
    to maximize F1-score on validation data.
    """
    best_f1 = -1.0
    best_threshold = target_event_threshold
    best_metrics = {}
    history = []

    true_warn = (y_true >= target_event_threshold).astype(int)

    for offset in candidate_offsets:
        cand = target_event_threshold + offset
        pred_warn = (y_pred >= cand).astype(int)

        tp = int(np.sum((true_warn == 1) & (pred_warn == 1)))
        fp = int(np.sum((true_warn == 0) & (pred_warn == 1)))
        tn = int(np.sum((true_warn == 0) & (pred_warn == 0)))
        fn = int(np.sum((true_warn == 1) & (pred_warn == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        far = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0

        m = {
            "decision_threshold": cand,
            "target_threshold": target_event_threshold,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_alarm_rate": round(far, 4),
        }
        history.append(m)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = cand
            best_metrics = m

    return {
        "target_event_threshold": target_event_threshold,
        "calibrated_threshold": best_threshold,
        "best_metrics": best_metrics,
        "history": history
    }


def run_warning_calibration(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    save_path: Path = EXPERIMENTS_DIR / "warning_calibration.json"
) -> Dict:
    """
    Run calibration across key regulatory tiers (75 ug/m^3 and 115 ug/m^3).
    """
    logger.info("Running warning threshold calibration on validation predictions...")
    if y_pred_val.ndim == 2:
        h24_idx = 3 if y_pred_val.shape[1] > 3 else 0
        y_t = y_true_val[:, h24_idx]
        y_p = y_pred_val[:, h24_idx]
    else:
        y_t = y_true_val
        y_p = y_pred_val

    calib_75 = calibrate_threshold(y_t, y_p, target_event_threshold=75.0)
    calib_115 = calibrate_threshold(y_t, y_p, target_event_threshold=115.0)

    results = {
        "horizon_evaluated": "24h",
        "tier_75_ugm3": calib_75,
        "tier_115_ugm3": calib_115,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Calibrated thresholds saved to {save_path}:")
    logger.info(f"  Tier 75 ug/m^3 -> Decision Threshold: {calib_75['calibrated_threshold']} ug/m^3 (F1: {calib_75['best_metrics']['f1_score']:.4f})")
    logger.info(f"  Tier 115 ug/m^3 -> Decision Threshold: {calib_115['calibrated_threshold']} ug/m^3 (F1: {calib_115['best_metrics']['f1_score']:.4f})")
    return results
