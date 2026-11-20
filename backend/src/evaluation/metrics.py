"""Scientific evaluation metrics for forecasting and early-warning validation."""
import numpy as np
from typing import Dict, List, Tuple


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Mean Absolute Percentage Error with numerical floor protection."""
    mask = np.abs(y_true) >= eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def forecast_skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """
    Skill score relative to reference baseline (Persistence):
    Skill = 1 - (RMSE_model / RMSE_baseline)
    Positive indicates genuine improvement over naive persistence.
    """
    if rmse_baseline == 0:
        return 0.0
    return float(1.0 - (rmse_model / rmse_baseline))


def warning_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   threshold: float = 75.0) -> Dict[str, float]:
    """
    Evaluate early-warning classification performance relative to PM2.5 threshold.
    Default threshold: 75.0 ug/m^3 (China Grade II / Sensitive Group alert level).
    
    Metrics:
    - TP, FP, TN, FN
    - Precision: TP / (TP + FP)
    - Recall / Hit Rate: TP / (TP + FN)
    - F1-Score: Harmonic mean of precision & recall
    - False Alarm Rate (FAR): FP / (FP + TP) or FP / (FP + TN)
    - Missed Warning Rate (MWR): FN / (TP + FN)
    - Accuracy: (TP + TN) / Total
    """
    true_warn = (y_true >= threshold).astype(int)
    pred_warn = (y_pred >= threshold).astype(int)

    tp = int(np.sum((true_warn == 1) & (pred_warn == 1)))
    fp = int(np.sum((true_warn == 0) & (pred_warn == 1)))
    tn = int(np.sum((true_warn == 0) & (pred_warn == 0)))
    fn = int(np.sum((true_warn == 1) & (pred_warn == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
    mwr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / max(1, len(true_warn)))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_alarm_rate": round(far, 4),
        "missed_warning_rate": round(mwr, 4),
        "accuracy": round(accuracy, 4),
        "specificity": round(specificity, 4),
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        baseline_rmse: float = None) -> Dict[str, float]:
    """Compute regression metrics and warning metrics."""
    m_mae = mae(y_true, y_pred)
    m_rmse = rmse(y_true, y_pred)
    m_r2 = r2(y_true, y_pred)
    m_mape = mape(y_true, y_pred)

    res = {
        "MAE": round(m_mae, 4),
        "RMSE": round(m_rmse, 4),
        "R2": round(m_r2, 4),
        "MAPE": round(m_mape, 2),
    }
    if baseline_rmse is not None:
        res["skill_score"] = round(forecast_skill_score(m_rmse, baseline_rmse), 4)

    # Add warning performance at standard alert threshold 75 ug/m^3
    warn_stats = warning_classification_metrics(y_true, y_pred, threshold=75.0)
    res.update({
        "warning_precision": warn_stats["precision"],
        "warning_recall": warn_stats["recall"],
        "warning_f1": warn_stats["f1_score"],
        "false_alarm_rate": warn_stats["false_alarm_rate"],
        "missed_warning_rate": warn_stats["missed_warning_rate"],
    })
    return res
