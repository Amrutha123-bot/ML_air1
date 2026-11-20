"""Scientific plotting and visualization routines for air quality forecasting."""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server/background execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from src.config.settings import DOCS_DIR

RESULTS_PLOT_DIR = DOCS_DIR / "results"
RESULTS_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_forecast_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "PM2.5 Forecast vs Actual",
                             lead_time_h: int = 24,
                             max_points: int = 300,
                             save_path: Optional[Path] = None):
    """Plot actual vs predicted time series snippet."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n = min(len(y_true), max_points)
    ax.plot(range(n), y_true[:n], label="Actual Observed PM2.5", color="#1f77b4", linewidth=1.8)
    ax.plot(range(n), y_pred[:n], label=f"Predicted PM2.5 (+{lead_time_h}h)", color="#d62728", linestyle="--", linewidth=1.8)
    ax.axhline(75.0, color="orange", linestyle=":", label="Warning Threshold (75 μg/m³)")
    ax.axhline(150.0, color="red", linestyle=":", label="High Alert (150 μg/m³)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Step (Hours)", fontsize=11)
    ax.set_ylabel("PM2.5 Concentration (μg/m³)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = save_path or (RESULTS_PLOT_DIR / f"forecast_vs_actual_h{lead_time_h}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_horizon_degradation(summary_df: pd.DataFrame,
                             save_path: Optional[Path] = None):
    """Plot MAE and RMSE degradation curves across lead times (1h to 72h)."""
    horizons = [1, 6, 12, 24, 48, 72]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for _, row in summary_df.iterrows():
        model = row["model"]
        maes = [row.get(f"MAE_{h}h", np.nan) for h in horizons]
        rmses = [row.get(f"RMSE_{h}h", np.nan) for h in horizons]

        ax1.plot(horizons, maes, marker="o", linewidth=2, label=model)
        ax2.plot(horizons, rmses, marker="s", linewidth=2, label=model)

    ax1.set_title("Forecast Error (MAE) vs Lead Time", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Lead Time (Hours)", fontsize=11)
    ax1.set_ylabel("MAE (μg/m³)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Forecast Error (RMSE) vs Lead Time", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Lead Time (Hours)", fontsize=11)
    ax2.set_ylabel("RMSE (μg/m³)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out = save_path or (RESULTS_PLOT_DIR / "horizon_error_degradation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_warning_confusion_matrix(cm_dict: Dict[str, int],
                                  model_name: str = "Best Model",
                                  save_path: Optional[Path] = None):
    """Plot 2x2 confusion matrix heatmap for alert classification."""
    matrix = np.array([
        [cm_dict["TN"], cm_dict["FP"]],
        [cm_dict["FN"], cm_dict["TP"]]
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal (<75)", "Alert (>=75)"],
                yticklabels=["Normal (<75)", "Alert (>=75)"])
    ax.set_title(f"Early Warning Confusion Matrix ({model_name})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Air Quality Status", fontsize=11)
    ax.set_xlabel("Predicted Air Quality Status", fontsize=11)
    plt.tight_layout()
    out = save_path or (RESULTS_PLOT_DIR / f"confusion_matrix_{model_name}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_residual_analysis(residuals: np.ndarray,
                           model_name: str = "Model",
                           save_path: Optional[Path] = None):
    """Plot residual histogram and QQ scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals.flatten()[:10000], kde=True, ax=ax1, color="#2ca02c", bins=50)
    ax1.axvline(0, color="red", linestyle="--", label="Zero Error")
    ax1.set_title(f"Residual Distribution ({model_name})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Residual: Predicted - Actual (μg/m³)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.legend()

    # Scatter of abs error
    sample = residuals.flatten()[:2000]
    ax2.scatter(range(len(sample)), sample, alpha=0.4, s=8, color="#1f77b4")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_title(f"Residual Scatter across Sequences ({model_name})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sequence Index", fontsize=11)
    ax2.set_ylabel("Error (μg/m³)", fontsize=11)

    plt.tight_layout()
    out = save_path or (RESULTS_PLOT_DIR / f"residuals_{model_name}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
