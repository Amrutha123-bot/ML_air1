"""Model evaluation metrics and benchmark routes."""
import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter
from src.api.models import ModelMetricsResponse, MetricSummaryItem
from src.config.settings import EXPERIMENTS_DIR

router = APIRouter(prefix="/metrics", tags=["Model Evaluation & Metrics"])


@router.get("", response_model=ModelMetricsResponse)
def get_model_metrics():
    report_file = EXPERIMENTS_DIR / "test_evaluation_report.json"
    summary_file = EXPERIMENTS_DIR / "test_metrics_summary.csv"

    summary_items = []
    details_data = None

    if report_file.exists():
        with open(report_file, "r") as f:
            details_data = json.load(f)

    if summary_file.exists():
        import pandas as pd
        df = pd.read_csv(summary_file)
        # Baseline RMSE for skill calculation
        base_rmse = 85.0
        p_row = df[df["model"] == "persistence"]
        if not p_row.empty:
            base_rmse = float(p_row["RMSE"].iloc[0])

        for _, row in df.iterrows():
            m_rmse = float(row["RMSE"])
            skill = 1.0 - (m_rmse / base_rmse) if base_rmse > 0 else 0.0
            summary_items.append(MetricSummaryItem(
                model=str(row["model"]),
                overall_MAE=round(float(row["MAE"]), 4),
                overall_RMSE=round(float(row["RMSE"]), 4),
                overall_R2=round(float(row["R2"]), 4),
                warning_F1=round(float(row.get("Warning_F1", 0.0)), 4),
                skill_score_vs_baseline=round(float(skill), 4)
            ))
    elif details_data and "models" in details_data:
        for m_name, m_info in details_data["models"].items():
            summary_items.append(MetricSummaryItem(
                model=m_name,
                overall_MAE=m_info.get("overall_MAE", 0.0),
                overall_RMSE=m_info.get("overall_RMSE", 0.0),
                overall_R2=m_info.get("overall_R2", 0.0),
                warning_F1=m_info.get("overall_warning_f1", 0.0),
                skill_score_vs_baseline=0.0
            ))

    return ModelMetricsResponse(
        benchmark_dataset="UCI Beijing Multi-Site Air Quality Dataset (Untouched Test Split)",
        summary=summary_items,
        details=details_data
    )
