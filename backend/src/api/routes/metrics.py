"""Model evaluation metrics, leaderboard, confusion matrix, and training fraction routes."""
from typing import Dict, Any, List
import json
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/metrics", tags=["Model Evaluation & Metrics"])

_EXP_DIR = Path(__file__).parent.parent.parent.parent / "artifacts" / "experiments"
_MASTER_CSV = _EXP_DIR / "master_model_comparison.csv"
_CONF_NORM_CSV = _EXP_DIR / "confusion_matrix_normalized.csv"
_CLASS_JSON = _EXP_DIR / "classification_metrics.json"
_FRACTIONS_JSON = _EXP_DIR / "training_fraction_experiments.json"


@router.get("")
def get_model_metrics():
    """Return all model evaluation records across horizons."""
    if not _MASTER_CSV.exists():
        raise HTTPException(status_code=404, detail="Master model comparison CSV not found")
    df = pd.read_csv(_MASTER_CSV)
    return {
        "benchmark_dataset": "AeroSense AI Delhi Air Quality Dataset (35,064 records)",
        "count": len(df),
        "metrics": df.to_dict(orient="records"),
    }


@router.get("/leaderboard")
def get_leaderboard():
    """Return 24h horizon model leaderboard sorted by test MAE."""
    if not _MASTER_CSV.exists():
        raise HTTPException(status_code=404, detail="Master model comparison CSV not found")
    df = pd.read_csv(_MASTER_CSV)
    h24_df = df[df["horizon"] == "+24h"].sort_values("mae").copy()
    h24_df["rank"] = range(1, len(h24_df) + 1)
    return {
        "horizon": "+24h",
        "champion_model": "Random Forest Ensemble (or XGBoost)",
        "leaderboard": h24_df.to_dict(orient="records")
    }


@router.get("/confusion")
def get_confusion_matrix():
    """Return CPCB 6-tier confusion matrix and classification scores for champion model (+24h)."""
    if not _CONF_NORM_CSV.exists() or not _CLASS_JSON.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix data not found")
    
    cm_df = pd.read_csv(_CONF_NORM_CSV, index_col=0)
    labels = list(cm_df.columns)
    matrix = cm_df.values.tolist()

    with open(_CLASS_JSON, "r") as f:
        metrics = json.load(f)

    return {
        "labels": labels,
        "matrix": matrix,
        "classification_metrics": metrics
    }


@router.get("/fractions")
def get_training_fractions():
    """Return empirical data efficiency experiments (15%, 30%, 45%, 60%, 100%)."""
    if not _FRACTIONS_JSON.exists():
        raise HTTPException(status_code=404, detail="Training fractions data not found")
    with open(_FRACTIONS_JSON, "r") as f:
        data = json.load(f)
    experiments = data if isinstance(data, list) else data.get("experiments", [])
    return {
        "experiments": experiments,
        "count": len(experiments)
    }
