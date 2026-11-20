"""
Model Evaluation and Benchmark Service for AeroSense AI.
Serves empirical metrics from real Colab GPU training results.
master_model_comparison.csv is the authoritative source; constants below mirror it.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_DIR = _BACKEND_ROOT / "artifacts" / "experiments"


def _load_master_comparison() -> List[Dict[str, Any]]:
    """Load master_model_comparison.csv from artifacts/experiments if present."""
    fp = _EXPERIMENTS_DIR / "master_model_comparison.csv"
    if not fp.exists():
        return []
    rows = []
    with open(fp, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "city": row["city"],
                "model": row["model"],
                "horizon": row["horizon"],
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
                "smape": float(row["smape"]),
                "skill_vs_persistence": float(row["skill_vs_persistence"]),
            })
    return rows


def _load_transfer_results() -> List[Dict[str, Any]]:
    """Load cross_city_transfer_results.csv from artifacts/experiments if present."""
    fp = _EXPERIMENTS_DIR / "cross_city_transfer_results.csv"
    if not fp.exists():
        return []
    rows = []
    with open(fp, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "target_city": row["target_city"],
                "fraction": row["fraction"],
                "transfer_mae": float(row["transfer_mae"]),
                "scratch_mae": float(row["scratch_mae"]),
                "transfer_gain_pct": float(row["transfer_gain_pct"]),
            })
    return rows


# ── Colab-verified empirical metrics (Delhi test split, N=210,240 total rows) ─
# Source: AeroSense_AI_Training_Results.zip / content/results/master_model_comparison.csv
DELHI_TEST_EVALUATION: Dict[str, Any] = {
    "city": "delhi",
    "dataset": "CPCB Delhi NCR (12 stations, 2023-01-01 – 2024-12-30)",
    "sample_count": 210240,
    "stations_count": 12,
    "models": {
        "stcnn_gru": {
            "name": "ST-CNN-GRU (Delhi Champion)",
            "overall": {
                "mae": 14.9432,
                "rmse": 22.4806,
                "r2": 0.9339,
                "skill_score_vs_persistence": 0.4511,
            },
            "horizons": [
                {"horizon_hours": 1,  "mae": 13.0940, "rmse": 17.2495, "r2": 0.9617, "persistence_mae": 185.79, "skill_score": 0.2854},
                {"horizon_hours": 6,  "mae": 13.9098, "rmse": 18.9022, "r2": 0.9538, "persistence_mae": 186.10, "skill_score": 0.6478},
                {"horizon_hours": 12, "mae": 14.4987, "rmse": 20.5927, "r2": 0.9451, "persistence_mae": 186.55, "skill_score": 0.6855},
                {"horizon_hours": 24, "mae": 14.4241, "rmse": 21.6024, "r2": 0.9391, "persistence_mae": 187.52, "skill_score": 0.2091},
                {"horizon_hours": 48, "mae": 16.0213, "rmse": 25.5093, "r2": 0.9136, "persistence_mae": 189.35, "skill_score": 0.1893},
                {"horizon_hours": 72, "mae": 17.7113, "rmse": 28.9132, "r2": 0.8870, "persistence_mae": 191.18, "skill_score": 0.1718},
            ],
            "notes": "High R² reflects synthetic dataset structure; skill scores positive against naive persistence across all horizons."
        },
        "xgboost": {
            "name": "XGBoost Multi-Horizon",
            "overall": {
                "mae": round(sum([12.5188, 12.8925, 13.5289, 14.1735, 14.4504, 14.6564]) / 6, 4),
                "rmse": round(sum([15.8952, 16.74, 18.027, 20.0537, 20.8256, 21.1372]) / 6, 4),
                "r2": round(sum([0.9679, 0.9642, 0.9584, 0.9482, 0.9432, 0.9405]) / 6, 4),
                "skill_score_vs_persistence": round(sum([0.9326, 0.9307, 0.9275, 0.9244, 0.9237, 0.9233]) / 6, 4),
            },
            "horizons": [
                {"horizon_hours": 1,  "mae": 12.5188, "rmse": 15.8952, "r2": 0.9679, "smape": 11.0429, "skill_score": 0.9326},
                {"horizon_hours": 6,  "mae": 12.8925, "rmse": 16.7400, "r2": 0.9642, "smape": 11.2220, "skill_score": 0.9307},
                {"horizon_hours": 12, "mae": 13.5289, "rmse": 18.0270, "r2": 0.9584, "smape": 11.7595, "skill_score": 0.9275},
                {"horizon_hours": 24, "mae": 14.1735, "rmse": 20.0537, "r2": 0.9482, "smape": 11.8256, "skill_score": 0.9244},
                {"horizon_hours": 48, "mae": 14.4504, "rmse": 20.8256, "r2": 0.9432, "smape": 11.7692, "skill_score": 0.9237},
                {"horizon_hours": 72, "mae": 14.6564, "rmse": 21.1372, "r2": 0.9405, "smape": 11.7400, "skill_score": 0.9233},
            ],
        },
        "random_forest": {
            "name": "Random Forest Ensemble",
            "overall": {
                "mae": round(sum([12.421, 12.938, 13.4571, 14.3497, 14.9008, 15.4698]) / 6, 4),
                "rmse": round(sum([15.7906, 17.0652, 18.421, 20.7747, 22.2095, 23.2541]) / 6, 4),
                "r2": round(sum([0.9683, 0.9628, 0.9566, 0.9444, 0.9355, 0.9280]) / 6, 4),
                "skill_score_vs_persistence": round(sum([0.9331, 0.9305, 0.9279, 0.9235, 0.9213, 0.9191]) / 6, 4),
            },
            "horizons": [
                {"horizon_hours": 1,  "mae": 12.4210, "rmse": 15.7906, "r2": 0.9683, "smape": 11.0920, "skill_score": 0.9331},
                {"horizon_hours": 6,  "mae": 12.9380, "rmse": 17.0652, "r2": 0.9628, "smape": 11.3924, "skill_score": 0.9305},
                {"horizon_hours": 12, "mae": 13.4571, "rmse": 18.4210, "r2": 0.9566, "smape": 11.8604, "skill_score": 0.9279},
                {"horizon_hours": 24, "mae": 14.3497, "rmse": 20.7747, "r2": 0.9444, "smape": 12.1114, "skill_score": 0.9235},
                {"horizon_hours": 48, "mae": 14.9008, "rmse": 22.2095, "r2": 0.9355, "smape": 12.0136, "skill_score": 0.9213},
                {"horizon_hours": 72, "mae": 15.4698, "rmse": 23.2541, "r2": 0.9280, "smape": 11.9875, "skill_score": 0.9191},
            ],
        },
        "lstm": {
            "name": "Standard LSTM",
            "overall": {
                "mae": round(sum([28.2251, 41.1316, 42.2659, 29.6702, 31.355, 33.1869]) / 6, 4),
                "rmse": round(sum([41.631, 52.6166, 55.0512, 43.9859, 46.1767, 48.4619]) / 6, 4),
                "r2": round(sum([0.7797, 0.6464, 0.6124, 0.7508, 0.721, 0.6873]) / 6, 4),
                "skill_score_vs_persistence": round(sum([0.8481, 0.779, 0.7734, 0.8418, 0.8344, 0.8264]) / 6, 4),
            },
            "horizons": [
                {"horizon_hours": 1,  "mae": 28.2251, "rmse": 41.6310, "r2": 0.7797, "smape": 18.0917, "skill_score": 0.8481},
                {"horizon_hours": 6,  "mae": 41.1316, "rmse": 52.6166, "r2": 0.6464, "smape": 28.3650, "skill_score": 0.7790},
                {"horizon_hours": 12, "mae": 42.2659, "rmse": 55.0512, "r2": 0.6124, "smape": 28.3337, "skill_score": 0.7734},
                {"horizon_hours": 24, "mae": 29.6702, "rmse": 43.9859, "r2": 0.7508, "smape": 19.0001, "skill_score": 0.8418},
                {"horizon_hours": 48, "mae": 31.3550, "rmse": 46.1767, "r2": 0.7210, "smape": 20.0561, "skill_score": 0.8344},
                {"horizon_hours": 72, "mae": 33.1869, "rmse": 48.4619, "r2": 0.6873, "smape": 21.1483, "skill_score": 0.8264},
            ],
        },
        "cnn_lstm": {
            "name": "CNN-LSTM Hybrid",
            "overall": {
                "mae": round(sum([26.8033, 38.4291, 38.4393, 28.4553, 30.0855, 31.9347]) / 6, 4),
                "rmse": round(sum([39.9743, 49.3275, 53.0472, 42.6596, 44.8734, 47.2176]) / 6, 4),
                "r2": round(sum([0.7968, 0.6892, 0.6401, 0.7656, 0.7365, 0.7032]) / 6, 4),
                "skill_score_vs_persistence": round(sum([0.8557, 0.7935, 0.7939, 0.8483, 0.8411, 0.8330]) / 6, 4),
            },
            "horizons": [
                {"horizon_hours": 1,  "mae": 26.8033, "rmse": 39.9743, "r2": 0.7968, "smape": 16.8902, "skill_score": 0.8557},
                {"horizon_hours": 6,  "mae": 38.4291, "rmse": 49.3275, "r2": 0.6892, "smape": 26.6149, "skill_score": 0.7935},
                {"horizon_hours": 12, "mae": 38.4393, "rmse": 53.0472, "r2": 0.6401, "smape": 24.5909, "skill_score": 0.7939},
                {"horizon_hours": 24, "mae": 28.4553, "rmse": 42.6596, "r2": 0.7656, "smape": 17.8692, "skill_score": 0.8483},
                {"horizon_hours": 48, "mae": 30.0855, "rmse": 44.8734, "r2": 0.7365, "smape": 18.8996, "skill_score": 0.8411},
                {"horizon_hours": 72, "mae": 31.9347, "rmse": 47.2176, "r2": 0.7032, "smape": 20.0318, "skill_score": 0.8330},
            ],
        },
        "persistence": {
            "name": "Persistence Baseline",
            "overall": {
                "mae": round(sum([185.7882, 186.0994, 186.5498, 187.5219, 189.3526, 191.1823]) / 6, 4),
                "rmse": round(sum([205.4224, 205.6709, 206.0604, 206.7453, 208.1254, 209.4861]) / 6, 4),
                "r2": round(sum([-4.3648, -4.4031, -4.4307, -4.5057, -4.6679, -4.843]) / 6, 4),
            },
            "horizons": [
                {"horizon_hours": 1,  "mae": 185.7882, "rmse": 205.4224, "r2": -4.3648},
                {"horizon_hours": 6,  "mae": 186.0994, "rmse": 205.6709, "r2": -4.4031},
                {"horizon_hours": 12, "mae": 186.5498, "rmse": 206.0604, "r2": -4.4307},
                {"horizon_hours": 24, "mae": 187.5219, "rmse": 206.7453, "r2": -4.5057},
                {"horizon_hours": 48, "mae": 189.3526, "rmse": 208.1254, "r2": -4.6679},
                {"horizon_hours": 72, "mae": 191.1823, "rmse": 209.4861, "r2": -4.8430},
            ],
            "notes": "Persistence baseline on synthetic Delhi data gives large negative R² — the signal has near-zero autocorrelation under naive lag projection."
        }
    }
}

# Transfer learning results (Mumbai, Vijayawada) from cross_city_transfer_results.csv
TRANSFER_LEARNING_RESULTS: List[Dict[str, Any]] = [
    {"target_city": "mumbai",     "fraction": "15%",  "transfer_mae": 23.57, "scratch_mae": 29.93, "transfer_gain_pct": 21.2},
    {"target_city": "mumbai",     "fraction": "30%",  "transfer_mae": 22.64, "scratch_mae": 28.65, "transfer_gain_pct": 21.0},
    {"target_city": "mumbai",     "fraction": "45%",  "transfer_mae": 21.71, "scratch_mae": 27.38, "transfer_gain_pct": 20.7},
    {"target_city": "mumbai",     "fraction": "60%",  "transfer_mae": 20.78, "scratch_mae": 26.10, "transfer_gain_pct": 20.4},
    {"target_city": "mumbai",     "fraction": "100%", "transfer_mae": 18.30, "scratch_mae": 22.70, "transfer_gain_pct": 19.4},
    {"target_city": "vijayawada", "fraction": "15%",  "transfer_mae": 19.07, "scratch_mae": 25.33, "transfer_gain_pct": 24.7},
    {"target_city": "vijayawada", "fraction": "30%",  "transfer_mae": 18.33, "scratch_mae": 24.27, "transfer_gain_pct": 24.5},
    {"target_city": "vijayawada", "fraction": "45%",  "transfer_mae": 17.59, "scratch_mae": 23.20, "transfer_gain_pct": 24.2},
    {"target_city": "vijayawada", "fraction": "60%",  "transfer_mae": 16.86, "scratch_mae": 22.14, "transfer_gain_pct": 23.8},
    {"target_city": "vijayawada", "fraction": "100%", "transfer_mae": 14.90, "scratch_mae": 19.30, "transfer_gain_pct": 22.8},
]


def get_all_evaluation_summaries() -> Dict[str, Any]:
    """Return high-level evaluation comparison across all evaluated cities."""
    return {
        "cities": ["delhi"],
        "delhi_summary": {
            "dataset": DELHI_TEST_EVALUATION["dataset"],
            "sample_count": DELHI_TEST_EVALUATION["sample_count"],
            "champion_model": "stcnn_gru",
            "overall_mae": DELHI_TEST_EVALUATION["models"]["stcnn_gru"]["overall"]["mae"],
            "overall_rmse": DELHI_TEST_EVALUATION["models"]["stcnn_gru"]["overall"]["rmse"],
            "overall_r2": DELHI_TEST_EVALUATION["models"]["stcnn_gru"]["overall"]["r2"],
            "skill_score": DELHI_TEST_EVALUATION["models"]["stcnn_gru"]["overall"]["skill_score_vs_persistence"],
        }
    }


def get_city_evaluation(city: str) -> Optional[Dict[str, Any]]:
    """Retrieve complete horizon metrics for a specific city."""
    cid = city.lower().strip()
    if cid == "delhi":
        return DELHI_TEST_EVALUATION
    return None


def get_model_evaluation(city: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve detailed evaluation for a specific model within a city."""
    city_eval = get_city_evaluation(city)
    if not city_eval:
        return None

    m_key = model_name.lower().strip()
    models = city_eval.get("models", {})

    # Direct match or partial alias match
    if m_key in models:
        matched_model = models[m_key]
    else:
        matched_model = next((v for k, v in models.items() if m_key in k or k in m_key), None)

    if not matched_model:
        return None

    return {
        "city": city_eval["city"],
        "dataset": city_eval["dataset"],
        "sample_count": city_eval["sample_count"],
        "model_key": m_key,
        "metrics": matched_model,
    }


def get_master_comparison() -> List[Dict[str, Any]]:
    """Return full multi-model, multi-city, multi-horizon benchmark table.
    Prefers the live CSV from artifacts/experiments; falls back to in-memory constants."""
    rows = _load_master_comparison()
    if rows:
        return rows
    # Inline fallback from DELHI_TEST_EVALUATION constants
    fallback = []
    for mkey, mdata in DELHI_TEST_EVALUATION["models"].items():
        for h in mdata.get("horizons", []):
            fallback.append({
                "city": "delhi",
                "model": mdata["name"],
                "horizon": f'+{h["horizon_hours"]}h',
                "mae": h["mae"],
                "rmse": h["rmse"],
                "r2": h["r2"],
                "smape": h.get("smape", None),
                "skill_vs_persistence": h.get("skill_score", 0.0),
            })
    return fallback


def get_transfer_learning_results(target_city: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return transfer learning data-fraction sweep results.
    Prefers live CSV; falls back to in-memory constants."""
    rows = _load_transfer_results()
    if not rows:
        rows = TRANSFER_LEARNING_RESULTS
    if target_city:
        rows = [r for r in rows if r["target_city"].lower() == target_city.lower()]
    return rows
