"""Central configuration for AeroSense AI (Indian-City Air Quality Forecasting Platform)."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ───────────────────────────────────────────────────────────────────────────
# Project Root
# ───────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"

# Create all active directories
for _dir in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, SCALERS_DIR,
             EXPERIMENTS_DIR, LOGS_DIR, DOCS_DIR / "results"]:
    _dir.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────
# Primary Active Indian City Network
# ───────────────────────────────────────────────────────────────────────────
DEFAULT_CITY = "delhi"
EXPECTED_STATIONS = [
    "Anand Vihar", "R K Puram", "Punjabi Bagh", "Mandir Marg",
    "IGI Airport (T3)", "Jawaharlal Nehru Std", "Bawana", "Okhla Phase 2",
    "Dwarka Sector 8", "Patparganj", "Wazirpur", "Rohini"
]
STATION_COORDS = {
    "Anand Vihar":             (28.6468, 77.3160),
    "R K Puram":               (28.5632, 77.1864),
    "Punjabi Bagh":            (28.6683, 77.1167),
    "Mandir Marg":             (28.6365, 77.2010),
    "IGI Airport (T3)":        (28.5621, 77.0984),
    "Jawaharlal Nehru Std":    (28.5802, 77.2334),
    "Bawana":                  (28.7762, 77.0510),
    "Okhla Phase 2":           (28.5307, 77.2713),
    "Dwarka Sector 8":         (28.5710, 77.0719),
    "Patparganj":              (28.6237, 77.2872),
    "Wazirpur":                (28.6997, 77.1654),
    "Rohini":                  (28.7325, 77.1199),
}
STATION_IDS = {name: f"DEL_{i+1:02d}" for i, name in enumerate(EXPECTED_STATIONS)}

# ───────────────────────────────────────────────────────────────────────────
# Target & Horizons
# ───────────────────────────────────────────────────────────────────────────
TARGET_COL = "PM2.5"
FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]  # hours ahead

# ───────────────────────────────────────────────────────────────────────────
# Features
# ───────────────────────────────────────────────────────────────────────────
METEOROLOGICAL_COLS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DIR_COL = "wd"
POLLUTANT_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [3, 6, 12, 24]
SEQUENCE_LENGTH = 24  # Default lookback window

# ───────────────────────────────────────────────────────────────────────────
# Warning Thresholds (CPCB National Ambient Air Quality Standards - NAAQS)
# ───────────────────────────────────────────────────────────────────────────
WARNING_THRESHOLDS = {
    "good":           (0, 30),
    "satisfactory":   (30, 60),
    "moderate":       (60, 90),
    "poor":           (90, 120),
    "very_poor":      (120, 250),
    "severe":         (250, float("inf")),
}
WARNING_AVERAGING_HOURS = 24  # Rolling average window for warnings
WARNING_LEAD_TIME = 24        # Minimum useful advance warning (hours)

# ───────────────────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────────────────
RANDOM_SEEDS = [42, 43, 44]
DEFAULT_SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
GRAD_CLIP = 1.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ───────────────────────────────────────────────────────────────────────────
# API Configuration
# ───────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ───────────────────────────────────────────────────────────────────────────
# Model Registry Config
# ───────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "stcnn_gru_delhi": {"artifact": "stcnn_gru_delhi.pt", "type": "pytorch"},
    "persistence":     {"artifact": "persistence_model.pkl", "type": "baseline"},
}
BEST_MODEL_KEY = "stcnn_gru_delhi"
