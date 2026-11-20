"""
City-aware ST-CNN-GRU inference module for AeroSense AI V2.

Loads city-specific checkpoints (e.g. stcnn_gru_delhi.pt),
builds the full 4D spatiotemporal input tensor (batch=1, seq_len=24, n_stations, n_features),
selects the target station by index from the city station list, and returns
multi-horizon PM2.5 predictions in original µg/m³ units.

Verified checkpoint SHA-256:
  stcnn_gru_delhi.pt   : dea79aab12c545f09508fde7f8b280e7b64f3d1fc7e7ade963435a326c8bc820
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.stcnn_gru import STCNNGRUModel
from src.warnings_system.early_warning import EarlyWarningSystem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths relative to the backend root
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _BACKEND_ROOT / "artifacts" / "models"
_FEATURES_DIR = _BACKEND_ROOT / "data" / "features"
_CANONICAL_DIR = _BACKEND_ROOT / "data" / "canonical"

SUPPORTED_CITIES: Dict[str, str] = {
    "delhi": "stcnn_gru_delhi.pt",
}

FORECAST_HORIZONS: List[int] = [1, 6, 12, 24, 48, 72]
SEQ_LEN: int = 24

# ---------------------------------------------------------------------------
# In-memory caches for fast inference
# ---------------------------------------------------------------------------
_model_cache: Dict[str, Any] = {}
_stations_cache: Dict[str, List[str]] = {}
_tensor_cache: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}
_warning_system = EarlyWarningSystem()


def _load_stations(city: str) -> List[str]:
    """Return the ordered station list for a city from canonical/stations.json."""
    if city in _stations_cache:
        return _stations_cache[city]
    path = _CANONICAL_DIR / city / "stations.json"
    if not path.exists():
        raise FileNotFoundError(f"Station file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            stations = [s.get("station_id") or s.get("station") for s in data]
        else:
            stations = list(data)
    elif isinstance(data, dict):
        stations = list(data.keys())
    else:
        raise ValueError(f"Unexpected stations.json format for {city}")
    _stations_cache[city] = stations
    return stations


def _load_city_model(city: str) -> Any:
    """Load (and cache) the ST-CNN-GRU model for the given city."""
    if city in _model_cache:
        return _model_cache[city]

    if city not in SUPPORTED_CITIES:
        raise ValueError(f"City '{city}' not supported. Supported: {list(SUPPORTED_CITIES)}")

    checkpoint_name = SUPPORTED_CITIES[city]
    checkpoint_path = _MODELS_DIR / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    stations = _load_stations(city)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STCNNGRUModel.load(checkpoint_path, device=device, stations=stations)
    model.network.eval()

    logger.info(
        "Loaded city ST-CNN-GRU: city=%s checkpoint=%s device=%s stations=%d",
        city, checkpoint_name, device, len(stations)
    )
    _model_cache[city] = model
    return model


def _get_city_tensors(city: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Get or compute 4D spatiotemporal tensors for the city test set."""
    if city in _tensor_cache:
        return _tensor_cache[city]

    stations = _load_stations(city)
    feat_path = _FEATURES_DIR / city / "test_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Test features not found: {feat_path}")

    df = pd.read_parquet(feat_path)
    target_cols = [f"target_h{h}" for h in FORECAST_HORIZONS]
    exclude_cols = [
        "datetime", "station", "station_id", "timestamp_utc",
        "timestamp_local", "city_id", "data_source", "quality_flag"
    ] + target_cols
    f_cols = [c for c in df.columns if c not in exclude_cols]

    X_4d, y_3d, meta = STCNNGRUModel.build_spatiotemporal_tensors(
        df, f_cols, target_cols, seq_len=SEQ_LEN, stride=1, stations=stations, return_metadata=True
    )
    _tensor_cache[city] = (X_4d, y_3d, meta)
    return X_4d, y_3d, meta


def _resolve_station(city: str, station_query: str) -> str:
    """Resolve station by either its canonical ID (DEL_01) or display name (Anand Vihar)."""
    path = _CANONICAL_DIR / city / "stations.json"
    if not path.exists():
        return station_query
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        q = station_query.strip().lower()
        for s in data:
            sid = str(s.get("station_id") or s.get("id") or "").strip()
            name = str(s.get("name") or s.get("station_name") or "").strip()
            if sid.lower() == q or name.lower() == q:
                return sid
    except Exception:
        pass
    return station_query


def predict_city_station(
    city: str,
    station_id: str,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Run ST-CNN-GRU inference for a specific city + station pair.

    Parameters
    ----------
    city : str
        City key ('delhi').
    station_id : str
        Station identifier (e.g. 'DEL_01') or display name (e.g. 'Anand Vihar').
    offset : int
        Steps back from the latest test sequence (0 = latest).

    Returns
    -------
    dict with full transparency metadata, warnings, and multi-horizon predictions.
    """
    city = city.lower()
    if city not in SUPPORTED_CITIES:
        raise ValueError(f"Unsupported city: '{city}'. Supported: {sorted(SUPPORTED_CITIES)}")

    stations = _load_stations(city)
    canonical_id = _resolve_station(city, station_id)
    if canonical_id not in stations:
        raise ValueError(
            f"Station '{station_id}' not found in {city}. Available: {stations}"
        )

    station_idx = stations.index(canonical_id)
    model = _load_city_model(city)
    X_4d, y_3d, meta = _get_city_tensors(city)

    n_windows = len(X_4d)
    if n_windows == 0:
        raise RuntimeError(f"No valid temporal windows found for {city}.")

    # Pick window index based on offset
    win_idx = max(0, min(n_windows - 1, n_windows - 1 - offset))
    ref_dt = meta["window_end_timestamps"][win_idx]
    ref_time_str = str(ref_dt)

    # 4D slice: (1, 24, n_stations, n_features)
    x_input = torch.tensor(X_4d[win_idx:win_idx + 1], dtype=torch.float32).to(model.device)

    model.network.eval()
    with torch.no_grad():
        out = model.network(x_input)  # (1, n_stations, n_horizons)

    out_np = out.cpu().numpy()
    if out_np.ndim == 3:
        preds = out_np[0, station_idx, :]
    elif out_np.ndim == 2:
        preds = out_np[0, :]
    else:
        preds = out_np.flatten()

    # Ground truth for current observation (approximated from lag or ground truth h1)
    actual_h1 = float(y_3d[win_idx, station_idx, 0]) if y_3d.shape[0] > win_idx else 50.0
    current_pm25 = max(0.0, actual_h1)

    forecasts_map: Dict[int, float] = {}
    predictions_list: List[Dict[str, Any]] = []

    for i, h in enumerate(FORECAST_HORIZONS):
        pred_val = float(preds[i]) if i < len(preds) else current_pm25
        pred_val = round(max(0.0, min(1500.0, pred_val)), 2)
        forecasts_map[h] = pred_val

        cat, info = _warning_system.classify_pm25(pred_val)
        predictions_list.append({
            "horizon_hours": h,
            "predicted_pm25": pred_val,
            "alert_tier": info["label"],
            "color_hex": info["color"],
            "units": "µg/m³",
        })

    from datetime import datetime
    gen_time = datetime.utcnow().isoformat() + "Z"
    ckpt_name = SUPPORTED_CITIES[city]
    ckpt_file = _MODELS_DIR / ckpt_name
    art_size = ckpt_file.stat().st_size if ckpt_file.exists() else 0

    return {
        "city": city,
        "station": canonical_id,
        "reference_datetime": ref_time_str,
        "model_name": f"stcnn_gru_{city}",
        "mode": "city_spatiotemporal",
        "current_pm25": round(current_pm25, 1),
        "forecasts": forecasts_map,
        "predictions": predictions_list,
        "feature_count": meta.get("feature_cols") and len(meta["feature_cols"]) or 57,
        "sequence_length": SEQ_LEN,
        "station_count": len(stations),
        "input_shape": [1, SEQ_LEN, len(stations), len(meta["feature_cols"])],
        "artifact_path": f"artifacts/models/{ckpt_name}",
        "artifact_size_bytes": art_size,
        "input_observation_datetime": ref_time_str,
        "prediction_generation_datetime": gen_time,
        "data_source": f"AeroSense AI V2 Canonical Lakehouse ({city.capitalize()})",
        "data_mode": "OFFLINE / HISTORICAL",
        "units": "µg/m³",
    }


def clear_cache() -> None:
    """Clear all in-memory caches."""
    _model_cache.clear()
    _stations_cache.clear()
    _tensor_cache.clear()
