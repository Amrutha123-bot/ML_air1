"""
Dataset and City Management Service for AeroSense AI V2.
Manages metadata, monitoring stations, canonical datasets, feature parquets,
dataset manifests, and pollutant availability matrices.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_CANONICAL_DIR = _BACKEND_ROOT / "data" / "canonical"
_FEATURES_DIR = _BACKEND_ROOT / "data" / "features"

CITY_CONFIG: Dict[str, Dict[str, Any]] = {
    "delhi": {
        "city_id": "delhi",
        "name": "Delhi NCR",
        "display_name": "Delhi NCR",
        "state": "National Capital Region",
        "country": "India",
        "lat": 28.6139,
        "lon": 77.2090,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Subtropical Semi-Arid (BSh)",
        "source": "CPCB National Air Quality Monitoring Programme (2023-01 to 2024-12)",
        "station_count": 12,
        "date_range": "2023-01-01 to 2024-12-31",
        "supported_models": ["stcnn_gru_delhi", "persistence"],
        "primary_model": "stcnn_gru_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {
            "canonical_total": 210240,
            "feature_total": 209088,
            "train": 146364,
            "val": 31368,
            "test": 31356,
        },
        "date_splits": {
            "train": "2023-01-02 to 2024-05-24",
            "val": "2024-05-24 to 2024-09-10",
            "test": "2024-09-10 to 2024-12-27",
        },
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "mumbai": {
        "city_id": "mumbai",
        "name": "Mumbai",
        "display_name": "Mumbai",
        "state": "Maharashtra",
        "country": "India",
        "lat": 19.0760,
        "lon": 72.8777,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Tropical Coastal (Aw)",
        "source": "CPCB / MPCB Air Quality Monitoring Network (2023-01 to 2023-12)",
        "station_count": 8,
        "date_range": "2023-01-01 to 2023-12-31",
        "supported_models": ["persistence"],
        "primary_model": "persistence",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "evaluation_completed",
        "records": {
            "canonical_total": 70080,
            "feature_total": 69312,
            "train": 48520,
            "val": 10400,
            "test": 10392,
        },
        "date_splits": {
            "train": "2023-01-02 to 2023-09-11",
            "val": "2023-09-11 to 2023-11-04",
            "test": "2023-11-04 to 2023-12-27",
        },
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "vijayawada": {
        "city_id": "vijayawada",
        "name": "Vijayawada",
        "display_name": "Vijayawada",
        "state": "Andhra Pradesh",
        "country": "India",
        "lat": 16.5062,
        "lon": 80.6480,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Tropical Savanna (Aw)",
        "source": "APPCB Ambient Air Monitoring Programme",
        "station_count": 2,
        "date_range": "2023-01-01 to 2024-12-31",
        "supported_models": ["persistence"],
        "primary_model": "persistence",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "training_pending",
        "records": {
            "canonical_total": 17520,
            "feature_total": 17328,
            "train": 12144,
            "val": 2604,
            "test": 2580,
        },
        "date_splits": {
            "train": "2023-01-02 to 2024-05-24",
            "val": "2024-05-24 to 2024-09-10",
            "test": "2024-09-10 to 2024-12-27",
        },
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "bengaluru": {
        "city_id": "bengaluru",
        "name": "Bengaluru",
        "display_name": "Bengaluru",
        "state": "Karnataka",
        "country": "India",
        "lat": 12.9716,
        "lon": 77.5946,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Tropical Savanna (As)",
        "source": "CPCB / KSPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 5,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "kolkata": {
        "city_id": "kolkata",
        "name": "Kolkata",
        "display_name": "Kolkata",
        "state": "West Bengal",
        "country": "India",
        "lat": 22.5726,
        "lon": 88.3639,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Tropical Wet-and-Dry (Aw)",
        "source": "CPCB / WBPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 5,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "hyderabad": {
        "city_id": "hyderabad",
        "name": "Hyderabad",
        "display_name": "Hyderabad",
        "state": "Telangana",
        "country": "India",
        "lat": 17.3850,
        "lon": 78.4867,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Tropical Semi-Arid (BSh)",
        "source": "CPCB / TSPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 4,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "chennai": {
        "city_id": "chennai",
        "name": "Chennai",
        "display_name": "Chennai",
        "state": "Tamil Nadu",
        "country": "India",
        "lat": 13.0827,
        "lon": 80.2707,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Tropical Wet-and-Dry (Aw)",
        "source": "CPCB / TNPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 4,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "ahmedabad": {
        "city_id": "ahmedabad",
        "name": "Ahmedabad",
        "display_name": "Ahmedabad",
        "state": "Gujarat",
        "country": "India",
        "lat": 23.0225,
        "lon": 72.5714,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Hot Semi-Arid (BSh)",
        "source": "CPCB / GPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 4,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "jaipur": {
        "city_id": "jaipur",
        "name": "Jaipur",
        "display_name": "Jaipur",
        "state": "Rajasthan",
        "country": "India",
        "lat": 26.9124,
        "lon": 75.7873,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Hot Semi-Arid (BSh)",
        "source": "CPCB / RSPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 3,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "lucknow": {
        "city_id": "lucknow",
        "name": "Lucknow",
        "display_name": "Lucknow",
        "state": "Uttar Pradesh",
        "country": "India",
        "lat": 26.8467,
        "lon": 80.9462,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Humid Subtropical (Cwa)",
        "source": "CPCB / UPPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 4,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
    "pune": {
        "city_id": "pune",
        "name": "Pune",
        "display_name": "Pune",
        "state": "Maharashtra",
        "country": "India",
        "lat": 18.5204,
        "lon": 73.8567,
        "timezone": "Asia/Kolkata",
        "climate_zone": "Tropical Semi-Arid (BSh)",
        "source": "CPCB / MPCB Ambient Air Quality Network (2015-2023)",
        "station_count": 4,
        "date_range": "2015-01-01 to 2023-12-31",
        "supported_models": ["random_forest_delhi", "xgboost_delhi", "stcnn_gru_delhi", "persistence"],
        "primary_model": "random_forest_delhi",
        "horizons": [1, 6, 12, 24, 48, 72],
        "feature_count": 57,
        "data_status": "historical_data_available",
        "live_status": "not_available",
        "pipeline_status": "forecasting_available",
        "records": {"canonical_total": 3287, "feature_total": 3287, "train": 2300, "val": 493, "test": 494},
        "date_splits": {"train": "2015-01-01 to 2021-04-18", "val": "2021-05-24 to 2022-09-28", "test": "2022-11-03 to 2023-12-31"},
        "data_mode": "OFFLINE / HISTORICAL",
    },
}

DATASET_MANIFEST: Dict[str, Dict[str, Any]] = {
    "cpcb_delhi_2023_2024": {
        "dataset_id": "cpcb_delhi_2023_2024",
        "city": "delhi",
        "source_name": "Central Pollution Control Board (CPCB) CAAQMS",
        "source_page_url": "https://cpcb.nic.in/air-quality-data/",
        "download_url_or_api_route": "https://app.cpcbccr.com/ccr/#/caaqms-dashboard",
        "original_filename": "delhi_cpcb_12st_hourly.parquet",
        "download_date": "2026-11-14",
        "local_path": "backend/data/canonical/delhi/observations.parquet",
        "drive_path": "AeroSense_AI_Artifacts/datasets/delhi/observations.parquet",
        "sha256": "a4b2977be2d133a04794f1a3fa11d90afa431cc157df683858e2a723a313f6e4",
        "coverage_start": "2023-01-01T00:00:00+05:30",
        "coverage_end": "2024-12-30T23:00:00+05:30",
        "stations": ["DEL_01", "DEL_02", "DEL_03", "DEL_04", "DEL_05", "DEL_06", "DEL_07", "DEL_08", "DEL_09", "DEL_10", "DEL_11", "DEL_12"],
        "pollutants": ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NH3"],
        "weather_vars": ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_sin", "wd_cos"],
        "row_count": 210240,
        "license": "Open Data Government of India (NDSAP)",
        "cleaning_pipeline_version": "2.1.0",
        "validation_status": "Verified and suitable",
    },
    "cpcb_mpcb_mumbai_2023": {
        "dataset_id": "cpcb_mpcb_mumbai_2023",
        "city": "mumbai",
        "source_name": "CPCB / Maharashtra Pollution Control Board (MPCB)",
        "source_page_url": "https://mpcb.gov.in/air-quality",
        "download_url_or_api_route": "https://app.cpcbccr.com/ccr/#/caaqms-dashboard",
        "original_filename": "mumbai_mpcb_8st_hourly.parquet",
        "download_date": "2026-11-14",
        "local_path": "backend/data/canonical/mumbai/observations.parquet",
        "drive_path": "AeroSense_AI_Artifacts/datasets/mumbai/observations.parquet",
        "sha256": "b2e12d443fcefbf621987d3ba3de8a62fec65e476c71365819ede9335341b3d1",
        "coverage_start": "2023-01-01T00:00:00+05:30",
        "coverage_end": "2023-12-31T23:00:00+05:30",
        "stations": ["MUM_01", "MUM_02", "MUM_03", "MUM_04", "MUM_05", "MUM_06", "MUM_07", "MUM_08"],
        "pollutants": ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NH3"],
        "weather_vars": ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_sin", "wd_cos"],
        "row_count": 70080,
        "license": "Open Data Government of India (NDSAP)",
        "cleaning_pipeline_version": "2.1.0",
        "validation_status": "Suitable after preprocessing",
    },
    "appcb_vijayawada_2023": {
        "dataset_id": "appcb_vijayawada_2023",
        "city": "vijayawada",
        "source_name": "Andhra Pradesh Pollution Control Board (APPCB)",
        "source_page_url": "https://pcb.ap.gov.in/",
        "download_url_or_api_route": "https://app.cpcbccr.com/ccr/#/caaqms-dashboard",
        "original_filename": "vijayawada_appcb_2st_hourly.parquet",
        "download_date": "2026-11-14",
        "local_path": "backend/data/canonical/vijayawada/observations.parquet",
        "drive_path": "AeroSense_AI_Artifacts/datasets/vijayawada/observations.parquet",
        "sha256": "be5e88f500c3b44c7a9aa82dad4f634daf22180dbc31a995fac0bcb964b2e70c",
        "coverage_start": "2023-01-01T00:00:00+05:30",
        "coverage_end": "2023-12-31T23:00:00+05:30",
        "stations": ["VJA_01", "VJA_02"],
        "pollutants": ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NH3"],
        "weather_vars": ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_sin", "wd_cos"],
        "row_count": 17520,
        "license": "Open Data Government of India (NDSAP)",
        "cleaning_pipeline_version": "2.1.0",
        "validation_status": "Suitable after preprocessing",
    },
}

POLLUTANT_AVAILABILITY: Dict[str, Dict[str, Any]] = {
    "delhi": {
        "PM2.5": {
            "valid_hourly_obs": 210240,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2024-12-30T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "µg/m³",
            "usable_144h_windows": 208512,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["stcnn_gru_delhi", "persistence"],
            "status": "forecasting_available",
        },
        "PM10": {
            "valid_hourly_obs": 210240,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2024-12-30T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "µg/m³",
            "usable_144h_windows": 208512,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["persistence"],
            "status": "data_available_training_pending",
        },
        "NO2": {
            "valid_hourly_obs": 210240,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2024-12-30T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "µg/m³",
            "usable_144h_windows": 208512,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["persistence"],
            "status": "data_available_training_pending",
        },
        "SO2": {
            "valid_hourly_obs": 210240,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2024-12-30T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "µg/m³",
            "usable_144h_windows": 208512,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["persistence"],
            "status": "data_available_training_pending",
        },
        "CO": {
            "valid_hourly_obs": 210240,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2024-12-30T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "mg/m³",
            "usable_144h_windows": 208512,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["persistence"],
            "status": "data_available_training_pending",
        },
        "O3": {
            "valid_hourly_obs": 210240,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2024-12-30T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "µg/m³",
            "usable_144h_windows": 208512,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["persistence"],
            "status": "data_available_training_pending",
        },
    },
    "mumbai": {
        "PM2.5": {
            "valid_hourly_obs": 70080,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2023-12-31T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "µg/m³",
            "usable_144h_windows": 68928,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["persistence"],
            "status": "evaluation_completed",
        },
    },
    "vijayawada": {
        "PM2.5": {
            "valid_hourly_obs": 17520,
            "coverage_start": "2023-01-01T00:00:00+05:30",
            "coverage_end": "2023-12-31T23:00:00+05:30",
            "missing_pct": 0.0,
            "unit": "µg/m³",
            "usable_144h_windows": 17232,
            "suitable_horizons": [1, 6, 12, 24, 48, 72],
            "recommended_models": ["persistence"],
            "status": "training_pending",
        },
    },
}

_stations_cache: Dict[str, List[Dict[str, Any]]] = {}


def get_available_cities() -> List[Dict[str, Any]]:
    """Return all configured cities with availability status conforming to Phase 2 registry spec."""
    cities_list = []
    csv_file = _BACKEND_ROOT / "data" / "india_city_aqi_2015_2023.csv"
    for cid, conf in CITY_CONFIG.items():
        feat_dir = _FEATURES_DIR / cid
        has_data = (feat_dir.exists() and (feat_dir / "test_features.parquet").exists()) or (csv_file.exists() and cid != "vijayawada")
        cities_list.append({
            "city_id": cid,
            "name": conf["name"],
            "display_name": conf["display_name"],
            "state": conf["state"],
            "country": conf["country"],
            "lat": conf["lat"],
            "lon": conf["lon"],
            "timezone": conf["timezone"],
            "climate_zone": conf["climate_zone"],
            "station_count": conf["station_count"],
            "supported_models": conf["supported_models"],
            "primary_model": conf.get("primary_model"),
            "data_mode": conf["data_mode"],
            "dataset_available": has_data,
            "data_status": conf["data_status"],
            "live_status": conf["live_status"],
            "pipeline_status": conf["pipeline_status"],
        })
    return cities_list


def get_city_details(city: str) -> Optional[Dict[str, Any]]:
    """Return comprehensive metadata for a specific city."""
    cid = city.lower().strip()
    if cid not in CITY_CONFIG:
        return None
    conf = dict(CITY_CONFIG[cid])
    conf["stations"] = get_city_stations(cid)
    return conf


def get_city_stations(city: str) -> List[Dict[str, Any]]:
    """Return ordered station list with geographic metadata for a city."""
    cid = city.lower().strip()
    if cid in _stations_cache:
        return _stations_cache[cid]

    path = _CANONICAL_DIR / cid / "stations.json"
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stations = []
    if isinstance(data, list):
        for s in data:
            if isinstance(s, dict):
                stations.append({
                    "station_id": s.get("station_id") or s.get("id"),
                    "name": s.get("name") or s.get("station_name") or s.get("station_id"),
                    "latitude": float(s.get("latitude", 0.0)),
                    "longitude": float(s.get("longitude", 0.0)),
                    "site_type": s.get("site_type") or s.get("description", "Urban Ambient Reference"),
                    "operator": s.get("operator", "CPCB / State Pollution Control Board"),
                    "first_obs": s.get("first_obs", "2023-01-01T00:00:00+05:30"),
                    "last_obs": s.get("last_obs", "2024-12-30T23:00:00+05:30"),
                })
            else:
                stations.append({
                    "station_id": str(s),
                    "name": str(s),
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "site_type": "Monitoring Station",
                    "operator": "CPCB",
                    "first_obs": "2023-01-01T00:00:00+05:30",
                    "last_obs": "2024-12-30T23:00:00+05:30",
                })
    _stations_cache[cid] = stations
    return stations


def get_all_datasets() -> List[Dict[str, Any]]:
    """Return dataset summaries for all configured metropolitan areas."""
    datasets = []
    for cid, conf in CITY_CONFIG.items():
        datasets.append({
            "city_id": cid,
            "city_name": conf["name"],
            "source": conf["source"],
            "date_range": conf["date_range"],
            "total_canonical_records": conf["records"]["canonical_total"],
            "total_feature_rows": conf["records"]["feature_total"],
            "station_count": conf["station_count"],
            "feature_count": conf["feature_count"],
            "preprocessing_status": "Normalized (70/15/15 Chronological Split, StandardScaler)",
            "supported_models": conf["supported_models"],
        })
    return datasets


def get_dataset_details(city: str) -> Optional[Dict[str, Any]]:
    """Return split-level dataset details for a specific city."""
    cid = city.lower().strip()
    if cid not in CITY_CONFIG:
        return None
    conf = CITY_CONFIG[cid]
    return {
        "city_id": cid,
        "city_name": conf["name"],
        "source": conf["source"],
        "station_count": conf["station_count"],
        "feature_count": conf["feature_count"],
        "preprocessing_status": "Normalized (StandardScaler fitted strictly on training split)",
        "splits": {
            "train": {
                "rows": conf["records"]["train"],
                "date_range": conf["date_splits"]["train"],
            },
            "validation": {
                "rows": conf["records"]["val"],
                "date_range": conf["date_splits"]["val"],
            },
            "test": {
                "rows": conf["records"]["test"],
                "date_range": conf["date_splits"]["test"],
            },
        },
        "boundary_row_reconciliation": {
            "canonical_rows": conf["records"]["canonical_total"],
            "feature_rows": conf["records"]["feature_total"],
            "dropped_boundary_rows": conf["records"]["canonical_total"] - conf["records"]["feature_total"],
            "rationale": f"Exactly 96 boundary rows per station ({conf['station_count']} x 96 = {conf['station_count'] * 96}): 24 leading rows (t-24h lag) + 72 trailing rows (t+72h target)",
        },
        "supported_models": conf["supported_models"],
    }


def get_dataset_manifest(dataset_id: Optional[str] = None) -> Any:
    """Return dataset manifest conforming to Phase 2 specification."""
    if dataset_id:
        did = dataset_id.lower().strip()
        return DATASET_MANIFEST.get(did)
    return list(DATASET_MANIFEST.values())


def get_pollutant_availability(city: Optional[str] = None, pollutant: Optional[str] = None) -> Any:
    """Return pollutant availability matrix conforming to Phase 2 specification."""
    if city:
        cid = city.lower().strip()
        c_avail = POLLUTANT_AVAILABILITY.get(cid, {})
        if pollutant:
            return c_avail.get(pollutant.upper())
        return c_avail
    return POLLUTANT_AVAILABILITY
