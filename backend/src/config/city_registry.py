"""
AeroSense AI — Delhi Metadata and Station Registry.

Defines geographical coordinates, CPCB regulatory standards, warning tiers,
and station metadata exclusively for Delhi NCR.
"""
from typing import Dict, List, Any, Tuple
import math
import numpy as np

# ─── Regulatory Frameworks ───────────────────────────────────────────────────

REGULATORY_STANDARDS = {
    "NAAQS_INDIA": {
        "standard_name": "Indian National Ambient Air Quality Standards (NAAQS)",
        "governing_body": "Central Pollution Control Board (CPCB)",
        "averaging_hours": 24,
        "tiers": {
            "good":           {"min": 0,   "max": 30,  "label": "Good",           "code": "TIER_I",   "color": "#DCFCE7", "text_color": "#166534"},
            "satisfactory":   {"min": 30,  "max": 60,  "label": "Satisfactory",   "code": "TIER_II",  "color": "#ECFDF5", "text_color": "#065F46"},
            "moderate":       {"min": 60,  "max": 90,  "label": "Moderate",       "code": "TIER_III", "color": "#FEF3C7", "text_color": "#92400E"},
            "poor":           {"min": 90,  "max": 120, "label": "Poor",           "code": "TIER_IV",  "color": "#FFEDD5", "text_color": "#9A3412"},
            "very_poor":      {"min": 120, "max": 250, "label": "Very Poor",      "code": "TIER_V",   "color": "#FEE2E2", "text_color": "#991B1B"},
            "severe":         {"min": 250, "max": 1000,"label": "Severe",         "code": "TIER_VI",  "color": "#FCE7F3", "text_color": "#9D174D"},
        }
    }
}

# ─── Delhi City & Station Registry ───────────────────────────────────────────

CITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "delhi": {
        "city_name": "Delhi NCR",
        "country": "India",
        "timezone": "Asia/Delhi",
        "utc_offset_hours": 5.5,
        "regulatory_standard": "NAAQS_INDIA",
        "climate_type": "Semiarid, continental with intense winter inversion",
        "network_summary": {
            "total_stations": 6,
            "total_observations": 210240,
            "rolling_72h_window_sequences": 209088,
            "hourly_cadence_per_station": 35040,
            "active_telemetry_station": "DEL_01",
            "date_range": "2020-01-01 to 2023-12-31"
        },
        "stations": [
            {
                "station_id": "DEL_01",
                "name": "Anand Vihar",
                "latitude": 28.6469,
                "longitude": 77.3152,
                "site_type": "Traffic / Commercial CAAQMS Hub",
                "district": "East Delhi",
                "status": "Active / Primary Telemetry",
                "records": 35064
            },
            {
                "station_id": "DEL_02",
                "name": "IGI Airport (T3)",
                "latitude": 28.5629,
                "longitude": 77.0984,
                "site_type": "Aviation & Suburban Transport CAAQMS Hub",
                "district": "South-West Delhi",
                "status": "Active Network Node",
                "records": 35040
            },
            {
                "station_id": "DEL_03",
                "name": "R.K. Puram",
                "latitude": 28.5632,
                "longitude": 77.1869,
                "site_type": "Dense Residential & Institutional CAAQMS Hub",
                "district": "South Delhi",
                "status": "Active Network Node",
                "records": 35040
            },
            {
                "station_id": "DEL_04",
                "name": "Punjabi Bagh",
                "latitude": 28.6741,
                "longitude": 77.1310,
                "site_type": "Urban Commercial & Residential Corridor",
                "district": "West Delhi",
                "status": "Active Network Node",
                "records": 35040
            },
            {
                "station_id": "DEL_05",
                "name": "Mandir Marg",
                "latitude": 28.6365,
                "longitude": 77.1997,
                "site_type": "Central Cultural & Administrative Zone",
                "district": "Central Delhi",
                "status": "Active Network Node",
                "records": 35040
            },
            {
                "station_id": "DEL_06",
                "name": "Shadipur",
                "latitude": 28.6515,
                "longitude": 77.1581,
                "site_type": "Industrial & Commercial Transit Hub",
                "district": "West-Central Delhi",
                "status": "Active Network Node",
                "records": 35040
            }
        ]
    }
}

SUPPORTED_CITIES = list(CITY_REGISTRY.keys())


def get_city_config(city_id: str = "delhi") -> Dict[str, Any]:
    return CITY_REGISTRY["delhi"]


def get_city_stations(city_id: str = "delhi") -> List[Dict[str, Any]]:
    return CITY_REGISTRY["delhi"]["stations"]


def get_regulatory_tiers(city_id: str = "delhi") -> Dict[str, Any]:
    std_key = CITY_REGISTRY["delhi"]["regulatory_standard"]
    return REGULATORY_STANDARDS[std_key]["tiers"]
