"""
CPCB / data.gov.in CAAQMS Data Ingestion Client.

Interfaces with the National Open Government Data (OGD) Platform India:
Resource ID: 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (Real-time & Archived CAAQMS AQI)
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
from loguru import logger

from src.config.settings import DATA_DIR

DEFAULT_CPCB_CACHE_DIR = DATA_DIR / "raw" / "cpcb_cache"


class CPCBClient:
    """Client for CPCB CAAQMS feeds via data.gov.in API."""

    BASE_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or os.getenv("DATAGOVIN_API_KEY", "")
        self.cache_dir = Path(cache_dir or DEFAULT_CPCB_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def fetch_records(self,
                      state: Optional[str] = None,
                      city: Optional[str] = None,
                      station: Optional[str] = None,
                      limit: int = 1000,
                      offset: int = 0,
                      use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch records from data.gov.in CAAQMS resource.
        Filters can include state, city, and station name.
        """
        cache_key = f"cpcb_{state}_{city}_{station}_{limit}_{offset}.json".replace(" ", "_")
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": offset,
        }
        if state:
            params["filters[state]"] = state
        if city:
            params["filters[city]"] = city
        if station:
            params["filters[station]"] = station

        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("records", [])

            if use_cache:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(records, f)

            return records
        except Exception as e:
            logger.warning(f"CPCB data.gov.in query failed ({e}). Returning empty list or cached fallback.")
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []

    @staticmethod
    def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw data.gov.in record into canonical format."""
        # Raw fields: 'country', 'state', 'city', 'station', 'last_update', 'pollutant_id', 'pollutant_min', 'pollutant_max', 'pollutant_avg'
        return {
            "city": rec.get("city"),
            "station": rec.get("station"),
            "timestamp": rec.get("last_update"),
            "pollutant": rec.get("pollutant_id"),
            "value": float(rec.get("pollutant_avg", 0.0)) if rec.get("pollutant_avg") else None,
            "latitude": float(rec.get("latitude")) if "latitude" in rec else None,
            "longitude": float(rec.get("longitude")) if "longitude" in rec else None,
        }
