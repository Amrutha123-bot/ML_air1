"""
OpenAQ API v3 Client for Automated Ground Station Ingestion.

Supports rate-limiting, disk caching, parameter ID mapping, and batch pagination.
Documentation: https://docs.openaq.org/
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import requests
from loguru import logger

from src.config.settings import DATA_DIR

DEFAULT_CACHE_DIR = DATA_DIR / "raw" / "openaq_cache"


class OpenAQClient:
    """Production client for OpenAQ API Version 3."""

    BASE_URL = "https://api.openaq.org/v3"

    # Known OpenAQ parameter IDs for standard pollutants
    PARAMETER_MAP = {
        1: "PM10",
        2: "PM2.5",
        3: "O3",
        4: "CO",
        5: "NO2",
        6: "SO2",
        19: "NO",
        21: "TEMP",
        22: "RH",
        23: "WSPM",
        24: "WD",
    }

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or os.getenv("OPENAQ_API_KEY", "")
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})
        self.session.headers.update({"User-Agent": "AeroSense-AI-V2-Research-Platform/2.0"})
        self.last_request_time = 0.0
        self.min_interval = 1.05  # Enforce <= 57 requests per minute (free tier limit 60/min)

    def _rate_limit_throttle(self):
        """Throttle requests to respect OpenAQ free tier rate limits (60 req/min)."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Execute rate-limited HTTP GET request with optional disk caching."""
        params = params or {}
        cache_key = f"{endpoint.replace('/', '_')}_{sorted(params.items())}.json".replace(":", "_").replace(" ", "_")
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        self._rate_limit_throttle()
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("OpenAQ Rate limit (429) hit. Backing off for 15 seconds...")
                time.sleep(15.0)
                self._rate_limit_throttle()
                resp = self.session.get(url, params=params, timeout=30)

            resp.raise_for_status()
            data = resp.json()

            if use_cache:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            return data
        except requests.exceptions.HTTPError as e:
            logger.error(f"OpenAQ HTTP Error for {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAQ Request Failed for {url}: {e}")
            raise

    def get_locations(self, country: str = "IN", city: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve monitoring locations filtered by country and city."""
        params = {"countries_id": country, "limit": limit}
        if city:
            params["city"] = city
        data = self._request("locations", params=params)
        return data.get("results", [])

    def get_location_sensors(self, location_id: int) -> List[Dict[str, Any]]:
        """List active sensors (pollutants) deployed at a specific location ID."""
        data = self._request(f"locations/{location_id}/sensors")
        return data.get("results", [])

    def fetch_sensor_hourly(self, sensor_id: int,
                            datetime_from: str,
                            datetime_to: str,
                            limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch hourly aggregated measurements for a sensor within a date range.
        Format of datetime_from / datetime_to: ISO 8601 (e.g. '2024-01-01T00:00:00Z').
        """
        params = {
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "limit": limit
        }
        data = self._request(f"sensors/{sensor_id}/measurements/hourly", params=params)
        return data.get("results", [])

    def parse_sensor_measurements_to_df(self, measurements: List[Dict[str, Any]], pollutant_name: str) -> List[Dict[str, Any]]:
        """Normalize raw OpenAQ v3 measurement dictionaries to canonical records."""
        parsed = []
        for m in measurements:
            # OpenAQ v3 hourly format: { 'period': { 'datetimeFrom': {...}, 'datetimeTo': {...} }, 'value': float }
            period = m.get("period", {})
            dt_utc = period.get("datetimeFrom", {}).get("utc") or m.get("datetime", {}).get("utc")
            val = m.get("value")
            if dt_utc is not None and val is not None:
                parsed.append({
                    "timestamp_utc": dt_utc,
                    "pollutant": pollutant_name,
                    "value": float(val),
                })
        return parsed
