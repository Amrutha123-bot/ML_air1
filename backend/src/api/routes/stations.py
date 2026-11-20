"""Monitoring station metadata and historical observations routes for Delhi."""
from typing import List, Optional
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from src.api.models import StationListResponse, StationInfo, HistoricalObservation
from src.config.city_registry import get_city_stations, CITY_REGISTRY

router = APIRouter(prefix="/stations", tags=["Monitoring Stations"])

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
CSV_PATH = _DATA_DIR / "delhi_air_quality.csv"
if not CSV_PATH.exists():
    CSV_PATH = _DATA_DIR / "delhi_synthetic_air_quality.csv"

_REGISTERED_STATIONS = [
    StationInfo(
        station_id=st["station_id"],
        name=f"{st['name']}, Delhi",
        latitude=st["latitude"],
        longitude=st["longitude"],
        description=f"{st['site_type']} ({st['district']}) — {st['status']}",
        district=st["district"],
        site_type=st["site_type"],
        status=st["status"],
        records=st["records"]
    )
    for st in get_city_stations("delhi")
]


@router.get("", response_model=StationListResponse)
def list_stations():
    """Return all Delhi CPCB continuous air quality monitoring stations in the network."""
    return StationListResponse(
        total_stations=len(_REGISTERED_STATIONS),
        stations=_REGISTERED_STATIONS
    )


@router.get("/network-summary")
def get_network_summary():
    """Return comprehensive Delhi monitoring network statistics."""
    return {
        "city": "Delhi NCR",
        "network_statistics": CITY_REGISTRY["delhi"]["network_summary"],
        "stations": _REGISTERED_STATIONS
    }


@router.get("/{station_name}/history", response_model=List[HistoricalObservation])
def get_station_history(
    station_name: str,
    limit: int = Query(24, ge=1, le=720, description="Number of recent hours"),
    offset: int = Query(0, ge=0, description="Offset back into test period")
):
    if not CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Delhi dataset not found")

    df = pd.read_csv(CSV_PATH)
    if offset > 0:
        df = df.iloc[:-offset]

    tail_df = df.tail(limit)
    obs = []
    for _, row in tail_df.iterrows():
        obs.append(HistoricalObservation(
            datetime=str(row.get("Timestamp", "")),
            station="Anand Vihar, Delhi",
            PM2_5=float(row.get("PM2.5", 0.0)),
            PM10=float(row.get("PM10", 0.0)),
            SO2=float(row.get("SO2", 0.0)),
            NO2=float(row.get("NO2", 0.0)),
            CO=float(row.get("CO", 0.0)),
            O3=float(row.get("O3", 0.0)),
            TEMP=float(row.get("Temperature", 25.0)),
            WSPM=float(row.get("Wind_Speed", 2.0)),
        ))
    return obs
