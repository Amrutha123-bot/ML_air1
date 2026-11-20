"""Environmental analytics and exploratory data analysis routes for Delhi."""
from typing import Any, Dict, List, Optional
import json
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/analytics", tags=["Environmental Analytics"])

_BACKEND_DIR = Path(__file__).parent.parent.parent.parent
_CSV_PATH = _BACKEND_DIR / "data" / "delhi_air_quality.csv"
if not _CSV_PATH.exists():
    _CSV_PATH = _BACKEND_DIR / "data" / "delhi_synthetic_air_quality.csv"
_EDA_PATH = _BACKEND_DIR / "artifacts" / "experiments" / "delhi_eda_summary.json"


@router.get("/trends", response_model=List[Dict[str, Any]])
def get_delhi_trends(
    limit: int = Query(168, ge=1, le=720, description="Number of recent hourly observations")
):
    """Return recent multi-pollutant trajectory for Delhi."""
    if not _CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Delhi dataset file not found")
    
    df = pd.read_csv(_CSV_PATH)
    tail = df.tail(limit)
    records = []
    for _, row in tail.iterrows():
        records.append({
            "datetime": str(row.get("Timestamp", "")),
            "timestamp": str(row.get("Timestamp", "")),
            "PM2.5": float(row.get("PM2.5", 0.0)),
            "PM10": float(row.get("PM10", 0.0)),
            "NO2": float(row.get("NO2", 0.0)),
            "SO2": float(row.get("SO2", 0.0)),
            "CO": float(row.get("CO", 0.0)),
            "O3": float(row.get("O3", 0.0)),
            "AQI": float(row.get("AQI", 0.0)),
            "AQI_Category": str(row.get("AQI_Category", "Moderate")),
            "Temperature": float(row.get("Temperature", 25.0)),
            "Humidity": float(row.get("Humidity", 50.0)),
            "Wind_Speed": float(row.get("Wind_Speed", 2.0)),
        })
    return records


@router.get("/trends/{city}/{station}", response_model=List[Dict[str, Any]])
def get_trends_compat(
    city: str,
    station: str,
    limit: int = Query(168, ge=1, le=720)
):
    return get_delhi_trends(limit=limit)


@router.get("/eda", response_model=Dict[str, Any])
def get_eda_summary():
    """Return genuine exploratory data analysis metrics computed during model training."""
    if not _EDA_PATH.exists():
        raise HTTPException(status_code=404, detail="EDA summary not found")
    with open(_EDA_PATH, "r") as f:
        return json.load(f)


@router.get("/correlation", response_model=List[Dict[str, Any]])
@router.get("/correlation/{city}", response_model=List[Dict[str, Any]])
def get_pollutant_correlation(city: Optional[str] = "delhi"):
    """Return Pearson correlation between all criteria pollutants."""
    if not _CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Delhi dataset file not found")
    df = pd.read_csv(_CSV_PATH)
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "Wind_Speed"]
    cols = [c for c in pollutants if c in df.columns]
    corr_df = df[cols].corr().round(3)
    
    out = []
    for pol in cols:
        row = {"pollutant": pol}
        for other in cols:
            row[other] = float(corr_df.loc[pol, other])
        out.append(row)
    return out


@router.get("/pollutants", response_model=Dict[str, Any])
def get_pollutant_stats():
    """Return descriptive statistics for each pollutant in Delhi dataset."""
    if not _CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Delhi dataset file not found")
    df = pd.read_csv(_CSV_PATH)
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    stats = {}
    for p in pollutants:
        if p in df.columns:
            stats[p] = {
                "mean": round(float(df[p].mean()), 2),
                "std": round(float(df[p].std()), 2),
                "min": round(float(df[p].min()), 2),
                "p25": round(float(df[p].quantile(0.25)), 2),
                "median": round(float(df[p].median()), 2),
                "p75": round(float(df[p].quantile(0.75)), 2),
                "max": round(float(df[p].max()), 2),
            }
    return stats
