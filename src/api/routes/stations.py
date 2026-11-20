"""Monitoring station metadata and historical observations routes."""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from src.api.models import StationListResponse, StationInfo, HistoricalObservation
from src.config.settings import EXPECTED_STATIONS, STATION_COORDS, STATION_IDS
from src.api.inference import InferenceEngine

router = APIRouter(prefix="/stations", tags=["Monitoring Stations"])

STATION_DESCRIPTIONS = {
    "Aotizhongxin": "Olympic Sports Center, North Chaoyang (Urban residential & athletic area)",
    "Changping": "Changping suburban district, Northwest Beijing (Mountain-basin boundary)",
    "Dingling": "Ming Tombs ecological background station, Northern suburban clean reference",
    "Dongsi": "Dongcheng inner city, Central Beijing (Dense commercial & traffic corridor)",
    "Guanyuan": "Xicheng central residential district (Governmental & educational center)",
    "Gucheng": "Shijingshan industrial heritage zone, Western Beijing (Near Western Hills)",
    "Huairou": "Huairou northern mountain resort & reservoir basin (Regional background)",
    "Nongzhanguan": "National Agricultural Exhibition Center, East Chaoyang (Embassy & commercial hub)",
    "Shunyi": "Northeastern industrial/airport logistics district (Capital International Airport area)",
    "Tiantan": "Temple of Heaven Park, South Dongcheng (Historical cultural preserve)",
    "Wanliu": "Haidian high-tech and university district, Northwest urban perimeter",
    "Wanshouxigong": "Wanshou West Palace, Xicheng Southern residential center"
}


@router.get("", response_model=StationListResponse)
def list_stations():
    stations_data = []
    for name in EXPECTED_STATIONS:
        lat, lon = STATION_COORDS.get(name, (39.9, 116.4))
        st_id = STATION_IDS.get(name, 0)
        desc = STATION_DESCRIPTIONS.get(name, "Urban monitoring station")
        stations_data.append(StationInfo(
            station_id=st_id,
            name=name,
            latitude=lat,
            longitude=lon,
            description=desc
        ))
    return StationListResponse(total_stations=len(stations_data), stations=stations_data)


@router.get("/{station_name}/history", response_model=List[HistoricalObservation])
def get_station_history(station_name: str,
                        limit: int = Query(24, ge=1, le=168, description="Number of recent hours (max 168)"),
                        offset: int = Query(0, ge=0, description="Offset back into test period")):
    engine = InferenceEngine.get_instance()
    if engine.test_df is None:
        raise HTTPException(status_code=503, detail="Historical dataset not loaded")

    st_df = engine.test_df[engine.test_df["station"] == station_name].sort_values("datetime")
    if len(st_df) == 0:
        raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")

    start_idx = max(0, len(st_df) - offset - limit)
    end_idx = max(0, len(st_df) - offset)
    slice_df = st_df.iloc[start_idx:end_idx]

    observations = []
    for _, row in slice_df.iterrows():
        obs = {
            "datetime": str(row["datetime"]),
            "station": row["station"],
            "PM2.5": float(row.get("PM2.5", 0.0)),
            "PM10": float(row.get("PM10", 0.0)) if "PM10" in row else None,
            "SO2": float(row.get("SO2", 0.0)) if "SO2" in row else None,
            "NO2": float(row.get("NO2", 0.0)) if "NO2" in row else None,
            "CO": float(row.get("CO", 0.0)) if "CO" in row else None,
            "O3": float(row.get("O3", 0.0)) if "O3" in row else None,
            "TEMP": float(row.get("TEMP", 0.0)) if "TEMP" in row else None,
            "PRES": float(row.get("PRES", 0.0)) if "PRES" in row else None,
            "DEWP": float(row.get("DEWP", 0.0)) if "DEWP" in row else None,
            "RAIN": float(row.get("RAIN", 0.0)) if "RAIN" in row else None,
            "WSPM": float(row.get("WSPM", 0.0)) if "WSPM" in row else None,
            "wd_sin": float(row.get("wd_sin", 0.0)) if "wd_sin" in row else None,
            "wd_cos": float(row.get("wd_cos", 0.0)) if "wd_cos" in row else None,
        }
        observations.append(HistoricalObservation(**obs))
    return observations
