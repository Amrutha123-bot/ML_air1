"""Pydantic request and response schemas for AeroSense AI API."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field("2.0.0", json_schema_extra={"example": "2.0.0"})
    timestamp: str
    device: str = "cpu"
    active_model: str = Field(..., json_schema_extra={"example": "stcnn_gru"})
    available_models: List[str]
    available_cities_count: int = 1
    available_cities: List[str] = ["delhi"]
    dataset_availability: Dict[str, bool] = {"delhi": True}
    data_mode: str = "OFFLINE / HISTORICAL"
    dataset_date_range: str = "2023-01-01 to 2024-12-31 (Indian-City Lakehouse)"
    real_time_ingestion: bool = False


class StationInfo(BaseModel):
    station_id: Union[str, int]
    name: str
    latitude: float
    longitude: float
    description: str
    district: Optional[str] = None
    site_type: Optional[str] = None
    status: Optional[str] = None
    records: Optional[int] = None


class StationListResponse(BaseModel):
    total_stations: int
    stations: List[StationInfo]


class HistoricalObservation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    datetime: str
    station: str
    PM2_5: float = Field(..., alias="PM2.5")
    PM10: Optional[float] = None
    SO2: Optional[float] = None
    NO2: Optional[float] = None
    CO: Optional[float] = None
    O3: Optional[float] = None
    TEMP: Optional[float] = None
    PRES: Optional[float] = None
    DEWP: Optional[float] = None
    RAIN: Optional[float] = None
    WSPM: Optional[float] = None
    wd_sin: Optional[float] = None
    wd_cos: Optional[float] = None


class ForecastHorizonPrediction(BaseModel):
    horizon_hours: int
    predicted_pm25: float
    alert_tier: str
    color_hex: str
    units: str = "µg/m³"


class ForecastResponse(BaseModel):
    station: str
    reference_datetime: str
    model_name: str
    mode: str = Field(default="historical_simulation", description="Historical test sequence replay")
    current_pm25: float
    predictions: List[ForecastHorizonPrediction]
    # Prediction transparency metadata
    artifact_path: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    input_observation_datetime: Optional[str] = None
    prediction_generation_datetime: Optional[str] = None
    data_source: str = "CPCB Indian National Ambient Air Quality Network (Delhi NCR)"
    data_mode: str = "OFFLINE / HISTORICAL"
    units: str = "µg/m³"
    feature_count: int = 57
    sequence_length: int = 24


class EarlyWarningItem(BaseModel):
    station: str
    lead_time_hours: int
    predicted_pm25: float
    warning_category: str
    alert_code: str
    severity_label: str
    color_hex: str
    trend_type: str
    is_acute_spike: bool
    rationale: str
    recommended_action: str
    reference_datetime: str


class WarningListResponse(BaseModel):
    station: str
    total_warnings: int
    highest_severity: str
    warnings: List[EarlyWarningItem]