"""Pydantic request and response schemas for Beijing Air Quality API."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    timestamp: str
    active_model: str = Field(..., json_schema_extra={"example": "stcnn_gru"})
    available_models: List[str]


class StationInfo(BaseModel):
    station_id: int
    name: str
    latitude: float
    longitude: float
    description: str


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


class ForecastResponse(BaseModel):
    station: str
    reference_datetime: str
    model_name: str
    mode: str = Field(default="historical_simulation", description="Historical test sequence replay")
    current_pm25: float
    predictions: List[ForecastHorizonPrediction]


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


class MetricSummaryItem(BaseModel):
    model: str
    overall_MAE: float
    overall_RMSE: float
    overall_R2: float
    warning_F1: float
    skill_score_vs_baseline: float


class ModelMetricsResponse(BaseModel):
    benchmark_dataset: str = "UCI Beijing Air Quality (2016-09 to 2017-02)"
    summary: List[MetricSummaryItem]
    details: Optional[Dict[str, Any]] = None
