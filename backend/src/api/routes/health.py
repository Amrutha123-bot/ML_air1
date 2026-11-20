"""Health and system status routes for AeroSense AI Delhi."""
from datetime import datetime
import torch
from fastapi import APIRouter
from src.api.models import HealthResponse
from src.api.inference import DelhiInferenceEngine

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    engine = DelhiInferenceEngine.get_instance()
    active_m = engine.get_active_model_name()
    available = engine.get_available_models()
    device_info = "cuda" if torch.cuda.is_available() else "cpu"

    return HealthResponse(
        status="healthy",
        version="2.1.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        device=device_info,
        active_model=active_m,
        available_models=available,
        available_cities_count=1,
        available_cities=["delhi"],
        dataset_availability={"delhi": True},
        data_mode="HISTORICAL / CPCB CALIBRATED",
        dataset_date_range="2020-01-01 to 2023-12-31 (35,064 records)",
        real_time_ingestion=False,
    )
