"""Health and system status routes."""
from datetime import datetime
from fastapi import APIRouter
from src.api.models import HealthResponse
from src.api.inference import InferenceEngine

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    engine = InferenceEngine.get_instance()
    active_m = engine.get_active_model_name()
    available = list(engine.models.keys())
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        active_model=active_m,
        available_models=available
    )
