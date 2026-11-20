"""Model registry and checkpoint discovery routes."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.services.model_registry_service import (
    get_all_models,
    get_model_by_id,
)

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("", response_model=List[Dict[str, Any]])
def list_models(
    include_experimental: bool = Query(
        True, description="Whether to include experimental seed replication checkpoints"
    )
):
    """Return all production models, benchmarks, and experimental checkpoints."""
    return get_all_models(include_experimental=include_experimental)


@router.get("/{model_name}")
def get_model(model_name: str):
    """Return architecture details, artifact status, and evaluation metrics for a model."""
    info = get_model_by_id(model_name)
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in registry."
        )
    return info
