"""Dataset specifications, split details, and schema routes."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.services.dataset_service import (
    get_all_datasets,
    get_dataset_details,
    get_dataset_manifest,
    get_pollutant_availability,
)

router = APIRouter(prefix="/datasets", tags=["Dataset Management"])


@router.get("", response_model=List[Dict[str, Any]])
def list_datasets():
    """Return summary statistics and preprocessing status for all canonical datasets."""
    return get_all_datasets()


@router.get("/manifest")
def get_manifest(dataset_id: Optional[str] = Query(None, description="Optional dataset ID")):
    """Return versioned dataset manifest records conforming to Phase 2 specification."""
    data = get_dataset_manifest(dataset_id)
    if dataset_id and not data:
        raise HTTPException(status_code=404, detail=f"Dataset manifest for '{dataset_id}' not found.")
    return data


@router.get("/pollutant-availability")
def get_pollutant_matrix(
    city: Optional[str] = Query(None, description="City ID (delhi, mumbai, vijayawada)"),
    pollutant: Optional[str] = Query(None, description="Pollutant (PM2.5, PM10, etc.)"),
):
    """Return pollutant availability matrix computed from canonical observation lakehouse."""
    data = get_pollutant_availability(city, pollutant)
    if (city or pollutant) and not data:
        raise HTTPException(
            status_code=404,
            detail=f"Pollutant availability not found for city '{city}' and pollutant '{pollutant}'.",
        )
    return data


@router.get("/{city}")
def get_dataset(city: str):
    """Return split-level record counts (train/val/test) and boundary loss reconciliation."""
    details = get_dataset_details(city)
    if not details:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset for city '{city}' not found."
        )
    return details
