"""City metadata and multi-city station discovery routes."""
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException

from src.services.dataset_service import (
    get_available_cities,
    get_city_details,
    get_city_stations,
)

router = APIRouter(prefix="/cities", tags=["Multi-City Management"])


@router.get("", response_model=List[Dict[str, Any]])
def list_cities():
    """Return configured metropolitan areas and their model/dataset support status."""
    return get_available_cities()


@router.get("/{city}")
def get_city(city: str):
    """Return comprehensive metadata, date ranges, and feature counts for a city."""
    details = get_city_details(city)
    if not details:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found. Supported: delhi, mumbai, vijayawada"
        )
    return details


@router.get("/{city}/stations")
def list_city_stations(city: str):
    """Return the ordered station network and coordinates for a city."""
    stations = get_city_stations(city)
    if not stations:
        raise HTTPException(
            status_code=404,
            detail=f"No stations found for city '{city}'"
        )
    return {
        "city": city.lower(),
        "total_stations": len(stations),
        "stations": stations,
    }
