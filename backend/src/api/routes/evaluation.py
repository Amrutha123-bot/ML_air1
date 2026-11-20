"""Model evaluation and benchmark metrics routes."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException

from src.services.evaluation_service import (
    get_all_evaluation_summaries,
    get_city_evaluation,
    get_model_evaluation,
    get_master_comparison,
    get_transfer_learning_results,
)

router = APIRouter(prefix="/evaluation", tags=["Model Evaluation & Benchmarks"])


@router.get("")
def list_evaluations():
    """Return high-level empirical evaluation comparison across supported cities."""
    return get_all_evaluation_summaries()


@router.get("/benchmark/all")
def get_full_benchmark():
    """Return full multi-model, multi-city, multi-horizon comparison table.
    Source: AeroSense_AI_Training_Results.zip / master_model_comparison.csv"""
    return {"rows": get_master_comparison(), "source": "master_model_comparison.csv"}


@router.get("/transfer/{target_city}")
def get_transfer_results(target_city: str):
    """Return data-fraction sweep transfer learning results for a target city (mumbai|vijayawada)."""
    allowed = {"mumbai", "vijayawada"}
    if target_city.lower() not in allowed:
        raise HTTPException(
            status_code=404,
            detail=f"Transfer results available for: {', '.join(allowed)}. Got: '{target_city}'"
        )
    return {"target_city": target_city, "rows": get_transfer_learning_results(target_city)}


@router.get("/{city}")
def get_city_eval(city: str):
    """Return complete multi-horizon evaluation metrics (MAE, RMSE, R²) for a city."""
    res = get_city_evaluation(city)
    if not res:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation metrics for city '{city}' not found. Supported: delhi"
        )
    return res


@router.get("/{city}/{model_name}")
def get_model_eval(city: str, model_name: str):
    """Return horizon-by-horizon metrics for a specific model in a city."""
    res = get_model_evaluation(city, model_name)
    if not res:
        raise HTTPException(
            status_code=404,
            detail=f"Model evaluation for '{model_name}' in city '{city}' not found."
        )
    return res
