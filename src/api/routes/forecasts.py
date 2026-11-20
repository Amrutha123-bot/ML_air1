"""Multi-horizon PM2.5 forecasting endpoints."""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from src.api.models import ForecastResponse, ForecastHorizonPrediction
from src.api.inference import InferenceEngine
from src.warnings_system.early_warning import WARNING_TIERS

router = APIRouter(prefix="/forecasts", tags=["Forecasting"])


@router.get("/{station_name}", response_model=ForecastResponse)
def get_station_forecast(station_name: str,
                         model: Optional[str] = Query(None, description="Model key: stcnn_gru|xgboost|random_forest|lstm|cnn_lstm|persistence"),
                         horizon: Optional[int] = Query(None, description="Specific horizon (1, 6, 12, 24, 48, 72) or omit for all"),
                         offset: int = Query(0, ge=0, description="Step back into test period")):
    # Handle potential direct function calls where FastAPI Query object is default
    if hasattr(offset, "default"):
        offset = int(offset.default) if offset.default is not ... else 0
    else:
        try:
            offset = int(offset)
        except (ValueError, TypeError):
            offset = 0

    if hasattr(model, "default"):
        model = model.default if model.default is not ... else None

    if hasattr(horizon, "default"):
        horizon = horizon.default if horizon.default is not ... else None

    engine = InferenceEngine.get_instance()
    try:
        res = engine.predict_for_station(station=station_name, model_name=model, offset=offset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    forecasts_map = res["forecasts"]
    predictions = []

    for h, val in forecasts_map.items():
        if horizon is not None and h != horizon:
            continue
        cat, info = engine.warning_system.classify_pm25(val)
        predictions.append(ForecastHorizonPrediction(
            horizon_hours=h,
            predicted_pm25=round(val, 1),
            alert_tier=info["label"],
            color_hex=info["color"]
        ))

    return ForecastResponse(
        station=res["station"],
        reference_datetime=res["reference_datetime"],
        model_name=res["model_name"],
        mode="historical_simulation",
        current_pm25=res["current_pm25"],
        predictions=predictions
    )
