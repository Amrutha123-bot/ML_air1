"""Early warning and threshold alert endpoints."""
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Query
from src.api.models import WarningListResponse, EarlyWarningItem
from src.api.inference import InferenceEngine
from src.warnings_system.early_warning import WARNING_TIERS

router = APIRouter(prefix="/warnings", tags=["Early Warnings"])


@router.get("/thresholds")
def get_regulatory_thresholds():
    """Return scientific and regulatory threshold definitions."""
    return {
        "standard": "China Ambient Air Quality Standard GB 3095-2012 / HJ 633-2012",
        "averaging_period": "24-hour average PM2.5 (ug/m^3)",
        "tiers": WARNING_TIERS
    }


@router.get("/{station_name}", response_model=WarningListResponse)
def get_station_warnings(station_name: str,
                         model: Optional[str] = Query(None, description="Model key for predictions"),
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

    engine = InferenceEngine.get_instance()
    try:
        res = engine.predict_for_station(station=station_name, model_name=model, offset=offset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warning generation error: {e}")

    raw_events = res["warnings"]
    items = []
    severities = ["good", "moderate", "unhealthy_sg", "unhealthy", "very_unhealthy", "hazardous"]
    highest_idx = 0

    for ev in raw_events:
        item = EarlyWarningItem(
            station=ev.station,
            lead_time_hours=ev.lead_time_hours,
            predicted_pm25=ev.predicted_pm25,
            warning_category=ev.warning_category,
            alert_code=ev.alert_code,
            severity_label=ev.severity_label,
            color_hex=ev.color_hex,
            trend_type=ev.trend_type,
            is_acute_spike=ev.is_acute_spike,
            rationale=ev.rationale,
            recommended_action=ev.recommended_action,
            reference_datetime=res["reference_datetime"]
        )
        items.append(item)
        if ev.warning_category in severities:
            highest_idx = max(highest_idx, severities.index(ev.warning_category))

    return WarningListResponse(
        station=station_name,
        total_warnings=len(items),
        highest_severity=severities[highest_idx],
        warnings=items
    )
