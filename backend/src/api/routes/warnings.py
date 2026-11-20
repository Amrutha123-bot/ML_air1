"""Early warning and threshold alert endpoints for Delhi."""
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from src.api.models import WarningListResponse, EarlyWarningItem
from src.api.inference import DelhiInferenceEngine

router = APIRouter(prefix="/warnings", tags=["Early Warnings"])

CPCB_WARNING_TIERS = {
    "tier_1_good": {
        "range_ug_m3": [0, 30],
        "category": "Good",
        "label": "Tier 1: Minimal Impact",
        "action": "Normal outdoor activities permitted.",
        "color": "#10B981"
    },
    "tier_2_satisfactory": {
        "range_ug_m3": [30, 60],
        "category": "Satisfactory",
        "label": "Tier 2: Minor Breathing Discomfort",
        "action": "Sensitive individuals consider limiting prolonged outdoor exertion.",
        "color": "#84CC16"
    },
    "tier_3_moderate": {
        "range_ug_m3": [60, 90],
        "category": "Moderate",
        "label": "Tier 3: Moderate Discomfort",
        "action": "People with lung/heart disease should reduce outdoor exertion.",
        "color": "#F59E0B"
    },
    "tier_4_poor": {
        "range_ug_m3": [90, 120],
        "category": "Poor",
        "label": "Tier 4: Breathing Discomfort for Most People",
        "action": "Avoid prolonged outdoor exposure; wear N95 mask in traffic corridors.",
        "color": "#F97316"
    },
    "tier_5_very_poor": {
        "range_ug_m3": [120, 250],
        "category": "Very Poor",
        "label": "Tier 5: Respiratory Illness on Prolonged Exposure",
        "action": "GRAP Stage III emergency protocols: cease construction dust activities, restrict BS-III petrol/BS-IV diesel vehicles.",
        "color": "#EF4444"
    },
    "tier_6_severe": {
        "range_ug_m3": [250, 1000],
        "category": "Severe",
        "label": "Tier 6: Healthy People Affected, Seriously Impacts Sick",
        "action": "GRAP Stage IV severe emergency: close non-essential diesel trucking, mandate work-from-home, school closures.",
        "color": "#7F1D1D"
    }
}


def _tier_from_pm25(val: float):
    if val <= 30:
        return "tier_1_good", CPCB_WARNING_TIERS["tier_1_good"]
    elif val <= 60:
        return "tier_2_satisfactory", CPCB_WARNING_TIERS["tier_2_satisfactory"]
    elif val <= 90:
        return "tier_3_moderate", CPCB_WARNING_TIERS["tier_3_moderate"]
    elif val <= 120:
        return "tier_4_poor", CPCB_WARNING_TIERS["tier_4_poor"]
    elif val <= 250:
        return "tier_5_very_poor", CPCB_WARNING_TIERS["tier_5_very_poor"]
    else:
        return "tier_6_severe", CPCB_WARNING_TIERS["tier_6_severe"]


@router.get("/thresholds")
def get_regulatory_thresholds():
    """Return scientific and regulatory threshold definitions."""
    return {
        "standard": "CPCB National Ambient Air Quality Standards (NAAQS) / Indian AQI",
        "averaging_period": "24-hour average PM2.5 (µg/m³)",
        "tiers": CPCB_WARNING_TIERS
    }


def _build_warnings(offset: int = 0) -> WarningListResponse:
    engine = DelhiInferenceEngine.get_instance()
    preds = engine.predict(model_name="random_forest", offset=offset)
    curr_data = engine.get_current_data()
    ref_time = str(curr_data.get("Timestamp", datetime.utcnow().isoformat()))
    curr_pm25 = float(curr_data.get("PM2.5", 85.0))

    horizon_map = {"+1h": 1, "+6h": 6, "+12h": 12, "+24h": 24, "+48h": 48, "+72h": 72}
    warnings: List[EarlyWarningItem] = []
    severities = []

    for lbl, pred_val in preds.items():
        h = horizon_map.get(lbl, 1)
        tier_key, t_info = _tier_from_pm25(pred_val)
        severities.append(t_info["category"])
        is_spike = (pred_val - curr_pm25) > 30.0

        warnings.append(EarlyWarningItem(
            station="Anand Vihar, Delhi",
            lead_time_hours=h,
            predicted_pm25=round(pred_val, 2),
            warning_category=t_info["category"],
            alert_code=tier_key.upper(),
            severity_label=t_info["label"],
            color_hex=t_info["color"],
            trend_type="Rapid Increase" if is_spike else "Steady",
            is_acute_spike=is_spike,
            rationale=f"Forecast PM2.5 of {pred_val:.1f} µg/m³ falls into {t_info['category']} tier.",
            recommended_action=t_info["action"],
            reference_datetime=ref_time
        ))

    highest_sev = "Severe" if "Severe" in severities else ("Very Poor" if "Very Poor" in severities else "Moderate")

    return WarningListResponse(
        station="Anand Vihar, Delhi",
        total_warnings=len(warnings),
        highest_severity=highest_sev,
        warnings=warnings
    )


@router.get("", response_model=WarningListResponse)
def get_delhi_warnings(
    offset: int = Query(0, ge=0, description="Step back into test period")
):
    """Generate regulatory early warnings from Delhi forecasts."""
    return _build_warnings(offset=offset)


@router.get("/city/{city}/{station_id}", response_model=WarningListResponse)
def get_city_warnings_compat(
    city: str,
    station_id: str,
    offset: int = Query(0, ge=0)
):
    return _build_warnings(offset=offset)


@router.get("/{station_name}", response_model=WarningListResponse)
def get_station_warnings_compat(
    station_name: str,
    offset: int = Query(0, ge=0)
):
    return _build_warnings(offset=offset)
