"""Automated integration tests for Delhi-only FastAPI backend routes."""
import pytest
from src.api.routes.health import get_health
from src.api.routes.stations import list_stations, get_station_history
from src.api.routes.forecasts import get_forecast, get_station_forecast
from src.api.routes.warnings import get_delhi_warnings, get_regulatory_thresholds
from src.api.routes.metrics import get_model_metrics, get_leaderboard, get_confusion_matrix, get_training_fractions
from src.api.routes.analytics import get_delhi_trends, get_eda_summary, get_pollutant_stats


def test_health_endpoint():
    resp = get_health()
    assert resp.status == "healthy"
    assert resp.version == "2.1.0"
    assert "random_forest" in resp.available_models
    assert [c.lower() for c in resp.available_cities] == ["delhi"]


def test_stations_endpoint():
    resp = list_stations()
    assert resp.total_stations == 6
    assert len(resp.stations) == 6
    st = resp.stations[0]
    assert "Anand Vihar" in st.name
    assert st.latitude > 20.0
    assert st.longitude > 70.0


def test_station_history_endpoint():
    resp = get_station_history("Anand Vihar", limit=10, offset=0)
    assert isinstance(resp, list)
    assert len(resp) == 10
    first = resp[0]
    assert hasattr(first, "PM2_5")
    assert first.PM2_5 >= 0.0


def test_forecast_endpoint():
    resp = get_forecast(model="random_forest", horizon=None, offset=0)
    assert "Anand Vihar" in resp.station
    assert len(resp.predictions) == 6
    p1 = resp.predictions[0]
    assert p1.horizon_hours in [1, 6, 12, 24, 48, 72]
    assert p1.predicted_pm25 >= 0.0
    assert p1.alert_tier != ""


def test_warnings_endpoint():
    resp = get_delhi_warnings(offset=0)
    assert "Anand Vihar" in resp.station
    assert len(resp.warnings) == 6
    w = resp.warnings[0]
    assert w.lead_time_hours in [1, 6, 12, 24, 48, 72]
    assert w.rationale != ""
    assert w.recommended_action != ""


def test_regulatory_thresholds_endpoint():
    resp = get_regulatory_thresholds()
    assert "tiers" in resp
    assert len(resp["tiers"]) == 6


def test_metrics_endpoints():
    m = get_model_metrics()
    assert "metrics" in m
    assert len(m["metrics"]) > 0

    lb = get_leaderboard()
    assert "leaderboard" in lb
    assert len(lb["leaderboard"]) > 0

    cm = get_confusion_matrix()
    assert "labels" in cm
    assert len(cm["labels"]) == 6

    fr = get_training_fractions()
    assert "experiments" in fr or isinstance(fr, list) or isinstance(fr, dict)


def test_analytics_endpoints():
    trends = get_delhi_trends(limit=24)
    assert len(trends) == 24
    assert "PM2.5" in trends[0]

    eda = get_eda_summary()
    assert "total_records" in eda or "eda_summary" in eda or len(eda) > 0

    stats = get_pollutant_stats()
    assert "PM2.5" in stats
    assert stats["PM2.5"]["mean"] > 0
