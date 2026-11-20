"""Integration and endpoint validation tests for the AeroSense AI V2 API rebuild.

Tests are written to match the ACTUAL service response schemas documented in:
- src/services/dataset_service.py (CITY_CONFIG)
- src/services/model_registry_service.py (MODEL_REGISTRY)
- src/services/evaluation_service.py (DELHI_TEST_EVALUATION)
- src/services/analytics_service.py (get_city_comparison, get_pollutant_trends)
- src/api/routes/forecasts.py (get_city_stcnn_forecast signature)
"""
import pytest
from fastapi import HTTPException

from src.api.routes.health import get_health
from src.api.routes.cities import list_cities, get_city, list_city_stations
from src.api.routes.datasets import list_datasets, get_dataset
from src.api.routes.models_registry import list_models, get_model
from src.api.routes.evaluation import list_evaluations, get_city_eval, get_model_eval
from src.api.routes.analytics import get_trends, get_correlation, get_comparison
from src.api.routes.forecasts import get_city_stcnn_forecast
from src.api.routes.warnings import get_city_warnings, get_station_warnings


def test_v2_health_endpoint():
    resp = get_health()
    assert resp.status == "healthy"
    assert resp.version == "2.0.0"
    assert resp.device in ["cuda", "cpu"]
    assert resp.available_cities_count >= 3
    assert {"delhi", "mumbai", "vijayawada"}.issubset(set(resp.available_cities))
    assert "beijing" not in resp.available_cities
    assert resp.dataset_availability["delhi"] is True
    assert len(resp.available_models) >= 1


def test_v2_cities_endpoints():
    cities = list_cities()
    assert len(cities) >= 3
    city_ids = [c["city_id"] for c in cities]
    assert "beijing" not in city_ids
    assert "delhi" in city_ids
    assert "mumbai" in city_ids
    assert "vijayawada" in city_ids

    # Beijing details raises 404
    with pytest.raises(HTTPException) as exc_bj:
        get_city(city="beijing")
    assert exc_bj.value.status_code == 404

    # Delhi details
    delhi = get_city(city="delhi")
    assert delhi["city_id"] == "delhi"
    assert delhi["station_count"] == 12
    assert delhi["primary_model"] == "stcnn_gru_delhi"

    # Non-existent city raises 404
    with pytest.raises(HTTPException) as exc_info:
        get_city(city="atlantis")
    assert exc_info.value.status_code == 404


def test_v2_city_stations():
    # Beijing stations raises 404
    with pytest.raises(HTTPException) as exc_bj:
        list_city_stations(city="beijing")
    assert exc_bj.value.status_code == 404

    delhi_st = list_city_stations(city="delhi")
    assert delhi_st["city"] == "delhi"
    assert delhi_st["total_stations"] == 12
    assert len(delhi_st["stations"]) == 12
    assert delhi_st["stations"][0]["station_id"] == "DEL_01"
    assert delhi_st["stations"][11]["station_id"] == "DEL_12"

    mumbai_st = list_city_stations(city="mumbai")
    assert mumbai_st["total_stations"] == 8
    assert mumbai_st["stations"][0]["station_id"] == "MUM_01"

    vja_st = list_city_stations(city="vijayawada")
    assert vja_st["total_stations"] == 2
    assert vja_st["stations"][0]["station_id"] == "VJA_01"


def test_v2_datasets_endpoints():
    datasets = list_datasets()
    assert len(datasets) >= 3
    city_ids = [d["city_id"] for d in datasets]
    assert "beijing" not in city_ids

    # Beijing dataset raises 404
    with pytest.raises(HTTPException):
        get_dataset(city="beijing")

    # Delhi — actual values from CITY_CONFIG
    delhi_ds = get_dataset(city="delhi")
    assert delhi_ds["city_id"] == "delhi"
    assert delhi_ds["splits"]["train"]["rows"] == 146364
    assert delhi_ds["splits"]["validation"]["rows"] == 31368
    assert delhi_ds["splits"]["test"]["rows"] == 31356
    del_reconciliation = delhi_ds["boundary_row_reconciliation"]
    assert del_reconciliation["dropped_boundary_rows"] == 1152

    with pytest.raises(HTTPException):
        get_dataset(city="non_existent_city")


def test_v2_model_registry_endpoints():
    all_models = list_models(include_experimental=True)
    assert len(all_models) >= 2
    model_ids = [m["model_id"] for m in all_models]
    assert "stcnn_gru_beijing" not in model_ids
    assert "stcnn_gru_delhi" in model_ids
    assert "persistence" in model_ids

    # Query specific model
    delhi_model = get_model(model_name="stcnn_gru_delhi")
    assert delhi_model["model_id"] == "stcnn_gru_delhi"
    assert delhi_model["category"] == "Deep Learning / Spatiotemporal"
    assert "delhi" in delhi_model["supported_cities"]
    assert delhi_model["artifact_available"] is True
    assert delhi_model["evaluation_metrics"]["overall_mae"] == 14.9432
    assert delhi_model["evaluation_metrics"]["overall_rmse"] == 22.4806

    with pytest.raises(HTTPException):
        get_model(model_name="stcnn_gru_beijing")


def test_v2_evaluation_endpoints():
    eval_summaries = list_evaluations()
    assert "cities" in eval_summaries
    assert "beijing" not in eval_summaries["cities"]
    assert "delhi" in eval_summaries["cities"]
    assert "beijing_summary" not in eval_summaries
    assert eval_summaries["delhi_summary"]["overall_mae"] == 14.9432

    # Beijing evaluation returns 404
    with pytest.raises(HTTPException):
        get_city_eval(city="beijing")

    delhi_eval = get_city_eval(city="delhi")
    assert delhi_eval["city"] == "delhi"
    assert "stcnn_gru" in delhi_eval["models"]
    assert delhi_eval["models"]["stcnn_gru"]["overall"]["mae"] == 14.9432
    assert delhi_eval["models"]["stcnn_gru"]["overall"]["rmse"] == 22.4806
    assert delhi_eval["models"]["stcnn_gru"]["overall"]["r2"] == 0.9339

    delhi_stcnn = get_model_eval(city="delhi", model_name="stcnn_gru")
    assert delhi_stcnn["model_key"] == "stcnn_gru"
    horizons = delhi_stcnn["metrics"]["horizons"]
    assert len(horizons) == 6
    h1 = next(h for h in horizons if h["horizon_hours"] == 1)
    assert h1["mae"] == 13.0940


def test_v2_analytics_endpoints():
    comp = get_comparison()
    assert len(comp) >= 3
    comp_city_ids = [c["city_id"] for c in comp]
    assert "beijing" not in comp_city_ids
    assert "delhi" in comp_city_ids
    comp_names = [c["name"] for c in comp]
    assert "Delhi NCR" in comp_names

    # Pollutant trends for Delhi
    trends = get_trends(city="delhi", station="DEL_01", limit=24)
    assert len(trends) == 24
    assert "PM2.5" in trends[0]
    assert "datetime" in trends[0]

    # Pollutant correlation matrix for Delhi
    corr = get_correlation(city="delhi")
    assert len(corr) == 6
    pollutants = [c["pollutant"] for c in corr]
    assert "PM2.5" in pollutants


def test_v2_delhi_stcnn_inference():
    resp = get_city_stcnn_forecast(city="delhi", station_id="DEL_01", offset=0)
    assert resp.city == "delhi"
    assert resp.station == "DEL_01"
    assert resp.model_name
    assert resp.mode == "city_spatiotemporal"
    assert len(resp.predictions) == 6
    for pred in resp.predictions:
        assert pred.horizon_hours in [1, 6, 12, 24, 48, 72]
        assert pred.predicted_pm25 >= 0.0
        assert isinstance(pred.alert_tier, str) and len(pred.alert_tier) > 0


def test_v2_city_warnings():
    delhi_warn = get_city_warnings(city="delhi", station_id="DEL_01", offset=0)
    assert delhi_warn.station == "DEL_01"
    assert hasattr(delhi_warn, "warnings")
    assert len(delhi_warn.warnings) > 0
    assert delhi_warn.highest_severity in ["safe", "caution", "warning", "hazardous"]

    # Test Beijing warnings raises 404
    with pytest.raises(HTTPException) as exc_bj:
        get_city_warnings(city="beijing", station_id="BJ_01", offset=0)
    assert exc_bj.value.status_code == 404

    # Test invalid station raises 404
    with pytest.raises(HTTPException) as exc_info:
        get_city_warnings(city="delhi", station_id="NON_EXISTENT_STATION", offset=0)
    assert exc_info.value.status_code == 404

    # Test unsupported city raises 404
    with pytest.raises(HTTPException) as exc_info:
        get_city_warnings(city="paris", station_id="ST_01", offset=0)
    assert exc_info.value.status_code == 404


def test_v2_station_name_resolution_and_routing():
    # Passing Delhi station name to get_station_warnings auto-resolves to Delhi DEL_01
    delhi_name_warn = get_station_warnings(station_name="Anand Vihar", offset=0)
    assert delhi_name_warn.station == "DEL_01"
    assert len(delhi_name_warn.warnings) > 0

    # Passing Delhi station ID directly
    delhi_id_warn = get_station_warnings(station_name="DEL_01", offset=0)
    assert delhi_id_warn.station == "DEL_01"


def test_v2_cross_city_isolation():
    # Invalid station requested under Delhi must raise 404
    with pytest.raises(HTTPException) as exc_info:
        get_city_stcnn_forecast(city="delhi", station_id="NON_EXISTENT", offset=0)
    assert exc_info.value.status_code == 404

    # Beijing city requested must raise 404
    with pytest.raises(HTTPException) as exc_info:
        get_city_stcnn_forecast(city="beijing", station_id="DEL_01", offset=0)
    assert exc_info.value.status_code == 404
