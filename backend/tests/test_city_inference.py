"""
Unit tests for city-aware ST-CNN-GRU inference and API endpoints in AeroSense AI V2.
"""
import pytest
from fastapi import HTTPException

from src.api.routes.forecasts import get_city_stcnn_forecast
from src.api.city_stcnn_inference import (
    predict_city_station,
    _load_stations,
    _load_city_model,
    SUPPORTED_CITIES,
    FORECAST_HORIZONS,
)


def test_supported_cities_configuration():
    """Verify supported cities and checkpoint bindings (Beijing must be absent)."""
    assert "beijing" not in SUPPORTED_CITIES
    assert "delhi" in SUPPORTED_CITIES
    assert SUPPORTED_CITIES["delhi"] == "stcnn_gru_delhi.pt"


def test_station_list_loading():
    """Verify station list loading for Delhi."""
    del_stations = _load_stations("delhi")
    assert len(del_stations) == 12
    assert "DEL_01" in del_stations
    assert "DEL_12" in del_stations


def test_city_model_loading():
    """Verify ST-CNN-GRU checkpoint loads cleanly for Delhi."""
    del_model = _load_city_model("delhi")
    assert del_model is not None
    assert del_model.network is not None


def test_delhi_inference_direct():
    """Verify direct ST-CNN-GRU inference on Delhi DEL_01."""
    res = predict_city_station("delhi", "DEL_01", offset=0)
    assert res["city"] == "delhi"
    assert res["station"] == "DEL_01"
    assert res["feature_count"] == 57
    assert res["sequence_length"] == 24
    assert res["station_count"] == 12
    assert res["input_shape"] == [1, 24, 12, 57]
    assert len(res["forecasts"]) == 6
    for h in FORECAST_HORIZONS:
        assert h in res["forecasts"]
        assert res["forecasts"][h] >= 0.0


def test_invalid_city_raises():
    """Verify unsupported city raises ValueError (including beijing)."""
    with pytest.raises(ValueError, match="Unsupported city"):
        predict_city_station("beijing", "BJ_01")

    with pytest.raises(ValueError, match="Unsupported city"):
        predict_city_station("paris", "DEL_01")


def test_invalid_station_raises():
    """Verify invalid station raises ValueError."""
    with pytest.raises(ValueError, match="Station 'INVALID_99' not found"):
        predict_city_station("delhi", "INVALID_99")


def test_api_city_forecast_delhi():
    """Verify endpoint handler for Delhi DEL_01."""
    data = get_city_stcnn_forecast(city="delhi", station_id="DEL_01", offset=0)
    assert data.city == "delhi"
    assert data.station == "DEL_01"
    assert data.model_name == "stcnn_gru_delhi"
    assert data.mode == "city_spatiotemporal"
    assert data.feature_count == 57
    assert data.sequence_length == 24
    assert data.station_count == 12
    assert data.input_shape == [1, 24, 12, 57]
    assert len(data.predictions) == 6
    assert 1 in data.forecasts


def test_api_city_forecast_beijing_rejected():
    """Verify 404 HTTPException for Beijing in API."""
    with pytest.raises(HTTPException) as exc:
        get_city_stcnn_forecast(city="beijing", station_id="BJ_01", offset=0)
    assert exc.value.status_code == 404


def test_api_city_forecast_invalid_station():
    """Verify 404 HTTPException for invalid station in API."""
    with pytest.raises(HTTPException) as exc:
        get_city_stcnn_forecast(city="delhi", station_id="NON_EXISTENT_STN", offset=0)
    assert exc.value.status_code == 404
    assert "not found" in str(exc.value.detail).lower()
