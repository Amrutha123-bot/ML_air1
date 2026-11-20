"""
Unit and integration tests for Phase 2: Central Registries and Validation.
Tests city registry, dataset manifest, pollutant availability, and HTTP 422 parameter validation.
"""
import pytest
from fastapi import HTTPException
from src.services.dataset_service import (
    get_available_cities,
    get_city_details,
    get_dataset_manifest,
    get_pollutant_availability,
)
from src.services.model_registry_service import (
    get_all_models,
    get_model_by_id,
    ModelStatus,
    STATUS_RANK,
)
from src.api.routes.forecasts import get_station_forecast, get_city_stcnn_forecast
from src.api.routes.datasets import get_manifest, get_pollutant_matrix


def test_city_registry_phase2_fields():
    """Verify city registry contains all Phase 2 specified fields."""
    cities = get_available_cities()
    assert len(cities) >= 3
    for c in cities:
        assert "city_id" in c
        assert "display_name" in c
        assert "state" in c
        assert "lat" in c and isinstance(c["lat"], (int, float))
        assert "lon" in c and isinstance(c["lon"], (int, float))
        assert c["timezone"] == "Asia/Kolkata"
        assert c["data_status"] in ["historical_data_available", "dataset_validation_pending", "data_unavailable"]
        assert c["live_status"] in ["live_observations_available", "not_available"]
        assert c["pipeline_status"] in ["training_pending", "fine_tuning_pending", "evaluation_completed", "forecasting_available"]

    delhi = next(c for c in cities if c["city_id"] == "delhi")
    assert delhi["pipeline_status"] == "forecasting_available"


def test_dataset_manifest_phase2():
    """Verify dataset manifest conforms to Phase 2 schema with valid sha256."""
    manifest = get_dataset_manifest()
    assert len(manifest) >= 3
    for entry in manifest:
        assert "dataset_id" in entry
        assert "city" in entry
        assert "source_name" in entry
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64
        assert "row_count" in entry and entry["row_count"] > 0
        assert "validation_status" in entry


def test_pollutant_availability_phase2():
    """Verify pollutant availability matrix returns valid computed coverage."""
    avail = get_pollutant_availability()
    assert "delhi" in avail
    assert "PM2.5" in avail["delhi"]
    delhi_pm25 = avail["delhi"]["PM2.5"]
    assert delhi_pm25["valid_hourly_obs"] > 0
    assert delhi_pm25["unit"] == "µg/m³"
    assert delhi_pm25["usable_144h_windows"] > 0
    assert delhi_pm25["suitable_horizons"] == [1, 6, 12, 24, 48, 72]
    assert delhi_pm25["status"] == "forecasting_available"


def test_model_registry_status_ordering():
    """Verify model registry status follows strict order specification."""
    models = get_all_models()
    assert len(models) >= 2
    for m in models:
        st = m["status"]
        assert st in [s.value for s in ModelStatus]
        # Rank must be defined
        assert ModelStatus(st) in STATUS_RANK

    delhi_stcnn = get_model_by_id("stcnn_gru_delhi")
    assert delhi_stcnn["status"] == ModelStatus.INFERENCE_READY.value
    assert STATUS_RANK[ModelStatus.INFERENCE_READY] > STATUS_RANK[ModelStatus.AVAILABLE]


def test_unsupported_horizon_returns_422():
    """Verify requesting an unsupported prediction horizon (e.g. 5h) returns HTTP 422."""
    with pytest.raises(HTTPException) as exc:
        get_station_forecast(station_name="Anand Vihar", horizon=5)
    assert exc.value.status_code == 422
    assert "horizon" in str(exc.value.detail).lower()


def test_unsupported_transfer_city_returns_422():
    """Verify requesting ST-CNN-GRU for a city whose model is not inference_ready returns HTTP 422."""
    with pytest.raises(HTTPException) as exc:
        get_city_stcnn_forecast(city="mumbai", station_id="MUM_01")
    assert exc.value.status_code == 422
    assert "fine-tuning" in str(exc.value.detail).lower() or "awaiting" in str(exc.value.detail).lower()
