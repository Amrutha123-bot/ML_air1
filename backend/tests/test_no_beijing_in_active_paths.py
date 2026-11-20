"""
Verification test: Ensure Beijing is absent from all active application, API, and inference paths
while properly preserved in archive/beijing_legacy/.
"""
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.api.city_stcnn_inference import SUPPORTED_CITIES
from src.api.routes.cities import list_cities, get_city, list_city_stations
from src.api.routes.forecasts import get_city_stcnn_forecast
from src.api.routes.evaluation import list_evaluations, get_city_eval
from src.services.dataset_service import get_available_cities
from src.services.evaluation_service import get_all_evaluation_summaries, get_city_evaluation
from src.services.model_registry_service import get_all_models


def test_archive_beijing_legacy_preserved():
    """Verify Beijing legacy research artifacts remain intact in archive/beijing_legacy/."""
    archive_dir = Path("archive/beijing_legacy")
    assert archive_dir.exists(), "Archive directory archive/beijing_legacy/ must exist"
    assert (archive_dir / "data").exists() or (archive_dir / "scripts").exists()


def test_no_beijing_in_active_inference_supported_cities():
    """Verify city_stcnn_inference only supports active Indian cities and does not include Beijing."""
    assert "beijing" not in SUPPORTED_CITIES
    assert "delhi" in SUPPORTED_CITIES


def test_active_forecast_route_rejects_beijing():
    """Verify active forecast route rejects Beijing with 404."""
    with pytest.raises(HTTPException) as exc:
        get_city_stcnn_forecast(city="beijing", station_id="DEL_01", offset=0)
    assert exc.value.status_code == 404
    assert "beijing" in str(exc.value.detail).lower()


def test_active_city_routes_reject_beijing():
    """Verify active city routes reject Beijing with 404."""
    with pytest.raises(HTTPException) as exc1:
        get_city(city="beijing")
    assert exc1.value.status_code == 404

    with pytest.raises(HTTPException) as exc2:
        list_city_stations(city="beijing")
    assert exc2.value.status_code == 404


def test_active_evaluation_route_rejects_beijing():
    """Verify active evaluation route rejects Beijing with 404."""
    with pytest.raises(HTTPException) as exc:
        get_city_eval(city="beijing")
    assert exc.value.status_code == 404


def test_active_dataset_service_excludes_beijing():
    """Verify dataset service does not list Beijing as an available active city."""
    cities = get_available_cities()
    city_ids = [c["city_id"] if isinstance(c, dict) else c for c in cities]
    assert "beijing" not in city_ids
    assert "delhi" in city_ids


def test_active_evaluation_service_excludes_beijing():
    """Verify evaluation service does not list Beijing under active evaluations."""
    eval_list = get_all_evaluation_summaries()
    assert "beijing" not in eval_list.get("cities", [])
    assert "delhi" in eval_list.get("cities", [])
    assert get_city_evaluation("beijing") is None


def test_active_model_registry_excludes_beijing_checkpoint():
    """Verify active model registry has no active Beijing model checkpoints."""
    models = get_all_models()
    model_ids = [m.get("model_id") for m in models]
    assert "stcnn_gru_beijing" not in model_ids
    assert "stcnn_gru_delhi" in model_ids
