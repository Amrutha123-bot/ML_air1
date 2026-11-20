"""Unit tests for DataQualityValidator in AeroSense AI V2."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.quality import DataQualityValidator
from src.data.harmonizer import CANONICAL_COLUMNS, CANONICAL_DIR


@pytest.fixture
def clean_test_df():
    """Generates a synthetic valid canonical DataFrame."""
    n_hours = 24
    dates = [datetime(2023, 1, 1, 0, 0) + timedelta(hours=i) for i in range(n_hours)]
    records = []
    for st_id in ["ST01", "ST02"]:
        for i, dt in enumerate(dates):
            records.append({
                "timestamp_utc": dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "timestamp_local": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "city_id": "test_city",
                "station_id": st_id,
                "station_name": f"Station {st_id}",
                "latitude": 28.5 + (0.1 if st_id == "ST02" else 0.0),
                "longitude": 77.2 + (0.1 if st_id == "ST02" else 0.0),
                "PM2.5": 45.0 + float(i),
                "PM10": 80.0 + float(i),
                "SO2": 10.0,
                "NO2": 25.0,
                "CO": 0.8,
                "O3": 35.0,
                "NH3": 15.0,
                "TEMP": 20.0 + float(i % 5),
                "PRES": 1012.0,
                "DEWP": 12.0,
                "RAIN": 0.0,
                "WSPM": 2.5,
                "wd_sin": 0.707,
                "wd_cos": 0.707,
                "data_source": "test_source",
                "quality_flag": 0,
            })
    return pd.DataFrame(records)


def test_validator_passes_on_clean_data(clean_test_df):
    """Clean data should satisfy all 5 validation checks."""
    validator = DataQualityValidator()
    report = validator.validate_dataset(clean_test_df, "test_city")
    assert report["passed_all"] is True
    assert report["checks"]["schema_conformance"]["passed"] is True
    assert report["checks"]["pm25_physical_bounds"]["passed"] is True
    assert report["checks"]["temporal_monotonicity"]["passed"] is True
    assert report["checks"]["sensor_freeze_detection"]["passed"] is True
    assert report["checks"]["missing_rate_audit"]["passed"] is True


def test_validator_catches_schema_violation(clean_test_df):
    """Dropping a canonical column must fail schema conformance."""
    validator = DataQualityValidator()
    bad_df = clean_test_df.drop(columns=["WSPM"])
    report = validator.validate_dataset(bad_df, "test_city")
    assert report["passed_all"] is False
    assert report["checks"]["schema_conformance"]["passed"] is False
    assert "WSPM" in report["checks"]["schema_conformance"]["missing_columns"]


def test_validator_catches_negative_pm25(clean_test_df):
    """Negative PM2.5 readings must fail physical bounds check."""
    validator = DataQualityValidator()
    bad_df = clean_test_df.copy()
    bad_df.loc[3, "PM2.5"] = -12.5
    report = validator.validate_dataset(bad_df, "test_city")
    assert report["passed_all"] is False
    assert report["checks"]["pm25_physical_bounds"]["passed"] is False
    assert report["checks"]["pm25_physical_bounds"]["negative_count"] >= 1


def test_validator_catches_exceeding_pm25(clean_test_df):
    """PM2.5 readings > 1000 µg/m³ must fail physical bounds check."""
    validator = DataQualityValidator()
    bad_df = clean_test_df.copy()
    bad_df.loc[5, "PM2.5"] = 1500.0
    report = validator.validate_dataset(bad_df, "test_city")
    assert report["passed_all"] is False
    assert report["checks"]["pm25_physical_bounds"]["passed"] is False
    assert report["checks"]["pm25_physical_bounds"]["exceed_1000_count"] >= 1


def test_validator_catches_non_monotonic_timestamps(clean_test_df):
    """Shuffled or non-monotonic timestamps must fail monotonicity check."""
    validator = DataQualityValidator()
    bad_df = clean_test_df.copy()
    # Swap timestamps at index 0 and 5
    t0 = bad_df.loc[0, "timestamp_utc"]
    bad_df.loc[0, "timestamp_utc"] = bad_df.loc[5, "timestamp_utc"]
    bad_df.loc[5, "timestamp_utc"] = t0
    report = validator.validate_dataset(bad_df, "test_city")
    assert report["passed_all"] is False
    assert report["checks"]["temporal_monotonicity"]["passed"] is False


def test_validator_catches_sensor_freeze(clean_test_df):
    """Sensor stuck at identical value for >= 12h must trigger sensor freeze detection."""
    validator = DataQualityValidator(freeze_threshold_hours=12)
    bad_df = clean_test_df.copy()
    # Set 15 consecutive identical non-zero values for ST01
    bad_df.loc[0:14, "PM2.5"] = 85.0
    report = validator.validate_dataset(bad_df, "test_city")
    assert report["checks"]["sensor_freeze_detection"]["passed"] is False
    assert report["checks"]["sensor_freeze_detection"]["frozen_sensor_incidents"] >= 1


def test_validator_catches_high_missing_rate(clean_test_df):
    """Missing rate > threshold must fail missing rate check."""
    validator = DataQualityValidator(max_allowed_missing_pct=25.0)
    bad_df = clean_test_df.copy()
    # Make 40% of PM2.5 null
    bad_df.loc[0:19, "PM2.5"] = np.nan
    report = validator.validate_dataset(bad_df, "test_city")
    assert report["passed_all"] is False
    assert report["checks"]["missing_rate_audit"]["passed"] is False


def test_canonical_lakehouse_quality():
    """Verify all 3 active Indian lakehouse canonical datasets pass automated quality validation."""
    validator = DataQualityValidator(max_allowed_missing_pct=25.0)
    for city_id in ["delhi", "mumbai", "vijayawada"]:
        parquet_path = CANONICAL_DIR / city_id / "observations.parquet"
        assert parquet_path.exists(), f"Missing {parquet_path}"
        df = pd.read_parquet(parquet_path)
        report = validator.validate_dataset(df, city_id)
        assert report["passed_all"] is True, f"City {city_id} failed quality checks: {report['checks']}"
