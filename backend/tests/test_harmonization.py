"""Unit tests for multi-city data harmonization and spatial graph construction."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config.city_registry import (
    CITY_REGISTRY,
    get_city_stations,
    compute_city_distance_matrix,
    compute_city_adjacency_matrix,
    compute_haversine_distance,
)
from src.data.harmonizer import CANONICAL_COLUMNS, PHYSICAL_BOUNDS, CANONICAL_DIR


def test_city_registry_completeness():
    """Verify all major candidate cities exist in the registry with valid metadata."""
    expected_cities = ["delhi", "mumbai", "bengaluru", "hyderabad", "chennai", "vijayawada"]
    for c in expected_cities:
        assert c in CITY_REGISTRY
        city = CITY_REGISTRY[c]
        assert "city_name" in city
        assert "country" in city
        assert "timezone" in city
        assert "regulatory_standard" in city
        assert len(city["stations"]) >= 2


def test_haversine_distance_calculation():
    """Verify mathematical calculation of Haversine distance between coordinates."""
    # Delhi Anand Vihar to R K Puram (~15.7 km)
    d = compute_haversine_distance(28.6468, 77.3160, 28.5632, 77.1864)
    assert 14.0 < d < 18.0

    # Same point distance must be zero
    assert compute_haversine_distance(28.6, 77.2, 28.6, 77.2) == 0.0


def test_distance_and_adjacency_matrix_properties():
    """Verify spatial matrices are square, symmetric, and row-normalized."""
    for city_id in ["delhi", "mumbai", "vijayawada"]:
        n = len(get_city_stations(city_id))
        dist = compute_city_distance_matrix(city_id)
        adj = compute_city_adjacency_matrix(city_id)

        assert dist.shape == (n, n)
        assert adj.shape == (n, n)
        # Distance matrix must be symmetric with zero diagonal
        np.testing.assert_allclose(dist, dist.T, atol=1e-5)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-5)
        # Adjacency matrix must have row sums equal to 1.0 (probability/stochastic matrix)
        row_sums = adj.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(n), atol=1e-4)


def test_canonical_datasets_exist_and_conform():
    """Verify saved canonical parquet files conform to CANONICAL_COLUMNS."""
    for city_id in ["delhi", "mumbai", "vijayawada"]:
        obs_file = CANONICAL_DIR / city_id / "observations.parquet"
        assert obs_file.exists(), f"Missing observations.parquet for {city_id}"
        df = pd.read_parquet(obs_file)
        assert len(df) > 0

        # Check all canonical columns are present
        for col in CANONICAL_COLUMNS:
            assert col in df.columns, f"Column {col} missing in {city_id} observations"

        # Check PM2.5 physical bounds
        valid_pm = df["PM2.5"].dropna()
        assert (valid_pm >= 0.0).all(), f"Found negative PM2.5 in {city_id}"
        assert (valid_pm <= 1000.0).all(), f"Found PM2.5 > 1000 in {city_id}"
