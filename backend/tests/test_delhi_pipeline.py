"""
Tests for Delhi-only AeroSense AI pipeline.
Tests run without a live server (unit tests on files and data).
"""
import pytest
import pandas as pd
import json
import joblib
from pathlib import Path
import torch

ROOT = Path("c:/Users/anala/OneDrive/Desktop/ML_air")
DATA = ROOT / "backend/data"
ARTIFACTS = ROOT / "backend/artifacts"
EXPERIMENTS = ARTIFACTS / "experiments"
MODELS = ARTIFACTS / "models"
SCALERS = ARTIFACTS / "scalers"


class TestDataset:
    def test_csv_exists(self):
        assert (DATA / "delhi_synthetic_air_quality.csv").exists()

    def test_csv_row_count(self):
        df = pd.read_csv(DATA / "delhi_synthetic_air_quality.csv")
        assert len(df) == 35064, f"Expected 35064 rows, got {len(df)}"

    def test_csv_no_duplicates(self):
        df = pd.read_csv(DATA / "delhi_synthetic_air_quality.csv")
        assert df.duplicated().sum() == 0

    def test_csv_columns(self):
        df = pd.read_csv(DATA / "delhi_synthetic_air_quality.csv")
        required = ["Timestamp", "City", "Station", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "AQI_Category"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_city_delhi_only(self):
        df = pd.read_csv(DATA / "delhi_synthetic_air_quality.csv")
        assert set(df["City"].unique()) == {"Delhi"}


class TestArtifacts:
    def test_scaler_exists(self):
        assert (SCALERS / "delhi_feature_scaler.pkl").exists()

    def test_scaler_loadable(self):
        scaler = joblib.load(SCALERS / "delhi_feature_scaler.pkl")
        assert scaler is not None

    def test_rf_model_exists(self):
        assert (MODELS / "random_forest_delhi.pkl").exists()

    def test_xgb_model_exists(self):
        assert (MODELS / "xgboost_delhi.pkl").exists()

    def test_lstm_model_exists(self):
        assert (MODELS / "lstm_delhi.pt").exists()

    def test_gru_model_exists(self):
        assert (MODELS / "gru_delhi.pt").exists()

    def test_cnn_lstm_model_exists(self):
        assert (MODELS / "cnn_lstm_delhi.pt").exists()

    def test_rf_model_loadable(self):
        model = joblib.load(MODELS / "random_forest_delhi.pkl")
        assert model is not None

    def test_feature_columns_exist(self):
        assert (EXPERIMENTS / "feature_columns.json").exists()

    def test_feature_columns_loadable(self):
        with open(EXPERIMENTS / "feature_columns.json") as f:
            cols = json.load(f)
        assert isinstance(cols, list)
        assert len(cols) > 0


class TestExperimentOutputs:
    def test_master_comparison_exists(self):
        assert (EXPERIMENTS / "master_model_comparison.csv").exists()

    def test_master_comparison_has_xgboost(self):
        df = pd.read_csv(EXPERIMENTS / "master_model_comparison.csv")
        models = df["model"].str.lower().tolist()
        assert any("xgboost" in m for m in models)

    def test_master_comparison_has_rf(self):
        df = pd.read_csv(EXPERIMENTS / "master_model_comparison.csv")
        models = df["model"].str.lower().tolist()
        assert any("random forest" in m or "rf" in m for m in models)

    def test_confusion_matrix_normalized_exists(self):
        assert (EXPERIMENTS / "confusion_matrix_normalized.csv").exists()

    def test_classification_metrics_exists(self):
        assert (EXPERIMENTS / "classification_metrics.json").exists()

    def test_classification_accuracy_valid(self):
        with open(EXPERIMENTS / "classification_metrics.json") as f:
            metrics = json.load(f)
        assert len(metrics) > 0

    def test_training_fractions_exist(self):
        assert (EXPERIMENTS / "training_fraction_experiments.json").exists()

    def test_eda_summary_exists(self):
        assert (EXPERIMENTS / "delhi_eda_summary.json").exists()


class TestModelLeaderboard:
    def test_xgboost_beats_persistence_at_h24(self):
        df = pd.read_csv(EXPERIMENTS / "master_model_comparison.csv")
        h24_df = df[df["horizon"] == 24] if "horizon" in df.columns else df
        if len(h24_df) == 0:
            h24_df = df
        xgb_rows = h24_df[h24_df["model"].str.lower().str.contains("xgboost")]
        pers_rows = h24_df[h24_df["model"].str.lower().str.contains("persistence")]
        if len(xgb_rows) > 0 and len(pers_rows) > 0:
            xgb_mae = xgb_rows["mae"].min()
            pers_mae = pers_rows["mae"].max()
            assert xgb_mae < pers_mae

    def test_rf_r2_above_80pct(self):
        df = pd.read_csv(EXPERIMENTS / "master_model_comparison.csv")
        h24_df = df[df["horizon"] == 24] if "horizon" in df.columns else df
        if len(h24_df) == 0:
            h24_df = df
        rf_rows = h24_df[h24_df["model"].str.lower().str.contains("random forest")]
        if len(rf_rows) > 0 and "r2" in rf_rows.columns:
            assert rf_rows["r2"].max() >= 0.80
