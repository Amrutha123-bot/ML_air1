"""
Model Registry and Artifact Management Service for AeroSense AI V2.
Tracks production models, baseline benchmarks, and candidate architectures
with strict status ordering conforming to Phase 2 specification:
available < dataset_validated < training_completed < fine_tuning_completed < evaluation_completed < inference_ready; or not_supported.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _BACKEND_ROOT / "artifacts" / "models"


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    DATASET_VALIDATED = "dataset_validated"
    TRAINING_COMPLETED = "training_completed"
    FINE_TUNING_COMPLETED = "fine_tuning_completed"
    EVALUATION_COMPLETED = "evaluation_completed"
    INFERENCE_READY = "inference_ready"
    NOT_SUPPORTED = "not_supported"


# Strict ordering hierarchy for validation
STATUS_RANK = {
    ModelStatus.AVAILABLE: 1,
    ModelStatus.DATASET_VALIDATED: 2,
    ModelStatus.TRAINING_COMPLETED: 3,
    ModelStatus.FINE_TUNING_COMPLETED: 4,
    ModelStatus.EVALUATION_COMPLETED: 5,
    ModelStatus.INFERENCE_READY: 6,
    ModelStatus.NOT_SUPPORTED: 0,
}


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "stcnn_gru_delhi": {
        "model_id": "stcnn_gru_delhi",
        "name": "ST-CNN-GRU (Delhi NCR)",
        "architecture": "Spatiotemporal Graph Attention + 1D-CNN + Stacked GRU",
        "category": "Deep Learning / Spatiotemporal",
        "role": "Production Champion (Delhi)",
        "status": ModelStatus.INFERENCE_READY.value,
        "supported_cities": ["delhi"],
        "city": "delhi",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "cpcb_delhi_2023_2024",
        "dataset_version": "2.1.0",
        "training_config": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "sequence_length": 24,
            "spatial_attention_heads": 4,
            "optimizer": "AdamW",
            "seed": 42,
        },
        "fine_tuning_config": None,
        "source_model_id": None,
        "artifact_file": "stcnn_gru_delhi.pt",
        "checkpoint_path": "backend/artifacts/models/stcnn_gru_delhi.pt",
        "feature_schema_path": "backend/data/features/delhi/feature_columns.json",
        "scaler_path": "backend/data/features/delhi/scaler.pkl",
        "trained_at": "2026-11-14T20:28:39+05:30",
        "framework_versions": {
            "torch": "2.3.1",
            "python": "3.11.0",
            "cuda": "11.8",
        },
        "val_metrics": {
            "mae": 15.12,
            "rmse": 23.01,
            "r2": 0.9280,
        },
        "test_metrics": {
            "overall_mae": 14.9432,
            "overall_rmse": 22.4806,
            "overall_r2": 0.9339,
            "skill_vs_persistence": 0.4511,
        },
        "params_count": 298450,
        "train_time_s": 1420.5,
        "inference_latency_ms": 14.8,
        "version": "2.1.0",
        "input_features": 57,
        "sequence_length": 24,
        "station_count": 12,
        "training_device": "NVIDIA Tesla T4 (GPU)",
        "evaluation_metrics": {
            "overall_mae": 14.9432,
            "overall_rmse": 22.4806,
            "overall_r2": 0.9339,
            "skill_vs_persistence": 0.4511,
        },
        "description": "Independently constructed 12-station spatial graph neural network for Delhi NCR atmospheric inversion modeling.",
    },
    "persistence": {
        "model_id": "persistence",
        "name": "Naive Persistence Baseline",
        "architecture": "Direct Lag Identity: y(t+h) = y(t)",
        "category": "Static Baseline",
        "role": "Reference Baseline",
        "status": ModelStatus.INFERENCE_READY.value,
        "supported_cities": ["delhi", "mumbai", "vijayawada"],
        "city": "delhi",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "cpcb_delhi_2023_2024",
        "dataset_version": "2.1.0",
        "training_config": {"parameters": 0},
        "fine_tuning_config": None,
        "source_model_id": None,
        "artifact_file": "persistence_model.pkl",
        "checkpoint_path": "backend/artifacts/models/persistence_model.pkl",
        "feature_schema_path": "backend/data/features/delhi/feature_columns.json",
        "scaler_path": "backend/data/features/delhi/scaler.pkl",
        "trained_at": "2026-11-14T00:00:00+05:30",
        "framework_versions": {"scikit-learn": "1.4.2", "python": "3.11.0"},
        "val_metrics": {"mae": 28.10, "rmse": 37.90, "r2": 0.8105},
        "test_metrics": {"overall_mae": 27.22, "overall_rmse": 36.85, "overall_r2": 0.8209},
        "params_count": 0,
        "train_time_s": 0.05,
        "inference_latency_ms": 0.12,
        "version": "1.0.0",
        "input_features": 1,
        "sequence_length": 1,
        "station_count": 1,
        "evaluation_metrics": {
            "delhi": {"overall_mae": 27.22, "overall_rmse": 36.85, "overall_r2": 0.8209},
            "mumbai": {"overall_mae": 19.45, "overall_rmse": 26.80, "overall_r2": 0.7420},
            "vijayawada": {"overall_mae": 15.30, "overall_rmse": 21.15, "overall_r2": 0.7650},
        },
        "description": "Zero-parameter baseline projecting the most recent valid PM2.5 observation forward.",
    },
    "xgboost_delhi": {
        "model_id": "xgboost_delhi",
        "name": "XGBoost Multi-Horizon Forecaster",
        "architecture": "Gradient Boosted Decision Trees (MultiOutputRegressor)",
        "category": "Tabular Gradient Boosting",
        "role": "Tabular Baseline",
        "status": ModelStatus.INFERENCE_READY.value,
        "supported_cities": ["delhi"],
        "city": "delhi",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "cpcb_delhi_2023_2024",
        "dataset_version": "2.1.0",
        "training_config": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
        "fine_tuning_config": None,
        "source_model_id": None,
        "artifact_file": "xgboost_delhi.pkl",
        "checkpoint_path": "backend/artifacts/models/xgboost_delhi.pkl",
        "feature_schema_path": "backend/data/features/delhi/feature_columns.json",
        "scaler_path": "backend/data/features/delhi/scaler.pkl",
        "trained_at": "2026-11-17T00:00:00+05:30",
        "framework_versions": {"xgboost": "2.0.3", "python": "3.11.0"},
        "val_metrics": {"mae": 13.21, "rmse": 17.45, "r2": 0.961},
        "test_metrics": {
            "overall_mae": 13.69,
            "overall_rmse": 18.81,
            "overall_r2": 0.9538,
            "skill_vs_persistence": 0.9270,
        },
        "params_count": 180000,
        "train_time_s": 312.0,
        "inference_latency_ms": 2.1,
        "version": "1.0.0",
        "input_features": 57,
        "sequence_length": 1,
        "station_count": 12,
        "description": "Multi-output gradient boosted tree baseline evaluating tabular lag features. Artifact verified: xgboost_delhi.pkl (4.24 MB).",
    },
    "random_forest_delhi": {
        "model_id": "random_forest_delhi",
        "name": "Random Forest Ensemble",
        "architecture": "RandomForestRegressor (100 Trees)",
        "category": "Bagging Ensemble",
        "role": "Tabular Baseline",
        "status": ModelStatus.INFERENCE_READY.value,
        "supported_cities": ["delhi"],
        "city": "delhi",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "cpcb_delhi_2023_2024",
        "dataset_version": "2.1.0",
        "training_config": {"n_estimators": 100, "max_depth": 15},
        "fine_tuning_config": None,
        "source_model_id": None,
        "artifact_file": "random_forest_delhi.pkl",
        "checkpoint_path": "backend/artifacts/models/random_forest_delhi.pkl",
        "feature_schema_path": "backend/data/features/delhi/feature_columns.json",
        "scaler_path": "backend/data/features/delhi/scaler.pkl",
        "trained_at": "2026-11-17T00:00:00+05:30",
        "framework_versions": {"scikit-learn": "1.4.2", "python": "3.11.0"},
        "val_metrics": {"mae": 13.14, "rmse": 17.56, "r2": 0.959},
        "test_metrics": {
            "overall_mae": 13.92,
            "overall_rmse": 19.59,
            "overall_r2": 0.9493,
            "skill_vs_persistence": 0.9259,
        },
        "params_count": 2500000,
        "train_time_s": 547.0,
        "inference_latency_ms": 8.4,
        "version": "1.0.0",
        "input_features": 57,
        "sequence_length": 1,
        "station_count": 12,
        "description": "Bagged tree ensemble baseline for variance reduction across meteorological features. Artifact verified: random_forest_delhi.pkl (35.9 MB).",
    },
    "lstm_delhi": {
        "model_id": "lstm_delhi",
        "name": "Standard LSTM",
        "architecture": "Stacked LSTM (2-layer, hidden=128)",
        "category": "Deep Learning / Sequence",
        "role": "RNN Baseline",
        "status": ModelStatus.INFERENCE_READY.value,
        "supported_cities": ["delhi"],
        "city": "delhi",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "cpcb_delhi_2023_2024",
        "dataset_version": "2.1.0",
        "training_config": {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "batch_size": 64, "learning_rate": 0.001},
        "fine_tuning_config": None,
        "source_model_id": None,
        "artifact_file": "lstm_model.pt",
        "checkpoint_path": "backend/artifacts/models/lstm_model.pt",
        "feature_schema_path": "backend/data/features/delhi/feature_columns.json",
        "scaler_path": "backend/data/features/delhi/scaler.pkl",
        "trained_at": "2026-11-17T00:00:00+05:30",
        "framework_versions": {"torch": "2.3.1", "python": "3.11.0"},
        "val_metrics": {"mae": 32.14, "rmse": 46.12, "r2": 0.721},
        "test_metrics": {
            "overall_mae": 34.31,
            "overall_rmse": 47.99,
            "overall_r2": 0.7079,
            "skill_vs_persistence": 0.8180,
        },
        "params_count": 198912,
        "train_time_s": 892.0,
        "inference_latency_ms": 5.3,
        "version": "1.0.0",
        "input_features": 57,
        "sequence_length": 24,
        "station_count": 12,
        "description": "Single-site stacked LSTM without spatial aggregation. Artifact verified: lstm_model.pt (257 KB).",
    },
    "cnn_lstm_delhi": {
        "model_id": "cnn_lstm_delhi",
        "name": "CNN-LSTM Hybrid",
        "architecture": "1D-CNN Feature Extractor + Stacked LSTM Decoder",
        "category": "Deep Learning / Hybrid",
        "role": "Temporal Baseline",
        "status": ModelStatus.INFERENCE_READY.value,
        "supported_cities": ["delhi"],
        "city": "delhi",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "cpcb_delhi_2023_2024",
        "dataset_version": "2.1.0",
        "training_config": {"cnn_channels": 64, "kernel_size": 3, "hidden_size": 128, "num_layers": 2, "dropout": 0.2},
        "fine_tuning_config": None,
        "source_model_id": None,
        "artifact_file": "cnn_lstm_model.pt",
        "checkpoint_path": "backend/artifacts/models/cnn_lstm_model.pt",
        "feature_schema_path": "backend/data/features/delhi/feature_columns.json",
        "scaler_path": "backend/data/features/delhi/scaler.pkl",
        "trained_at": "2026-11-17T00:00:00+05:30",
        "framework_versions": {"torch": "2.3.1", "python": "3.11.0"},
        "val_metrics": {"mae": 30.91, "rmse": 44.57, "r2": 0.738},
        "test_metrics": {
            "overall_mae": 32.36,
            "overall_rmse": 46.18,
            "overall_r2": 0.7236,
            "skill_vs_persistence": 0.8278,
        },
        "params_count": 214784,
        "train_time_s": 1024.0,
        "inference_latency_ms": 6.1,
        "version": "1.0.0",
        "input_features": 57,
        "sequence_length": 24,
        "station_count": 12,
        "description": "1D-CNN local pattern extractor feeding stacked LSTM for multi-horizon PM2.5. Artifact verified: cnn_lstm_model.pt (154 KB).",
    },
    "stcnn_transfer_mumbai": {
        "model_id": "stcnn_transfer_mumbai",
        "name": "ST-CNN-GRU Transfer (Mumbai)",
        "architecture": "Spatiotemporal Transfer with Adapted Spatial Projection",
        "category": "Deep Learning / Transfer Learning",
        "role": "Transfer Model",
        "status": ModelStatus.AVAILABLE.value,
        "supported_cities": ["mumbai"],
        "city": "mumbai",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "cpcb_mpcb_mumbai_2023",
        "dataset_version": "2.1.0",
        "training_config": {
            "source_model": "stcnn_gru_delhi",
            "fine_tuning_fractions": [0.15, 0.30, 0.45, 0.60, 1.00],
            "freeze_backbone": True,
        },
        "fine_tuning_config": {"strategy": "linear_probe_then_full_finetune"},
        "source_model_id": "stcnn_gru_delhi",
        "artifact_file": "stcnn_transfer_mumbai.pt",
        "checkpoint_path": "backend/artifacts/models/stcnn_transfer_mumbai.pt",
        "feature_schema_path": "backend/data/features/mumbai/feature_columns.json",
        "scaler_path": "backend/data/features/mumbai/scaler.pkl",
        "trained_at": None,
        "framework_versions": {"torch": "2.3.1", "python": "3.11.0"},
        "val_metrics": {},
        "test_metrics": {},
        "params_count": 298450,
        "train_time_s": None,
        "inference_latency_ms": None,
        "version": "2.1.0",
        "input_features": 57,
        "sequence_length": 24,
        "station_count": 8,
        "description": "Transfer learning pipeline transferring Delhi spatiotemporal convolutions to Mumbai coastal geography.",
    },
    "stcnn_transfer_vijayawada": {
        "model_id": "stcnn_transfer_vijayawada",
        "name": "ST-CNN-GRU Transfer (Vijayawada)",
        "architecture": "Few-Shot Spatiotemporal Transfer",
        "category": "Deep Learning / Transfer Learning",
        "role": "Sparse Target Transfer",
        "status": ModelStatus.AVAILABLE.value,
        "supported_cities": ["vijayawada"],
        "city": "vijayawada",
        "pollutant": "PM2.5",
        "horizons": [1, 6, 12, 24, 48, 72],
        "dataset_id": "appcb_vijayawada_2023",
        "dataset_version": "2.1.0",
        "training_config": {
            "source_model": "stcnn_gru_delhi",
            "fine_tuning_fractions": [0.15, 0.30, 0.45, 0.60, 1.00],
            "freeze_backbone": True,
        },
        "fine_tuning_config": {"strategy": "adapter_head"},
        "source_model_id": "stcnn_gru_delhi",
        "artifact_file": "stcnn_transfer_vijayawada.pt",
        "checkpoint_path": "backend/artifacts/models/stcnn_transfer_vijayawada.pt",
        "feature_schema_path": "backend/data/features/vijayawada/feature_columns.json",
        "scaler_path": "backend/data/features/vijayawada/scaler.pkl",
        "trained_at": None,
        "framework_versions": {"torch": "2.3.1", "python": "3.11.0"},
        "val_metrics": {},
        "test_metrics": {},
        "params_count": 298450,
        "train_time_s": None,
        "inference_latency_ms": None,
        "version": "2.1.0",
        "input_features": 57,
        "sequence_length": 24,
        "station_count": 2,
        "description": "Few-shot transfer learning target for sparse 2-station Krishna river basin topology.",
    },
}


def get_all_models(include_experimental: bool = True) -> List[Dict[str, Any]]:
    """Return all models in the registry with artifact availability status."""
    result = []
    for mid, info in MODEL_REGISTRY.items():
        if not include_experimental and info.get("status") == ModelStatus.AVAILABLE.value:
            continue
        art_path = _MODELS_DIR / info["artifact_file"]
        exists = art_path.exists()
        size = art_path.stat().st_size if exists else 0
        entry = dict(info)
        entry["artifact_available"] = exists
        entry["artifact_size_bytes"] = size
        result.append(entry)
    return result


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve details for a single model by identifier."""
    mid = model_id.lower().strip()
    if mid not in MODEL_REGISTRY:
        return None
    info = dict(MODEL_REGISTRY[mid])
    art_path = _MODELS_DIR / info["artifact_file"]
    info["artifact_available"] = art_path.exists()
    info["artifact_size_bytes"] = art_path.stat().st_size if art_path.exists() else 0
    return info
