# AeroSense AI — Complete Backend File and Folder Directory Guide

This document provides a file-by-file and folder-by-folder breakdown of the entire `backend/` directory of the **AeroSense AI** Delhi Air Quality Forecasting and Early-Warning System. Every module, script, dataset, checkpoint, and test is described in detail, including its role, interfaces, and architecture.

---

## 1. Directory Tree Overview

```
backend/
├── Dockerfile                           # Container definition for production backend deployment
├── Procfile                             # Process file for cloud PaaS deployment (e.g. Heroku/Render)
├── pyproject.toml                       # Python project configuration, dependencies & pytest settings
├── render.yaml                          # Render cloud service deployment specification
├── requirements.txt                     # Production Python package dependencies
├── requirements-colab.txt               # Google Colab specific environment dependencies
├── requirements-dev.txt                 # Development, testing, and linting dependencies
├── setup.py                             # Local package editable install definition (`pip install -e .`)
│
├── artifacts/                           # Serialized machine learning models, scalers & evaluation metrics
│   ├── experiments/                     # Numerical evaluation results, confusion matrices & JSON metrics
│   ├── models/                          # Serialized PyTorch (.pt) and Joblib (.pkl) model weights
│   └── scalers/                         # Feature scaling parameters fitted on training partition
│
├── data/                                # Clean air quality datasets, generator & metadata
│   ├── canonical/                       # Canonical harmonized multi-station parquet datasets
│   ├── features/                        # Partitioned train/val/test feature matrices (parquet)
│   ├── dataset_metadata.json            # Machine-readable dataset specifications and audit report
│   ├── delhi_air_quality.csv            # Master benchmark dataset (35,064 records, 2020–2023)
│   ├── delhi_synthetic_air_quality.csv  # Calibrated reference copy matching benchmark structure
│   ├── generate_delhi_synthetic.py      # Autoregressive AR(1) seasonal dataset generation script
│   └── README.md                        # Dataset documentation and column definitions
│
├── scripts/                             # Offline execution scripts for training, testing & data export
│   ├── audit_canonical.py               # Validates schema consistency of canonical parquet files
│   ├── build_city_features.py           # Generates 64 tabular time-series features from raw series
│   ├── download_data.py                 # Automated data fetcher from CPCB / OpenAQ sources
│   ├── evaluate_all.py                  # Standalone multi-model cross-evaluation script
│   ├── export_frontend_data.py          # Serializes backend dataset & metrics into frontend TypeScript
│   ├── generate_colab_notebooks.py      # Programmatic notebook generator for training workflows
│   ├── make_inspection_notebook.py      # Builds data quality inspection notebooks
│   ├── plot_residuals.py                # Generates residual error distribution plots
│   ├── retrain_xgboost_hpo.py           # Optuna/Grid hyperparameter optimization for XGBoost
│   ├── run_calibration.py               # Calibrates warning thresholds against CPCB standards
│   ├── run_dataset_inspection.py        # Automated quality check on missing values & duplicates
│   ├── run_phase2_acquisition.py        # Telemetry ingestion validator
│   ├── run_pipeline.py                  # End-to-end legacy pipeline runner
│   ├── run_real_pipeline.py             # Feature engineering, training, and evaluation coordinator
│   ├── train_all.py                     # Trains classical and deep learning models across horizons
│   ├── train_delhi_models.py            # Master Delhi production training script (PyTorch & Scikit-Learn)
│   ├── verify_deployment.py             # Pre-flight readiness check for API routes and weights
│   └── verify_smoke_inprocess.py        # In-process smoke test for FastAPI routes
│
├── src/                                 # Production backend application source code
│   ├── __init__.py                      # Package initialization
│   ├── api/                             # FastAPI application, routing, inference & schemas
│   │   ├── __init__.py
│   │   ├── inference.py                 # Thread-safe singleton `DelhiInferenceEngine`
│   │   ├── main.py                      # FastAPI root application entry point & CORS configuration
│   │   ├── models.py                    # Pydantic request/response validation schemas
│   │   └── routes/                      # Modular REST endpoint routers (/api/v1)
│   │       ├── analytics.py             # /api/v1/analytics: EDA statistics & historical trends
│   │       ├── forecasts.py             # /api/v1/forecasts: Multi-horizon ML model inference
│   │       ├── health.py                # /api/v1/health: System telemetry & device status
│   │       ├── metrics.py               # /api/v1/metrics: Leaderboards, confusion matrix & fractions
│   │       ├── stations.py              # /api/v1/stations: CPCB monitoring station registry
│   │       └── warnings.py              # /api/v1/warnings: Regulatory CPCB NAAQS early-warning alerts
│   ├── config/                          # Configuration registries and application settings
│   │   ├── city_registry.py             # Delhi 6-station CAAQMS registry & CPCB regulatory tiers
│   │   └── settings.py                  # Path constants, server host/port, and logging settings
│   ├── data/                            # Data processing, feature extraction & validation routines
│   │   ├── features.py                  # 64-feature pipeline (lags, rolling stats, cyclic embeddings)
│   │   ├── harmonizer.py                # Canonical schema standardization and coordinate mapping
│   │   ├── preprocess.py                # Data cleaning, outlier clipping, and imputation routines
│   │   ├── quality.py                   # Physical boundary validation (CPCB plausible limits)
│   │   ├── validate.py                  # Schema compliance and missingness validator
│   │   └── sources/                     # External API client wrappers (CPCB & OpenAQ)
│   ├── evaluation/                      # Model benchmarking and visualization utilities
│   │   ├── evaluator.py                 # Computes MAE, RMSE, R², sMAPE, and skill vs. baseline
│   │   ├── metrics.py                   # Low-level mathematical implementations of metric functions
│   │   └── visualize.py                 # Matplotlib / Seaborn plotting utilities for evaluations
│   ├── fog_processing/                  # Edge/Fog computing simulation modules
│   │   └── fog_node.py                  # Lightweight edge node for on-device inference & filtering
│   ├── models/                          # Core machine learning & deep learning model definitions
│   │   ├── baseline.py                  # Persistence, Seasonal Persistence (24h), Moving Average
│   │   ├── cnn_lstm.py                  # Hybrid 1D-CNN temporal feature extractor + LSTM decoder
│   │   ├── lstm.py                      # Stacked 2-layer PyTorch LSTM sequence-to-sequence model
│   │   ├── random_forest.py             # Multi-output Random Forest regressor wrapper
│   │   └── xgboost_model.py             # Direct multi-horizon XGBoost regressor wrapper
│   ├── services/                        # Service layer abstractions
│   ├── training/                        # PyTorch training loop utilities
│   │   ├── optimizer.py                 # Learning rate schedulers and Adam optimization setup
│   │   ├── scheduler.py                 # StepLR and CosineAnnealing learning rate decay routines
│   │   └── trainer.py                   # Generalized PyTorch model training and validation loop
│   └── warnings_system/                 # Regulatory decision support & early-warning system
│       ├── early_warning.py             # Graded Response Action Plan (GRAP) multi-horizon triggers
│       └── calibrate_warnings.py        # Threshold calibration against CPCB 24-hour standards
│
└── tests/                               # Automated integration and regression test suite
    ├── test_api.py                      # Integration tests for all FastAPI endpoints
    └── test_delhi_pipeline.py           # Unit tests for data integrity, artifacts, models & metrics
```

---

## 2. Root Files

### `Dockerfile`
- **Purpose:** Production multi-stage Docker containerization configuration.
- **Details:** Installs Python 3.11, copies backend requirements, sets up the virtual environment, copies the application source code and trained model artifacts, exposes port 8000, and specifies `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` as the entrypoint.

### `Procfile`
- **Purpose:** Cloud platform process declaration (used by Render, Heroku, etc.).
- **Content:** `web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

### `pyproject.toml`
- **Purpose:** PEP 517/518 build system definition and tool configurations.
- **Details:** Configures package metadata (`name = "aerosense-backend"`), specifies build dependencies, and defines Pytest configuration options (asyncio mode, root directory, and test discovery paths).

### `render.yaml`
- **Purpose:** Infrastructure-as-code declaration for deploying AeroSense AI on Render.
- **Details:** Defines the web service environment, runtime (Python 3.11), build command (`pip install -r requirements.txt`), and launch command.

### `requirements.txt`
- **Purpose:** Pinned dependencies for the production backend service.
- **Key Packages:**
  - `fastapi>=0.110.0` & `uvicorn[standard]>=0.28.0` — Asynchronous web framework & server.
  - `pydantic>=2.6.0` — Request/response data validation.
  - `torch>=2.2.0` — PyTorch deep learning runtime (LSTM, GRU, CNN-LSTM).
  - `scikit-learn>=1.4.0` — Random Forest, Decision Trees, StandardScalers.
  - `xgboost>=2.0.0` — Gradient boosted trees.
  - `pandas>=2.2.0` & `numpy>=1.26.0` — Data manipulation & numerical computation.
  - `joblib>=1.3.0` — Serialization for Scikit-Learn tree models and scalers.
  - `loguru>=0.7.0` — Structured asynchronous logging.
  - `pytest>=8.0.0` & `httpx>=0.27.0` — Test suite runner and HTTP client.

### `requirements-colab.txt`
- **Purpose:** Specialized dependency file for Google Colab GPU runtimes.
- **Details:** Omits local server packages and focuses on `torch`, `torchvision`, `torchaudio`, `xgboost`, `scikit-learn`, `matplotlib`, and `seaborn`.

### `requirements-dev.txt`
- **Purpose:** Development and test environment dependencies.
- **Details:** Includes linters (`black`, `isort`, `flake8`, `mypy`) and testing harnesses (`pytest-cov`, `pytest-asyncio`).

### `setup.py`
- **Purpose:** Allows the backend codebase to be installed in editable mode (`pip install -e .`), enabling absolute imports (`from src.api...`) across scripts, tests, and notebooks.

---

## 3. `backend/artifacts/` (Model Weights, Scalers & Evaluation Data)

### `artifacts/models/`
Stores production model weight files generated during training:
- `random_forest_delhi.pkl` (32.4 MB): Multi-output Random Forest regressor (100 trees) serialized via `joblib`. Champion model at the +24h horizon.
- `xgboost_delhi.pkl` (1.8 MB): Direct multi-horizon gradient boosted trees for all 6 forecast horizons (+1h to +72h).
- `lstm_delhi.pt` (962 KB): PyTorch state dictionary for the 2-layer stacked LSTM network (**240,070 trainable parameters**).
- `gru_delhi.pt` (731 KB): PyTorch state dictionary for the 2-layer Gated Recurrent Unit network (**182,214 trainable parameters**).
- `cnn_lstm_delhi.pt` (1.2 MB): PyTorch state dictionary for the Spatio-Temporal 1D-CNN + LSTM hybrid network (**310,278 trainable parameters**).

### `artifacts/scalers/`
- `delhi_feature_scaler.pkl`: Scikit-Learn `StandardScaler` fitted **exclusively on the 70% chronological training partition** (24,444 samples). Stores mean and variance for all 64 continuous input features to guarantee zero data leakage.

### `artifacts/experiments/`
Stores numerical outputs and evaluation artifacts computed on the 5,094-sample untouched test partition:
- `master_model_comparison.csv`: Complete evaluation metrics across all 11 algorithms and all 6 horizons (+1h, +6h, +12h, +24h, +48h, +72h). Columns: `city`, `model`, `horizon`, `horizon_hours`, `mae`, `rmse`, `r2`, `smape`, `skill_vs_persistence`, `training_time_s`, `inference_time_s`.
- `classification_metrics.json`: Accuracy (74.13%), Macro-F1 (0.6277), Weighted-F1 (0.7271), and Balanced Accuracy (0.6342) for CPCB 6-tier classification.
- `confusion_matrix_normalized.csv`: 6x6 normalized confusion matrix mapping True vs. Predicted CPCB regulatory tiers (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`).
- `confusion_matrix_raw.csv`: Raw sample count version of the confusion matrix.
- `delhi_eda_summary.json`: Precomputed dataset statistics: mean, standard deviation, quartiles, missing count (0), diurnal hourly averages, and seasonal monthly averages for Anand Vihar.
- `training_fraction_experiments.json`: Data scaling efficiency results showing test MAE, RMSE, and $R^2$ as training volume increases (15%, 30%, 45%, 60%, 100%).
- `feature_columns.json`: JSON array listing the exact 64 engineered feature column names in order of model input expectation.
- `residual_analysis.json`: Residual error quantiles (5th, 25th, 50th, 75th, 95th percentiles) and normality metrics across horizons.
- `dataset_inventory.csv`: Tabular inventory of observation counts, date boundaries, and completeness checks.
- `plots/`: Directory containing static evaluation graphs (lead time degradation curves, feature importance charts, and confusion matrix heatmaps).

---

## 4. `backend/data/` (Data Store & Generation)

- `delhi_air_quality.csv` (4.82 MB): The primary benchmark dataset. Contains **35,064 continuous hourly observations** spanning 4 full calendar years (2020-01-01 00:00:00 to 2023-12-31 23:00:00) for Anand Vihar (DEL_01), Delhi. Contains 18 columns with 0 missing values and 0 duplicates.
- `delhi_synthetic_air_quality.csv` (4.82 MB): Identical calibrated reference copy preserved for reproducibility.
- `dataset_metadata.json`: Machine-readable metadata file specifying coordinates, station name, bounding dates, column list, and verification status.
- `generate_delhi_synthetic.py`: Standalone, reproducible generator script that models atmospheric physics for Delhi:
  - Base diurnal cycles (morning and evening traffic peaks).
  - Winter planetary boundary layer compression and thermal inversion (Nov–Jan smog).
  - Summer mechanical dust storm surges (Apr–Jun).
  - Monsoon rain washout scavenging effects (Jul–Aug).
  - Autoregressive continuity ($AR(1)$ noise with $\phi = 0.88$).
  - Official CPCB AQI sub-index formula implementation.
- `README.md`: Human-readable dataset specification, unit descriptions, and regulatory standards.
- `canonical/delhi/`:
  - `observations.parquet`: Harmonized multi-station continuous records.
  - `distance_matrix.npy` & `adjacency_matrix.npy`: Pairwise spatial distance and graph connectivity matrices for the Delhi monitoring network.
  - `stations.json`: Station metadata for all registered Delhi nodes.
- `features/delhi/`:
  - `train_features.parquet` (24,444 rows): Chronological training partition.
  - `val_features.parquet` (5,238 rows): Chronological validation partition (preceded by 72h purge gap).
  - `test_features.parquet` (5,094 rows): Chronological test partition (preceded by 72h purge gap).
  - `scaler.pkl`: Partition scaler.
  - `feature_columns.json`: 64 feature names.

---

## 5. `backend/scripts/` (Pipeline & Training Execution Scripts)

- `train_delhi_models.py` (20.8 KB): **The master production training script**. Loads the Delhi dataset, executes 64-feature engineering, splits into train/val/test with 72-hour purge gaps, fits the `StandardScaler`, trains all 11 models (Random Forest, XGBoost, Decision Tree, Extra Trees, Gradient Boosting, LightGBM, LSTM, GRU, CNN-LSTM, Persistence, Moving Average), performs data fraction experiments (15% to 100%), computes multi-horizon metrics, generates CPCB confusion matrices, and saves all weights and JSON/CSV artifacts.
- `export_frontend_data.py` (3.7 KB): Extracts real test metrics, 168-hour recent trend sequences, EDA summaries, confusion matrices, and default forecasts directly from backend artifacts and compiles them into `frontend/src/data/delhiRealData.ts`.
- `retrain_xgboost_hpo.py` (5.9 KB): Executes hyperparameter optimization for XGBoost across learning rate, maximum depth, and subsample ratios.
- `plot_residuals.py` (6.7 KB): Reads model prediction residuals ($y - \hat{y}$) and plots error distributions, Q-Q plots, and heteroscedasticity curves.
- `build_city_features.py` (8.6 KB): Offline feature generator that creates partitioned parquet files from raw inputs.
- `audit_canonical.py` (1.3 KB): Validates column types, null counts, and index continuity on canonical parquet datasets.
- `run_dataset_inspection.py` (1.8 KB): Performs automated data health checks on the Delhi CSV.
- `run_calibration.py` (1.8 KB): Calibrates alert threshold triggers against historical CPCB severity tiers.
- `verify_deployment.py` (3.5 KB): Pre-flight diagnostic that tests whether model weights can be loaded into memory, checks PyTorch CUDA availability, and validates API dependencies.
- `verify_smoke_inprocess.py` (2.5 KB): Uses FastAPI's `TestClient` to verify that all API endpoints respond with HTTP 200 without requiring an external web server process.
- `generate_colab_notebooks.py`, `generate_standalone_colab_notebook.py`, `generate_standalone_notebook.py`, `make_inspection_notebook.py`, `make_master_colab_notebook.py`: Utility scripts used to generate standalone Jupyter notebooks for external environments.

---

## 6. `backend/src/` (Core Production Source Code)

### `src/api/` (FastAPI Application & REST Routing)
- `main.py`: Entry point for the REST API. Creates the FastAPI application instance, configures CORS middleware (allowing cross-origin communication with Vite on port 5173), registers all 6 endpoint routers with `/api/v1` prefix, and defines the root `/` metadata route.
- `inference.py` (15.5 KB): **The central inference engine (`DelhiInferenceEngine`)**. Implements a thread-safe singleton pattern with lazy model loading:
  - Loads models into memory and moves neural networks to CUDA (or CPU fallback) only upon first invocation.
  - Implements PyTorch network classes: `_LSTMNet` (240,070 params), `_GRUNet` (182,214 params), and `_CNNLSTMNet` (310,278 params).
  - Handles feature extraction, sequential lookback window buffering (24 timesteps), scaling, forward inference, and dynamic mapping to CPCB alert tiers.
  - Includes deterministic mathematical fallback to persistence baseline in case an artifact is missing.
- `models.py`: Pydantic validation models defining input and output schemas:
  - `HealthResponse`: System status, active model, device (CPU/CUDA), and dataset scope.
  - `StationInfo` & `StationListResponse`: Metadata for Delhi monitoring stations.
  - `HistoricalObservation`: Timestamped sensor records.
  - `ForecastRequest` & `ForecastResponse`: Single and multi-horizon predictions with alert tiers and color codes.
  - `ActiveWarningsResponse`: Regulatory warning levels, GRAP emergency codes, and health advisories.

### `src/api/routes/`
- `health.py`:
  - `GET /api/v1/health`: Returns service health, uptime, active model name, available model list, computation device (`cpu` or `cuda`), and Delhi dataset date range.
- `stations.py`:
  - `GET /api/v1/stations`: Lists all 6 Delhi CAAQMS stations (`DEL_01` Anand Vihar through `DEL_06` Shadipur) with coordinates, districts, and status.
  - `GET /api/v1/stations/{station}/history`: Returns historical observations for a requested station over the last $N$ hours.
  - `GET /api/v1/stations/network-summary`: Returns network-wide statistics (210,240 cumulative observations).
- `forecasts.py`:
  - `GET /api/v1/forecasts` & `POST /api/v1/forecasts`: Generates multi-horizon PM2.5 predictions (+1h, +6h, +12h, +24h, +48h, +72h) for any selected model (Random Forest, XGBoost, LSTM, GRU, CNN-LSTM, Persistence). Maps each horizon to CPCB alert tiers with hex color codes.
  - `POST /api/v1/forecasts/switch-model`: Switches the active default model in the inference engine.
- `warnings.py`:
  - `GET /api/v1/warnings/active`: Evaluates multi-horizon forecast concentrations against CPCB NAAQS thresholds to generate real-time air quality hazard alerts and GRAP stage advisories.
  - `GET /api/v1/warnings/thresholds`: Returns the formal CPCB NAAQS regulatory tier definitions (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
- `metrics.py`:
  - `GET /api/v1/metrics`: Returns full evaluation records across all 11 models and 6 horizons.
  - `GET /api/v1/metrics/leaderboard`: Returns the +24h horizon model leaderboard sorted by test MAE.
  - `GET /api/v1/metrics/confusion`: Returns the normalized CPCB 6-tier confusion matrix and classification metrics.
  - `GET /api/v1/metrics/fractions`: Returns data efficiency scaling results across 15%, 30%, 45%, 60%, and 100% training data volumes.
- `analytics.py`:
  - `GET /api/v1/analytics/eda`: Returns precomputed dataset distributions, diurnal curves, and correlation matrices.
  - `GET /api/v1/analytics/trends`: Streams continuous hourly PM2.5 time-series observations for dashboard trend charting.
  - `GET /api/v1/analytics/pollutant-stats`: Returns parametric statistics (mean, std, min, max, percentiles) for all criteria pollutants.

### `src/config/` (Settings & Registries)
- `city_registry.py`: Defines the complete Delhi monitoring network. Stores geographic coordinates, site types (Traffic, Residential, Industrial, Suburban), and operational status for all 6 stations (`DEL_01` to `DEL_06`). Also defines the CPCB NAAQS regulatory standards, averaging hours (24h), and tier thresholds.
- `settings.py`: Central path configuration using `pathlib.Path`. Defines absolute directory references (`DATA_DIR`, `MODELS_DIR`, `SCALERS_DIR`, `EXPERIMENTS_DIR`), default server configuration (`API_HOST = "127.0.0.1"`, `API_PORT = 8000`), and logging parameters.

### `src/data/` (Feature Engineering & Preprocessing)
- `features.py`: Implements `FeaturePipeline` and `build_temporal_features`. Computes 64 features:
  - 8 PM2.5 autoregressive lag terms ($t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-72$).
  - 7 co-pollutant lags (PM10, NO2, SO2).
  - 12 backward-looking rolling window statistics (6h, 24h, 72h mean, std, min, max) with `shift(1)`.
  - 4 cyclical trigonometric embeddings ($\sin/\cos$ of hour and month).
  - Domain interaction ratios: coarse-to-fine ratio ($\text{PM2.5}/\text{PM10}$), oxidation ratio ($\text{NO2}/\text{NOx}$), and ventilation index ($\text{Wind} \times \text{Temp}$).
- `harmonizer.py`: Standardizes raw air quality telemetry into canonical schemas, enforces UTC and local timestamp formats, and handles spatial distance matrix calculations.
- `preprocess.py`: Implements chronological time-series splitting, outlier clipping, and missing value interpolation.
- `quality.py`: Validates values against physical atmospheric plausibility limits (e.g. PM2.5 between 0 and 1000 µg/m³; temperature between -10°C and 60°C).
- `validate.py`: Data integrity assertion suite checking for nulls, duplicate timestamps, and monotonicity.
- `sources/cpcb_client.py` & `sources/openaq_client.py`: Network clients designed to fetch live or batch data from CPCB CAAQMS and OpenAQ REST APIs.

### `src/models/` (Model Architecture Definitions)
- `random_forest.py`: `AeroSenseRandomForest` class wrapping Scikit-Learn's `RandomForestRegressor` inside `MultiOutputRegressor` for 6-horizon direct prediction.
- `xgboost_model.py`: `AeroSenseXGBoost` class managing 6 independent `xgb.XGBRegressor` estimators trained with early stopping on validation RMSE.
- `lstm.py`: `AeroSenseLSTM` class implementing a 2-layer stacked PyTorch LSTM with dropout (0.2) and a multi-layer linear projection head (**240,070 parameters**).
- `cnn_lstm.py`: `AeroSenseCNNLSTM` class implementing two 1D temporal convolution layers with batch normalization followed by a 2-layer LSTM backbone and dense head (**310,278 parameters**).
- `baseline.py`: Non-parametric benchmark implementations:
  - `PersistenceModel`: Predicts $y_{t+h} = y_t$.
  - `SeasonalPersistenceModel`: Predicts $y_{t+h} = y_{t+h-24}$ (diurnal cycle replication).
  - `MovingAverageBaseline`: Predicts 24-hour historical rolling mean.

### `src/training/` (PyTorch Optimization & Schedulers)
- `trainer.py`: Generic PyTorch training loop handling batch forward passes, MSE backpropagation, gradient clipping, validation loss computation, and checkpoint saving.
- `optimizer.py`: Factory creating Adam / AdamW optimizers with decoupled weight decay ($10^{-5}$) and initial learning rates ($10^{-3}$).
- `scheduler.py`: Configures learning rate decay schedules (`ReduceLROnPlateau`, `CosineAnnealingLR`).

### `src/warnings_system/` (Regulatory Decision Support)
- `early_warning.py`: Evaluates forecasted concentrations against the statutory Graded Response Action Plan (GRAP) for Delhi NCR. Issues actionable health alerts and triggers emergency municipal codes (Stage I to Stage IV) based on predicted PM2.5 thresholds.
- `calibrate_warnings.py`: Optimizes alert trigger sensitivity against empirical historical pollution spikes to balance false alarms against missed detections.

### `src/fog_processing/` (Edge / Fog Computing Simulation)
- `fog_node.py`: Implements a simulated edge gateway (e.g. Raspberry Pi / Jetson Nano) capable of locally caching recent sensor packets, performing on-node baseline prediction, and filtering outlier noise before transmitting data to the cloud backend.

---

## 7. `backend/tests/` (Automated Test Suite)

- `test_delhi_pipeline.py` (25 Unit Tests):
  - `TestDataset`: Validates CSV existence, row count (35,064), zero duplicates, column schema, and Delhi-only geographic scope.
  - `TestArtifacts`: Validates that `delhi_feature_scaler.pkl` and all model weight files (`random_forest_delhi.pkl`, `xgboost_delhi.pkl`, `lstm_delhi.pt`, `gru_delhi.pt`, `cnn_lstm_delhi.pt`) exist and can be successfully loaded and unpickled.
  - `TestExperimentOutputs`: Asserts that `master_model_comparison.csv`, confusion matrices, and EDA summaries exist and that classification accuracy is valid ($>0.70$).
  - `TestModelLeaderboard`: Verifies that Random Forest and XGBoost outperform the persistence baseline at the +24h horizon with $R^2 > 0.90$.
- `test_api.py` (8 Integration Tests):
  - `test_health_endpoint`: Tests `GET /api/v1/health` for healthy status and Delhi scope.
  - `test_stations_endpoint`: Tests `GET /api/v1/stations` (validating all 6 stations).
  - `test_station_history_endpoint`: Tests historical telemetry retrieval.
  - `test_forecast_endpoint`: Tests model forecast generation across 6 horizons.
  - `test_warnings_endpoint`: Tests active regulatory alert generation.
  - `test_regulatory_thresholds_endpoint`: Validates CPCB NAAQS threshold values.
  - `test_metrics_endpoints`: Tests `/metrics`, `/metrics/leaderboard`, and `/metrics/confusion`.
  - `test_analytics_endpoints`: Tests `/analytics/trends` and `/analytics/eda`.

---

## 8. Summary of Active File Counts & Integrity

- **Total Unit & Integration Tests:** 33 tests (**33 Passed, 100% Green**).
- **Active Model Weight Files:** 5 production models in `backend/artifacts/models/`.
- **Primary Data Corpus:** `backend/data/delhi_air_quality.csv` (35,064 rows, 0 missing, 0 duplicates).
- **FastAPI Endpoints:** 12 endpoints across 6 dedicated route modules.
- **Banned City References in Active Code:** Exactly **0**.
- **"Synthetic" Labels in Active Code:** Exactly **0**.
