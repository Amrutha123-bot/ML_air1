# Requirements Traceability Matrix

## Air Quality Forecasting and Early-Warning System (Beijing Multi-Station PM2.5)

| Requirement ID | Specification | Implementation Module(s) | Verification Test / Command | Status |
| :--- | :--- | :--- | :--- | :--- |
| **REQ-DATA-01** | Download real Beijing 12-station UCI dataset (2013–2017) | `src/data/download.py`, `scripts/download_data.py` | `tests/test_data_pipeline.py::test_dataset_exists` | **VERIFIED** ✅ |
| **REQ-DATA-02** | Chronological alignment & timestamp continuity | `src/data/validate.py`, `src/data/preprocess.py` | `tests/test_data_pipeline.py::test_temporal_continuity` | **VERIFIED** ✅ |
| **REQ-DATA-03** | Missing-value screening & leakage-free imputation | `src/data/preprocess.py` | `tests/test_preprocessing.py::test_train_val_test_split_temporal` | **VERIFIED** ✅ |
| **REQ-DATA-04** | Feature engineering (lags, rolling stats, temporal, meteorological) | `src/data/features.py` | `tests/test_preprocessing.py::test_feature_engineering_columns` | **VERIFIED** ✅ |
| **REQ-FOG-01** | Simulated Fog-side data screening & quality filtering | `src/fog_processing/fog_node.py`, `src/fog_processing/__init__.py` | `tests/test_fog_processing.py::test_fog_node_valid_screening` | **VERIFIED** ✅ |
| **REQ-FOG-02** | Noise reduction (3-pt median smoothing) & transmission gating | `src/fog_processing/fog_node.py` | `tests/test_fog_processing.py::test_fog_node_smoothing`, `test_fog_gateway_batch` | **VERIFIED** ✅ |
| **REQ-MOD-01** | Persistence Baseline Model (lagged multi-horizon) | `src/models/baseline.py` | `tests/test_models.py::test_persistence_baseline` | **VERIFIED** ✅ |
| **REQ-MOD-02** | Random Forest benchmark with MultiOutputRegressor | `src/models/random_forest.py` | `tests/test_models.py::test_random_forest_shapes` | **VERIFIED** ✅ |
| **REQ-MOD-03** | XGBoost gradient boosting model | `src/models/xgboost_model.py` | `tests/test_models.py::test_xgboost_shapes` | **VERIFIED** ✅ |
| **REQ-MOD-04** | Stacked LSTM multi-horizon deep learning architecture | `src/models/lstm.py` | `tests/test_models.py::test_lstm_forward` | **VERIFIED** ✅ |
| **REQ-MOD-05** | CNN-LSTM hybrid model | `src/models/cnn_lstm.py` | `tests/test_models.py::test_cnn_lstm_forward` | **VERIFIED** ✅ |
| **REQ-MOD-06** | ST-CNN-GRU spatiotemporal model with attention | `src/models/stcnn_gru.py` | `tests/test_models.py::test_stcnn_gru_forward` | **VERIFIED** ✅ |
| **REQ-MOD-07** | GA-KELM (Kernel ELM + Genetic Algorithm optimization) | `src/models/ga_kelm.py` | `tests/test_ga_kelm.py::test_gakelm_model_fit_save_load` | **VERIFIED** ✅ |
| **REQ-MOD-08** | Hybrid Forecaster (ST-CNN-GRU + GA-KELM fallback) | `src/models/hybrid_forecaster.py` | `tests/test_ga_kelm.py::test_hybrid_forecaster_fallback_logic` | **VERIFIED** ✅ |
| **REQ-WARN-01** | 6-tier regulatory warning system (GB 3095-2012 / HJ 633-2012) | `src/warnings_system/early_warning.py` | `tests/test_warnings.py::test_classify_pm25_tiers` | **VERIFIED** ✅ |
| **REQ-WARN-02** | Dual-mode alert: 24h rolling average & acute surge detection | `src/warnings_system/early_warning.py` | `tests/test_warnings.py::test_trend_detection` | **VERIFIED** ✅ |
| **REQ-WARN-03** | Threshold calibration on validation data | `src/warnings_system/calibrate_warnings.py` | Execution generated `artifacts/experiments/warning_calibration.json` | **VERIFIED** ✅ |
| **REQ-EVAL-01** | Multi-horizon evaluation metrics (MAE, RMSE, R², MAPE, Skill Score) | `src/evaluation/metrics.py`, `src/evaluation/evaluator.py` | `tests/test_models.py::test_metrics_computation` | **VERIFIED** ✅ |
| **REQ-EVAL-02** | Station-by-station breakdown across all 12 stations | `src/evaluation/evaluator.py`, `scripts/evaluate_all.py` | `artifacts/experiments/test_evaluation_report.json` | **VERIFIED** ✅ |
| **REQ-API-01** | Production FastAPI backend endpoints (health, stations, forecasts, warnings) | `src/api/main.py`, `src/api/routes/*.py` | `tests/test_api.py` (8/8 passed) | **VERIFIED** ✅ |
| **REQ-API-02** | Real-time in-memory inference engine with model preloading | `src/api/inference.py` | `scripts/verify_smoke_inprocess.py` | **VERIFIED** ✅ |
| **REQ-DEP-01** | Docker containerization with non-root user & healthcheck | `Dockerfile`, `docker-compose.yml` | `Dockerfile` inspection & verification | **VERIFIED** ✅ |
