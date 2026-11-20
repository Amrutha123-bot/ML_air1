# AeroSense AI — Phase 0 Repository Audit Report

**Date of Audit:** 2026-11-17  
**Auditor:** Antigravity AI (Master Implementation Engine)  
**Status:** Read-Only Complete — Strictly Evidence-Based (Zero Fabrication)  
**Execution Context:** Local Repository (`c:\Users\anala\OneDrive\Desktop\ML_air`)

---

## 1. Executive Summary & Critical Ground-Truth Findings

Before touching any code or making any architectural modifications, an exhaustive read-only inspection was executed across the entire repository, checkpoints, datasets, pipelines, and frontends.

### Key Revelations:
1. **The Beijing Reality:**
   - The only genuine historical sensor observations currently in the repository belong to Beijing (UCI PRSA Multi-Site dataset: 12 monitoring stations, 420,768 records, March 2013 to February 2017).
   - 6 of the 8 models (`xgboost`, `random_forest`, `lstm`, `cnn_lstm`, `ga_kelm`, `persistence`) were trained exclusively on Beijing features.
   - The primary spatiotemporal graph checkpoint `stcnn_gru_beijing.pt` was trained on Beijing's 12-station adjacency matrix.

2. **The Indian-City Data Reality (Delhi, Mumbai, Vijayawada):**
   - **Critical Integrity Discovery:** The canonical datasets currently residing in `backend/data/canonical/{delhi,mumbai,vijayawada}/observations.parquet` were **NOT** fetched from the CPCB portal or OpenAQ. 
   - They were generated synthetically in `backend/src/data/harmonizer.py` lines 183–340 via mathematical formulas using `np.random.RandomState(42, 43, 44)`, seasonal sinusoidal curves, and uniform random distributions.
   - This directly explains why Delhi, Mumbai, and Vijayawada have exactly **0.00% missing values** across all columns and why Delhi achieved an implausibly high $R^2$ of **0.9339** (0.8870 at 72h).
   - `stcnn_gru_delhi.pt` is a real neural checkpoint, but it was trained on synthetic sinusoidal data, not real atmospheric measurements.
   - For Mumbai (8 stations) and Vijayawada (2 stations), only synthetic features exist (`train/val/test_features.parquet`); **no dedicated neural checkpoints exist** for them.

3. **Codebase Quality & Test Foundation:**
   - The backend currently has **78 passing unit and integration tests** under `backend/tests/` (pytest 8.2.0, Python 3.11).
   - The frontend builds with **0 TypeScript errors** (`tsc -b && vite build`, React 19, Vite 8, Tailwind CSS).
   - However, Beijing and the synthetic Indian datasets are deeply intertwined across 65 backend and frontend files.

---

## 2. Repository Structure & Dependency Mapping

### Directory Tree
```
ML_air/
├── backend/
│   ├── artifacts/
│   │   ├── experiments/     # JSON/CSV evaluation reports & residual curves
│   │   ├── models/          # 16 trained model checkpoints (.pt, .pkl)
│   │   └── scalers/         # feature_scaler.pkl
│   ├── colab_backups/       # AeroSense_V2_GPU_Training.ipynb (30 cells)
│   ├── data/
│   │   ├── canonical/       # Normalized city lakehouse (beijing, delhi, mumbai, vijayawada)
│   │   ├── features/        # Parquet train/val/test feature splits & scalers
│   │   ├── processed/       # Legacy processed parquet files
│   │   └── raw/             # 12 UCI PRSA CSV files (Beijing only)
│   ├── notebooks/           # Working copies of training notebooks
│   ├── scripts/             # Data build, training, calibration, validation scripts
│   ├── src/
│   │   ├── api/             # FastAPI routes (cities, forecasts, warnings, metrics, etc.)
│   │   ├── config/          # Environment & path settings
│   │   ├── data/            # Harmonizer, downloader, validator, feature pipeline
│   │   ├── models/          # Model implementations (ST-CNN-GRU, XGBoost, RF, LSTM, KELM)
│   │   ├── services/        # Registry, dataset, evaluation, and analytics services
│   │   ├── training/        # Multi-seed trainers, Optuna HPO, schedulers
│   │   └── warnings_system/ # CPCB/GB 3095 regulatory early warning engine
│   └── tests/               # 78 passing test cases across 12 test suites
├── frontend/
│   ├── src/
│   │   ├── api/             # Axios client & TypeScript endpoint contracts
│   │   ├── components/      # Layout (Header, Sidebar), UI (Badge, Card)
│   │   ├── pages/           # 8 dashboard pages (frozen lavender theme)
│   │   └── types/           # TypeScript interfaces (Forecast, Warning, City, etc.)
│   └── package.json         # React 19, Vite 8, Recharts 3.10, Tailwind 3.4
└── docs/                    # Project documentation & audit trail
```

### Dependency List
- **Backend (Python 3.11):** `torch==2.3.1`, `xgboost==2.0.3`, `scikit-learn==1.4.2`, `optuna==3.6.1`, `pandas==2.2.2`, `numpy==1.26.4`, `fastapi==0.111.0`, `uvicorn==0.30.1`, `pydantic==2.7.1`, `pyarrow`, `joblib==1.4.2`, `loguru==0.7.2`, `pytest==8.2.0`.
- **Frontend (Node 20+):** `react@19.3.1`, `react-dom@19.3.1`, `react-router-dom@7.18.5`, `recharts@3.10.1`, `lucide-react@1.72.0`, `axios@1.21.1`, `tailwindcss@3.4.19`, `vite@8.3.6`.

---

## 3. Eight-Model Audit Table

Every model selectable in the UI or codebase was forensically verified for its implementation, checkpoint presence, size, parameter configuration, input features, and training authenticity.

| Model Architecture | Implementation File | Training Script | Checkpoint File (Size, mtime) | Framework | Input Features | Supported Cities/Pollutants/Horizons | Evaluation Code Path | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1. ST-CNN-GRU (Beijing)** | `backend/src/models/stcnn_gru.py` | `backend/scripts/train_all.py` & Colab Cell 15 | `backend/artifacts/models/stcnn_gru_beijing.pt` (1,181,337 B, Nov 14 2026) | PyTorch 2.3.1 | 57 tabular features $\times$ 12 stations (4D tensor: $B \times T \times N \times F$) | Beijing / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_evaluation_report.json` | **Real & trained** (Trained on real Beijing UCI data) |
| **2. ST-CNN-GRU (Delhi)** | `backend/src/models/stcnn_gru.py` | Colab Cell 19 (`AeroSense_V2_GPU_Training.ipynb`) | `backend/artifacts/models/stcnn_gru_delhi.pt` (1,181,345 B, Nov 14 2026) | PyTorch 2.3.1 | 57 tabular features $\times$ 12 stations (4D tensor: $B \times T \times N \times F$) | Delhi / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/colab_run_20261114_202839/delhi_final_test_metrics.json` | **Real & trained** (Trained on synthetic sinusoidal Delhi data) |
| **3. XGBoost** | `backend/src/models/xgboost_model.py` | `backend/scripts/train_all.py` | `backend/artifacts/models/xgboost_model.pkl` (1,409,623 B, Nov 14 2026) | XGBoost 2.0.3 / scikit-learn MultiOutputRegressor | 57 tabular features (lags, rolling means, weather, cyclic temporal) | Beijing only / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_metrics_summary.csv` | **Real & trained** (Optuna tuned on Beijing data) |
| **4. Random Forest** | `backend/src/models/random_forest.py` | `backend/scripts/train_all.py` | `backend/artifacts/models/random_forest_model.pkl` (51,038,087 B, Nov 14 2026) | scikit-learn 1.4.2 RandomForestRegressor | 57 tabular features | Beijing only / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_metrics_summary.csv` | **Real & trained** (100 trees, Beijing data) |
| **5. Standard LSTM** | `backend/src/models/lstm.py` | `backend/scripts/train_all.py` | `backend/artifacts/models/lstm_model.pt` (274,037 B, Nov 14 2026) | PyTorch 2.3.1 (2 layers, hidden=64, seq_len=24) | 57 tabular features ($B \times 24 \times 57$) | Beijing only / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_metrics_summary.csv` | **Real & trained** (Multi-seed 42, 43, 44 available) |
| **6. CNN-LSTM Hybrid** | `backend/src/models/cnn_lstm.py` | `backend/scripts/train_all.py` | `backend/artifacts/models/cnn_lstm_model.pt` (1,246,869 B, Nov 14 2026) | PyTorch 2.3.1 (1D-CNN + 2-layer LSTM, hidden=128) | 57 tabular features ($B \times 24 \times 57$) | Beijing only / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_metrics_summary.csv` | **Real & trained** (Multi-seed 42, 43, 44 available) |
| **7. GA-KELM** | `backend/src/models/ga_kelm.py` | `backend/scripts/train_all.py` & `src/models/ga_kelm.py` | `backend/artifacts/models/ga_kelm_model.pkl` (3,026,356 B, Nov 14 2026) | Custom Kernel ELM with RBF kernel ($C=4.457, \gamma=0.0114$) | 57 tabular features ($N \times 57$) | Beijing only / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_metrics_summary.csv` | **Real & trained** (Fitted on 6,000 Beijing subsamples) |
| **8. Persistence Baseline** | `backend/src/models/baseline.py` | `backend/scripts/train_all.py` | `backend/artifacts/models/persistence_model.pkl` (124 B, Nov 14 2026) | Python/Joblib ($t_0$ forward propagation) | Current PM2.5 reading (`PM2.5_lag1h` or `PM2.5`) | All cities / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_metrics_summary.csv` | **Real & trained** (Deterministic benchmark) |
| **Hybrid Forecaster** | `backend/src/models/hybrid_forecaster.py` | Dynamic composite wrapper | None (dynamic wrapper around ST-CNN-GRU + GA-KELM) | Runtime orchestration | 57 tabular features | Beijing only / PM2.5 / 1, 6, 12, 24, 48, 72h | `backend/artifacts/experiments/test_metrics_summary.csv` | **Real, algorithmic routing** (No standalone weight file) |

---

## 4. Multi-City Data Audit (Computed by Code)

### Exact Measurements on Canonical and Feature Lakehouse

| Territory | Stations | Canonical Rows | Date Range (Local Time) | PM2.5 Range ($\mu$g/m$^3$) | Missing % (PM2.5 / Other) | Feature Splits (Train / Val / Test) | Data Provenance & Integrity Status |
|---|---|---|---|---|---|---|---|
| **Beijing** | 12 | 420,768 | 2013-03-01 00:00 to 2017-02-28 23:00 CST | Min: 2.00, Mean: 79.79, Max: 999.00 | PM2.5: 2.08%, PM10: 1.53%, SO2: 2.14%, NO2: 2.88%, CO: 4.92%, NH3: 100.00% | Train: 293,736<br>Val: 62,940<br>Test: 62,940 (71 cols) | **Real empirical data** from UCI PRSA Multi-Site. 96 rows per station dropped at boundary during sequence-window construction. |
| **Delhi NCR** | 12 | 210,240 | 2023-01-01 00:00 to 2024-12-30 23:00 IST | Min: 5.00, Mean: 137.15, Max: 392.70 | **0.00% across all columns** | Train: 146,364<br>Val: 31,368<br>Test: 31,356 (71 cols) | **Synthetic data**: Generated by `DataHarmonizer.generate_delhi_canonical()` using `np.random.RandomState(42)`. No real CPCB raw data exists in repo. |
| **Mumbai** | 8 | 70,080 | 2023-01-01 00:00 to 2023-12-31 23:00 IST | Min: 5.00, Mean: 60.82, Max: 154.40 | **0.00% across all columns** | Train: 48,520<br>Val: 10,400<br>Test: 10,392 (71 cols) | **Synthetic data**: Generated by `DataHarmonizer.generate_mumbai_canonical()` using `np.random.RandomState(43)`. No dedicated model trained. |
| **Vijayawada** | 2 | 17,520 | 2023-01-01 00:00 to 2023-12-31 23:00 IST | Min: 5.00, Mean: 49.53, Max: 110.60 | **0.00% across all columns** | Train: 12,130<br>Val: 2,600<br>Test: 2,598 (71 cols) | **Synthetic data**: Generated by `DataHarmonizer.generate_vijayawada_canonical()` using `np.random.RandomState(44)`. No dedicated model trained. |

---

## 5. The 14-Point Prediction Integrity Audit (§3)

Walkthrough of all 14 checks against the existing implementation:

| # | Integrity Check | Status | Evidence & Details |
|---|---|---|---|
| **1** | Outputs come from loaded trained weights (assert checkpoint hash logged per prediction; fail if random init) | **FAIL / PARTIAL** | Model weights are loaded from disk (`stcnn_gru_{city}.pt`, `.pkl`), but neither `inference.py` nor `city_stcnn_inference.py` computes or logs a SHA-256 hash of the checkpoint during inference. |
| **2** | Timestamp alignment between inputs, targets, and displayed x-axis | **PASS** | `city_stcnn_inference.py` lines 180–195 calculates horizon timestamps as $t_{\text{ref}} + h$ hours and passes `target_datetime` explicitly to API response and chart components. |
| **3** | A $t+24$ forecast is compared with observation exactly 24 h after origin | **PASS** | Evaluated in Colab notebook Cell 17 & 21; unit test in `test_api_v2_rebuild.py` verifies $t+24$ is 24 hours ahead of input end. |
| **4** | Actual future observations exist for evaluated windows | **FAIL** | Real observations exist **only for Beijing**. For Delhi, Mumbai, and Vijayawada, "future observations" are synthetically generated values from `harmonizer.py`. |
| **5** | Predictions inverse-transformed to original units | **PASS** | `city_stcnn_inference.py` line 204 calls `scaler.inverse_transform(raw_preds)` using `backend/data/features/{city}/scaler.pkl`. |
| **6** | Correct city/station scaler used | **PASS** | `city_stcnn_inference.py` dynamically resolves `_FEATURES_DIR / city / "scaler.pkl"`. Beijing and Delhi use distinct scaler instances. |
| **7** | Feature names and order identical in training and inference | **PASS** | `city_stcnn_inference.py` loads `feature_columns.json` (57 features) and orders columns identically to `train_features.parquet`. |
| **8** | Missing-value handling identical in training and inference | **PASS** | Both `trainer.py` (`fillna(0)`) and `city_stcnn_inference.py` (`fillna(0)`) apply identical zero-imputation policy. |
| **9** | Predicted series never rendered as "actual" | **PASS** | `AIForecasting.tsx` and `Overview.tsx` render "Historical (Actual)" with solid line and "Forecast (Predicted)" with dashed violet line. |
| **10** | No hardcoded or fabricated metrics anywhere | **FAIL** | `backend/src/services/evaluation_service.py` lines 11–125 hardcodes dictionary metrics (`overall_mae`, `r2`, etc.). Furthermore, underlying Indian datasets are synthetically generated. |
| **11** | No future information in features (shifted-feature test) | **PASS** | `scripts/build_city_features.py` constructs features strictly via backward lags (`lag1h`, `lag2h`, etc.) and rolling statistics (`roll12h`, `roll24h`). Targets are forward-shifted. |
| **12** | Test data excluded from training, scaling, tuning, early stopping, and fine-tuning | **PASS** | `scripts/build_city_features.py` lines 90–120 fits `StandardScaler` strictly on `train_df`. Val and test splits are strictly out-of-sample. |
| **13** | Frontend and backend show the same model version/hash | **FAIL** | Neither frontend nor backend displays a cryptographic model checkpoint hash. |
| **14** | Unsupported city/pollutant/model combinations are not presented as available | **PASS** | API returns HTTP 404/422 for unsupported requests; UI disables deep learning models for Mumbai and Vijayawada. |

---

## 6. Beijing Removal Impact Analysis (Preparation for Phase 1)

Grep scanning identified **65 files** referencing Beijing, UCI, PRSA, or Beijing station names.

### High-Risk Files & Required Architectural Changes

1. **Model Checkpoints & Weights:**
   - Files to quarantine:
     - `backend/artifacts/models/stcnn_gru_beijing.pt`
     - `backend/artifacts/models/stcnn_gru_model.pt`
     - `backend/artifacts/models/stcnn_gru_seed42.pt`
     - `backend/artifacts/models/xgboost_model.pkl` (trained on Beijing)
     - `backend/artifacts/models/random_forest_model.pkl` (trained on Beijing)
     - `backend/artifacts/models/lstm_model.pt` (trained on Beijing)
     - `backend/artifacts/models/cnn_lstm_model.pt` (trained on Beijing)
     - `backend/artifacts/models/ga_kelm_model.pkl` (trained on Beijing)
   - Risk: Removing these will leave the multi-model Beijing endpoint (`GET /api/v1/forecasts/{station}`) without weights.
   - Mitigation: Quarantine to `archive/beijing_legacy/artifacts/models/`. Replace active routes with city-specific registry endpoints.

2. **Data & Features:**
   - Files to quarantine:
     - `backend/data/raw/PRSA_Data_*.csv` (12 files)
     - `backend/data/canonical/beijing/` (entire directory)
     - `backend/data/features/beijing/` (entire directory)
     - `backend/data/processed/` (contains Beijing splits)
   - Risk: Breakage in `download.py`, `run_phase2_acquisition.py`, `build_city_features.py`.
   - Mitigation: Decouple data pipeline from hardcoded Beijing files; point exclusively to Indian city ingestion.

3. **Backend API Routes & Services:**
   - Files requiring refactor:
     - `backend/src/api/routes/cities.py`: Currently returns Beijing as first city.
     - `backend/src/api/routes/forecasts.py`: Default station fallback is `Dongsi`.
     - `backend/src/api/routes/metrics.py`: Reads `test_metrics_summary.csv` (Beijing).
     - `backend/src/api/routes/warnings.py`: Default station fallback is `Dongsi`.
     - `backend/src/api/city_stcnn_inference.py`: `SUPPORTED_CITIES` includes Beijing.
     - `backend/src/api/inference.py`: Legacy multi-model inference hardcoded to Beijing.
     - `backend/src/services/dataset_service.py`: `CITY_CONFIG["beijing"]`.
     - `backend/src/services/model_registry_service.py`: Beijing models registered.
     - `backend/src/services/evaluation_service.py`: `BEIJING_TEST_EVALUATION`.
     - `backend/src/services/analytics_service.py`: Beijing entry in `get_city_comparison()`.

4. **Frontend Dashboard Pages:**
   - Files requiring refactor:
     - `frontend/src/pages/Overview.tsx`: Defaults to Beijing stations (`Dongsi`).
     - `frontend/src/pages/MonitoringStations.tsx`: Beijing station list and coordinates.
     - `frontend/src/pages/AIForecasting.tsx`: Beijing multi-model dropdown.
     - `frontend/src/pages/ModelLaboratory.tsx`: Beijing benchmark metrics and charts.
     - `frontend/src/pages/EarlyWarningCentre.tsx`: Beijing station selector.
     - `frontend/src/pages/AnalyticsReports.tsx`: Beijing diurnal and topographical charts.

5. **Test Suite:**
   - 10 test files currently assert Beijing stations (`BJ_01`..`BJ_12` or `Dongsi`).
   - Need updating to test genuine Indian city stations (e.g. `DEL_01` Anand Vihar).

---

## 7. Acceptance Checklist for Phase 0

- [x] Map the repository: backend framework, routes, services, model loading code, data folders, frontend framework, pages, city lists, chart data sources.
- [x] Grep for hardcoded numbers, random generators, mock/dummy/placeholder strings in prediction and data paths.
- [x] Grep every occurrence of Beijing, UCI, PRSA, Beijing station names, ST-CNN-GRU (65 files mapped).
- [x] Grep and list all 18 files that enumerate cities.
- [x] Construct Eight-Model Audit Table with implementation, training script, checkpoint, framework, features, and verdicts.
- [x] Audit Mumbai and Vijayawada data files, computed row counts, date ranges, missing %, and model status.
- [x] Complete 14-point prediction integrity audit with PASS/FAIL evidence.
- [x] Deliver `docs/AUDIT_REPORT.md` including Beijing removal impact analysis.

---

## 8. Questions Requiring User Decision Before Phase 1

1. **Quarantine Scope:** 
   In Phase 1, we will move all Beijing datasets, checkpoints, metrics, and scripts to `archive/beijing_legacy/`. Confirm approval to proceed with Phase 1.
2. **Delhi Real Data Acquisition:**
   Since the existing Delhi dataset was proven to be mathematically generated (`np.random.RandomState(42)` in `src/data/harmonizer.py`), we must replace it with authentic CPCB/OpenCity/Kaggle data in Phase 4. Confirm that you want the synthetic Delhi data completely replaced with real empirical observations.

---
*End of Phase 0 Audit Report. Standing by for user approval to proceed to Phase 1.*
