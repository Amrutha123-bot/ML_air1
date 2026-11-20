# AeroSense AI V2 — Project Changelog & Implementation History

This document details the chronological evolution of the AeroSense AI project from its initial Beijing-only single-station proof-of-concept (V1) to its current scientifically rigorous, multi-city spatiotemporal deep learning and early-warning platform (V2).

---

## Evolution Timeline Summary

```
[Phase 1: V1 Baseline] ──> [Phase 2: Multi-City Lakehouse] ──> [Phase 3: Spatial Learning]
  - Beijing UCI (12 stns)    - 4 Cities (Beijing, Delhi,        - Non-hardcoded Adjacency
  - Single-station models     Mumbai, Vijayawada)              - 4D Tensors (B, 24, N, 57)
  - 38 Passing Tests         - 716,344 feature rows            - Timestamp alignment check
                                                                        │
[Phase 5: Local Integration] <── [Phase 4: GPU Optimization] <──────────┘
  - city_stcnn_inference.py        - Colab GPU Tesla T4 Runs
  - FastAPI City Endpoints         - Beijing & Delhi Checkpoints
  - 67 Passing Tests               - Aligned Baselines & Optuna HPO
```

---

## Detailed Chronological History

### Phase 1: Beijing-Only Single-Station Baseline (V1)
* **Scope**: Initial implementation focused on the UCI Beijing Multi-Site Air Quality Dataset (2013-03-01 to 2017-02-28).
* **Dataset**: 12 monitoring stations, 420,768 canonical records.
* **Architectures**:
  - Implemented initial tabular baselines (Persistence, Random Forest, XGBoost).
  - Neural baselines: LSTM, CNN-LSTM, and an early prototype of ST-CNN-GRU.
  - Secondary fallback prototype: GA-KELM (trained on a 6,000-sample subsample).
* **Identified Architectural Weaknesses**:
  - The initial ST-CNN-GRU implementation relied on repeated single-station tensors rather than true cross-station graph operations.
  - The geographic adjacency matrix was hardcoded solely to Beijing station coordinates.
  - Test evaluations occasionally evaluated mismatched arrays or arbitrary positional slices.
* **Test Baseline**: 38 passing tests in `backend/tests/`.

---

### Phase 2: Multi-City Data Lakehouse & Canonical Feature Pipeline
* **Scope**: Expanding geographic and climatological scope from Beijing to major Indian metropolitan and tier-2 centers.
* **Cities Integrated**:
  1. **Beijing** (12 stations): 420,768 canonical rows → 419,616 feature rows.
  2. **Delhi NCR** (12 stations): 210,240 canonical rows → 209,088 feature rows.
  3. **Mumbai** (8 stations): 70,080 canonical rows → 69,312 feature rows.
  4. **Vijayawada** (2 stations): 17,520 canonical rows → 17,328 feature rows.
* **Boundary Analysis & Verification**:
  - **Identified Reconciliation Gap**: An exact difference of 96 rows per station between canonical records and engineered features.
  - **Root Cause**: 72 trailing rows dropped due to the $t+72\text{h}$ multi-horizon forward shift, plus 24 leading rows dropped due to the 24-hour backward lag ($\text{PM2.5\_lag24h}$).
  - **Verdict**: Proven mathematical boundary consequence; zero silent data loss.
* **Standardized Features (57 Input Features)**:
  - Six criteria pollutants ($\text{PM}_{2.5}, \text{PM}_{10}, \text{SO}_2, \text{NO}_2, \text{CO}, \text{O}_3$), meteorological variables ($\text{TEMP}, \text{PRES}, \text{DEWP}, \text{RAIN}, \text{WSPM}$), cyclical wind representations ($\text{wd\_sin}, \text{wd\_cos}$), temporal trigonometric cycles (hour, day of week, month, day of year), autoregressive lags ($1, 2, 3, 6, 12, 24, 48$ hours), and rolling window statistics ($3, 6, 12, 24$ hours mean and standard deviation).
* **Leakage-Free Preprocessing**:
  - Strict 70% Train, 15% Validation, 15% Test chronological splits.
  - `StandardScaler` fitted strictly on the chronological training split and transformed forward onto validation and test splits.

---

### Phase 3: Spatial Graph & ST-CNN-GRU Refactoring
* **Graph Structure**:
  - Replaced hardcoded Beijing distance buffers with dynamic, station-aware adjacency construction.
  - Normalized Gaussian thresholded distance kernel:
    $$A_{ij} = \exp\left(-\frac{d(s_i, s_j)^2}{\sigma^2}\right)$$
    with row-normalization $\sum_j A_{ij} = 1$.
  - Station count parameterization: Beijing ($N=12$), Delhi ($N=12$), Mumbai ($N=8$), Vijayawada ($N=2$).
* **Model Tensor Upgrades**:
  - Upgraded `build_spatiotemporal_tensors` to produce genuine 4D sliding window tensors: $(B, T=24, N, F=57)$.
  - MultiIndex reindexing guarantees temporal-spatial grid alignment even if individual stations experience missing hourly timestamps.
  - Implemented `return_metadata=True` returning exact sequence end timestamps, station order, and feature lists.
* **Target Alignment Validation**:
  - Created `tests/test_spatial_learning.py::test_prediction_target_timestamp_alignment`.
  - Proved that input window $[t-23, t]$ is mapped strictly to prediction targets $[t+1, t+6, t+12, t+24, t+48, t+72]$.
  - Backend test suite increased to 56 passing tests.

---

### Phase 4: Google Colab GPU Optimization & Training Run
* **Hardware & Runtime**:
  - Compute platform: Google Colab Tesla T4 GPU (16 GB VRAM).
  - Software environment: PyTorch 2.14.1+cu130, XGBoost 3.4.1, CUDA 13.0.
* **Separated Model Training**:
  - **Beijing ST-CNN-GRU**: Trained for 25 epochs with Cosine Annealing learning rate schedule. Best validation checkpoint saved at epoch 8.
  - **Delhi ST-CNN-GRU**: Trained for 25 epochs independently using the Delhi 12-station spatial graph.
  - **XGBoost Hyperparameter Optimization (Optuna)**: 15 trials optimizing tree depth, learning rate, subsampling, and feature column subsampling. Best parameters: depth=9, lr=0.0359, subsample=0.7, colsample=0.6.
* **Aligned Baselines Benchmark**:
  - Evaluated Persistence ($y(t+h) = y(t)$) and Seasonal Naive ($y(t+h) = y(t+h-24)$) on the exact identical test targets and validity masks as the neural models.
  - Resolved historical evaluation discrepancies where baselines had been compared on different sample counts.

---

### Phase 5: Verification, Local Inference Integration & Artifact Packaging
* **Inference Module (`backend/src/api/city_stcnn_inference.py`)**:
  - Created standalone city inference pipeline loading verified checkpoints.
  - Builds full 4D spatiotemporal tensors $(1, 24, 12, 57)$ from canonical test features.
  - Selects station slice by index from the city station configuration.
  - Formats physical, unscaled PM2.5 forecasts across all 6 horizons.
* **FastAPI City Route Integration**:
  - Added endpoint `GET /api/v1/forecasts/city/{city}/{station_id}/stcnn` to `src/api/routes/forecasts.py`.
  - Added `CityForecastResponse` schema to `src/api/models.py`.
  - Validates city and station parameters with informative 404 HTTP errors.
* **Frontend Verification**:
  - Added `getCitySTCNNForecast` helper to `frontend/src/api/client.ts`.
  - Production build compiled cleanly with `pnpm run build` in 11.46s.
* **Test Suite Expansion**:
  - Added 11 automated unit tests in `backend/tests/test_city_inference.py`.
  - Full backend test suite now stands at **67 passing tests** (0 failures, 8 warnings).
