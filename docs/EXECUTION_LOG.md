# Engineering Execution Log

## Beijing Multi-Station Air Quality Forecasting System

### Phase 1: Environment & Pipeline Initialization
- Verified Python 3.11 environment on Windows.
- Installed required production packages: `xgboost`, `optuna`, `seaborn`, `pytest`, `pytest-asyncio`.
- Executed `scripts/download_data.py` retrieving all 12 station CSV files from GitHub mirror after UCI archive connection instability.
- Executed `src/data/validate.py`: verified all 12 stations, 420,768 records, correct temporal bounds, no duplicate records.
- Executed `src/data/preprocess.py`: generated clean temporal splits (train=298,368, val=70,272, test=52,128 records) with leak-free imputer fitting.
- Executed `src/data/features.py`: extracted 57 features (temporal cycles, lagged pollutants, rolling means/stds, meteorological interactions).

### Phase 2: Stage 3 Implementation (GA-KELM & Hybrid Forecaster)
- Created `src/models/ga_kelm.py`:
  - Mathematical implementation of Kernel Extreme Learning Machine (KELM) using closed-form RBF kernel least-squares inversion.
  - Genetic Algorithm continuous optimizer (`GAKELMOptimizer`) tuning $(C, \gamma)$ parameters.
  - Scikit-learn API compatibility for multi-output regression.
  - Trained on multi-horizon prediction, saving artifact to `artifacts/models/ga_kelm_model.pkl` ($C=4.4572, \gamma=0.011427$).
- Created `src/models/hybrid_forecaster.py`:
  - Adaptive routing between primary ST-CNN-GRU deep learning model and secondary GA-KELM.
  - Tracks rolling 48-hour error window per station against calibrated Random Forest benchmark threshold ($49.4\,\mu\text{g}/\text{m}^3$).
- Registered new models into `src/models/__init__.py` and `src/api/inference.py`.

### Phase 3: Stage 1 Fog Layer Implementation
- Created `src/fog_processing/fog_node.py` and `src/fog_processing/__init__.py`:
  - Implemented simulated edge data quality screening (`FogNode`, `FogGateway`).
  - Gated transmission based on signal delta ($>2\%$) and scheduled checkpoints.
  - Implemented 3-point running median filter for localized sensor noise reduction.

### Phase 4: Stage 4 Early Warning & Calibration
- Created `src/warnings_system/calibrate_warnings.py` and `scripts/run_calibration.py`:
  - Implemented threshold calibration against validation set predictions.
  - Generated `artifacts/experiments/warning_calibration.json`:
    - Tier 75 $\mu\text{g}/\text{m}^3$ calibrated to decision boundary $60.0\,\mu\text{g}/\text{m}^3$ ($F_1=0.5332$).
    - Tier 115 $\mu\text{g}/\text{m}^3$ calibrated to decision boundary $100.0\,\mu\text{g}/\text{m}^3$ ($F_1=0.3341$).
- Verified dual-mode warning logic: 24-hour rolling averages and acute hourly spike detection ($>30\,\mu\text{g}/\text{m}^3/\text{h}$).

### Phase 5: Scientific Evaluation on Untouched Test Set
- Executed `scripts/evaluate_all.py` across all 8 model paradigms:
  - Models evaluated: Persistence Baseline, Random Forest, XGBoost, LSTM, CNN-LSTM, ST-CNN-GRU, GA-KELM, Hybrid Forecaster.
  - Generated evaluation reports: `artifacts/experiments/test_evaluation_report.json` and `test_metrics_summary.csv`.
  - Metrics logged across all 6 horizons: MAE, RMSE, $R^2$, MAPE, Skill Score, and Early Warning $F_1$.

### Phase 6: Automated Testing & Pydantic V2 Migration
- Created unit tests `tests/test_ga_kelm.py` (5 tests) and `tests/test_fog_processing.py` (4 tests).
- Verified PyTorch 2.6 checkpoint loading compatibility (`weights_only=False`) across all deep models.
- Updated `src/api/models.py` to use Pydantic v2 `ConfigDict` and `json_schema_extra`, eliminating schema deprecation warnings.
- Robustified FastAPI route handlers (`forecasts.py`, `warnings.py`) to safely handle direct Python function calls.
- Executed full test suite: **38 passed, 0 failed** in 26.69s.

### Phase 7: Deployment Smoke Testing & Verification
- Created `scripts/verify_smoke_inprocess.py` to verify in-process deployment health without requiring open networking ports.
- Verified all 5 core API routes:
  1. `/health`: Healthy status, active model `hybrid_forecaster`, 8 preloaded models.
  2. `/stations`: 12 verified stations, historical sequence access.
  3. `/forecasts/{station}`: 6 forecast horizons ($1, 6, 12, 24, 48, 72$ hours) generated for Dongsi station.
  4. `/warnings/{station}`: 6 regulatory alerts with severity labels and actionable advisories.
  5. `/metrics`: Complete 8-model benchmark summary returned.

### Phase 8: Independent Scientific & Engineering Audit
- Executed rigorous audit across all 12 operational dimensions (see `docs/INDEPENDENT_AUDIT_REPORT.md`).
- Confirmed zero data leakage across train, val, and test splits.
- Re-verified test suite: 38 passed, 0 failed in 46.58s.
- Scientifically investigated performance dynamics: ST-CNN-GRU outperforms tree models at long horizons (h48, h72) while trees dominate short horizons (h1, h6) due to scalar lag features.
- Identified evaluation harness defect where static batch prediction never fed rolling ground truth errors to HybridForecaster (resulting in 0 fallback calls during batch evaluation).
- Identified architectural discrepancy in ST-CNN-GRU: `proximity` buffer is unreferenced in forward pass; attention is temporal over lookback window T=24 rather than multi-station spatial graph attention.
- Verified live dynamic FastAPI inference across all 8 models with clean subprocess restart.
