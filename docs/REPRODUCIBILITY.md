# AeroSense AI V2 — Reproducibility Guide

> Phase 19 deliverable. Last updated: 2026-11-17.

## Determinism Policy

All random seeds are fixed at `seed=42` throughout the pipeline:

```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The XGBoost and RandomForest models also use `random_state=42`.

## Data Split Specification

Chronological, **no shuffling**, with a **96-hour purge gap** between splits:

| Split | Rows | Date Range (Delhi) |
|-------|------|--------------------|
| Train | 70% | 2023-01-01 – ~2024-07-07 |
| Val   | 15% | (+ 96h gap) ~2024-07-11 – ~2024-09-10 |
| Test  | 15% | (+ 96h gap) ~2024-09-14 – 2024-12-30 |

**Rule:** Test data is never used for hyperparameter tuning or early-stopping decisions.

## Environment Pinning

```text
# backend/requirements.txt (key packages)
torch==2.3.1
xgboost==2.0.3
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
fastapi==0.111.0
```

Colab environment: Python 3.10 (Colab default), CUDA 11.8, NVIDIA Tesla T4.

## Running the Full Pipeline

### Step 1 — Dataset Inspection (local)
```bash
$env:PYTHONPATH="backend"
.venv\Scripts\python backend/scripts/run_dataset_inspection.py
# Expected: 14/14 audit checks pass
```

### Step 2 — Feature Engineering (local)
```bash
.venv\Scripts\python -m aerosense_ml.preprocessing.features
# Writes backend/data/features/delhi/feature_columns.json
```

### Step 3 — Training (Colab GPU required)
1. Upload `AeroSense_Colab_Bundle.zip` to Colab via Files panel
2. Run `AeroSense_AI_Master_Colab_Pipeline.ipynb` cells sequentially
3. Download `AeroSense_AI_Training_Results.zip` from `/content/`
4. Extract and place artifacts as described in `docs/PROJECT_SETUP.md §4`

### Step 4 — Backend Verification
```bash
$env:PYTHONPATH="backend"
.venv\Scripts\python -m pytest backend/tests -q
# Expected: 90 passed, 0 failed
```

### Step 5 — Frontend Build
```bash
pnpm --prefix frontend run build
# Expected: 0 TypeScript errors
```

## Experiment Tracking

All results are stored in `backend/artifacts/experiments/`:
- `master_model_comparison.csv` — 42-row benchmark table (7 models × 6 horizons)
- `cross_city_transfer_results.csv` — 10-row transfer sweep (2 cities × 5 fractions)
- `dataset_inventory.csv` — 3-row dataset audit (Delhi, Mumbai, Vijayawada)

Plots are in `reports/plots/`.

## Known Non-Reproducible Elements

1. **Synthetic data generator:** `harmonizer.py` uses `np.random.RandomState(seed=42)` but the exact row count depends on the generator's date range parameters. Any change to `start_date` or `end_date` will change the dataset.

2. **Colab environment drift:** Google may update Colab's Python/CUDA versions without notice. Pin the runtime: `Runtime → Change runtime type → Python 3.10` and select T4 GPU.

3. **Checkpoint epoch:** The ST-CNN-GRU checkpoint was saved at `best_epoch` during training; re-training from scratch will produce a different epoch number and slightly different weights due to CUDA non-determinism even with `deterministic=True`.
