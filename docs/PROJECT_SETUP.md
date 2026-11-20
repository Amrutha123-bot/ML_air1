# AeroSense AI V2 — Project Setup Guide

> Phase 19 deliverable. Last updated: 2026-11-17.

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| Python | 3.11 | 3.12+ not tested |
| Node.js | 18.x | Required for frontend |
| pnpm | 8.x | `npm install -g pnpm` |
| Git | Any | — |
| (Optional) NVIDIA GPU | CUDA 11.8+ | For Colab training only |

## 1 · Clone and Python Environment

```bash
git clone <repo-url> ML_air
cd ML_air

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install backend dependencies
pip install -r backend/requirements.txt

# Install aerosense_ml package (editable)
pip install -e .
```

## 2 · Backend (FastAPI)

```bash
cd backend
uvicorn src.main:app --reload --port 8000
```

API will be available at `http://localhost:8000`. Interactive docs at `/docs`.

**Key environment variables** (`.env` in `backend/`):

| Variable | Default | Description |
|----------|---------|-------------|
| `AEROSENSE_ENV` | `development` | Switch to `production` for no debug logging |
| `MODELS_DIR` | `artifacts/models` | Absolute or relative path to checkpoint directory |

## 3 · Frontend (React + Vite)

```bash
cd frontend
pnpm install
pnpm run dev        # Development server on :5173
pnpm run build      # Production build (checks TypeScript)
```

## 4 · Model Artifacts

Place trained checkpoints in `backend/artifacts/models/`:

| File | Size | Source |
|------|------|--------|
| `stcnn_gru_delhi.pt` | 1.18 MB | Colab GPU training (Phase 7) |
| `xgboost_delhi.pkl` | 4.24 MB | Colab CPU training |
| `random_forest_delhi.pkl` | 35.9 MB | Colab CPU training |
| `lstm_model.pt` | 257 KB | Colab GPU training |
| `cnn_lstm_model.pt` | 154 KB | Colab GPU training |
| `persistence_model.pkl` | 124 B | Generated locally |

All artifacts are verified in `backend/tests/test_registries_and_validation.py`.

## 5 · Run Tests

```bash
$env:PYTHONPATH="backend"
.venv\Scripts\python -m pytest backend/tests -q
# Expected: 90+ passed, 0 failed
```

## 6 · Google Colab Training

See `docs/COLAB_EXECUTION_GUIDE.md` for the step-by-step guide to training on Colab.
Upload `AeroSense_Colab_Bundle.zip` via the Colab Files panel (no Drive mount needed).
