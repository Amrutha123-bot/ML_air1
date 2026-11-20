# Stage 5: Deployment, FastAPI Architecture & Real-Time Monitoring

## 1. System Architecture Overview
The AeroSense AI system deploys a decoupled FastAPI asynchronous microservice backend alongside a React 18 / Vite / TailwindCSS single-page application. The system operates on a lavender/indigo aesthetic theme and serves real-time predictions and analytics exclusively for Delhi NCR.

```
       +-------------------------------------------------------------+
       |             React 18 / Vite Frontend Dashboard             |
       |  (Overview, Forecasting, Stations, Lab, Alerts, Analytics)  |
       +------------------------------+------------------------------+
                                      | HTTP REST JSON (/api/v1)
                                      v
       +-------------------------------------------------------------+
       |                   FastAPI Backend Service                   |
       |    (Health, Stations, Forecasts, Warnings, Metrics, EDA)    |
       +------------------------------+------------------------------+
                                      |
                 +--------------------+--------------------+
                 v                                         v
       +----------------------+                  +--------------------+
       | Trained Models (.pt) |                  | Scalers & Feature  |
       |  Random Forest, XGB, |                  | Columns JSON Store |
       |  LSTM, GRU, CNN-LSTM |                  +--------------------+
       +----------------------+
```

---

## 2. API Endpoint Matrix (`/api/v1`)

| Endpoint Route | HTTP Method | Purpose & Description | Input Parameters | Output Schema |
|---|---|---|---|---|
| `/api/v1/health` | `GET` | Service telemetry, active model, device (CPU/CUDA) | None | `HealthResponse` |
| `/api/v1/stations` | `GET` | List all 6 Delhi monitoring stations in network | None | `StationListResponse` |
| `/api/v1/stations/{station}/history` | `GET` | Retrieve chronological station observation history | `limit` (int), `offset` (int) | `List[HistoricalObservation]` |
| `/api/v1/forecasts` | `GET` / `POST` | Generate multi-horizon predictions (+1h to +72h) | `model_name`, `hours_ahead`, `offset` | `ForecastResponse` |
| `/api/v1/warnings/active` | `GET` | CPCB NAAQS early warning alerts & health actions | `horizon` (str) | `ActiveWarningsResponse` |
| `/api/v1/metrics` | `GET` | Comprehensive evaluation results across all models | None | `dict` (count, metrics list) |
| `/api/v1/metrics/leaderboard` | `GET` | +24h leaderboard sorted by test MAE | None | `dict` (rank, model, MAE, R2) |
| `/api/v1/metrics/confusion` | `GET` | CPCB 6-tier normalized confusion matrix & F1 | None | `dict` (labels, matrix, metrics) |
| `/api/v1/metrics/fractions` | `GET` | Data scaling efficiency results (15% to 100%) | None | `dict` (experiments list) |
| `/api/v1/analytics/eda` | `GET` | Summary statistics, diurnal curves, correlations | None | `dict` (distributions, stats) |
| `/api/v1/analytics/trends` | `GET` | Continuous PM2.5 trend series | `limit` (default 168) | `List[dict]` |

---

## 3. Inference Engine Runtime Design
The `DelhiInferenceEngine` (`backend/src/api/inference.py`) implements a thread-safe singleton design pattern with lazy model loading:
- **Lazy Loading:** Models are loaded into memory and bound to PyTorch CUDA tensors only upon the first forecast request, avoiding startup latency.
- **Model Agnostic Interface:** Common `.predict_horizons(features, horizon_steps)` method for tree ensembles, PyTorch neural networks, and baselines.
- **CPCB NAAQS Categorization:** Automatic dynamic mapping of continuous forecast values to CPCB alert tiers.
- **Fallback Guarantee:** Deterministic mathematical fallback to persistence baseline ensures 100% uptime even if a model weight file is missing.
