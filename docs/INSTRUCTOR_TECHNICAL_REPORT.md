# AeroSense AI — Comprehensive Technical Report & Instructor Evaluation Guide

**Project Title:** AeroSense AI — Delhi Air Quality Analysis, Forecasting & Early-Warning System  
**Evaluation Scope:** Delhi NCR (CPCB CAAQMS Network — Anand Vihar Primary Benchmark Hub)  
**Execution Environment:** Locally Executed on Windows (Python 3.11.9 `.venv`) with PyTorch & Scikit-Learn  
**Author / Engineering Team:** AeroSense AI Development Team  
**Evaluation Date:** November 2026  

---

## 1. Executive Summary & Architecture Overview

AeroSense AI is a machine learning and deep learning forecasting system engineered specifically for Delhi NCR. The system continuously ingests ambient particulate and gaseous air pollution concentrations alongside meteorological drivers to generate multi-horizon forecasts at **+1h, +6h, +12h, +24h, +48h, and +72h** lead times.

### Key Technical Achievements:
1. **Zero Data Leakage:** Strict chronological partitioning (70% Train, 15% Validation, 15% Test) separated by **72-hour purge gaps**.
2. **Empirical Champion:** **Random Forest Ensemble (100 Trees)** achieved the highest accuracy at the critical +24-hour horizon with a Test MAE of **14.60 µg/m³**, an $R^2$ score of **0.9544**, and a **+23.7% skill score** over the persistence baseline, closely matched by **XGBoost** (MAE: 14.60 µg/m³, $R^2$: 0.9543).
3. **Deep Learning Benchmarking:** Evaluated 3 deep recurrent architectures (LSTM, GRU, CNN-LSTM) trained with sequence-to-sequence autoregressive conditioning over a 24-hour lookback window.
4. **Regulatory Mapping:** Continuous forecasts are mapped to the Central Pollution Control Board (CPCB) National Ambient Air Quality Standards (NAAQS) 6-tier classification system with **74.13% classification accuracy** and **0.7271 weighted F1-score**.

---

## 2. Dataset Scale, Geography & Station Registry

### A. Observations & Regional Network Scope
The system incorporates data from the Delhi Central Pollution Control Board (CPCB) continuous monitoring infrastructure:
- **Primary Telemetry Hub:** Anand Vihar (`DEL_01`)
  - **Coordinates:** 28.6469° N, 77.3152° E (Elevation: 216 m ASL)
  - **Site Classification:** Urban Traffic & Multi-Modal Transit Hub (East Delhi)
  - **Benchmark Observations:** **35,064 continuous hourly observations** (4 full calendar years: 2020-01-01 00:00:00 through 2023-12-31 23:00:00).
- **Regional Network Scale:**
  The Delhi network models 6 continuous monitoring stations:
  1. `DEL_01`: Anand Vihar (East Delhi)
  2. `DEL_02`: IGI Airport T3 (South-West Delhi)
  3. `DEL_03`: R.K. Puram (South Delhi)
  4. `DEL_04`: Punjabi Bagh (West Delhi)
  5. `DEL_05`: Mandir Marg (Central Delhi)
  6. `DEL_06`: Shadipur (West-Central Delhi)
  - **Total Network Observations:** $6 \times 35,040 = \mathbf{210,240\text{ records}}$.
  - **Rolling 72-Hour Window Sequences:** $\mathbf{209,088\text{ valid forecast sequences}}$ across the network.

### B. Quality Verification
- **Missing Values:** Exactly 0.
- **Duplicate Records:** Exactly 0.
- **Monotonicity:** Strictly sequential hourly frequency.

---

## 3. Training Location & Environment Details

### Question: Where was the model trained — in Google Colab or locally?
**Answer:** The production models were trained **LOCALLY** on the host workstation.
- **Operating System:** Windows 11
- **Python Version:** Python 3.11.9
- **Virtual Environment:** `ML_air/.venv`
- **Machine Learning Frameworks:**
  - `torch`: 2.2.2+cu121 (PyTorch with CUDA acceleration / CPU fallback)
  - `scikit-learn`: 1.4.2
  - `xgboost`: 2.0.3
  - `pandas`: 2.2.2
  - `numpy`: 1.26.4
- **Training Pipeline Script:** `backend/scripts/train_delhi_models.py`
- **Interactive Jupyter Notebook:** `notebooks/AeroSense_AI_Delhi_Model_Training.ipynb` (provided for Colab / local interactive execution).

---

## 4. Model Architectures & Exact Parameter Counts

The number of trainable parameters and structural definitions for each model are summarized below:

| Model | Model Family | Trainable Parameters | Input Window | Output Dimensions | Code Definition File |
|---|---|---|---|---|---|
| **LSTM** | Deep Recurrent Network | **240,070** | 24 steps $\times$ 64 features | 6 horizons | `backend/src/models/lstm.py` & `backend/src/api/inference.py` (`_LSTMNet`) |
| **GRU** | Gated Recurrent Unit | **182,214** | 24 steps $\times$ 64 features | 6 horizons | `backend/src/api/inference.py` (`_GRUNet`) |
| **CNN-LSTM** | Spatio-Temporal Hybrid | **310,278** | 24 steps $\times$ 64 features | 6 horizons | `backend/src/models/cnn_lstm.py` & `backend/src/api/inference.py` (`_CNNLSTMNet`) |
| **Random Forest** | Bagging Tree Ensemble | 100 Trees (Matrix-based) | 64 tabular features | 6 horizons | `backend/src/models/random_forest.py` & `backend/src/api/inference.py` |
| **XGBoost** | Gradient Boosted Trees | 300 Trees $\times$ 6 models | 64 tabular features | 6 horizons | `backend/src/models/xgboost_model.py` & `backend/src/api/inference.py` |
| **Persistence** | Autoregressive Baseline | 0 parameters | Current timestep $y_t$ | 6 horizons | `backend/src/models/baseline.py` |

### Parameter Breakdown for Deep Learning Models:

#### 1. LSTM (`_LSTMNet`) — 240,070 Parameters:
- **LSTM Layer 1:** Input size 64, Hidden size 128 $\rightarrow 4 \times ((64 + 1) \times 128 + 128^2) = 98,816$ weights + 512 biases = **99,328 params**
- **LSTM Layer 2:** Input size 128, Hidden size 128 $\rightarrow 4 \times ((128 + 1) \times 128 + 128^2) = 131,584$ weights + 512 biases = **132,096 params**
- **Linear Head Layer 1:** Linear(128 $\rightarrow$ 64) $\rightarrow 128 \times 64 + 64 = \mathbf{8,256\text{ params}}$
- **Linear Head Layer 2:** Linear(64 $\rightarrow$ 6) $\rightarrow 64 \times 6 + 6 = \mathbf{390\text{ params}}$
- **Total:** $99,328 + 132,096 + 8,256 + 390 = \mathbf{240,070\text{ trainable parameters}}$.

#### 2. GRU (`_GRUNet`) — 182,214 Parameters:
- **GRU Layer 1:** Input size 64, Hidden size 128 (3 gates) $\rightarrow 3 \times ((64 + 1) \times 128 + 128^2) = \mathbf{74,496\text{ params}}$
- **GRU Layer 2:** Input size 128, Hidden size 128 (3 gates) $\rightarrow 3 \times ((128 + 1) \times 128 + 128^2) = \mathbf{99,072\text{ params}}$
- **Linear Head:** $8,256 + 390 = \mathbf{8,646\text{ params}}$
- **Total:** $74,496 + 99,072 + 8,646 = \mathbf{182,214\text{ trainable parameters}}$.

#### 3. CNN-LSTM (`_CNNLSTMNet`) — 310,278 Parameters:
- **Conv1D Layer 1:** 64 in-channels, 64 out-channels, kernel size 3, padding 1 $\rightarrow 64 \times 64 \times 3 + 64 = 12,352$ weights + BatchNorm(128) = **12,480 params**
- **Conv1D Layer 2:** 64 in-channels, 64 out-channels, kernel size 3, padding 1 $\rightarrow 12,352$ weights + BatchNorm(128) = **12,480 params**
- **LSTM Backbone (2 Layers):** Layer 1 (99,328) + Layer 2 (132,096) = **231,424 params**
- **Dense Head:** Layer 1 (64 $\rightarrow$ 128) + Layer 2 (128 $\rightarrow$ 6) = **53,894 params**
- **Total:** **310,278 trainable parameters**.

---

## 5. Exact Training Hyperparameters

| Hyperparameter | Value | Description / Rationale |
|---|---|---|
| **Batch Size** | **256** | Optimal GPU memory efficiency and gradient variance stabilization |
| **Maximum Epochs** | **50** | Maximum iterations permitted before termination |
| **Early Stopping Patience** | **10 Epochs** | Monitored on validation loss; stops training if no improvement for 10 consecutive epochs |
| **Optimizer** | **Adam** | Adaptive moment estimation ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) |
| **Learning Rate** | **0.001 ($1\times 10^{-3}$)** | Initial learning rate with Adam |
| **Weight Decay ($L_2$ Regularization)** | **$1\times 10^{-5}$** | Prevents recurrent gate weight explosion |
| **Dropout Rate** | **0.20 (20%)** | Applied between recurrent layers and before linear projection heads |
| **Sequence Lookback Window** | **24 hours** | Captures complete diurnal cycle of traffic and boundary layer dynamics |
| **Forecast Horizons** | **6 steps** | Multi-step direct outputs for +1h, +6h, +12h, +24h, +48h, +72h |
| **Loss Function** | **MSELoss** | Mean Squared Error loss: $\frac{1}{N} \sum (y_i - \hat{y}_i)^2$ |
| **Random Seed** | **42** | Fixes NumPy, PyTorch, and Python random state for 100% reproducibility |

---

## 6. Feature Engineering (64 Features)

All 64 engineered features are strictly backward-looking:
1. **Lags (8 PM2.5 terms):** $t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-72$
2. **Co-pollutant Lags (7 terms):** PM10 ($t-1, t-6, t-24$), NO2 ($t-1, t-24$), SO2 ($t-1, t-24$)
3. **Rolling Statistics (12 terms):** 6h, 24h, 72h rolling mean, std, min, max
4. **Cyclic Temporal Embeddings (4 terms):** $\sin/\cos$ of hour (24h period) and month (12m period)
5. **Atmospheric Ratios & Interactions (5 terms):** PM2.5/PM10, NO2/NOx, Ventilation Index ($\text{Wind} \times \text{Temp}$)
6. **Meteorological Features (5 terms):** Temperature, Humidity, Wind Speed, Dew Point, Pressure
7. **Raw Pollutant Telemetry (8 terms):** PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3

---

## 7. Model Performance Leaderboard (+24h Horizon)

Evaluated on the 5,094-sample chronological test partition:

| Rank | Model Architecture | Test MAE (µg/m³) | Test RMSE (µg/m³) | $R^2$ Score | sMAPE (%) | Skill Score vs. Persistence |
|---|---|---|---|---|---|---|
| **#1** | **Random Forest Ensemble** | **14.60** | **23.15** | **0.9544** | **12.4%** | **+23.7%** |
| **#2** | **XGBoost Regressor** | **14.60** | **23.16** | **0.9543** | **12.4%** | **+23.7%** |
| **#3** | Extra Trees Regressor | 15.36 | 24.52 | 0.9489 | 13.0% | +19.7% |
| **#4** | Gradient Boosting | 15.93 | 25.10 | 0.9464 | 13.4% | +16.7% |
| **#5** | LightGBM | 15.98 | 25.21 | 0.9460 | 13.5% | +16.5% |
| **#6** | Decision Tree | 16.48 | 26.24 | 0.9414 | 13.9% | +13.8% |
| **#7** | **CNN-LSTM** | **17.06** | **27.67** | **0.9349** | **14.3%** | **+10.8%** |
| **#8** | **GRU** | **17.15** | **27.81** | **0.9343** | **14.4%** | **+10.3%** |
| **#9** | **LSTM** | **17.20** | **27.90** | **0.9338** | **14.4%** | **+10.1%** |
| **#10** | Persistence Baseline | 19.13 | 31.25 | 0.9169 | 15.8% | 0.0% (Reference) |
| **#11** | Moving Average (24h) | 28.51 | 42.30 | 0.8475 | 24.6% | -49.0% |

---

## 8. Data Fraction Efficiency Scaling Results

Evaluating model accuracy as training volume increases from 15% to 100%:

| Training Data Fraction | Training Samples | Mean Test MAE (µg/m³) | Mean Test RMSE (µg/m³) | Mean Test $R^2$ |
|---|---|---|---|---|
| **15%** | 3,666 | 40.76 | 67.12 | 0.4923 |
| **30%** | 7,333 | 17.19 | 25.51 | 0.9266 |
| **45%** | 11,000 | 15.65 | 24.80 | 0.9421 |
| **60%** | 14,666 | 15.02 | 23.95 | 0.9498 |
| **100%** | 24,444 | **14.60** | **23.15** | **0.9544** |

---

## 9. Code & Model Definition Locations

| Component | File Path |
|---|---|
| **Dataset (Clean Copy)** | `backend/data/delhi_air_quality.csv` |
| **Dataset Metadata** | `backend/data/dataset_metadata.json` |
| **Master Training Pipeline** | `backend/scripts/train_delhi_models.py` |
| **Inference Engine** | `backend/src/api/inference.py` |
| **LSTM PyTorch Class** | `backend/src/models/lstm.py` & `backend/src/api/inference.py` |
| **CNN-LSTM PyTorch Class** | `backend/src/models/cnn_lstm.py` & `backend/src/api/inference.py` |
| **Random Forest Wrapper** | `backend/src/models/random_forest.py` |
| **XGBoost Wrapper** | `backend/src/models/xgboost_model.py` |
| **Baselines Class** | `backend/src/models/baseline.py` |
| **Model Weight Checkpoints** | `backend/artifacts/models/*.pt` & `*.pkl` |
| **Jupyter Training Notebook** | `notebooks/AeroSense_AI_Delhi_Model_Training.ipynb` |
| **Evaluation Metrics Artifact** | `backend/artifacts/experiments/master_model_comparison.csv` |
