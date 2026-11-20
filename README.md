# AeroSense AI — Delhi (Anand Vihar) Air Quality Forecasting & Early-Warning System

[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-teal.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3.1-61dafb.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/Tests-33%2F33%20Passed%20(100%25)-brightgreen.svg)]()
[![Target Station](https://img.shields.io/badge/Station-Anand%20Vihar%20(Delhi)-purple.svg)]()

> **AeroSense AI** is an end-to-end, data-driven machine learning and deep learning air quality forecasting and regulatory decision-support system engineered specifically for **Delhi (Anand Vihar)**.

The system continuously processes criteria air pollutants ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3, \text{NH}_3$) alongside atmospheric boundary layer meteorology to generate calibrated multi-horizon predictions at **+1h, +6h, +12h, +24h, +48h, and +72h** lead times. Predictions are directly mapped to the Central Pollution Control Board (CPCB) National Ambient Air Quality Standards (NAAQS) and Graded Response Action Plan (GRAP) alert stages.

---

## 📑 Table of Contents
1. [Project Overview & Key Highlights](#1-project-overview--key-highlights)
2. [Monitoring Station Specification — Anand Vihar, Delhi](#2-monitoring-station-specification--anand-vihar-delhi)
3. [Dataset Architecture & Quality Verification](#3-dataset-architecture--quality-verification)
4. [Time-Series Feature Engineering Architecture](#4-time-series-feature-engineering-architecture)
5. [Data Partitioning & Leakage Prevention](#5-data-partitioning--leakage-prevention)
6. [Model Architectures & Exact Parameter Counts](#6-model-architectures--exact-parameter-counts)
7. [Training Hyperparameters & Environment Details](#7-training-hyperparameters--environment-details)
8. [Comprehensive Empirical Evaluation & Results](#8-comprehensive-empirical-evaluation--results)
9. [CPCB Regulatory Mapping & Confusion Matrix](#9-cpcb-regulatory-mapping--confusion-matrix)
10. [Data Scaling Efficiency Experiments](#10-data-scaling-efficiency-experiments)
11. [System Architecture & REST API Endpoints](#11-system-architecture--rest-api-endpoints)
12. [Project File & Directory Guide](#12-project-file--directory-guide)
13. [Complete Step-by-Step Setup & Reproduction Guide](#13-complete-step-by-step-setup--reproduction-guide)

---

## 1. Project Overview & Key Highlights

- **Dedicated Geographic Scope:** Exclusively focused on **Delhi (Anand Vihar)**.
- **Continuous Temporal Scale:** **35,064 continuous hourly observations** spanning 4 full calendar years (2020-01-01 00:00:00 to 2023-12-31 23:00:00).
- **Strict Chronological Discipline:** 70% Train, 15% Validation, and 15% Test partitions separated by **72-hour purge gaps** with feature scalers fitted out-of-sample on training data only to guarantee zero data leakage.
- **Empirical Champion Model:** **Random Forest Ensemble (100 Trees)** achieved the highest forecasting accuracy at the critical +24-hour horizon with a Test MAE of **14.60 µg/m³**, an $R^2$ score of **0.9544**, and a **+23.7% skill score** over the persistence baseline, closely matched by **XGBoost** (MAE: 14.60 µg/m³, $R^2$: 0.9543).
- **Deep Recurrent Benchmarking:** Implemented and evaluated PyTorch deep learning architectures (LSTM, GRU, CNN-LSTM) conditioned on a 24-hour sequence lookback window.
- **Regulatory Decision Support:** Direct CPCB NAAQS 6-tier classification mapping with **74.13% classification accuracy** and **0.7271 weighted F1-score**.
- **Automated Verification:** Comprehensive test suite with **33 out of 33 tests passing (100% green)** and a production frontend build with 0 errors.

---

## 2. Monitoring Station Specification — Anand Vihar, Delhi

The entire system is calibrated, trained, evaluated, and deployed exclusively on continuous telemetry from the Anand Vihar monitoring hub:

- **City:** Delhi
- **Station Name:** Anand Vihar (`DEL_01`)
- **Geographic Coordinates:** $28.6469^\circ\text{ N}, 77.3152^\circ\text{ E}$
- **Elevation:** $216\text{ m ASL}$
- **Station Classification:** Continuous Ambient Air Quality Monitoring Station (CAAQMS) — Urban Traffic & Commercial Transport Hub
- **Dataset Duration:** 4 Full Calendar Years (2020–2023)
- **Continuous Observations:** **35,064 hourly records**
- **Temporal Resolution:** Strictly sequential 1-hour intervals ($\Delta t = 1\text{ hour}$)
- **Data Quality:** 0 missing values, 0 duplicate timestamps, 100% verified chronological monotonicity

---

## 3. Dataset Architecture & Quality Verification

- **Primary Data File:** `backend/data/delhi_air_quality.csv` (4.82 MB)
- **Temporal Span:** 2020-01-01 00:00:00 through 2023-12-31 23:00:00 (35,064 hourly samples).
- **Temporal Resolution:** Continuous hourly intervals.

### Column Specifications (18 Raw Variables):

| Column Name | Physical Quantity | Standard Units | Physical Bounds | Description |
|---|---|---|---|---|
| `Timestamp` | Chronological Timestamp | ISO-8601 String | 2020-01-01 to 2023-12-31 | Observation date and time |
| `City` | Geographic Entity | String | Delhi | Target jurisdiction |
| `Station` | Monitoring Station Code | String | DEL_01 | Anand Vihar hub |
| `PM2.5` | Fine Particulate Matter ($\le 2.5\,\mu\text{m}$) | $\mu\text{g/m}^3$ | $[0.0, 1000.0]$ | **Primary Target Variable** |
| `PM10` | Coarse Particulate Matter ($\le 10\,\mu\text{m}$) | $\mu\text{g/m}^3$ | $[0.0, 2000.0]$ | Inhalable particulate fraction |
| `NO` | Nitric Oxide | $\mu\text{g/m}^3$ | $[0.0, 500.0]$ | Primary combustion byproduct |
| `NO2` | Nitrogen Dioxide | $\mu\text{g/m}^3$ | $[0.0, 1000.0]$ | Traffic emission tracer |
| `NOx` | Total Nitrogen Oxides | $\text{ppb}$ | $[0.0, 1000.0]$ | Cumulative nitrogen oxides |
| `NH3` | Ammonia | $\mu\text{g/m}^3$ | $[0.0, 1000.0]$ | Agricultural & secondary aerosol precursor |
| `SO2` | Sulfur Dioxide | $\mu\text{g/m}^3$ | $[0.0, 1000.0]$ | Industrial emission tracer |
| `CO` | Carbon Monoxide | $\text{mg/m}^3$ | $[0.0, 15.0]$ | Incomplete combustion tracer |
| `O3` | Ground-level Ozone | $\mu\text{g/m}^3$ | $[0.0, 1000.0]$ | Photochemical smog indicator |
| `AQI` | Air Quality Index | Dimensionless (0–500) | $[0, 500]$ | CPCB National AQI sub-index |
| `AQI_Category` | CPCB Severity Tier | Categorical | 6 Tiers | Good, Satisfactory, Moderate, Poor, Very Poor, Severe |
| `Temperature` | Ambient Surface Temperature | $^\circ\text{C}$ | $[-5.0, 55.0]$ | Inversion and boundary layer driver |
| `Humidity` | Relative Humidity | $\%$ | $[5.0, 100.0]$ | Aerosol hygroscopic growth driver |
| `Wind_Speed` | Surface Wind Speed | $\text{m/s}$ | $[0.0, 40.0]$ | Atmospheric dispersion driver |
| `Hour` | Diurnal Observation Hour | Integer (0–23) | $[0, 23]$ | Daily traffic & boundary layer cycle |

### Automated Quality Verification Results:
1. **Missing Value Count:** Exactly **0** missing values across all 35,064 records.
2. **Duplicate Check:** Exactly **0** duplicate timestamps.
3. **Temporal Monotonicity:** Continuous chronological progression verified without gaps.

---

## 4. Time-Series Feature Engineering Architecture

### Why Did Features Expand from 18 Raw Columns to 56 Model Features?

In static tabular machine learning, each row is independent. **In time-series air quality forecasting, an instantaneous row has zero memory.**

If a model only observes $\text{PM}_{2.5} = 150\,\mu\text{g/m}^3$ at the current hour:
- It cannot tell if pollution is surging from $40\,\mu\text{g/m}^3$ (a rapid morning traffic spike).
- It cannot tell if pollution is collapsing from $350\,\mu\text{g/m}^3$ (rain washout).
- It cannot observe diurnal recurrence (yesterday's rush-hour cycle).

To supply velocity, acceleration, and historical trajectory to machine learning algorithms, we apply **time-delay embedding** (Takens' Theorem) and backward-looking rolling statistics:

```
Raw Instantaneous Telemetry (18 Columns)
        │
        ▼  [Domain Pruning: Remove non-predictive string columns & noise]
        ▼  [Autoregressive Lag Extraction: t-1h to t-72h]
        ▼  [Rolling Dispersion Windows: 6h, 24h, 72h with shift(1)]
        ▼  [Atmospheric Interaction Ratios & Cyclical Trigonometric Encodings]
        │
        ▼
56 Engineered Predictive Inputs (Saved in feature_columns.json)
```

### Complete Breakdown of the 56 Engineered Model Features:

1. **Autoregressive $\text{PM}_{2.5}$ Lags (8 features):**
   `PM2.5_lag_1h`, `PM2.5_lag_2h`, `PM2.5_lag_3h`, `PM2.5_lag_6h`, `PM2.5_lag_12h`, `PM2.5_lag_24h`, `PM2.5_lag_48h`, `PM2.5_lag_72h`
2. **Co-Pollutant $\text{PM}_{10}$ Lags (8 features):**
   `PM10_lag_1h`, `PM10_lag_2h`, `PM10_lag_3h`, `PM10_lag_6h`, `PM10_lag_12h`, `PM10_lag_24h`, `PM10_lag_48h`, `PM10_lag_72h`
3. **Chemical Combustion $\text{NO}_2$ Lags (8 features):**
   `NO2_lag_1h`, `NO2_lag_2h`, `NO2_lag_3h`, `NO2_lag_6h`, `NO2_lag_12h`, `NO2_lag_24h`, `NO2_lag_48h`, `NO2_lag_72h`
4. **Thermal Inversion Temperature Lags (8 features):**
   `Temperature_lag_1h`, `Temperature_lag_2h`, `Temperature_lag_3h`, `Temperature_lag_6h`, `Temperature_lag_12h`, `Temperature_lag_24h`, `Temperature_lag_48h`, `Temperature_lag_72h`
5. **Dispersion Wind Speed Lags (8 features):**
   `Wind_Speed_lag_1h`, `Wind_Speed_lag_2h`, `Wind_Speed_lag_3h`, `Wind_Speed_lag_6h`, `Wind_Speed_lag_12h`, `Wind_Speed_lag_24h`, `Wind_Speed_lag_48h`, `Wind_Speed_lag_72h`
6. **Backward-Looking Rolling Statistics (6 features):**
   - $6\text{h}$ Window: `PM2.5_roll_mean_6h`, `PM2.5_roll_std_6h`
   - $24\text{h}$ Window: `PM2.5_roll_mean_24h`, `PM2.5_roll_std_24h`
   - $72\text{h}$ Window: `PM2.5_roll_mean_72h`, `PM2.5_roll_std_72h`  
   *(Strictly computed using `shift(1)` to eliminate lookahead bias).*
7. **Atmospheric Ratios & Physical Interactions (4 features):**
   - `pm25_to_pm10_ratio` ($\text{PM}_{2.5} / (\text{PM}_{10} + \epsilon)$) — Differentiates vehicular combustion from mechanical dust.
   - `no2_to_nox_ratio` ($\text{NO}_2 / (\text{NO}_x + \epsilon)$) — Atmospheric photochemical oxidation index.
   - `temp_wind_interaction` ($\text{Temperature} \times \text{Wind Speed}$) — Ventilation volume proxy.
   - `humidity_wind_stagnation` — Boundary layer trapping index.
8. **Trigonometric Cyclical Embeddings (6 features):**
   - Diurnal Hour: `hour_sin`, `hour_cos` ($\sin/\cos(2\pi h / 24)$)
   - Seasonal Day of Year: `doy_sin`, `doy_cos` ($\sin/\cos(2\pi \text{doy} / 365.25)$)
   - Weekly Traffic Cycle: `dow_sin`, `dow_cos` ($\sin/\cos(2\pi \text{dow} / 7)$)

$$\mathbf{8 + 8 + 8 + 8 + 8 + 6 + 4 + 6 = 56 \text{ Predictive Input Features}}$$

> **Feature Selection:** Had all 18 raw columns been lagged across all 8 steps with full statistics, the input space would have exceeded **200+ features**. Feature selection pruned non-informative variables (e.g., categorical labels, uninformative secondary gas lags), isolating the 56 most predictive features.

---

## 5. Data Partitioning & Leakage Prevention

To guarantee scientific validity, the data was partitioned using strict chronological time-series splitting:

```
[2020-01-01 00:00:00]                                                              [2023-12-31 23:00:00]
├─────────────────────────────────────────┼──────┼──────────────────┼──────┼──────────────────┤
│           TRAIN SET (70%)               │PURGE │  VAL SET (15%)   │PURGE │  TEST SET (15%)  │
│             24,444 samples              │ 72h  │  5,238 samples   │ 72h  │  5,094 samples   │
└─────────────────────────────────────────┴──────┴──────────────────┴──────┴──────────────────┘
```

1. **Chronological Discipline:** No random shuffling was performed anywhere in the pipeline.
2. **72-Hour Purge Gaps:** To prevent overlap of target sequences across boundaries, a $72\text{-hour}$ purge window was placed between Train and Validation, and between Validation and Test.
3. **Out-of-Sample Scaler Fitting:** The `StandardScaler` (`delhi_feature_scaler.pkl`) was fitted **exclusively on the training partition** ($N=24,444$). The validation and test partitions were transformed using training parameters.

---

## 6. Model Architectures & Exact Parameter Counts

All models predict multi-step forward horizons simultaneously: **+1h, +6h, +12h, +24h, +48h, and +72h**.

| Model Architecture | Model Family | Parameter Count / Complexity | Input Lookback | Model Definition Location |
|---|---|---|---|---|
| **LSTM** | Deep Recurrent Network | **64,902 parameters** *(240,070 for 128-dim)* | 24 hours ($24 \times 56$) | `backend/src/models/lstm.py` & `backend/src/api/inference.py` (`_LSTMNet`) |
| **GRU** | Gated Recurrent Unit | **48,774 parameters** *(182,214 for 128-dim)* | 24 hours ($24 \times 56$) | `backend/src/api/inference.py` (`_GRUNet`) |
| **CNN-LSTM** | Spatio-Temporal Hybrid | **30,886 parameters** *(310,278 for 128-dim)* | 24 hours ($24 \times 56$) | `backend/src/models/cnn_lstm.py` & `backend/src/api/inference.py` (`_CNNLSTMNet`) |
| **Random Forest** | Bagging Ensemble | **100 Decision Trees** | 56 tabular features | `backend/src/models/random_forest.py` & `backend/src/api/inference.py` |
| **XGBoost** | Gradient Boosted Trees | **1,800 Trees** (300 per horizon $\times 6$) | 56 tabular features | `backend/src/models/xgboost_model.py` & `backend/src/api/inference.py` |
| **Persistence Baseline** | Naive Autoregressive | 0 parameters | Current $y_t$ | `backend/src/models/baseline.py` & `backend/src/api/inference.py` |
| **Seasonal Persistence** | 24h Diurnal Baseline | 0 parameters | $y_{t+h-24}$ | `backend/src/models/baseline.py` |
| **Moving Average** | 24h Rolling Mean | 0 parameters | $\bar{y}_{t-24:t}$ | `backend/src/models/baseline.py` |

### Detailed Layer Parameter Breakdown (Production Checkpoints):

#### 1. LSTM (`lstm_delhi.pt`) — 64,902 Parameters:
- **Layer 1:** `lstm.weight_ih_l0` $[256, 56] = 14,336$ | `lstm.weight_hh_l0` $[256, 64] = 16,384$ | Biases $= 512$
- **Layer 2:** `lstm.weight_ih_l1` $[256, 64] = 16,384$ | `lstm.weight_hh_l1` $[256, 64] = 16,384$ | Biases $= 512$
- **Linear Head:** `fc.weight` $[6, 64] = 384$ | `fc.bias` $[6] = 6$
- **Total:** $14,336 + 16,384 + 512 + 16,384 + 16,384 + 512 + 384 + 6 = \mathbf{64,902\text{ trainable parameters}}$.

#### 2. GRU (`gru_delhi.pt`) — 48,774 Parameters:
- **Layer 1:** `gru.weight_ih_l0` $[192, 56] = 10,752$ | `gru.weight_hh_l0` $[192, 64] = 12,288$ | Biases $= 384$
- **Layer 2:** `gru.weight_ih_l1` $[192, 64] = 12,288$ | `gru.weight_hh_l1` $[192, 64] = 12,288$ | Biases $= 384$
- **Linear Head:** `fc.weight` $[6, 64] = 384$ | `fc.bias` $[6] = 6$
- **Total:** $10,752 + 12,288 + 384 + 12,288 + 12,288 + 384 + 384 + 6 = \mathbf{48,774\text{ trainable parameters}}$.

#### 3. CNN-LSTM (`cnn_lstm_delhi.pt`) — 30,886 Parameters:
- **Conv1D Layer:** `conv.0.weight` $[32, 56, 3] = 5,376$ | `conv.0.bias` $[32] = 32$
- **LSTM Backbone:** `lstm.weight_ih_l0` $[256, 32] = 8,192$ | `lstm.weight_hh_l0` $[256, 64] = 16,384$ | Biases $= 512$
- **Linear Head:** `fc.weight` $[6, 64] = 384$ | `fc.bias` $[6] = 6$
- **Total:** $5,376 + 32 + 8,192 + 16,384 + 512 + 384 + 6 = \mathbf{30,886\text{ trainable parameters}}$.

---

## 7. Training Hyperparameters & Environment Details

### Training Location:
- **Execution:** Trained **LOCALLY on Windows 11** inside the dedicated Python virtual environment (`.venv`).
- **Python Version:** Python 3.11.9
- **Acceleration:** PyTorch CUDA GPU with automatic CPU fallback.
- **Master Training Pipeline:** `backend/scripts/train_delhi_models.py`
- **Interactive Jupyter Notebook:** `notebooks/AeroSense_AI_Delhi_Model_Training.ipynb`

### Pinned Hyperparameters:

| Hyperparameter | PyTorch Deep Learning (LSTM / GRU / CNN-LSTM) | Random Forest Regressor | XGBoost Regressor |
|---|---|---|---|
| **Batch Size** | **256** | N/A (Full matrix) | Subsample mini-batches |
| **Max Epochs** | **50 Epochs** | 100 Trees | 300 Boosting Rounds per Horizon |
| **Early Stopping** | **Patience = 10 Epochs** on Validation Loss | N/A | Early stopping on validation RMSE |
| **Learning Rate** | **0.001 ($1\times 10^{-3}$)** | N/A | **0.05** |
| **Optimizer** | **Adam** ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | N/A | Second-order Taylor gradient descent |
| **Weight Decay** | $1 \times 10^{-5}$ ($L_2$ regularization) | N/A | `reg_lambda = 1.0` |
| **Dropout** | **0.20 (20%)** | N/A | N/A |
| **Sequence Window** | **24 hours lookback** | N/A | N/A |
| **Criterion / Loss** | Mean Squared Error (MSELoss) | MSE split criterion | `reg:squarederror` |
| **Random Seed** | 42 (100% reproducible) | 42 | 42 |

---

## 8. Comprehensive Empirical Evaluation & Results

All models were evaluated on the **5,094-sample untouched chronological test partition** (representing 5,022 valid 72-hour forecast evaluations).

### Benchmark Leaderboard (+24-Hour Horizon):

```
===================================================================================================================
Rank   Model Architecture               Test MAE (µg/m³)   Test RMSE   R² Score   sMAPE (%)   Skill vs Baseline  Time
===================================================================================================================
#1     Random Forest Ensemble (100)          14.60          23.15       0.9544      12.4%         +23.7%        22.4s
#2     XGBoost Regressor                     14.60          23.16       0.9543      12.4%         +23.7%         3.2s
#3     Extra Trees Regressor                 15.36          24.52       0.9489      13.0%         +19.7%        18.7s
#4     Gradient Boosting Regressor           15.93          25.10       0.9464      13.4%         +16.7%        14.1s
#5     LightGBM Regressor                    15.98          25.21       0.9460      13.5%         +16.5%         1.8s
#6     Decision Tree Regressor               16.48          26.24       0.9414      13.9%         +13.8%         0.9s
#7     CNN-LSTM Hybrid                       17.06          27.67       0.9349      14.3%         +10.8%        48.2s
#8     GRU Network                           17.15          27.81       0.9343      14.4%         +10.3%        39.5s
#9     LSTM Network                          17.20          27.90       0.9338      14.4%         +10.1%        42.1s
#10    Persistence Baseline                  19.13          31.25       0.9169      15.8%           0.0% (Ref)   0.0s
#11    Moving Average (24h)                  28.51          42.30       0.8475      24.6%         -49.0%         0.0s
===================================================================================================================
```

> **Champion Model:** **Random Forest Ensemble** achieved the lowest test error ($14.60\,\mu\text{g/m}^3$) and highest correlation ($R^2 = 0.9544$), closely followed by **XGBoost** ($R^2 = 0.9543$). Both tree ensembles outperformed deep neural networks on tabular lag features while executing inference in under $5\,\text{ms}$.

### Multi-Horizon Error Degradation Curve:

As prediction lead time increases from +1 hour to +72 hours, error scales smoothly with atmospheric uncertainty:

| Forecast Lead Horizon | Random Forest MAE | XGBoost MAE | LSTM MAE | Persistence MAE | Error Growth Rate |
|---|---|---|---|---|---|
| **+1 Hour Ahead** | **4.12 µg/m³** | 4.15 µg/m³ | 4.88 µg/m³ | 6.82 µg/m³ | Baseline |
| **+6 Hours Ahead** | **7.84 µg/m³** | 7.86 µg/m³ | 8.92 µg/m³ | 11.45 µg/m³ | $+90\%$ |
| **+12 Hours Ahead** | **10.95 µg/m³** | 10.98 µg/m³ | 12.30 µg/m³ | 15.10 µg/m³ | $+165\%$ |
| **+24 Hours Ahead** | **14.60 µg/m³** | 14.60 µg/m³ | 17.20 µg/m³ | 19.13 µg/m³ | $+254\%$ |
| **+48 Hours Ahead** | **19.82 µg/m³** | 19.85 µg/m³ | 22.45 µg/m³ | 25.40 µg/m³ | $+381\%$ |
| **+72 Hours Ahead** | **24.31 µg/m³** | 24.35 µg/m³ | 27.10 µg/m³ | 31.85 µg/m³ | $+490\%$ |

---

## 9. CPCB Regulatory Mapping & Confusion Matrix

Continuous forecasts at the +24h horizon were mapped to statutory Central Pollution Control Board (CPCB) National Ambient Air Quality Standards (NAAQS) tiers:

- **Classification Accuracy:** **74.13%**
- **Weighted F1-Score:** **0.7271**
- **Macro-Averaged F1-Score:** **0.6277**
- **Balanced Accuracy:** **0.6342**

### Normalized Confusion Matrix (True vs. Predicted Tiers):

| True Regulatory Tier | Good (0–30) | Satisfactory (30–60) | Moderate (60–90) | Poor (90–120) | Very Poor (120–250) | Severe (250+) |
|---|---|---|---|---|---|---|
| **Good** | **0.78** | 0.18 | 0.04 | 0.00 | 0.00 | 0.00 |
| **Satisfactory** | 0.14 | **0.64** | 0.19 | 0.03 | 0.00 | 0.00 |
| **Moderate** | 0.02 | 0.15 | **0.69** | 0.12 | 0.02 | 0.00 |
| **Poor** | 0.00 | 0.04 | 0.16 | **0.71** | 0.09 | 0.00 |
| **Very Poor** | 0.00 | 0.00 | 0.02 | 0.11 | **0.84** | 0.03 |
| **Severe** | 0.00 | 0.00 | 0.00 | 0.01 | 0.20 | **0.79** |

> **Safety Assurance:** Zero catastrophic errors (e.g., $0.00\%$ of Severe episodes misclassified as Good or Satisfactory).

---

## 10. Data Scaling Efficiency Experiments

To test data efficiency, models were evaluated across training dataset volume fractions from 15% to 100%:

| Training Data Fraction | Training Samples | Test MAE (+24h) | Test RMSE (+24h) | Test $R^2$ Score | Convergence Note |
|---|---|---|---|---|---|
| **15%** | 3,666 | 40.76 µg/m³ | 67.12 | 0.4923 | Underfitting; seasonal dynamics unlearned |
| **30%** | 7,333 | 17.19 µg/m³ | 25.51 | 0.9266 | Rapid convergence to core diurnal cycle |
| **45%** | 11,000 | 15.65 µg/m³ | 24.80 | 0.9421 | Winter inversion patterns captured |
| **60%** | 14,666 | 15.02 µg/m³ | 23.95 | 0.9498 | Minor incremental gains |
| **100%** | **24,444** | **14.60 µg/m³** | **23.15** | **0.9544** | **Optimal generalized champion model** |

---

## 11. System Architecture & REST API Endpoints

AeroSense AI operates as a decoupled microservices architecture with a FastAPI backend and a React 18 single-page application:

```
+─────────────────────────────────────────────────────────────+
│            React 18 / Vite / TailwindCSS Web UI             │
│   (Overview, Forecasting, Stations, Lab, Alerts, Reports)   │
+──────────────────────────────┬──────────────────────────────+
                               │ HTTP REST JSON (/api/v1)
                               ▼
+─────────────────────────────────────────────────────────────+
│                   FastAPI Backend Service                   │
│   (Health, Stations, Forecasts, Warnings, Metrics, EDA)     │
+──────────────────────────────┬──────────────────────────────+
                               │
               +───────────────┴───────────────+
               ▼                               ▼
+──────────────────────────────+ +────────────────────────────+
│   Trained Model Artifacts    │ │ Scalers & Feature Registry │
│  RF, XGB, LSTM, GRU, CNN-LSTM│ │ delhi_feature_scaler.pkl   │
+──────────────────────────────+ +────────────────────────────+
```

### Registered REST Endpoints (`/api/v1`):

| Endpoint Route | HTTP Method | Functionality | Response Output |
|---|---|---|---|
| `/api/v1/health` | `GET` | Service status, active model, device (CPU/CUDA) | `HealthResponse` |
| `/api/v1/stations` | `GET` | Returns Anand Vihar (Delhi) monitoring station metadata | `StationListResponse` |
| `/api/v1/stations/{station}/history` | `GET` | Recent chronological hourly sensor records | `List[HistoricalObservation]` |
| `/api/v1/forecasts` | `GET` / `POST` | Multi-horizon forecast (+1h to +72h) with CPCB tiers | `ForecastResponse` |
| `/api/v1/warnings/active` | `GET` | Real-time CPCB NAAQS alerts and GRAP stage actions | `ActiveWarningsResponse` |
| `/api/v1/metrics` | `GET` | Complete model evaluation records across horizons | `dict` |
| `/api/v1/metrics/leaderboard` | `GET` | +24h leaderboard sorted by test MAE | `dict` |
| `/api/v1/metrics/confusion` | `GET` | Normalized CPCB 6-tier confusion matrix | `dict` |
| `/api/v1/metrics/fractions` | `GET` | Data efficiency results (15% to 100%) | `dict` |
| `/api/v1/analytics/trends` | `GET` | Continuous PM2.5 hourly time series for charting | `List[dict]` |
| `/api/v1/analytics/eda` | `GET` | Summary statistics, quartiles, diurnal curves | `dict` |

---

## 12. Project File & Directory Guide

```
ML_air/
├── backend/
│   ├── artifacts/
│   │   ├── experiments/           # Benchmark metrics, confusion matrices, fractions JSON
│   │   ├── models/                # Trained production checkpoints (*.pt, *.pkl)
│   │   └── scalers/               # Scaler fitted exclusively on train set (StandardScaler)
│   ├── data/
│   │   ├── delhi_air_quality.csv  # Clean benchmark corpus (35,064 records, 2020–2023)
│   │   ├── dataset_metadata.json  # Comprehensive dataset specification & audit
│   │   └── README.md              # Dataset documentation & column definitions
│   ├── scripts/
│   │   ├── train_delhi_models.py  # Master model training pipeline
│   │   └── export_frontend_data.py# Telemetry exporter for web client
│   ├── src/
│   │   ├── api/                   # FastAPI REST backend, routing, and inference engine
│   │   │   ├── inference.py       # DelhiInferenceEngine with lazy loading and fallback
│   │   │   ├── main.py            # FastAPI entry point & CORS configuration
│   │   │   ├── models.py          # Pydantic schemas
│   │   │   └── routes/            # Modular endpoint handlers
│   │   ├── config/                # Delhi station registry & CPCB NAAQS standards
│   │   └── models/                # Random Forest, XGBoost, LSTM, CNN-LSTM class definitions
│   └── tests/                     # 33 automated integration & unit tests
├── docs/                          # Exactly 1 markdown file per ML lifecycle stage
│   ├── 01_DATA_COLLECTION_AND_INGESTION.md
│   ├── 02_FEATURE_ENGINEERING_AND_PREPROCESSING.md
│   ├── 03_MODEL_ARCHITECTURE_AND_TRAINING.md
│   ├── 04_MODEL_EVALUATION_AND_LEADERBOARD.md
│   ├── 05_DEPLOYMENT_API_AND_MONITORING.md
│   ├── BACKEND_STRUCTURE_AND_FILE_GUIDE.md
│   └── INSTRUCTOR_TECHNICAL_REPORT.md
├── frontend/                      # React 18 / Vite / TailwindCSS web application
│   └── src/
│       ├── pages/                 # Overview, Forecasting, Stations, Lab, Alerts, Reports
│       └── data/                  # Real data exports & benchmark curves
├── notebooks/
│   └── AeroSense_AI_Delhi_Model_Training.ipynb  # Standalone training & fine-tuning notebook
└── README.md                      # This comprehensive master project report
```

---

## 13. Complete Step-by-Step Setup & Reproduction Guide

Follow these exact instructions to set up, test, and run the entire AeroSense AI project locally from scratch.

### A. System Prerequisites
- **Operating System:** Windows 10/11, macOS, or Ubuntu Linux
- **Python:** Python 3.11.x installed and added to PATH (`python --version`)
- **Node.js & Package Manager:** Node.js 18+ and `pnpm` (or `npm`) installed
- **Git:** Git installed (`git --version`)

---

### B. Step 1: Clone the Repository & Open Project Root
```bash
# Clone your repository from GitHub
git clone <YOUR_GITHUB_REPOSITORY_URL>

# Enter the project directory
cd ML_air
```

---

### C. Step 2: Set Up Python Backend Environment

#### On Windows (PowerShell):
```powershell
# 1. Create a dedicated Python 3.11 virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install backend dependencies
pip install -r backend/requirements.txt
```

#### On Linux / macOS (Bash):
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install backend dependencies
pip install -r backend/requirements.txt
```

---

### D. Step 3: Run the Automated Test Suite (33 Passing Tests)
Before starting servers, verify the system integrity:
```bash
# Run all unit and integration tests
python -m pytest backend/tests -v
```
**Expected Output:** `33 passed in ~9.8s (100% green)`

---

### E. Step 4: Launch the FastAPI Backend Server
```bash
# Navigate to the backend folder
cd backend

# Launch the FastAPI service with Uvicorn on port 8000
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Backend API Base:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Schema Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### F. Step 5: Set Up and Launch the Frontend Web Dashboard
Open a **new terminal window** while leaving the backend terminal running:

```bash
# Navigate to the frontend directory
cd ML_air/frontend

# Install Node dependencies
pnpm install
# (or 'npm install' if not using pnpm)

# Optional: Run production build check to ensure zero TypeScript errors
pnpm run build

# Start the Vite local development server
pnpm exec vite --host 127.0.0.1 --port 5173
```
- **Live Web Dashboard:** [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

### G. Step 6: Interactive Model Training (Jupyter Notebook or Python Script)

#### Option 1: Run the Complete Training Pipeline via Terminal
To re-train all 11 models from scratch, generate scalers, and export evaluation metrics:
```bash
# From project root with .venv activated
python backend/scripts/train_delhi_models.py
```

#### Option 2: Run via Interactive Jupyter Notebook (Local or Google Colab)
Open the self-contained training and fine-tuning notebook:
```
notebooks/AeroSense_AI_Delhi_Model_Training.ipynb
```
Select a Python 3.11 / GPU kernel and click **Run All**. The notebook sequentially executes:
1. Environment setup and reproducibility configuration (`seed=42`).
2. Data loading and validation of `backend/data/delhi_air_quality.csv`.
3. 56-feature engineering with strictly backward-looking lags and rolling stats.
4. Chronological 70/15/15 train/val/test split with 72-hour purge gaps.
5. Training Random Forest (100 trees) and direct multi-horizon XGBoost.
6. Training PyTorch deep neural networks (LSTM, GRU, CNN-LSTM) with early stopping.
7. Generating the +24h benchmark leaderboard and multi-horizon error curves.
8. Evaluating CPCB 6-tier classification and rendering the normalized confusion matrix.
9. Running data fraction scaling experiments (15% to 100%).
10. Exporting production model checkpoints (`.pt`, `.pkl`) and scalers.

---

## 14. Verification Summary for GitHub Submission

- **Target Jurisdiction:** Delhi (Anand Vihar Station `DEL_01`)
- **Other Cities/Stations Mentioned:** None (100% focused on Anand Vihar, Delhi)
- **Automated Tests:** 33 / 33 Passing
- **Frontend Build:** Built cleanly in 3.19s with 0 errors
- **Standalone Training Notebook:** `notebooks/AeroSense_AI_Delhi_Model_Training.ipynb`
- **Documentation:** Exactly 1 file per ML lifecycle stage in `docs/`
