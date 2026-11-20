# Stage 3: Model Architecture, Training & Hyperparameters

## 1. Overview & Training Environment
All models in AeroSense AI were trained and validated **locally** on Windows using Python 3.11 in the dedicated `.venv` virtual environment. PyTorch 2.2.2 with CUDA acceleration (with automatic CPU fallback) and Scikit-Learn 1.4.2 / XGBoost 2.0.3 were utilized.

- **Training Pipeline Script:** `backend/scripts/train_delhi_models.py`
- **Inference Runtime Engine:** `backend/src/api/inference.py` (`DelhiInferenceEngine`)
- **Trained Artifacts Directory:** `backend/artifacts/models/`
- **Target Horizons:** 6 multi-step horizons: +1h, +6h, +12h, +24h, +48h, +72h

---

## 2. Model Zoo & Parameter Counts

| Model Architecture | Model Family | Parameter Count / Complexity | Training Mechanism | Code Definition Location |
|---|---|---|---|---|
| **Random Forest** | Tree Ensemble | 100 Trees (No weight tensors) | `MultiOutputRegressor(RandomForestRegressor)` | `backend/src/models/random_forest.py` & `backend/src/api/inference.py` |
| **XGBoost** | Gradient Boosted Trees | 300 Boosting Trees ($\times 6$ horizons) | `xgb.XGBRegressor` per horizon | `backend/src/models/xgboost_model.py` & `backend/src/api/inference.py` |
| **LSTM** | Deep Recurrent Network | **240,070** trainable parameters | PyTorch Sequence-to-Sequence (24h lookback) | `backend/src/models/lstm.py` & `backend/src/api/inference.py` (`_LSTMNet`) |
| **GRU** | Gated Recurrent Unit | **182,214** trainable parameters | PyTorch Sequence-to-Sequence (24h lookback) | `backend/src/api/inference.py` (`_GRUNet`) |
| **CNN-LSTM** | Spatio-Temporal Hybrid | **310,278** trainable parameters | 1D Conv feature extraction + LSTM recurrent decoder | `backend/src/models/cnn_lstm.py` & `backend/src/api/inference.py` (`_CNNLSTMNet`) |
| **Persistence Baseline** | Naive Baseline | 0 parameters | $y_{t+h} = y_t$ | `backend/src/models/baseline.py` & `backend/src/api/inference.py` |
| **Seasonal Persistence** | 24-Hour Diurnal Baseline | 0 parameters | $y_{t+h} = y_{t+h-24}$ | `backend/src/models/baseline.py` |
| **Moving Average** | Smoothing Baseline | 0 parameters | $\bar{y}_{t-24:t}$ | `backend/src/models/baseline.py` |

---

## 3. Deep Learning Architecture Specifications & Layer Parameters

### A. LSTM Network (`_LSTMNet`) — 240,070 Parameters
- **Input Dimension:** 64 engineered features
- **Sequence Lookback Window:** 24 continuous hourly timesteps
- **Recurrent Hidden Dimension:** 128 units
- **Number of Recurrent Layers:** 2 stacked LSTM layers
- **Recurrent Dropout:** 0.20
- **Fully Connected Projection Head:**
  - Layer 1: Linear(128 $\rightarrow$ 64) with ReLU activation
  - Layer 2: Dropout(0.20)
  - Layer 3: Linear(64 $\rightarrow$ 6) multi-horizon forecast output

### B. GRU Network (`_GRUNet`) — 182,214 Parameters
- **Input Dimension:** 64 engineered features
- **Sequence Lookback Window:** 24 timesteps
- **Hidden Dimension:** 128 units (3 gates per cell instead of 4 in LSTM)
- **Number of Layers:** 2 stacked GRU layers
- **Recurrent Dropout:** 0.20
- **Projection Head:** Linear(128 $\rightarrow$ 64) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(64 $\rightarrow$ 6)

### C. CNN-LSTM Network (`_CNNLSTMNet`) — 310,278 Parameters
- **Input Dimension:** 64 features over 24 timesteps ($B \times 64 \times 24$)
- **Temporal Conv1D Layer 1:** 64 in-channels $\rightarrow$ 64 out-channels, kernel size = 3, padding = 1, BatchNorm1d, ReLU
- **Temporal Conv1D Layer 2:** 64 in-channels $\rightarrow$ 64 out-channels, kernel size = 3, padding = 1, BatchNorm1d, ReLU
- **LSTM Recurrent Backbone:** 64 input features $\rightarrow$ 128 hidden units, 2 layers, dropout = 0.20
- **Dense Head:** Linear(128 $\rightarrow$ 64) $\rightarrow$ ReLU $\rightarrow$ Linear(64 $\rightarrow$ 6)

---

## 4. Hyperparameters & Training Execution Details

| Hyperparameter | PyTorch Deep Learning (LSTM / GRU / CNN-LSTM) | Random Forest Regressor | XGBoost Regressor |
|---|---|---|---|
| **Batch Size** | **256** | N/A (Batch Matrix) | Full Batch / Histogram Subsamples |
| **Max Epochs** | **50** (with Early Stopping) | 100 Trees | 300 Boosting Rounds |
| **Early Stopping Patience** | **10 Epochs** on Validation Loss | N/A | Early stopping on validation RMSE |
| **Learning Rate** | **1e-3 ($0.001$)** | N/A | **0.05** |
| **Optimizer** | **Adam** ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | N/A | Gradient descent with second-order Taylor expansion |
| **Loss Function** | Mean Squared Error (MSELoss) | MSE Criterion | `reg:squarederror` |
| **Regularization** | Weight Decay = $10^{-5}$, Dropout = 0.20 | `min_samples_split=2`, `min_samples_leaf=1` | `subsample=0.8`, `colsample_bytree=0.8`, `max_depth=6` |
| **Random Seed** | 42 (Reproducible) | 42 | 42 |
| **Hardware Device** | CUDA GPU with CPU fallback | CPU Multi-core (`n_jobs=-1`) | CPU Histogram / GPU Hist (`tree_method='hist'`) |
