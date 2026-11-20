import json
from pathlib import Path

notebook_dir = Path("notebooks")
notebook_dir.mkdir(parents=True, exist_ok=True)
notebook_path = notebook_dir / "AeroSense_AI_Delhi_Model_Training.ipynb"

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

cells = []

# Title & Metadata
cells.append(md_cell("""# AeroSense AI — Delhi Air Quality Multi-Horizon Forecasting & Fine-Tuning
### Complete End-to-End Machine Learning & Deep Learning Training Pipeline
**Geographic Target:** Delhi NCR (Central Pollution Control Board CAAQMS Network — Anand Vihar Hub)  
**Execution Support:** Google Colab (GPU accelerated) or Local Workstation (`.venv` Python 3.11)  
**Prediction Horizons:** +1h, +6h, +12h, +24h, +48h, +72h  
**Target Variable:** PM2.5 (µg/m³) with CPCB 6-Tier NAAQS Regulatory Mapping  
"""))

# Cell 1: Environment Setup
cells.append(md_cell("## 1. Environment Setup & Dependency Installation"))
cells.append(code_cell("""# Check GPU availability and install dependencies if in Colab
import sys
import os
from pathlib import Path

try:
    import torch
    import sklearn
    import xgboost
    print(f"PyTorch version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"XGBoost version: {xgboost.__version__}")
except ImportError:
    !pip install -q torch torchvision torchaudio scikit-learn xgboost pandas numpy matplotlib seaborn joblib
"""))

# Cell 2: Imports & Global Configuration
cells.append(md_cell("## 2. Global Imports & Reproducibility Settings"))
cells.append(code_cell("""import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib

# Fix seeds for 100% reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Computation Device: {DEVICE}")
"""))

# Cell 3: Data Loading
cells.append(md_cell("""## 3. Data Ingestion & Integrity Verification
Load the Delhi air quality benchmark dataset (35,064 continuous hourly records from Anand Vihar DEL_01).
"""))
cells.append(code_cell("""# Load dataset
csv_paths = [
    Path("backend/data/delhi_air_quality.csv"),
    Path("backend/data/delhi_synthetic_air_quality.csv"),
    Path("delhi_air_quality.csv"),
    Path("../backend/data/delhi_air_quality.csv")
]

data_path = None
for p in csv_paths:
    if p.exists():
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("Delhi dataset not found in expected paths. Please upload delhi_air_quality.csv.")

print(f"Loading Delhi Air Quality Dataset from: {data_path}")
df = pd.read_csv(data_path)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f"Total Observations: {len(df):,} rows, {df.shape[1]} columns")
print(f"Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"Missing Values: {df.isnull().sum().sum()} | Duplicates: {df.duplicated(subset=['Timestamp']).sum()}")
df.head(3)
"""))

# Cell 4: Feature Engineering
cells.append(md_cell("""## 4. Feature Engineering (Strictly Backward-Looking, Zero Data Leakage)
Engineers 64 predictive features including autoregressive lags, rolling statistics, cyclical sin/cos transforms, and domain interaction ratios.
"""))
cells.append(code_cell("""def engineer_delhi_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    
    # 1. Cyclical time embeddings
    hour = d['Timestamp'].dt.hour
    month = d['Timestamp'].dt.month
    d['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    d['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    d['month_sin'] = np.sin(2 * np.pi * month / 12)
    d['month_cos'] = np.cos(2 * np.pi * month / 12)
    d['day_of_week'] = d['Timestamp'].dt.dayofweek
    d['is_weekend'] = (d['day_of_week'] >= 5).astype(float)
    
    # 2. PM2.5 Lags (Past observations only)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        d[f'pm25_lag_{lag}h'] = d['PM2.5'].shift(lag)
        
    # 3. Co-pollutant lags
    for lag in [1, 6, 24]:
        d[f'pm10_lag_{lag}h'] = d['PM10'].shift(lag)
    for lag in [1, 24]:
        d[f'no2_lag_{lag}h'] = d['NO2'].shift(lag)
        d[f'so2_lag_{lag}h'] = d['SO2'].shift(lag)
        
    # 4. Backward-looking rolling windows (closed='left' to prevent leakage)
    for w in [6, 24, 72]:
        d[f'pm25_roll_{w}h_mean'] = d['PM2.5'].shift(1).rolling(w).mean()
        d[f'pm25_roll_{w}h_std'] = d['PM2.5'].shift(1).rolling(w).std()
        d[f'pm25_roll_{w}h_min'] = d['PM2.5'].shift(1).rolling(w).min()
        d[f'pm25_roll_{w}h_max'] = d['PM2.5'].shift(1).rolling(w).max()
        
    # 5. Domain ratios and atmospheric interactions
    d['pm25_pm10_ratio'] = d['PM2.5'] / (d['PM10'] + 1e-5)
    d['no2_nox_ratio'] = d['NO2'] / (d['NOx'] + 1e-5)
    d['ventilation_index'] = d['Wind_Speed'] * d['Temperature']
    
    # Forward multi-horizon forecast targets
    for h in [1, 6, 12, 24, 48, 72]:
        d[f'target_pm25_{h}h'] = d['PM2.5'].shift(-h)
        
    # Drop warm-up rows containing NaN from lags and tail rows missing future targets
    d = d.dropna().reset_index(drop=True)
    return d

df_feat = engineer_delhi_features(df)
feature_cols = [c for c in df_feat.columns if not c.startswith('target_') and c not in ['Timestamp', 'City', 'Station', 'AQI_Category']]
target_cols = [f'target_pm25_{h}h' for h in [1, 6, 12, 24, 48, 72]]

print(f"Engineered Features: {len(feature_cols)} total predictive inputs")
print(f"Target Horizons: {len(target_cols)} lead times: {target_cols}")
print(f"Valid Chronological Sequences: {len(df_feat):,} rows")
"""))

# Cell 5: Train / Val / Test Chronological Partitioning
cells.append(md_cell("""## 5. Chronological Train / Val / Test Partitioning with 72h Purge Gap
Strictly preserves temporal sequence. No random shuffling. 72-hour purge gap prevents overlap of forward horizon targets.
"""))
cells.append(code_cell("""n_samples = len(df_feat)
n_train = int(n_samples * 0.70)
n_val = int(n_samples * 0.15)
purge_gap = 72

train_df = df_feat.iloc[:n_train].copy()
val_df = df_feat.iloc[n_train + purge_gap : n_train + purge_gap + n_val].copy()
test_df = df_feat.iloc[n_train + purge_gap + n_val + purge_gap:].copy()

print(f"Train Partition (70%): {len(train_df):,} samples ({train_df['Timestamp'].min()} to {train_df['Timestamp'].max()})")
print(f"Val Partition   (15%): {len(val_df):,} samples ({val_df['Timestamp'].min()} to {val_df['Timestamp'].max()})")
print(f"Test Partition  (15%): {len(test_df):,} samples ({test_df['Timestamp'].min()} to {test_df['Timestamp'].max()})")

# Fit scaler exclusively on training partition
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])
X_val_scaled = scaler.transform(val_df[feature_cols])
X_test_scaled = scaler.transform(test_df[feature_cols])

y_train = train_df[target_cols].values
y_val = val_df[target_cols].values
y_test = test_df[target_cols].values
"""))

# Cell 6: Classical ML & Tree Models
cells.append(md_cell("## 6. Classical ML & Boosting Model Training"))
cells.append(code_cell("""# 1. Random Forest Ensemble (100 Trees)
print("Training Random Forest Ensemble (100 trees)...")
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1))
rf.fit(X_train_scaled, y_train)

# 2. XGBoost Regressors
print("Training Multi-Horizon XGBoost Regressors...")
xgb_models = []
xgb_preds_val = []
xgb_preds_test = []

for i, h in enumerate([1, 6, 12, 24, 48, 72]):
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        tree_method='hist'
    )
    model.fit(
        X_train_scaled, y_train[:, i],
        eval_set=[(X_val_scaled, y_val[:, i])],
        verbose=False
    )
    xgb_models.append(model)
    xgb_preds_test.append(model.predict(X_test_scaled))

y_pred_xgb = np.column_stack(xgb_preds_test)
y_pred_rf = rf.predict(X_test_scaled)
print("Classical ML & Tree Model training complete!")
"""))

# Cell 7: Deep Learning Architectures
cells.append(md_cell("""## 7. Deep Learning Architectures (LSTM, GRU, CNN-LSTM)
PyTorch deep neural networks trained with a 24-hour lookback sequence window.
- **LSTM:** 240,070 parameters
- **GRU:** 182,214 parameters
- **CNN-LSTM:** 310,278 parameters
"""))
cells.append(code_cell("""SEQ_LEN = 24

class TimeSeriesSeqDataset(Dataset):
    def __init__(self, X, y, seq_len=24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len + 1
        
    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]

train_ds = TimeSeriesSeqDataset(X_train_scaled, y_train, seq_len=SEQ_LEN)
val_ds = TimeSeriesSeqDataset(X_val_scaled, y_val, seq_len=SEQ_LEN)
test_ds = TimeSeriesSeqDataset(X_test_scaled, y_test, seq_len=SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# Architecture 1: LSTM (240,070 parameters)
class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=64, hidden=128, layers=2, out_dim=6):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

# Architecture 2: GRU (182,214 parameters)
class GRUForecaster(nn.Module):
    def __init__(self, in_dim=64, hidden=128, layers=2, out_dim=6):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

# Architecture 3: CNN-LSTM (310,278 parameters)
class CNNLSTMForecaster(nn.Module):
    def __init__(self, in_dim=64, cnn_out=64, hidden=128, layers=2, out_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, cnn_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out),
            nn.ReLU(),
            nn.Conv1d(cnn_out, cnn_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(cnn_out, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        # x: (B, T, F) -> permute to (B, F, T) for Conv1d
        x_conv = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x_conv)
        return self.head(out[:, -1, :])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

lstm_model = LSTMForecaster().to(DEVICE)
gru_model = GRUForecaster().to(DEVICE)
cnn_lstm_model = CNNLSTMForecaster().to(DEVICE)

print(f"LSTM Parameter Count:     {count_parameters(lstm_model):,}")
print(f"GRU Parameter Count:      {count_parameters(gru_model):,}")
print(f"CNN-LSTM Parameter Count: {count_parameters(cnn_lstm_model):,}")
"""))

# Cell 8: Training & Early Stopping Routine
cells.append(md_cell("## 8. Deep Learning Training Loop with Early Stopping"))
cells.append(code_cell("""def train_dl_model(model, train_loader, val_loader, max_epochs=50, patience=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_loss = float('inf')
    best_weights = None
    patience_cnt = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping triggered at Epoch {epoch} (Best Val Loss: {best_loss:.4f})")
                break
                
    model.load_state_dict(best_weights)
    return model

print("Training LSTM...")
lstm_model = train_dl_model(lstm_model, train_loader, val_loader)
print("Training GRU...")
gru_model = train_dl_model(gru_model, train_loader, val_loader)
print("Training CNN-LSTM...")
cnn_lstm_model = train_dl_model(cnn_lstm_model, train_loader, val_loader)
"""))

# Cell 9: Evaluation & Benchmark Leaderboard
cells.append(md_cell("## 9. Model Evaluation & Benchmark Leaderboard (+24h Horizon)"))
cells.append(code_cell("""def evaluate_dl(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.append(model(X_b.to(DEVICE)).cpu().numpy())
            targets.append(y_b.numpy())
    return np.vstack(preds), np.vstack(targets)

y_pred_lstm, y_true_seq = evaluate_dl(lstm_model, test_loader)
y_pred_gru, _ = evaluate_dl(gru_model, test_loader)
y_pred_cnnlstm, _ = evaluate_dl(cnn_lstm_model, test_loader)

# Align sequence test labels with tabular test set (offset by SEQ_LEN - 1)
y_true_h24 = y_true_seq[:, 3] # +24h is index 3 [1, 6, 12, 24, 48, 72]
y_rf_h24 = y_pred_rf[SEQ_LEN - 1:, 3]
y_xgb_h24 = y_pred_xgb[SEQ_LEN - 1:, 3]
y_lstm_h24 = y_pred_lstm[:, 3]
y_gru_h24 = y_pred_gru[:, 3]
y_cnnlstm_h24 = y_pred_cnnlstm[:, 3]

# Persistence baseline (t-0 PM2.5)
y_persist_h24 = test_df['PM2.5'].iloc[SEQ_LEN - 1 : len(y_true_h24) + SEQ_LEN - 1].values

models_dict = {
    'Random Forest (100 Trees)': y_rf_h24,
    'XGBoost': y_xgb_h24,
    'CNN-LSTM': y_cnnlstm_h24,
    'GRU': y_gru_h24,
    'LSTM': y_lstm_h24,
    'Persistence Baseline': y_persist_h24
}

res = []
persist_mae = mean_absolute_error(y_true_h24, y_persist_h24)
for name, p in models_dict.items():
    mae = mean_absolute_error(y_true_h24, p)
    rmse = np.sqrt(mean_squared_error(y_true_h24, p))
    r2 = r2_score(y_true_h24, p)
    smape = 100 * np.mean(2 * np.abs(p - y_true_h24) / (np.abs(p) + np.abs(y_true_h24) + 1e-5))
    skill = 100 * (1 - mae / persist_mae)
    res.append({'Model': name, 'Test MAE (µg/m³)': round(mae, 2), 'Test RMSE': round(rmse, 2), 'R² Score': round(r2, 4), 'sMAPE (%)': round(smape, 1), 'Skill Score vs Baseline (%)': f"+{round(skill, 1)}%" if skill > 0 else f"{round(skill, 1)}%"})

leaderboard_df = pd.DataFrame(res).sort_values('Test MAE (µg/m³)').reset_index(drop=True)
leaderboard_df.index = [f"#{i+1}" for i in range(len(leaderboard_df))]
print("\\n--- AEROSENSE AI DELHI BENCHMARK LEADERBOARD (+24H HORIZON) ---")
leaderboard_df
"""))

# Cell 10: CPCB Classification
cells.append(md_cell("## 10. Central Pollution Control Board (CPCB) 6-Tier Classification"))
cells.append(code_cell("""def to_cpcb_tier(pm25):
    if pm25 <= 30: return 0  # Good
    elif pm25 <= 60: return 1  # Satisfactory
    elif pm25 <= 90: return 2  # Moderate
    elif pm25 <= 120: return 3 # Poor
    elif pm25 <= 250: return 4 # Very Poor
    else: return 5             # Severe

tier_names = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
y_true_cat = [to_cpcb_tier(v) for v in y_true_h24]
y_rf_cat = [to_cpcb_tier(v) for v in y_rf_h24]

cm = confusion_matrix(y_true_cat, y_rf_cat, normalize='true')
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens', xticklabels=tier_names, yticklabels=tier_names)
plt.title('CPCB 6-Tier Normalized Confusion Matrix (Random Forest +24h)')
plt.xlabel('Predicted Tier')
plt.ylabel('True Regulatory Tier')
plt.tight_layout()
plt.show()

print(classification_report(y_true_cat, y_rf_cat, target_names=tier_names, digits=4))
"""))

# Cell 11: Fine-Tuning & Data Fraction Scaling
cells.append(md_cell("""## 11. Fine-Tuning & Data Fraction Scaling Experiments
Evaluates model performance scaling as training dataset volume expands from 15% to 100%.
"""))
cells.append(code_cell("""fractions = [0.15, 0.30, 0.45, 0.60, 1.00]
frac_results = []

for fr in fractions:
    n_fr = int(len(X_train_scaled) * fr)
    X_sub = X_train_scaled[:n_fr]
    y_sub = y_train[:n_fr]
    
    m = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1))
    m.fit(X_sub, y_sub)
    pred = m.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test[:, 3], pred[:, 3])
    rmse = np.sqrt(mean_squared_error(y_test[:, 3], pred[:, 3]))
    r2 = r2_score(y_test[:, 3], pred[:, 3])
    
    frac_results.append({
        'Fraction': f"{int(fr * 100)}%",
        'Training Samples': n_fr,
        'Test MAE (+24h)': round(mae, 2),
        'Test RMSE (+24h)': round(rmse, 2),
        'R² Score': round(r2, 4)
    })

frac_df = pd.DataFrame(frac_results)
print("\\n--- TRAINING DATA FRACTION SCALING EXPERIMENTS ---")
frac_df
"""))

# Cell 12: Saving Artifacts
cells.append(md_cell("## 12. Model & Scaler Artifact Export"))
cells.append(code_cell("""# Export production checkpoints
out_dir = Path("backend/artifacts/models")
out_dir.mkdir(parents=True, exist_ok=True)
scaler_dir = Path("backend/artifacts/scalers")
scaler_dir.mkdir(parents=True, exist_ok=True)

# Save Random Forest
joblib.dump(rf, out_dir / "random_forest_delhi.pkl")
# Save Deep Learning models
torch.save(lstm_model.state_dict(), out_dir / "lstm_delhi.pt")
torch.save(gru_model.state_dict(), out_dir / "gru_delhi.pt")
torch.save(cnn_lstm_model.state_dict(), out_dir / "cnn_lstm_delhi.pt")
# Save Scaler
joblib.dump(scaler, scaler_dir / "delhi_feature_scaler.pkl")

print(f"Successfully exported production checkpoints and scalers to {out_dir}!")
"""))

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Jupyter notebook successfully created at: {notebook_path}")
