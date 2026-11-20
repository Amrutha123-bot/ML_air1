"""
AeroSense AI — Delhi-Only Model Training & Experimentation Pipeline
Trains and evaluates all regression and classification models on the synthetic Delhi dataset.
Horizons: +1h, +6h, +12h, +24h, +48h, +72h.
Includes: Persistence, Seasonal Persistence, Moving Average, Linear Regression,
Ridge Regression, Decision Tree, Random Forest, XGBoost, LSTM, GRU, CNN-LSTM.
Generates master comparison CSV, classification metrics, confusion matrices, scalers, and checkpoints.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Seeds & Paths
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "backend" / "data" / "delhi_synthetic_air_quality.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"

for d in [MODELS_DIR, EXPERIMENTS_DIR, SCALERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 6, 12, 24, 48, 72]
TARGET_COL = "PM2.5"

# CPCB Category Helper
def aqi_category_from_pm25(pm25_val: float) -> str:
    if pm25_val <= 30:
        return "Good"
    elif pm25_val <= 60:
        return "Satisfactory"
    elif pm25_val <= 90:
        return "Moderate"
    elif pm25_val <= 120:
        return "Poor"
    elif pm25_val <= 250:
        return "Very Poor"
    else:
        return "Severe"

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


# PyTorch Sequence Models
class PyTorchLSTM(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, out_horizons=6):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, out_horizons)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PyTorchGRU(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, out_horizons=6):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, out_horizons)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class PyTorchCNNLSTM(nn.Module):
    def __init__(self, in_features, conv_filters=32, hidden_dim=64, out_horizons=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(conv_filters, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_horizons)

    def forward(self, x):
        # x: (batch, seq_len, features) -> conv1d needs (batch, features, seq_len)
        x_conv = x.permute(0, 2, 1)
        c_out = self.conv(x_conv)
        # permute back: (batch, seq_len/2, conv_filters)
        c_out = c_out.permute(0, 2, 1)
        l_out, _ = self.lstm(c_out)
        return self.fc(l_out[:, -1, :])


def run_pipeline():
    print(f"Loading synthetic Delhi air quality dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    n_rows = len(df)
    print(f"Loaded {n_rows:,} rows. Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    # 1. Compute & Export EDA Summary
    print("Computing exploratory data analysis metrics...")
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "Temperature", "Humidity", "Wind_Speed"]
    corr_matrix = df[pollutants].corr().round(4).to_dict()

    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

    hourly_pm25 = df.groupby("Hour")["PM2.5"].mean().round(2).to_dict()
    monthly_pm25 = df.groupby("Month")["PM2.5"].mean().round(2).to_dict()
    cat_distribution = df["AQI_Category"].value_counts().to_dict()

    eda_summary = {
        "total_records": n_rows,
        "date_range": [str(df["Timestamp"].min()), str(df["Timestamp"].max())],
        "category_distribution": cat_distribution,
        "hourly_diurnal_pm25": hourly_pm25,
        "monthly_seasonal_pm25": monthly_pm25,
        "correlation_matrix": corr_matrix,
        "variables_summary": df[pollutants + ["AQI"]].describe().round(2).to_dict(),
    }

    with open(EXPERIMENTS_DIR / "delhi_eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"Saved EDA summary to: {EXPERIMENTS_DIR / 'delhi_eda_summary.json'}")

    # 2. Feature Engineering
    print("Engineering features (lags, rolling stats, cyclic calendar)...")
    feats_df = pd.DataFrame(index=df.index)
    feats_df["datetime"] = df["Timestamp"]
    feats_df["raw_pm25"] = df["PM2.5"]

    # Autoregressive Lags
    lag_steps = [1, 2, 3, 6, 12, 24, 48, 72]
    for p in ["PM2.5", "PM10", "NO2", "Temperature", "Wind_Speed"]:
        for lag in lag_steps:
            feats_df[f"{p}_lag_{lag}h"] = df[p].shift(lag)

    # Rolling Windows (past only, min_periods=1)
    for w in [6, 24, 72]:
        feats_df[f"PM2.5_roll_mean_{w}h"] = df["PM2.5"].rolling(window=w, min_periods=1).mean()
        feats_df[f"PM2.5_roll_std_{w}h"] = df["PM2.5"].rolling(window=w, min_periods=1).std().fillna(0)

    # Physical Interaction Ratios
    feats_df["pm25_to_pm10_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-4)
    feats_df["no2_to_nox_ratio"] = df["NO2"] / (df["NOx"] + 1e-4)
    feats_df["temp_wind_interaction"] = df["Temperature"] * df["Wind_Speed"]
    feats_df["humidity_wind_stagnation"] = df["Humidity"] / (df["Wind_Speed"] + 0.1)

    # Cyclic Time Embeddings
    doy = df["Timestamp"].dt.day_of_year.values
    hr = df["Timestamp"].dt.hour.values
    feats_df["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
    feats_df["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
    feats_df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats_df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats_df["dow_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.dayofweek.values / 7.0)
    feats_df["dow_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.dayofweek.values / 7.0)

    # Multi-Horizon Targets: y(t+h)
    for h in HORIZONS:
        feats_df[f"target_h{h}"] = df["PM2.5"].shift(-h)

    # Drop NaNs created by max lag (72) and max lead (72)
    max_lag = max(lag_steps)
    max_lead = max(HORIZONS)
    clean_df = feats_df.iloc[max_lag:-max_lead].copy().reset_index(drop=True)
    print(f"Clean features dataset: {clean_df.shape[0]:,} rows, {clean_df.shape[1]} columns")

    feature_cols = [c for c in clean_df.columns if c not in ["datetime", "raw_pm25"] and not c.startswith("target_h")]
    target_cols = [f"target_h{h}" for h in HORIZONS]

    with open(EXPERIMENTS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # 3. Chronological 70/15/15 Split with Purge Gap (72h)
    total_samples = len(clean_df)
    n_train = int(total_samples * 0.70)
    n_val = int(total_samples * 0.15)
    purge_gap = 72

    train_df = clean_df.iloc[:n_train].copy()
    val_df = clean_df.iloc[n_train + purge_gap : n_train + purge_gap + n_val].copy()
    test_df = clean_df.iloc[n_train + purge_gap + n_val + purge_gap :].copy()

    print(f"Train samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    print(f"Test samples: {len(test_df):,}")

    # Fit scaler on TRAIN ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    Y_train = train_df[target_cols].values

    X_val = scaler.transform(val_df[feature_cols].values)
    Y_val = val_df[target_cols].values

    X_test = scaler.transform(test_df[feature_cols].values)
    Y_test = test_df[target_cols].values

    joblib.dump(scaler, SCALERS_DIR / "delhi_feature_scaler.pkl")
    print(f"Scaler saved to: {SCALERS_DIR / 'delhi_feature_scaler.pkl'}")

    # Persistence Reference Baselines
    # Lag 1 identity: y(t+h) = y(t)
    persistence_preds = np.tile(test_df["raw_pm25"].values[:, None], (1, len(HORIZONS)))

    # Seasonal Persistence (24h lag): y(t+h) = y(t+h-24)
    seasonal_pers_preds = np.zeros_like(Y_test)
    for j, h in enumerate(HORIZONS):
        lag_24_idx = max(0, 24 - h)
        seasonal_pers_preds[:, j] = test_df[f"PM2.5_lag_24h"].values

    # Moving Average (24h roll mean): y(t+h) = roll_mean_24h(t)
    moving_avg_preds = np.tile(test_df["PM2.5_roll_mean_24h"].values[:, None], (1, len(HORIZONS)))

    # Compute baseline metrics
    baseline_maes = [mean_absolute_error(Y_test[:, j], persistence_preds[:, j]) for j in range(len(HORIZONS))]

    results_records = []

    def log_results(model_name: str, preds: np.ndarray, train_time: float, infer_time: float):
        for j, h in enumerate(HORIZONS):
            y_true = Y_test[:, j]
            y_pred = preds[:, j]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
            smape = np.mean(np.where(denom == 0, 0, np.abs(y_pred - y_true) / denom)) * 100.0
            base_mae = baseline_maes[j]
            skill = (base_mae - mae) / base_mae if base_mae > 0 else 0.0

            results_records.append({
                "city": "Delhi",
                "model": model_name,
                "horizon": f"+{h}h",
                "horizon_hours": h,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "smape": round(smape, 4),
                "skill_vs_persistence": round(skill, 4),
                "training_time_s": round(train_time, 2),
                "inference_time_s": round(infer_time, 4),
            })

    # Evaluate Baselines
    log_results("Persistence Baseline", persistence_preds, 0.0, 0.001)
    log_results("Seasonal Persistence (24h)", seasonal_pers_preds, 0.0, 0.001)
    log_results("Moving Average (24h)", moving_avg_preds, 0.0, 0.001)

    # 4. Train Classical Models
    # Linear Regression
    print("Training Linear Regression...")
    t0 = time.time()
    lr = MultiOutputRegressor(LinearRegression(), n_jobs=1)
    lr.fit(X_train, Y_train)
    t_train = time.time() - t0
    t0 = time.time()
    lr_preds = lr.predict(X_test)
    t_infer = time.time() - t0
    log_results("Linear Regression", lr_preds, t_train, t_infer)

    # Ridge Regression
    print("Training Ridge Regression...")
    t0 = time.time()
    ridge = MultiOutputRegressor(Ridge(alpha=10.0), n_jobs=1)
    ridge.fit(X_train, Y_train)
    t_train = time.time() - t0
    t0 = time.time()
    ridge_preds = ridge.predict(X_test)
    t_infer = time.time() - t0
    log_results("Ridge Regression", ridge_preds, t_train, t_infer)

    # Decision Tree
    print("Training Decision Tree...")
    t0 = time.time()
    dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=SEED)
    dt.fit(X_train, Y_train)
    t_train = time.time() - t0
    t0 = time.time()
    dt_preds = dt.predict(X_test)
    t_infer = time.time() - t0
    log_results("Decision Tree", dt_preds, t_train, t_infer)

    # Random Forest (Champion)
    print("Training Random Forest Ensemble (100 trees)...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, Y_train)
    t_train = time.time() - t0
    t0 = time.time()
    rf_preds = rf.predict(X_test)
    t_infer = time.time() - t0
    joblib.dump(rf, MODELS_DIR / "random_forest_delhi.pkl")
    log_results("Random Forest (Champion)", rf_preds, t_train, t_infer)

    # XGBoost
    if HAS_XGB:
        print("Training Multi-Output XGBoost...")
        t0 = time.time()
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=SEED, n_jobs=-1),
            n_jobs=1,
        )
        xgb_model.fit(X_train, Y_train)
        t_train = time.time() - t0
        t0 = time.time()
        xgb_preds = xgb_model.predict(X_test)
        t_infer = time.time() - t0
        joblib.dump(xgb_model, MODELS_DIR / "xgboost_delhi.pkl")
        log_results("XGBoost", xgb_preds, t_train, t_infer)

    # 5. Train Deep Learning Models (PyTorch)
    # Prepare 24-step sequences for sequence architectures
    print("Preparing sequence tensors for Deep Learning models...")
    seq_len = 24
    
    def make_sequences(X_arr, Y_arr):
        xs, ys = [], []
        for i in range(len(X_arr) - seq_len):
            xs.append(X_arr[i : i + seq_len])
            ys.append(Y_arr[i + seq_len])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_train_seq, Y_train_seq = make_sequences(X_train, Y_train)
    X_test_seq, Y_test_seq = make_sequences(X_test, Y_test)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(Y_train_seq)), batch_size=128, shuffle=True)
    test_tensor_x = torch.from_numpy(X_test_seq)

    n_features = X_train.shape[1]

    # Train LSTM
    print("Training PyTorch LSTM (10 epochs)...")
    lstm = PyTorchLSTM(in_features=n_features, hidden_dim=64, num_layers=2, out_horizons=6)
    opt = torch.optim.Adam(lstm.parameters(), lr=0.003)
    criterion = nn.HuberLoss()

    t0 = time.time()
    lstm.train()
    for ep in range(10):
        for bx, by in train_loader:
            opt.zero_grad()
            out = lstm(bx)
            loss = criterion(out, by)
            loss.backward()
            opt.step()
    t_train = time.time() - t0

    lstm.eval()
    t0 = time.time()
    with torch.no_grad():
        lstm_preds_seq = lstm(test_tensor_x).numpy()
    t_infer = time.time() - t0

    # Align sequence predictions with test set
    lstm_preds = np.zeros_like(Y_test)
    lstm_preds[:seq_len] = persistence_preds[:seq_len]
    lstm_preds[seq_len:] = lstm_preds_seq
    torch.save(lstm.state_dict(), MODELS_DIR / "lstm_delhi.pt")
    log_results("Standard LSTM", lstm_preds, t_train, t_infer)

    # Train GRU
    print("Training PyTorch GRU (10 epochs)...")
    gru = PyTorchGRU(in_features=n_features, hidden_dim=64, num_layers=2, out_horizons=6)
    opt_gru = torch.optim.Adam(gru.parameters(), lr=0.003)

    t0 = time.time()
    gru.train()
    for ep in range(10):
        for bx, by in train_loader:
            opt_gru.zero_grad()
            loss = criterion(gru(bx), by)
            loss.backward()
            opt_gru.step()
    t_train = time.time() - t0

    gru.eval()
    t0 = time.time()
    with torch.no_grad():
        gru_preds_seq = gru(test_tensor_x).numpy()
    t_infer = time.time() - t0

    gru_preds = np.zeros_like(Y_test)
    gru_preds[:seq_len] = persistence_preds[:seq_len]
    gru_preds[seq_len:] = gru_preds_seq
    torch.save(gru.state_dict(), MODELS_DIR / "gru_delhi.pt")
    log_results("Gated Recurrent Unit (GRU)", gru_preds, t_train, t_infer)

    # Train CNN-LSTM
    print("Training PyTorch CNN-LSTM Hybrid (10 epochs)...")
    cnn_lstm = PyTorchCNNLSTM(in_features=n_features, conv_filters=32, hidden_dim=64, out_horizons=6)
    opt_cnn = torch.optim.Adam(cnn_lstm.parameters(), lr=0.003)

    t0 = time.time()
    cnn_lstm.train()
    for ep in range(10):
        for bx, by in train_loader:
            opt_cnn.zero_grad()
            loss = criterion(cnn_lstm(bx), by)
            loss.backward()
            opt_cnn.step()
    t_train = time.time() - t0

    cnn_lstm.eval()
    t0 = time.time()
    with torch.no_grad():
        cnn_lstm_preds_seq = cnn_lstm(test_tensor_x).numpy()
    t_infer = time.time() - t0

    cnn_lstm_preds = np.zeros_like(Y_test)
    cnn_lstm_preds[:seq_len] = persistence_preds[:seq_len]
    cnn_lstm_preds[seq_len:] = cnn_lstm_preds_seq
    torch.save(cnn_lstm.state_dict(), MODELS_DIR / "cnn_lstm_delhi.pt")
    log_results("CNN-LSTM Hybrid", cnn_lstm_preds, t_train, t_infer)

    # 6. Training Fraction Experiment (15%, 30%, 45%, 60%, 100%)
    print("Running training data fraction experiments...")
    fractions = [0.15, 0.30, 0.45, 0.60, 1.00]
    frac_results = []

    for frac in fractions:
        n_frac = int(len(X_train) * frac)
        sub_x = X_train[:n_frac]
        sub_y = Y_train[:n_frac]
        sub_rf = RandomForestRegressor(n_estimators=40, max_depth=12, random_state=SEED, n_jobs=-1)
        sub_rf.fit(sub_x, sub_y)
        sub_preds = sub_rf.predict(X_test)

        mean_mae = mean_absolute_error(Y_test, sub_preds)
        mean_rmse = np.sqrt(mean_squared_error(Y_test, sub_preds))
        mean_r2 = r2_score(Y_test, sub_preds)

        frac_results.append({
            "fraction_pct": int(frac * 100),
            "training_samples": n_frac,
            "mean_test_mae": round(mean_mae, 4),
            "mean_test_rmse": round(mean_rmse, 4),
            "mean_test_r2": round(mean_r2, 4),
        })

    with open(EXPERIMENTS_DIR / "training_fraction_experiments.json", "w", encoding="utf-8") as f:
        json.dump(frac_results, f, indent=2)

    # 7. Save Master Model Comparison CSV
    res_df = pd.DataFrame(results_records)
    res_df.to_csv(EXPERIMENTS_DIR / "master_model_comparison.csv", index=False)
    print(f"Saved master model comparison to: {EXPERIMENTS_DIR / 'master_model_comparison.csv'}")

    # 8. CPCB 6-Tier Discretization & Confusion Matrix (Random Forest Champion, Horizon +24h)
    print("Computing CPCB NAAQS 6-tier confusion matrices and classification metrics...")
    h24_idx = HORIZONS.index(24)
    y_test_h24 = Y_test[:, h24_idx]
    y_pred_h24 = rf_preds[:, h24_idx]

    cat_true = [aqi_category_from_pm25(v) for v in y_test_h24]
    cat_pred = [aqi_category_from_pm25(v) for v in y_pred_h24]

    raw_cm = confusion_matrix(cat_true, cat_pred, labels=CATEGORIES)
    norm_cm = confusion_matrix(cat_true, cat_pred, labels=CATEGORIES, normalize="true")

    raw_cm_df = pd.DataFrame(raw_cm, index=CATEGORIES, columns=CATEGORIES)
    norm_cm_df = pd.DataFrame(np.round(norm_cm, 4), index=CATEGORIES, columns=CATEGORIES)

    raw_cm_df.to_csv(EXPERIMENTS_DIR / "confusion_matrix_raw.csv")
    norm_cm_df.to_csv(EXPERIMENTS_DIR / "confusion_matrix_normalized.csv")

    cls_metrics = {
        "model_evaluated": "Random Forest (Champion)",
        "horizon": "+24h",
        "sample_count": len(cat_true),
        "overall_accuracy": round(accuracy_score(cat_true, cat_pred), 4),
        "macro_precision": round(precision_score(cat_true, cat_pred, labels=CATEGORIES, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(cat_true, cat_pred, labels=CATEGORIES, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(cat_true, cat_pred, labels=CATEGORIES, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(cat_true, cat_pred, labels=CATEGORIES, average="weighted", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(cat_true, cat_pred), 4),
        "per_class": {},
    }

    for idx, cat in enumerate(CATEGORIES):
        cls_metrics["per_class"][cat] = {
            "support": int(raw_cm[idx].sum()),
            "true_positive": int(raw_cm[idx, idx]),
            "sensitivity_recall": round(float(norm_cm[idx, idx]), 4),
        }

    with open(EXPERIMENTS_DIR / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cls_metrics, f, indent=2)

    # 9. Model Leaderboard Summary
    print("\n================== AEROSENSE AI DELHI MODEL LEADERBOARD (+24h Horizon) ==================")
    lead_df = res_df[res_df["horizon"] == "+24h"].sort_values("mae").reset_index(drop=True)
    print(lead_df[["model", "mae", "rmse", "r2", "skill_vs_persistence", "training_time_s", "inference_time_s"]].to_string())

    print("\nTraining & evaluation pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
