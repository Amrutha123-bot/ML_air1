# AeroSense AI V2 — Machine Learning Artifact Inventory

This document provides a comprehensive, verified audit of all Machine Learning model checkpoints, training archives, experiment outputs, and dataset files within AeroSense AI V2.

---

## 1. Colab Backup Archives Directory (`backend/colab_backups/`)

All files in `backend/colab_backups/` have been audited, cataloged, and verified for data integrity without modifying the original source archives.

| File Name | Size (Bytes) | Category | Description | Verification Status |
| :--- | :---: | :--- | :--- | :---: |
| `AeroSense_Final_Artifacts.zip` | 58,528,008 | Master Backup | Complete training dataset splits, scalers, station configs, and final models | Verified ✅ |
| `AeroSense_AI_V2_Final_ML_Results.zip` | 17,418,298 | ML Results | Optuna HPO trials, aligned test prediction CSVs, metrics, and models | Verified ✅ |
| `aerosense_colab_run_20261114_202839 (1).zip` | 15,255,247 | Run Archive | Checkpoints, Optuna history, and aligned predictions from Colab GPU run | Verified ✅ |
| `aerosense_aligned_baseline_results.csv` | 1,878 | Baseline Eval | Horizon-by-horizon comparison of ST-CNN-GRU vs Persistence on identical samples | Verified ✅ |
| `AeroSense_V2_GPU_Training.ipynb` | 102,007 | Notebook | Verified 30-cell multi-city GPU training and evaluation pipeline | Verified ✅ |
| `stcnn_gru_beijing (1).pt` | 1,181,337 | Checkpoint | Duplicate download of the Beijing ST-CNN-GRU checkpoint | Verified (Identical) ✅ |

### Archive Deduplication & Content Analysis
- **`stcnn_gru_beijing (1).pt` vs `backend/artifacts/models/stcnn_gru_beijing.pt`**:
  - Size: 1,181,337 bytes
  - SHA-256: `6561736efe1fd17a2b71d8e2879c94920fc6f6f73a5824f76fbe362cfdf46b1f`
  - Result: 100% binary match; safely retained in backup without overwriting production.
- **`AeroSense_AI_V2_Final_ML_Results.zip` vs `aerosense_colab_run_20261114_202839 (1).zip`**:
  - Both archives contain the identical Optuna history (`xgb_optuna_history.json`, 7,715 bytes) and aligned prediction CSVs for Beijing (30,035,979 bytes) and Delhi (16,000,701 bytes).
  - `AeroSense_AI_V2_Final_ML_Results.zip` contains additional per-horizon metrics files (`beijing_final_test_metrics.csv`, `delhi_final_test_metrics.csv`, `beijing_corrected_validation_metrics.csv`, `delhi_checkpoint_validation_metrics.csv`).

---

## 2. Integrated Experiment Directory (`backend/artifacts/experiments/colab_run_20261114_202839/`)

Extracted selectively from Colab run archives to provide local traceability without overwriting core models:

| Artifact Name | Size | Purpose / Content |
| :--- | :---: | :--- |
| `experiment_manifest.json` | 988 B | Run metadata: Tesla T4 GPU, PyTorch 2.14.1+cu130, XGBoost 3.4.1, dataset SHA256 hashes |
| `dataset_inventory.json` | 4,188 B | City split summary (rows, columns, stations, date ranges, and split fingerprints) |
| `baselines_report.json` | 224 B | Evaluated Persistence and Seasonal Naive (24h) baselines on identical test targets |
| `xgb_optuna_history.json` | 7,715 B | 15-trial Optuna hyperparameter optimization log for GPU-accelerated XGBoost |
| `beijing_final_test_metrics.csv` | 816 B | 6-horizon test metrics for Beijing ST-CNN-GRU vs Persistence (N=62,664) |
| `delhi_final_test_metrics.csv` | 821 B | 6-horizon test metrics for Delhi ST-CNN-GRU vs Persistence (N=31,080) |
| `beijing_corrected_validation_metrics.csv` | 1,097 B | Validation checkpoint metrics with valid target masking |
| `delhi_checkpoint_validation_metrics.csv` | 420 B | Delhi validation checkpoint metrics across all 6 horizons |
| `delhi_validation_model_vs_persistence.csv` | 822 B | Delhi validation comparison against Persistence baseline |
| `aligned_predictions_beijing.csv` | 30,035,979 B | Full test-set aligned predictions and ground truth across all 12 Beijing stations |
| `aligned_predictions_delhi.csv` | 16,000,701 B | Full test-set aligned predictions and ground truth across all 12 Delhi stations |

---

## 3. Production Model Artifacts (`backend/artifacts/models/`)

All checkpoints in `backend/artifacts/models/` have been verified for architecture consistency, weights integrity, and non-destructive preservation:

| Checkpoint Name | Size (Bytes) | Architecture / Model Type | Horizon Support | Key Hyperparameters / Config |
| :--- | :---: | :--- | :---: | :--- |
| `stcnn_gru_beijing.pt` | 1,181,337 | Spatiotemporal CNN + Spatial Attention + 2-layer GRU | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, cnn_channels=64, gru_hidden=128, n_stations=12 |
| `stcnn_gru_delhi.pt` | 1,181,345 | Spatiotemporal CNN + Spatial Attention + 2-layer GRU | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, cnn_channels=64, gru_hidden=128, n_stations=12 |
| `stcnn_gru_model.pt` | 1,182,788 | ST-CNN-GRU Baseline Checkpoint | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, cnn_channels=64, gru_hidden=128, n_stations=12 |
| `stcnn_gru_seed42.pt` | 1,112,599 | ST-CNN-GRU Seed 42 Replication | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, cnn_channels=64, gru_hidden=128, n_stations=12 |
| `xgboost_model.pkl` | 1,409,623 | MultiOutputRegressor(XGBRegressor) | 1h, 6h, 12h, 24h, 48h, 72h | n_estimators=100, max_depth=6, lr=0.05, subsample=0.8 |
| `random_forest_model.pkl` | 51,038,087 | MultiOutputRegressor(RandomForestRegressor) | 1h, 6h, 12h, 24h, 48h, 72h | n_estimators=100, max_depth=15, min_samples_split=5 |
| `lstm_model.pt` | 274,037 | 2-layer LSTM with multi-horizon linear heads | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, hidden_size=64, num_layers=2, dropout=0.2 |
| `lstm_seed42.pt` | 273,671 | LSTM Seed 42 Replication Checkpoint | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, hidden_size=64, num_layers=2 |
| `lstm_seed43.pt` | 273,671 | LSTM Seed 43 Replication Checkpoint | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, hidden_size=64, num_layers=2 |
| `lstm_seed44.pt` | 273,607 | LSTM Seed 44 Replication Checkpoint | 1h, 6h, 12h, 24h, 48h, 72h | input_size=57, hidden_size=64, num_layers=2 |
| `cnn_lstm_model.pt` | 1,246,869 | 1D Temporal CNN + 2-layer LSTM | 1h, 6h, 12h, 24h, 48h, 72h | conv_channels=64, lstm_hidden=128, kernel_size=3 |
| `cnn_lstm_seed42.pt` | 1,246,869 | CNN-LSTM Seed 42 Replication | 1h, 6h, 12h, 24h, 48h, 72h | conv_channels=64, lstm_hidden=128, kernel_size=3 |
| `cnn_lstm_seed43.pt` | 1,246,613 | CNN-LSTM Seed 43 Replication | 1h, 6h, 12h, 24h, 48h, 72h | conv_channels=64, lstm_hidden=128, kernel_size=3 |
| `cnn_lstm_seed44.pt` | 1,246,869 | CNN-LSTM Seed 44 Replication | 1h, 6h, 12h, 24h, 48h, 72h | conv_channels=64, lstm_hidden=128, kernel_size=3 |
| `ga_kelm_model.pkl` | 3,026,356 | GA-Optimized Kernel Extreme Learning Machine | 1h, 6h, 12h, 24h, 48h, 72h | RBF Kernel, C=4.4572, gamma=0.011427, 6,000 prototypes |
| `persistence_model.pkl` | 124 | Naive Persistence Baseline Serializer | 1h, 6h, 12h, 24h, 48h, 72h | Direct lag projection: y(t+h) = y(t) |

---

## 4. Cryptographic Checksum Reference (SHA-256)

```text
stcnn_gru_beijing.pt : 6561736efe1fd17a2b71d8e2879c94920fc6f6f73a5824f76fbe362cfdf46b1f
stcnn_gru_delhi.pt   : dea79aab12c545f09508fde7f8b280e7b64f3d1fc7e7ade963435a326c8bc820
```
