# AeroSense AI V2 — Google Colab Master Execution Guide

This guide describes how to run the entire AeroSense AI V2 training, transfer learning, and evaluation pipeline using the **consolidated single notebook** (`AeroSense_AI_Master_Colab_Pipeline.ipynb`).

---

## 1. Zero Google Drive Dependency

You **do not need** to mount Google Drive or resolve `drive.mount` credential issues. The pipeline runs completely inside Google Colab's fast local `/content/` environment and packages all trained models and results into a single download zip at the end.

---

## 2. 3-Step Execution Instructions

### Step 1: Open Google Colab & Select GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload:
   `c:\Users\anala\OneDrive\Desktop\ML_air\AeroSense_AI_Master_Colab_Pipeline.ipynb`
3. In Colab, select: **Runtime** -> **Change runtime type** -> **T4 GPU** (or CPU).

### Step 2: Upload the Pre-Packaged Data Bundle
1. In the left sidebar of Colab, click the **Folder icon** (`Files`).
2. Drag and drop this single zip file into the Files area:
   `c:\Users\anala\OneDrive\Desktop\ML_air\AeroSense_Colab_Bundle.zip` (33.86 MB)

### Step 3: Run the Pipeline
1. Click **Runtime** -> **Run all** (or press `Ctrl+F9`).
2. The notebook will automatically:
   - Extract the datasets and `aerosense_ml` library into `/content/`.
   - Run the 14-point scientific integrity audit.
   - Evaluate the Persistence baseline across all 6 horizons (+1h, +6h, +12h, +24h, +48h, +72h) for Delhi, Mumbai, Vijayawada.
   - Train and evaluate XGBoost & Random Forest tabular models.
   - Train and evaluate Sequential LSTM and 1D-CNN-LSTM deep neural networks.
   - Train/evaluate the Champion ST-CNN-GRU spatiotemporal graph model.
   - Run cross-city transfer learning (Delhi -> Mumbai & Vijayawada) across fine-tuning data fractions [15%, 30%, 45%, 60%, 100%].
   - Generate master comparative benchmark tables and lead-time degradation plots.
   - Automatically package everything and trigger a browser download for **`AeroSense_AI_Training_Results.zip`**!

---

## 3. Bringing Results Back to Local Repository

When the browser finishes downloading `AeroSense_AI_Training_Results.zip`:
1. Extract the zip into your local `ML_air` project:
   - Checkpoints (`.pt`, `.pkl`) go into `backend/artifacts/models/`
   - Metric summaries (`.csv`) go into `backend/artifacts/experiments/` or `results/`
   - Verification plots (`.png`) go into `results/plots/`
2. Run backend validation:
   ```powershell
   $env:PYTHONPATH=".;backend"
   .venv\Scripts\python -m pytest backend/tests -q
   ```
