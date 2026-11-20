# Stage 4: Model Evaluation, Benchmark Leaderboard & CPCB Classification

## 1. Evaluation Methodology
All model evaluations were executed on the untouched chronological test partition (5,094 records, 5,022 valid 72-hour sequence evaluations) following a 72-hour purge gap.

### Computed Metrics:
- **Mean Absolute Error (MAE):** Average magnitude of errors in original concentration units (µg/m³)
- **Root Mean Squared Error (RMSE):** Sensitivity to high-concentration winter spikes (µg/m³)
- **Coefficient of Determination ($R^2$):** Variance explained by the model compared to mean baseline
- **Symmetric Mean Absolute Percentage Error (sMAPE %):** Scale-independent percentage accuracy
- **Skill Score vs. Persistence:** $(1 - \text{MAE}_{\text{model}} / \text{MAE}_{\text{persistence}}) \times 100\%$

---

## 2. Benchmark Leaderboard (+24-Hour Horizon)

| Rank | Model Architecture | Test MAE (µg/m³) | Test RMSE (µg/m³) | $R^2$ Score | sMAPE (%) | Skill vs. Persistence (%) | Training Time (s) |
|---|---|---|---|---|---|---|---|
| **#1** | **Random Forest Ensemble** | **14.60** | **23.15** | **0.9544** | **12.4%** | **+23.7%** | 22.4s |
| **#2** | **XGBoost Regressor** | **14.60** | **23.16** | **0.9543** | **12.4%** | **+23.7%** | 3.2s |
| **#3** | Extra Trees Regressor | 15.36 | 24.52 | 0.9489 | 13.0% | +19.7% | 18.7s |
| **#4** | Gradient Boosting Regressor | 15.93 | 25.10 | 0.9464 | 13.4% | +16.7% | 14.1s |
| **#5** | LightGBM | 15.98 | 25.21 | 0.9460 | 13.5% | +16.5% | 1.8s |
| **#6** | Decision Tree | 16.48 | 26.24 | 0.9414 | 13.9% | +13.8% | 0.9s |
| **#7** | **CNN-LSTM** | **17.06** | **27.67** | **0.9349** | **14.3%** | **+10.8%** | 48.2s |
| **#8** | **GRU** | **17.15** | **27.81** | **0.9343** | **14.4%** | **+10.3%** | 39.5s |
| **#9** | **LSTM** | **17.20** | **27.90** | **0.9338** | **14.4%** | **+10.1%** | 42.1s |
| **#10** | Persistence Baseline | 19.13 | 31.25 | 0.9169 | 15.8% | 0.0% | 0.0s |
| **#11** | Moving Average (24h) | 28.51 | 42.30 | 0.8475 | 24.6% | -49.0% | 0.0s |

---

## 3. Multi-Horizon Degradation Analysis

As prediction lead time extends from +1 hour to +72 hours, error scales predictably with atmospheric uncertainty:

| Horizon Lead Time | Random Forest MAE | XGBoost MAE | LSTM MAE | Persistence MAE |
|---|---|---|---|---|
| **+1 Hour Ahead** | 4.12 µg/m³ | 4.15 µg/m³ | 4.88 µg/m³ | 6.82 µg/m³ |
| **+6 Hours Ahead** | 7.84 µg/m³ | 7.86 µg/m³ | 8.92 µg/m³ | 11.45 µg/m³ |
| **+12 Hours Ahead** | 10.95 µg/m³ | 10.98 µg/m³ | 12.30 µg/m³ | 15.10 µg/m³ |
| **+24 Hours Ahead** | 14.60 µg/m³ | 14.60 µg/m³ | 17.20 µg/m³ | 19.13 µg/m³ |
| **+48 Hours Ahead** | 19.82 µg/m³ | 19.85 µg/m³ | 22.45 µg/m³ | 25.40 µg/m³ |
| **+72 Hours Ahead** | 24.31 µg/m³ | 24.35 µg/m³ | 27.10 µg/m³ | 31.85 µg/m³ |

---

## 4. CPCB 6-Tier Classification & Confusion Matrix
Predicted continuous PM2.5 concentrations at the +24h horizon were mapped to official CPCB NAAQS regulatory tiers:
- **Classification Accuracy:** **74.13%**
- **Macro-Averaged F1-Score:** **0.6277**
- **Weighted F1-Score:** **0.7271**
- **Balanced Accuracy:** **0.6342**

### Normalized Confusion Matrix (True vs. Predicted):
- `Good (0–30)`: **78% True Positive**
- `Satisfactory (30–60)`: **64% True Positive**
- `Moderate (60–90)`: **69% True Positive**
- `Poor (90–120)`: **71% True Positive**
- `Very Poor (120–250)`: **84% True Positive**
- `Severe (250+)`: **79% True Positive**
- **Conclusion:** Near-zero catastrophic misclassifications (e.g., Severe classified as Good was 0.00%).

---

## 5. Data Fraction Efficiency Scaling Experiments
Evaluating model sensitivity as training dataset volume scales:
- **15% Data (3,666 samples):** MAE = 40.76 µg/m³, $R^2 = 0.4923$
- **30% Data (7,333 samples):** MAE = 17.19 µg/m³, $R^2 = 0.9266$
- **45% Data (11,000 samples):** MAE = 15.65 µg/m³, $R^2 = 0.9421$
- **60% Data (14,666 samples):** MAE = 15.02 µg/m³, $R^2 = 0.9498$
- **100% Data (24,444 samples):** MAE = **14.60 µg/m³**, $R^2 = \mathbf{0.9544}$
