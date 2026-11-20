# Stage 2: Feature Engineering & Time-Series Preprocessing

## 1. Preprocessing Philosophy & Leakage Prevention
Air quality time-series forecasting requires strict chronological discipline to ensure models are evaluated realistically without data leakage:
- **No Future Leakage:** All rolling statistics, moving averages, and lag terms use backward-looking windows exclusively (`closed='left'` or historical indices `t-1, t-2, ...`).
- **Partition Independence:** MinMax and Standard scalers are fitted exclusively on the training partition ($t \le \text{Train End}$) and applied out-of-sample to validation and test partitions.
- **Purge Gap:** A 72-hour purge gap is inserted between train, validation, and test splits to prevent overlap of 72-hour forward forecast targets.

## 2. Engineered Feature Architecture (64 Total Features)
From the base 18 variables, 64 predictive features are constructed:

### A. Temporal & Cyclical Embeddings
- `hour_sin`, `hour_cos`: Continuous diurnal representations ($\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$)
- `month_sin`, `month_cos`: Seasonal cycle representations ($\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$)
- `day_of_week`: Weekly traffic and industrial activity cycles (0 = Monday, 6 = Sunday)
- `is_weekend`: Binary flag for commercial transport variations

### B. Autoregressive Lag Features
- **PM2.5 Lags:** $t-1\text{h}, t-2\text{h}, t-3\text{h}, t-6\text{h}, t-12\text{h}, t-24\text{h}, t-48\text{h}, t-72\text{h}$
- **PM10 Lags:** $t-1\text{h}, t-6\text{h}, t-24\text{h}$
- **NO2 & SO2 Lags:** $t-1\text{h}, t-24\text{h}$

### C. Backward-Looking Rolling Statistics
- **6-Hour Rolling Windows:** Mean, standard deviation, min, max
- **24-Hour Rolling Windows:** Mean, standard deviation, exponential moving average (EMA)
- **72-Hour Rolling Windows:** Long-term trajectory mean and standard deviation

### D. Atmospheric Interaction & Domain Ratios
- **Coarse-to-Fine Ratio:** $\text{PM2.5} / (\text{PM10} + \epsilon)$ (identifies combustion vs. mechanical dust)
- **Oxidation Ratio:** $\text{NO2} / (\text{NOx} + \epsilon)$
- **Ventilation Index Proxy:** $\text{Wind Speed} \times \text{Temperature}$

## 3. Chronological Dataset Partitions
- **Total Dataset Samples:** 35,064 records (Anand Vihar Hub, Delhi)
- **Train Partition (70%):** 24,444 samples (2020-01-01 00:00 to 2022-10-16 12:00)
- **Purge Gap:** 72 hours excluded
- **Validation Partition (15%):** 5,238 samples (2022-10-19 13:00 to 2023-05-25 18:00)
- **Purge Gap:** 72 hours excluded
- **Test Partition (15%):** 5,094 samples (2023-05-28 19:00 to 2023-12-31 23:00)
- **Test Evaluation Sequences:** 5,022 valid 72-hour prediction sequences
