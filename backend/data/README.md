# AeroSense AI — Delhi Synthetic Air Quality Dataset

> **IMPORTANT DISCLOSURE:**
> This dataset is **COMPLETELY SYNTHETIC**. It was generated for AeroSense AI using atmospheric and meteorological simulation models incorporating physical diurnal cycles, seasonal winter inversions, planetary boundary layer dynamics, and official Central Pollution Control Board (CPCB) NAAQS sub-index formulas.
>
> **It is NOT historical sensor data from CPCB or any official government monitoring station.**

---

## Dataset Overview

- **Location:** Delhi (Station: `DEL_01` / Anand Vihar)
- **Time Span:** `2020-01-01 00:00:00` to `2023-12-31 23:00:00` (4 years)
- **Resolution:** 1-hour intervals
- **Total Records:** **35,064** unbroken hourly observations
- **Missing Values:** **0 (0.00%)**
- **Duplicates:** **0 (0.00%)**
- **File Format:** CSV (`delhi_synthetic_air_quality.csv`, ~4.60 MB)

---

## Variable Dictionary

| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `Timestamp` | datetime | YYYY-MM-DD HH:MM:SS | Hourly timestamp (Asia/Kolkata) |
| `City` | string | - | Always `Delhi` |
| `Station` | string | - | Primary monitoring code (`DEL_01`) |
| `Station_Name` | string | - | `Anand Vihar, Delhi` |
| `PM2.5` | float | µg/m³ | Fine particulate matter (≤ 2.5 µm) |
| `PM10` | float | µg/m³ | Inhalable particulate matter (≤ 10 µm) |
| `NO` | float | µg/m³ | Nitric Oxide |
| `NO2` | float | µg/m³ | Nitrogen Dioxide |
| `NOx` | float | ppb | Total Nitrogen Oxides |
| `NH3` | float | µg/m³ | Ammonia |
| `SO2` | float | µg/m³ | Sulfur Dioxide |
| `CO` | float | mg/m³ | Carbon Monoxide |
| `O3` | float | µg/m³ | Ground-level Ozone |
| `AQI` | float | - | CPCB NAAQS Air Quality Index (0 - 500) |
| `AQI_Category` | string | - | Regulatory category (Good to Severe) |
| `Temperature` | float | °C | Ambient 2m air temperature |
| `Humidity` | float | % | Relative humidity |
| `Wind_Speed` | float | m/s | 10m wind velocity |

---

## Statistical Summary

- **PM2.5 Mean:** 83.48 µg/m³ (Std: 84.1)
- **PM2.5 Range:** [5.0 – 515.83] µg/m³
- **AQI Mean:** 192.22 (Std: 119.77)
- **AQI Range:** [31.5 – 500.0]

### AQI Category Breakdown

| Category | Count | Percentage |
| :--- | :--- | :--- |
| Moderate | 16,161 | 46.1% |
| Satisfactory | 7,975 | 22.7% |
| Very Poor | 6,291 | 17.9% |
| Severe | 2,902 | 8.3% |
| Poor | 1,632 | 4.7% |
| Good | 103 | 0.3% |

---

## Atmospheric Simulation Details

1. **Winter Inversion Dynamics:** Elevated baseline emissions during Nov–Jan reflecting atmospheric boundary layer contraction, nocturnal radiation cooling, and biomass burning episodes.
2. **Diurnal Rush-Hour Signatures:** Dual-peak emission profile at 08:00–10:00 (morning traffic) and 19:00–22:00 (evening traffic + boundary layer collapse).
3. **Monsoon Wet Deposition:** Dramatic particulate reduction (washout) during July–August with elevated relative humidity.
4. **Summer Dust & Ozone:** Pre-monsoon coarse fraction elevation (PM10) and peak afternoon photochemical Ozone generation.
5. **Autoregressive Memory:** AR(1) autoregression ($0.82 persistence) preserving realistic multi-hour weather inertia.
