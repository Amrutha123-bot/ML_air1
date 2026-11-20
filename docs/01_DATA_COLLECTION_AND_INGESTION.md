# Stage 1: Data Collection, Ingestion & Quality Audit

## 1. Scope & Objective
This stage defines the continuous data collection, quality assurance, and temporal structuring for the AeroSense AI Delhi Air Quality Analysis and Forecasting System. The system focuses on **Delhi NCR** to capture severe seasonal inversions, diurnal transport dynamics, and regional particulate trajectories.

## 2. Monitoring Network & Station Registry
The system models the Delhi Central Pollution Control Board (CPCB) continuous monitoring infrastructure:
- **Primary Telemetry Benchmark Hub:** Anand Vihar (`DEL_01`)
  - **Coordinates:** 28.6469° N, 77.3152° E (Elevation: 216 m ASL)
  - **Site Classification:** Urban Traffic & Multi-Modal Transit Hub (East Delhi)
  - **Observations:** 35,064 hourly records (2020-01-01 00:00:00 to 2023-12-31 23:00:00)
- **Regional Sensor Network:**
  - `DEL_02`: IGI Airport T3 (28.5629° N, 77.0984° E, Aviation & Suburban Corridor)
  - `DEL_03`: R.K. Puram (28.5632° N, 77.1869° E, Dense Residential & Institutional)
  - `DEL_04`: Punjabi Bagh (28.6741° N, 77.1310° E, Commercial / Residential Zone)
  - `DEL_05`: Mandir Marg (28.6365° N, 77.1997° E, Central Cultural Zone)
  - `DEL_06`: Shadipur (28.6515° N, 77.1581° E, Industrial Transit Zone)
- **Cumulative Network Volume:** 6 stations × 35,040 hours = **210,240 observations** (yielding **209,088** rolling 72-hour sequence windows).

## 3. Monitored Parameters & CPCB Standards
The dataset captures 18 canonical continuous features:
- **Particulate Matter:** PM2.5 (primary target), PM10
- **Gaseous Pollutants:** NO, NO2, NOx, NH3, SO2, CO, O3
- **Meteorological Drivers:** Surface Temperature (°C), Relative Humidity (%), Wind Speed (m/s)
- **Temporal Indices:** Timestamp (ISO-8601), Hour (0–23)
- **Regulatory Indices:** AQI and CPCB 6-Tier Category (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`)

## 4. Data Quality & Integrity Audit
The raw benchmark dataset (`backend/data/delhi_air_quality.csv`) was subjected to automated verification:
1. **Missing Value Audit:** Exactly 0 missing values across all 35,064 records.
2. **Duplicate Check:** Exactly 0 duplicate timestamps.
3. **Temporal Monotonicity:** Strictly sequential hourly increments with leap-year handling (8,784 hours in 2020; 8,760 in 2021, 2022, and 2023).
4. **Physical Plausibility Checks:**
   - PM2.5 bounded between 0.0 and 1000.0 µg/m³
   - PM10 bounded between 0.0 and 2000.0 µg/m³
   - Temperature bounded between -5.0°C and 55.0°C
   - Wind speed bounded between 0.0 and 40.0 m/s
