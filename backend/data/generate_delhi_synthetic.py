"""
AeroSense AI — Synthetic Delhi Air Quality Dataset Generator
Generates ~35,000 hourly records of Delhi air quality data (2020-01-01 to 2023-12-31).
Uses documented physical, seasonal, meteorological, and diurnal atmospheric modeling
coupled with official CPCB NAAQS sub-index AQI computation.

DISCLOSURE: THIS DATASET IS ENTIRELY SYNTHETIC AND DESIGNED FOR ML BENCHMARKING.
IT DOES NOT REPRESENT DIRECT SENSOR MEASUREMENTS.
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Set fixed seed for 100% reproducibility
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "delhi_synthetic_air_quality.csv"
METADATA_PATH = OUTPUT_DIR / "dataset_metadata.json"
README_PATH = OUTPUT_DIR / "README.md"

# CPCB Sub-Index Breakpoints
# (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 600, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2000, 401, 500),
    ],
    "CO": [
        (0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500),
    ],
    "O3": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50),
        (200, 400, 51, 100),
        (400, 800, 101, 200),
        (800, 1200, 201, 300),
        (1200, 1800, 301, 400),
        (1800, 2400, 401, 500),
    ],
}


def calculate_sub_index(conc: float, pollutant: str) -> float:
    """Calculate CPCB sub-index for a single pollutant concentration."""
    if pollutant not in CPCB_BREAKPOINTS or np.isnan(conc) or conc < 0:
        return 0.0

    bps = CPCB_BREAKPOINTS[pollutant]
    if conc > bps[-1][1]:
        # Cap or extrapolate linearly
        return 500.0

    for c_low, c_high, i_low, i_high in bps:
        if c_low <= conc <= c_high:
            sub = i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
            return round(sub, 1)

    return 0.0


def aqi_to_category(aqi: float) -> str:
    """Return regulatory category for AQI value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def generate_dataset():
    print("Generating synthetic Delhi hourly air quality dataset...")

    # Date range: 2020-01-01 00:00:00 to 2023-12-31 23:00:00
    dates = pd.date_range("2020-01-01 00:00:00", "2023-12-31 23:00:00", freq="h")
    n = len(dates)
    print(f"Total timestamps generated: {n}")

    # Extract time components
    day_of_year = dates.dayofyear.values
    hour = dates.hour.values
    month = dates.month.values
    dayofweek = dates.dayofweek.values

    # 1. Meteorology Simulation
    # Temperature: seasonal wave (cool winter ~12°C, hot summer ~36°C) + diurnal wave
    temp_seasonal = 25.0 - 13.0 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)
    temp_diurnal = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    temperature = temp_seasonal + temp_diurnal + np.random.normal(0, 1.8, n)
    temperature = np.clip(temperature, 4.0, 48.0)

    # Relative Humidity: high in monsoon (Jul-Aug ~78%), low in May (~30%), winter fog (~80%)
    humidity_base = 55.0 + 20.0 * np.sin(2 * np.pi * (day_of_year - 180) / 365.25)
    # Add winter fog boost in Nov-Jan
    winter_mask = (month >= 11) | (month <= 1)
    humidity_base[winter_mask] += 12.0
    # Diurnal humidity: peaks early morning, dips in afternoon
    humidity_diurnal = -12.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    humidity = humidity_base + humidity_diurnal + np.random.normal(0, 4.5, n)
    humidity = np.clip(humidity, 12.0, 98.0)

    # Wind Speed (m/s): low in winter inversion (~1.2 m/s), higher in pre-monsoon & monsoon (~3.5 m/s)
    wind_base = 2.4 + 1.1 * np.sin(2 * np.pi * (day_of_year - 120) / 365.25)
    wind_diurnal = 0.6 * np.sin(2 * np.pi * (hour - 8) / 24.0)
    wind_speed = wind_base + wind_diurnal + np.random.exponential(0.6, n)
    wind_speed = np.clip(wind_speed, 0.2, 12.0)

    # 2. Atmospheric Dispersion Factor (Inversion / Ventilation Index)
    # High in winter, low wind speed, nocturnal
    dispersion_factor = (wind_speed / 3.0) * (temperature / 25.0)
    stagnation = np.clip(1.0 / (dispersion_factor + 0.3), 0.3, 3.2)

    # 3. Diurnal Emission Cycles
    # Double rush hour peak (8-10 AM, 7-10 PM)
    morning_peak = np.exp(-((hour - 8.5) ** 2) / (2 * 1.5**2))
    evening_peak = np.exp(-((hour - 20.0) ** 2) / (2 * 2.0**2))
    diurnal_emission = 0.5 + 0.9 * morning_peak + 1.2 * evening_peak
    # Weekend reduction (~15% on Sunday)
    weekend_factor = np.where(dayofweek == 6, 0.85, np.where(dayofweek == 5, 0.92, 1.0))
    diurnal_emission *= weekend_factor

    # 4. Seasonal Base for PM2.5
    # Winter inversion & crop residue burning (Oct 25 - Dec 15)
    doy = day_of_year
    post_monsoon_spike = np.exp(-((doy - 315) ** 2) / (2 * 18.0**2)) * 140.0
    winter_base = np.where(winter_mask, 140.0, 50.0)
    monsoon_mask = (month >= 7) & (month <= 9)
    monsoon_washout = np.where(monsoon_mask, 0.45, 1.0)

    pm25_raw = (winter_base + post_monsoon_spike) * stagnation * diurnal_emission * monsoon_washout

    # Add temporal autocorrelation (AR(1) process for physical persistence)
    pm25 = np.zeros(n)
    pm25[0] = max(10.0, pm25_raw[0])
    for t in range(1, n):
        # 80% autoregressive persistence + 20% new physical forcing + random shock
        noise = np.random.normal(0, 10.0)
        pm25[t] = 0.82 * pm25[t - 1] + 0.18 * pm25_raw[t] + noise
        if pm25[t] < 5.0:
            pm25[t] = 5.0 + np.random.uniform(0, 5)

    # Episodic spike injections (severe dust storms in May/June or intense smog in November)
    spike_indices = np.random.choice(n, size=85, replace=False)
    for idx in spike_indices:
        span = np.random.randint(4, 18)
        spike_mag = np.random.uniform(80, 220)
        end_idx = min(n, idx + span)
        pm25[idx:end_idx] += spike_mag * np.hanning(end_idx - idx)

    pm25 = np.clip(pm25, 5.0, 580.0)

    # 5. Coupled Pollutant Synthesis
    # PM10: PM2.5 + coarse fraction (coarse elevated in summer dry/dust conditions)
    summer_dust = np.where((month >= 4) & (month <= 6), 90.0, 35.0)
    pm10 = pm25 * np.random.uniform(1.4, 1.9, n) + summer_dust * (wind_speed / 2.5) + np.random.normal(0, 12, n)
    pm10 = np.clip(pm10, pm25 + 5.0, 750.0)

    # NO, NO2, NOx: Traffic-dominated, diurnal rush hours
    no2 = (pm25 * 0.28 + morning_peak * 25.0 + evening_peak * 35.0) * stagnation + np.random.normal(0, 6, n)
    no2 = np.clip(no2, 5.0, 220.0)

    no = no2 * np.random.uniform(0.3, 0.7, n) + morning_peak * 20.0 + np.random.normal(0, 4, n)
    no = np.clip(no, 2.0, 180.0)

    nox = no + no2

    # NH3: Agricultural & waste emissions
    nh3 = 25.0 + 0.15 * pm25 + np.random.normal(0, 5, n)
    nh3 = np.clip(nh3, 4.0, 180.0)

    # SO2: Background industrial + winter elevation
    so2 = 12.0 + 0.08 * pm25 + np.random.normal(0, 3.5, n)
    so2 = np.clip(so2, 2.0, 95.0)

    # CO: Combustion/exhaust (mg/m^3)
    co = 0.5 + 0.009 * pm25 + (morning_peak + evening_peak) * 0.6 + np.random.normal(0, 0.15, n)
    co = np.clip(co, 0.1, 8.5)

    # O3: Photochemical secondary pollutant (high in hot sunny summer afternoons, suppressed at night by NO)
    o3_photochem = 20.0 + 1.2 * temperature + 28.0 * np.exp(-((hour - 14.5) ** 2) / (2 * 2.5**2))
    o3_titration = np.clip(no * 0.25, 0, 30)
    o3 = o3_photochem - o3_titration + np.random.normal(0, 6, n)
    o3 = np.clip(o3, 3.0, 240.0)

    # 6. Compute Official CPCB Sub-Indices and Combined AQI
    print("Computing CPCB NAAQS sub-indices and AQI...")
    aqi_values = np.zeros(n)
    categories = []

    for i in range(n):
        sub_pm25 = calculate_sub_index(pm25[i], "PM2.5")
        sub_pm10 = calculate_sub_index(pm10[i], "PM10")
        sub_no2 = calculate_sub_index(no2[i], "NO2")
        sub_so2 = calculate_sub_index(so2[i], "SO2")
        sub_co = calculate_sub_index(co[i], "CO")
        sub_o3 = calculate_sub_index(o3[i], "O3")
        sub_nh3 = calculate_sub_index(nh3[i], "NH3")

        # Master AQI is max of sub-indices
        overall_aqi = max(sub_pm25, sub_pm10, sub_no2, sub_so2, sub_co, sub_o3, sub_nh3)
        overall_aqi = round(overall_aqi, 1)
        aqi_values[i] = overall_aqi
        categories.append(aqi_to_category(overall_aqi))

    # Assemble DataFrame
    df = pd.DataFrame({
        "Timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "City": "Delhi",
        "Station": "DEL_01",
        "Station_Name": "Anand Vihar, Delhi",
        "PM2.5": np.round(pm25, 2),
        "PM10": np.round(pm10, 2),
        "NO": np.round(no, 2),
        "NO2": np.round(no2, 2),
        "NOx": np.round(nox, 2),
        "NH3": np.round(nh3, 2),
        "SO2": np.round(so2, 2),
        "CO": np.round(co, 2),
        "O3": np.round(o3, 2),
        "AQI": np.round(aqi_values, 1),
        "AQI_Category": categories,
        "Temperature": np.round(temperature, 1),
        "Humidity": np.round(humidity, 1),
        "Wind_Speed": np.round(wind_speed, 2),
    })

    # Save CSV
    df.to_csv(CSV_PATH, index=False)
    file_size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
    print(f"Dataset saved successfully: {CSV_PATH} ({file_size_mb:.2f} MB)")

    # Category counts
    cat_dist = df["AQI_Category"].value_counts().to_dict()

    # 7. Generate Dataset Metadata JSON
    metadata = {
        "dataset_name": "AeroSense AI Delhi Synthetic Air Quality Dataset",
        "dataset_type": "SYNTHETIC_SIMULATION",
        "disclosure": "THIS DATASET IS ENTIRELY SYNTHETIC AND WAS GENERATED MATHEMATICALLY FOR MACHINE LEARNING TRAINING AND FORECASTING BENCHMARKS. IT DOES NOT REPRESENT DIRECT SENSOR MEASUREMENTS FROM CPCB OR GOVERNMENT MONITORING HARNESSES.",
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "random_seed": SEED,
        "location": {
            "city": "Delhi",
            "country": "India",
            "station_id": "DEL_01",
            "station_name": "Anand Vihar, Delhi",
            "latitude": 28.6468,
            "longitude": 77.3160,
        },
        "time_coverage": {
            "start": "2020-01-01 00:00:00",
            "end": "2023-12-31 23:00:00",
            "sampling_interval": "1 hour",
            "total_records": len(df),
            "missing_values": 0,
            "duplicate_records": 0,
        },
        "variables": {
            "particulate": ["PM2.5 (µg/m³)", "PM10 (µg/m³)"],
            "gaseous": ["NO (µg/m³)", "NO2 (µg/m³)", "NOx (ppb)", "NH3 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)", "O3 (µg/m³)"],
            "regulatory": ["AQI (unitless)", "AQI_Category (6 tiers)"],
            "meteorological": ["Temperature (°C)", "Humidity (%)", "Wind_Speed (m/s)"],
        },
        "statistics": {
            "pm25": {
                "mean": float(np.round(df["PM2.5"].mean(), 2)),
                "std": float(np.round(df["PM2.5"].std(), 2)),
                "min": float(df["PM2.5"].min()),
                "p25": float(np.round(df["PM2.5"].quantile(0.25), 2)),
                "median": float(np.round(df["PM2.5"].median(), 2)),
                "p75": float(np.round(df["PM2.5"].quantile(0.75), 2)),
                "max": float(df["PM2.5"].max()),
            },
            "aqi": {
                "mean": float(np.round(df["AQI"].mean(), 2)),
                "std": float(np.round(df["AQI"].std(), 2)),
                "min": float(df["AQI"].min()),
                "median": float(np.round(df["AQI"].median(), 2)),
                "max": float(df["AQI"].max()),
            },
            "category_distribution": cat_dist,
        },
        "cpcb_methodology": "Indian National Air Quality Index (NAAQS) piecewise linear sub-index calculation",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {METADATA_PATH}")

    # Format category table in markdown
    cat_rows = "\n".join([f"| {cat} | {count:,} | {count/len(df)*100:.1f}% |" for cat, count in cat_dist.items()])
    cat_table = f"| Category | Count | Percentage |\n| :--- | :--- | :--- |\n{cat_rows}"

    # 8. Generate README.md
    readme_content = f"""# AeroSense AI — Delhi Synthetic Air Quality Dataset

> **IMPORTANT DISCLOSURE:**
> This dataset is **COMPLETELY SYNTHETIC**. It was generated for AeroSense AI using atmospheric and meteorological simulation models incorporating physical diurnal cycles, seasonal winter inversions, planetary boundary layer dynamics, and official Central Pollution Control Board (CPCB) NAAQS sub-index formulas.
>
> **It is NOT historical sensor data from CPCB or any official government monitoring station.**

---

## Dataset Overview

- **Location:** Delhi (Station: `DEL_01` / Anand Vihar)
- **Time Span:** `2020-01-01 00:00:00` to `2023-12-31 23:00:00` (4 years)
- **Resolution:** 1-hour intervals
- **Total Records:** **{len(df):,}** unbroken hourly observations
- **Missing Values:** **0 (0.00%)**
- **Duplicates:** **0 (0.00%)**
- **File Format:** CSV (`delhi_synthetic_air_quality.csv`, ~{file_size_mb:.2f} MB)

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

- **PM2.5 Mean:** {metadata['statistics']['pm25']['mean']} µg/m³ (Std: {metadata['statistics']['pm25']['std']})
- **PM2.5 Range:** [{metadata['statistics']['pm25']['min']} – {metadata['statistics']['pm25']['max']}] µg/m³
- **AQI Mean:** {metadata['statistics']['aqi']['mean']} (Std: {metadata['statistics']['aqi']['std']})
- **AQI Range:** [{metadata['statistics']['aqi']['min']} – {metadata['statistics']['aqi']['max']}]

### AQI Category Breakdown

{cat_table}

---

## Atmospheric Simulation Details

1. **Winter Inversion Dynamics:** Elevated baseline emissions during Nov–Jan reflecting atmospheric boundary layer contraction, nocturnal radiation cooling, and biomass burning episodes.
2. **Diurnal Rush-Hour Signatures:** Dual-peak emission profile at 08:00–10:00 (morning traffic) and 19:00–22:00 (evening traffic + boundary layer collapse).
3. **Monsoon Wet Deposition:** Dramatic particulate reduction (washout) during July–August with elevated relative humidity.
4. **Summer Dust & Ozone:** Pre-monsoon coarse fraction elevation (PM10) and peak afternoon photochemical Ozone generation.
5. **Autoregressive Memory:** AR(1) autoregression ($0.82 persistence) preserving realistic multi-hour weather inertia.
"""

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Documentation saved: {README_PATH}")


if __name__ == "__main__":
    generate_dataset()
