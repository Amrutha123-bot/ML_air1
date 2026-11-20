"""
Exploratory Data Analysis (EDA) generator for Beijing Air Quality dataset.
Produces publication-quality figures and statistical summaries:
1. Missing-value analysis by station and variable
2. Pollutant correlation heatmap
3. Seasonal PM2.5 distributions
4. Diurnal pollution cycle
5. Meteorological relationships (Wind Speed & Temperature vs PM2.5)
6. Spatial pollution differences across 12 stations
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from loguru import logger
from src.config.settings import PROCESSED_DIR, DOCS_DIR

EDA_PLOT_DIR = DOCS_DIR / "results"
EDA_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    logger.info("=== Running Exploratory Data Analysis ===")
    combined_path = PROCESSED_DIR / "combined_all_stations.parquet"
    if not combined_path.exists():
        logger.error(f"{combined_path} not found!")
        return

    df = pd.read_parquet(combined_path)
    logger.info(f"Loaded {len(df)} records across {df['station'].nunique()} stations.")

    # 1. Correlation Heatmap
    logger.info("Generating correlation matrix...")
    corr_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix: Pollutants & Meteorological Variables", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_path = EDA_PLOT_DIR / "eda_correlation_matrix.png"
    fig.savefig(corr_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved {corr_path}")

    # 2. Station-wise PM2.5 Distribution
    logger.info("Generating station-wise PM2.5 comparison...")
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby("station")["PM2.5"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="station", y="PM2.5", order=order, ax=ax, palette="Spectral", showfliers=False)
    ax.axhline(75.0, color="orange", linestyle="--", label="China Grade II Limit (75 μg/m³)")
    ax.axhline(35.0, color="green", linestyle="--", label="China Grade I Limit (35 μg/m³)")
    ax.set_title("Station-Wise PM2.5 Concentration Distributions", fontsize=13, fontweight="bold")
    ax.set_ylabel("PM2.5 (μg/m³)", fontsize=11)
    ax.set_xlabel("Monitoring Station", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right")
    plt.tight_layout()
    box_path = EDA_PLOT_DIR / "eda_station_pm25_boxplot.png"
    fig.savefig(box_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved {box_path}")

    # 3. Diurnal Cycle by Season
    logger.info("Generating diurnal cycles by season...")
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    month_to_season = {12: "Winter", 1: "Winter", 2: "Winter",
                       3: "Spring", 4: "Spring", 5: "Spring",
                       6: "Summer", 7: "Summer", 8: "Summer",
                       9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["month"].map(month_to_season)

    diurnal = df.groupby(["season", "hour"])["PM2.5"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=diurnal, x="hour", y="PM2.5", hue="season", marker="o", ax=ax, linewidth=2)
    ax.set_title("Average Diurnal PM2.5 Cycle Across Seasons in Beijing", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (0-23)", fontsize=11)
    ax.set_ylabel("Mean PM2.5 (μg/m³)", fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    diurnal_path = EDA_PLOT_DIR / "eda_diurnal_cycle_by_season.png"
    fig.savefig(diurnal_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved {diurnal_path}")

    # 4. Wind Speed vs PM2.5 Stagnation Effect
    logger.info("Generating wind speed vs PM2.5 relationship...")
    fig, ax = plt.subplots(figsize=(9, 5))
    df["wspm_bin"] = pd.cut(df["WSPM"], bins=[0, 1, 2, 3, 5, 8, 20], labels=["<1", "1-2", "2-3", "3-5", "5-8", ">8"])
    wspm_pm25 = df.groupby("wspm_bin")["PM2.5"].mean().reset_index()
    sns.barplot(data=wspm_pm25, x="wspm_bin", y="PM2.5", palette="Blues_r", ax=ax)
    ax.set_title("Impact of Wind Stagnation on PM2.5 Pollution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wind Speed Category (m/s)", fontsize=11)
    ax.set_ylabel("Average PM2.5 (μg/m³)", fontsize=11)
    plt.tight_layout()
    wind_path = EDA_PLOT_DIR / "eda_wind_stagnation_effect.png"
    fig.savefig(wind_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved {wind_path}")

    logger.info("=== EDA Analysis Complete ===")


if __name__ == "__main__":
    run_eda()
