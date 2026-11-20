#!/usr/bin/env python
"""
City-Level Feature Engineering Pipeline for AeroSense AI V2.

Transforms canonical lakehouse observations (data/canonical/{city}/observations.parquet)
into ML-ready feature matrices with chronological 70/15/15 train/val/test splits,
lag features, temporal cyclic features, rolling statistics, and leakage-free scaling.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

CANONICAL_DIR = ROOT_DIR / "data" / "canonical"
FEATURES_BASE_DIR = ROOT_DIR / "data" / "features"

TARGET = "PM2.5"
FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [3, 6, 12, 24]


def parse_args():
    parser = argparse.ArgumentParser(description="Build city-level feature matrices")
    parser.add_argument(
        "--city",
        type=str,
        default="delhi",
        choices=["beijing", "delhi", "mumbai", "vijayawada", "all"],
        help="Target city to process",
    )
    return parser.parse_args()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"]

    df["hour"] = dt.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    dow = dt.dt.day_of_week if hasattr(dt.dt, "day_of_week") else dt.dt.dayofweek
    df["dayofweek"] = dow
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month"] = dt.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    month_to_season = {
        12: 3, 1: 3, 2: 3,
        3: 0, 4: 0, 5: 0,
        6: 1, 7: 1, 8: 1,
        9: 2, 10: 2, 11: 2,
    }
    df["season"] = df["month"].map(month_to_season).fillna(0).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    doy = dt.dt.day_of_year if hasattr(dt.dt, "day_of_year") else dt.dt.dayofyear
    df["dayofyear"] = doy
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, group_col: str = "station") -> pd.DataFrame:
    df = df.copy().sort_values([group_col, "datetime"])
    cols = [col for col in [TARGET, "TEMP", "WSPM"] if col in df.columns]
    for col in cols:
        for lag in LAG_HOURS:
            df[f"{col}_lag{lag}h"] = df.groupby(group_col)[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, group_col: str = "station") -> pd.DataFrame:
    df = df.copy().sort_values([group_col, "datetime"])
    if TARGET not in df.columns:
        return df

    for w in ROLLING_WINDOWS:
        # Strictly shift(1) before rolling to avoid lookahead leakage
        shifted = df.groupby(group_col)[TARGET].shift(1)
        df[f"{TARGET}_roll{w}h_mean"] = shifted.groupby(df[group_col]).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"{TARGET}_roll{w}h_std"] = shifted.groupby(df[group_col]).transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0.0)
        )
    return df


def add_target_horizons(df: pd.DataFrame, group_col: str = "station") -> pd.DataFrame:
    df = df.copy().sort_values([group_col, "datetime"])
    for h in FORECAST_HORIZONS:
        df[f"target_h{h}"] = df.groupby(group_col)[TARGET].shift(-h)
    return df


def encode_station(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stations = sorted(df["station"].unique())
    s_map = {s: i for i, s in enumerate(stations)}
    df["station_id"] = df["station"].map(s_map).fillna(0).astype(int)
    return df


def process_city(city: str):
    logger.info(f"=== Processing City: {city.upper()} ===")
    can_path = CANONICAL_DIR / city / "observations.parquet"
    if not can_path.exists():
        logger.error(f"Canonical file missing: {can_path}")
        return

    df = pd.read_parquet(can_path)
    logger.info(f"Loaded {len(df)} canonical observations from {can_path.name}")

    # Standardize column naming
    if "station_id" in df.columns and "station" not in df.columns:
        df = df.rename(columns={"station_id": "station"})

    if "timestamp_local" in df.columns:
        clean_local = df["timestamp_local"].astype(str).str.replace(r"\s+[A-Za-z]+$", "", regex=True)
        df["datetime"] = pd.to_datetime(clean_local)
    elif "timestamp_utc" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize(None)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError(f"No valid datetime column found in {can_path}")

    # Sort strictly by station and time
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Missing value handling (interpolate up to 6 hours forward, then backfill)
    numeric_candidates = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = df.groupby("station")[col].transform(lambda x: x.interpolate(method="linear", limit=6))
            df[col] = df.groupby("station")[col].transform(lambda x: x.bfill().ffill())
            df[col] = df[col].fillna(df[col].median() if not np.isnan(df[col].median()) else 0.0)

    # Feature additions
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target_horizons(df)
    df = encode_station(df)

    # Drop edge rows where targets or fundamental lags are NaN
    target_cols = [f"target_h{h}" for h in FORECAST_HORIZONS]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    main_lags = [f"{TARGET}_lag{l}h" for l in [1, 24] if f"{TARGET}_lag{l}h" in df.columns]
    if main_lags:
        df = df.dropna(subset=main_lags).reset_index(drop=True)

    # Chronological 70% Train, 15% Validation, 15% Test split
    unique_dts = sorted(df["datetime"].unique())
    n_dts = len(unique_dts)
    train_end_idx = int(0.70 * n_dts)
    val_end_idx = int(0.85 * n_dts)

    train_end_dt = unique_dts[train_end_idx]
    val_end_dt = unique_dts[val_end_idx]

    train_df = df[df["datetime"] <= train_end_dt].copy()
    val_df = df[(df["datetime"] > train_end_dt) & (df["datetime"] <= val_end_dt)].copy()
    test_df = df[df["datetime"] > val_end_dt].copy()

    logger.info(f"Split sizes: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
                f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
                f"Test={len(test_df)} ({len(test_df)/len(df):.1%})")

    # Exclude metadata and targets from scaling
    exclude_cols = ["datetime", "station", "station_id", "hour", "dayofweek",
                    "month", "season", "is_weekend", "dayofyear",
                    "timestamp_utc", "timestamp_local", "city_id", "data_source", "quality_flag"] + target_cols
    scale_cols = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]

    # Fit scaler ONLY on train_df
    scaler = StandardScaler()
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols].fillna(0.0))
    val_df[scale_cols] = scaler.transform(val_df[scale_cols].fillna(0.0))
    test_df[scale_cols] = scaler.transform(test_df[scale_cols].fillna(0.0))

    # Save output to data/features/{city}/
    out_dir = FEATURES_BASE_DIR / city
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_dir / "train_features.parquet", index=False)
    val_df.to_parquet(out_dir / "val_features.parquet", index=False)
    test_df.to_parquet(out_dir / "test_features.parquet", index=False)
    joblib.dump(scaler, out_dir / "scaler.pkl")

    feature_cols = [c for c in train_df.columns if c not in ["datetime", "station"] + target_cols]
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump({"features": feature_cols, "targets": target_cols, "scale_cols": scale_cols}, f, indent=2)

    logger.info(f"Successfully built and saved features for {city} in {out_dir}")
    logger.info(f"Feature count: {len(feature_cols)}, Target horizons: {len(target_cols)}")


def main():
    args = parse_args()
    if args.city == "all":
        for c in ["delhi", "mumbai", "vijayawada", "beijing"]:
            process_city(c)
    else:
        process_city(args.city)


if __name__ == "__main__":
    main()
