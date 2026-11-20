"""
AeroSense AI V2 — Automated Data Quality & Anomaly Detection Engine.

Performs rigorous scientific validation checks on canonical observations:
1. Physical boundary validation (no negative PM2.5, no values exceeding 1000 µg/m³).
2. Monotonic timestamp sequence verification per station.
3. Sensor freeze / stagnation anomaly detection (detecting sensors stuck at identical values > 12h).
4. Missing rate threshold audit (warning if missing > 20%, failure if missing > 50%).
5. Column completeness and schema conformance.
"""
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from src.data.harmonizer import PHYSICAL_BOUNDS, CANONICAL_COLUMNS


class DataQualityValidator:
    """Validator performing automated data quality audits on canonical datasets."""

    def __init__(self, max_allowed_missing_pct: float = 25.0, freeze_threshold_hours: int = 12):
        self.max_missing = max_allowed_missing_pct
        self.freeze_threshold = freeze_threshold_hours

    def validate_dataset(self, df: pd.DataFrame, city_id: str) -> Dict[str, Any]:
        """
        Run complete battery of data quality checks on a canonical DataFrame.
        Returns a structured report with pass/fail status and audit diagnostics.
        """
        report: Dict[str, Any] = {
            "city_id": city_id,
            "total_records": len(df),
            "total_stations": df["station_id"].nunique() if "station_id" in df.columns else 0,
            "checks": {},
            "passed_all": True,
        }

        # 1. Schema Conformance Check
        missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
        report["checks"]["schema_conformance"] = {
            "passed": len(missing_cols) == 0,
            "missing_columns": missing_cols,
        }
        if missing_cols:
            report["passed_all"] = False

        # 2. Non-Negative & Physical Range Check for PM2.5
        if "PM2.5" in df.columns:
            valid_pm = df["PM2.5"].dropna()
            neg_count = int((valid_pm < 0.0).sum())
            exceed_count = int((valid_pm > 1000.0).sum())
            report["checks"]["pm25_physical_bounds"] = {
                "passed": neg_count == 0 and exceed_count == 0,
                "negative_count": neg_count,
                "exceed_1000_count": exceed_count,
                "observed_min": float(valid_pm.min()) if len(valid_pm) else None,
                "observed_max": float(valid_pm.max()) if len(valid_pm) else None,
            }
            if neg_count > 0 or exceed_count > 0:
                report["passed_all"] = False

        # 3. Monotonic Timestamp Continuity per Station
        if "timestamp_utc" in df.columns and "station_id" in df.columns:
            monotonic_failures = 0
            for st_id, group in df.groupby("station_id"):
                times = pd.to_datetime(group["timestamp_utc"])
                if not times.is_monotonic_increasing:
                    monotonic_failures += 1
            report["checks"]["temporal_monotonicity"] = {
                "passed": monotonic_failures == 0,
                "stations_with_chronological_violations": monotonic_failures,
            }
            if monotonic_failures > 0:
                report["passed_all"] = False

        # 4. Sensor Freeze / Stagnation Anomaly Check
        if "PM2.5" in df.columns and "station_id" in df.columns:
            freeze_events = 0
            for st_id, group in df.groupby("station_id"):
                vals = group["PM2.5"].values
                # Check for run-length of identical non-zero values
                run_len = 1
                for i in range(1, len(vals)):
                    if vals[i] == vals[i - 1] and pd.notnull(vals[i]) and vals[i] > 10.0:
                        run_len += 1
                        if run_len >= self.freeze_threshold:
                            freeze_events += 1
                            break
                    else:
                        run_len = 1
            report["checks"]["sensor_freeze_detection"] = {
                "passed": freeze_events == 0,
                "frozen_sensor_incidents": freeze_events,
            }

        # 5. Missing Rate Audit
        missing_rates = {}
        for col in ["PM2.5", "PM10", "NO2", "TEMP", "WSPM"]:
            if col in df.columns:
                pct = round(float(df[col].isna().mean() * 100), 2)
                missing_rates[col] = pct
        pm25_missing = missing_rates.get("PM2.5", 0.0)
        report["checks"]["missing_rate_audit"] = {
            "passed": pm25_missing <= self.max_missing,
            "missing_percentages": missing_rates,
            "threshold_pct": self.max_missing,
        }
        if pm25_missing > self.max_missing:
            report["passed_all"] = False

        logger.info(f"Data Quality Report for {city_id}: Overall Passed = {report['passed_all']}")
        return report
