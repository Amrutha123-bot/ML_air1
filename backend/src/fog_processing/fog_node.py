"""
Fog-side processing layer — simulated software module.

NOTE: This is a SOFTWARE SIMULATION of fog-side (edge) processing.
In a real deployment, this module would run on lightweight edge devices
co-located with monitoring stations, performing local quality-screening
before transmitting data to the cloud forecasting service.

Data-flow
---------
Monitoring station → FogNode.screen_incoming() → quality flags
                   → FogNode.reduce_transmission() → filtered batch
                   → Cloud-side integration → ST-CNN-GRU + GA-KELM
                   → Early-warning system → API response

Design rationale
----------------
- Reduces network traffic by suppressing redundant or invalid readings.
- Catches obvious sensor errors early (saves cloud processing time).
- Applies lightweight 3-point median smoothing (computationally cheap).
- Preserves the original raw value alongside the screened value for audit.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validity ranges for each measurement variable
# (physiologically and instrumentally plausible bounds)
# ---------------------------------------------------------------------------
_VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "PM2.5": (0.0,    999.0),    # µg/m³
    "PM10":  (0.0,   2000.0),    # µg/m³
    "SO2":   (0.0,    500.0),    # µg/m³
    "NO2":   (0.0,    500.0),    # µg/m³
    "CO":    (0.0,  10000.0),    # µg/m³
    "O3":    (0.0,    500.0),    # µg/m³
    "TEMP":  (-40.0,   60.0),    # °C
    "PRES":  (900.0, 1100.0),    # hPa
    "DEWP":  (-40.0,   40.0),    # °C
    "RAIN":  (0.0,    200.0),    # mm/h
    "WSPM":  (0.0,     50.0),    # m/s
}

# Extreme outlier multiplier: flag if value > OUTLIER_Z × IQR from median
_OUTLIER_Z = 4.0

# Fraction of mandatory fields that must be present to allow transmission
_MIN_FIELD_PRESENCE = 0.7

# Change-threshold for suppressing redundant transmissions (fraction of range)
_CHANGE_THRESHOLD_FRAC = 0.02   # 2 % of valid range must change to retransmit
_SCHEDULED_INTERVAL   = 6       # Always transmit at least every 6 observations


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QualityFlags:
    """Quality flags attached to a screened observation."""
    missing_fields: List[str]     = field(default_factory=list)
    out_of_range:   List[str]     = field(default_factory=list)
    extreme_outlier:List[str]     = field(default_factory=list)
    smoothed_fields:List[str]     = field(default_factory=list)
    overall_quality: str          = "good"      # "good" | "moderate" | "poor"
    transmit: bool                = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missing_fields":  self.missing_fields,
            "out_of_range":    self.out_of_range,
            "extreme_outlier": self.extreme_outlier,
            "smoothed_fields": self.smoothed_fields,
            "overall_quality": self.overall_quality,
            "transmit":        self.transmit,
        }


@dataclass
class ScreenedObservation:
    """Result of fog-side screening for one observation."""
    station:        str
    timestamp:      Any                   # original timestamp passthrough
    raw:            Dict[str, float]
    screened:       Dict[str, Optional[float]]
    quality:        QualityFlags
    obs_index:      int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station":    self.station,
            "timestamp":  str(self.timestamp),
            "screened":   self.screened,
            "quality":    self.quality.to_dict(),
            "obs_index":  self.obs_index,
        }


# ---------------------------------------------------------------------------
# Fog Node
# ---------------------------------------------------------------------------

class FogNode:
    """
    Simulated fog-side processing node for one monitoring station.

    Responsibilities
    ----------------
    1. Range-validity checks — flag out-of-range readings as NaN.
    2. Missing-field detection — record which fields are absent.
    3. Extreme-outlier flagging — statistical check vs. a rolling buffer.
    4. Lightweight 3-point median smoothing — reduce sensor noise.
    5. Transmission gating — suppress observations with insufficient change.
    """

    def __init__(self, station_id: str, buffer_size: int = 72):
        self.station_id  = station_id
        self.buffer_size = buffer_size
        self._obs_count  = 0

        # Circular buffer of recent values for outlier detection and smoothing
        # {field: list of recent values}
        self._buffer: Dict[str, List[float]] = {}
        self._last_transmitted: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    def _check_range(
        self, field: str, value: float
    ) -> Tuple[float, bool]:
        """
        Returns (value_or_nan, out_of_range_flag).
        """
        if field not in _VALID_RANGES:
            return value, False
        lo, hi = _VALID_RANGES[field]
        if not math.isfinite(value) or value < lo or value > hi:
            return float("nan"), True
        return value, False

    def _check_outlier(self, field: str, value: float) -> bool:
        """
        Returns True if value is an extreme outlier relative to recent buffer.
        Uses interquartile range (IQR) method.
        """
        buf = self._buffer.get(field, [])
        if len(buf) < 10:
            return False
        arr  = np.array(buf)
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-6:
            return False
        return bool(abs(value - np.median(arr)) > _OUTLIER_Z * iqr)

    def _smooth_3pt(self, field: str, value: float) -> float:
        """
        3-point median smoothing using the buffer.
        Returns smoothed value if at least 2 prior values exist.
        """
        buf = self._buffer.get(field, [])
        if len(buf) >= 2:
            window = [buf[-2], buf[-1], value]
            return float(np.median(window))
        return value

    def _update_buffer(self, field: str, value: float) -> None:
        if field not in self._buffer:
            self._buffer[field] = []
        self._buffer[field].append(value)
        if len(self._buffer[field]) > self.buffer_size:
            self._buffer[field].pop(0)

    def _should_transmit(self, screened: Dict[str, Optional[float]]) -> bool:
        """
        Gate: suppress transmission if change since last transmission is minimal.
        Always transmit on scheduled interval or if last_transmitted is None.
        """
        if self._last_transmitted is None:
            return True
        if self._obs_count % _SCHEDULED_INTERVAL == 0:
            return True

        # Check if any key pollutant changed significantly
        key_fields = ["PM2.5", "PM10", "NO2", "CO", "O3"]
        for f in key_fields:
            cur = screened.get(f)
            prev = self._last_transmitted.get(f)
            if cur is None or prev is None:
                continue
            lo, hi = _VALID_RANGES.get(f, (0, 1))
            full_range = hi - lo
            if abs(cur - prev) > _CHANGE_THRESHOLD_FRAC * full_range:
                return True
        return False

    # ------------------------------------------------------------------
    def screen_incoming(
        self,
        raw_obs: Dict[str, Any],
        timestamp: Any = None,
    ) -> ScreenedObservation:
        """
        Screen a single raw observation dictionary.

        Parameters
        ----------
        raw_obs : dict mapping field name → numeric value (or None/NaN)
        timestamp : optional timestamp passthrough

        Returns
        -------
        ScreenedObservation with cleaned data and quality flags.
        """
        self._obs_count += 1
        flags = QualityFlags()
        screened: Dict[str, Optional[float]] = {}
        raw_copy: Dict[str, float] = {}

        for field, raw_val in raw_obs.items():
            # Treat None / NaN as missing
            if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
                flags.missing_fields.append(field)
                screened[field] = None
                raw_copy[field] = float("nan")
                continue

            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                flags.missing_fields.append(field)
                screened[field] = None
                raw_copy[field] = float("nan")
                continue

            raw_copy[field] = val

            # 1. Range check
            val, oor = self._check_range(field, val)
            if oor:
                flags.out_of_range.append(field)
                screened[field] = None
                continue

            # 2. Extreme outlier check (against historical buffer)
            if self._check_outlier(field, val):
                flags.extreme_outlier.append(field)
                # Keep flagged value but mark it; do NOT set to NaN so
                # downstream can decide whether to impute

            # 3. 3-point median smoothing (only on clean, non-extreme values)
            if field not in flags.extreme_outlier:
                smoothed_val = self._smooth_3pt(field, val)
                if abs(smoothed_val - val) > 1e-6:
                    flags.smoothed_fields.append(field)
                val = smoothed_val

            self._update_buffer(field, val)
            screened[field] = val

        # Determine overall quality
        n_total = len(raw_obs)
        n_bad   = len(flags.missing_fields) + len(flags.out_of_range)
        frac_bad = n_bad / max(n_total, 1)
        if frac_bad > 0.4:
            flags.overall_quality = "poor"
        elif frac_bad > 0.15:
            flags.overall_quality = "moderate"
        else:
            flags.overall_quality = "good"

        # Transmission gate
        present_frac = 1.0 - (len(flags.missing_fields) + len(flags.out_of_range)) / max(n_total, 1)
        if present_frac < _MIN_FIELD_PRESENCE:
            flags.transmit = False
            flags.overall_quality = "poor"
        else:
            flags.transmit = self._should_transmit(
                {k: v for k, v in screened.items() if v is not None}
            )

        if flags.transmit:
            self._last_transmitted = {k: v for k, v in screened.items() if v is not None}

        return ScreenedObservation(
            station   = self.station_id,
            timestamp = timestamp,
            raw       = raw_copy,
            screened  = screened,
            quality   = flags,
            obs_index = self._obs_count,
        )

    # ------------------------------------------------------------------
    def reduce_transmission(
        self, obs_list: List[Dict[str, Any]]
    ) -> List[ScreenedObservation]:
        """
        Screen a batch of raw observations and filter to only those
        that pass transmission gate.

        Parameters
        ----------
        obs_list : list of raw observation dicts (may include 'timestamp' key)

        Returns
        -------
        List of ScreenedObservation objects with transmit=True.
        """
        results = []
        for raw in obs_list:
            ts  = raw.pop("timestamp", None)
            obs = self.screen_incoming(raw, timestamp=ts)
            if obs.quality.transmit:
                results.append(obs)
        return results


# ---------------------------------------------------------------------------
# Multi-station fog gateway
# ---------------------------------------------------------------------------

class FogGateway:
    """
    Simulated fog gateway coordinating multiple FogNode instances.

    In deployment: one FogNode per monitoring station on edge hardware.
    """

    def __init__(self, station_ids: List[str]):
        self.nodes: Dict[str, FogNode] = {
            sid: FogNode(station_id=sid) for sid in station_ids
        }
        logger.info("FogGateway: initialised with %d stations", len(station_ids))

    def process_station(
        self,
        station_id: str,
        raw_obs: Dict[str, Any],
        timestamp: Any = None,
    ) -> ScreenedObservation:
        """Screen a single observation for the given station."""
        if station_id not in self.nodes:
            self.nodes[station_id] = FogNode(station_id=station_id)
        return self.nodes[station_id].screen_incoming(raw_obs, timestamp)

    def process_batch(
        self,
        station_id: str,
        obs_list: List[Dict[str, Any]],
    ) -> List[ScreenedObservation]:
        """Screen a batch and return only observations passing transmission gate."""
        if station_id not in self.nodes:
            self.nodes[station_id] = FogNode(station_id=station_id)
        return self.nodes[station_id].reduce_transmission(obs_list)

    def gateway_summary(self) -> Dict[str, Any]:
        """Return per-station observation counts."""
        return {
            sid: {"obs_count": node._obs_count}
            for sid, node in self.nodes.items()
        }
