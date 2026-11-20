"""
Hybrid Forecaster: primary ST-CNN-GRU + GA-KELM fallback + RF benchmark.

The hybrid system monitors recent validation error and switches to the
GA-KELM when the primary model's rolling MAE exceeds a threshold.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default threshold multiplier: switch to fallback when primary MAE exceeds
# 1.3 × baseline_mae (the RF benchmark's validation MAE)
DEFAULT_FALLBACK_MULTIPLIER = 1.3
_ROLLING_WINDOW = 48  # hours over which rolling error is tracked


class HybridForecaster:
    """
    Hybrid forecaster that selects between primary and fallback models.

    Decision logic
    --------------
    1. Each prediction, compute absolute error on the most recent observation.
    2. Maintain a rolling deque of recent errors per station.
    3. If rolling_MAE > threshold, route to GA-KELM (fallback).
    4. Otherwise, route to primary (ST-CNN-GRU).
    5. The RF model serves as a benchmark, never the runtime model.

    Parameters
    ----------
    primary_model : model with .predict(X) → array
        ST-CNN-GRU (or any primary model).
    fallback_model : GAKELMModel
        GA-KELM model used when primary underperforms.
    benchmark_mae : float
        Validation MAE of the RF benchmark model.
    fallback_multiplier : float
        Fallback activated when rolling MAE > multiplier × benchmark_mae.
    """

    def __init__(
        self,
        primary_model,
        fallback_model,
        benchmark_mae: float = 40.0,
        fallback_multiplier: float = DEFAULT_FALLBACK_MULTIPLIER,
    ):
        self.primary_model       = primary_model
        self.fallback_model      = fallback_model
        self.benchmark_mae       = benchmark_mae
        self.fallback_threshold  = fallback_multiplier * benchmark_mae
        self.fallback_multiplier = fallback_multiplier

        # Per-station rolling error windows
        self._errors: Dict[str, deque] = {}
        self._decisions: Dict[str, str] = {}  # "primary" or "fallback"
        self._fallback_count = 0
        self._primary_count  = 0

        logger.info(
            "HybridForecaster: threshold=%.2f (%.1f× RF benchmark=%.2f)",
            self.fallback_threshold, fallback_multiplier, benchmark_mae,
        )

    # ------------------------------------------------------------------
    def _get_rolling_mae(self, station: str) -> Optional[float]:
        errs = self._errors.get(station)
        if errs and len(errs) > 0:
            return float(np.mean(errs))
        return None

    def _record_error(self, station: str, error: float) -> None:
        if station not in self._errors:
            self._errors[station] = deque(maxlen=_ROLLING_WINDOW)
        self._errors[station].append(abs(error))

    def _select_model(self, station: str) -> str:
        """Return 'primary' or 'fallback' based on recent errors."""
        rolling_mae = self._get_rolling_mae(station)
        if rolling_mae is not None and rolling_mae > self.fallback_threshold:
            return "fallback"
        return "primary"

    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        station: str = "unknown",
        true_y: Optional[np.ndarray] = None,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Dict]:
        """
        Produce forecast using dynamically selected model.

        Parameters
        ----------
        X : array (n_samples, n_features) or DataFrame
        station : str  — station identifier for per-station tracking
        true_y : array optional — if provided, errors are tracked to inform
                 future model-selection decisions.
        return_meta : bool — whether to return metadata dict or raw predictions array.

        Returns
        -------
        If return_meta=True: dict with keys: predictions, model_used, rolling_mae, threshold
        If return_meta=False: np.ndarray of predictions
        """
        model_choice = self._select_model(station)

        try:
            if model_choice == "fallback":
                preds = self.fallback_model.predict(X)
                self._fallback_count += 1
            else:
                preds = self.primary_model.predict(X)
                self._primary_count  += 1
        except Exception as exc:
            logger.warning(
                "HybridForecaster: %s model failed (%s), falling back to GA-KELM",
                model_choice, exc,
            )
            preds = self.fallback_model.predict(X)
            model_choice = "fallback"
            self._fallback_count += 1

        self._decisions[station] = model_choice

        # Update rolling errors if ground truth available
        if true_y is not None:
            errs = np.abs(np.asarray(preds).ravel() - np.asarray(true_y).ravel())
            for e in errs:
                self._record_error(station, float(e))

        if return_meta:
            return {
                "predictions": preds,
                "model_used":  model_choice,
                "rolling_mae": self._get_rolling_mae(station),
                "threshold":   self.fallback_threshold,
                "fallback_total":  self._fallback_count,
                "primary_total":   self._primary_count,
            }
        return np.asarray(preds)

    # ------------------------------------------------------------------
    def status_report(self) -> Dict:
        """Return summary of model selection decisions."""
        total = self._primary_count + self._fallback_count
        return {
            "total_calls":      total,
            "primary_calls":    self._primary_count,
            "fallback_calls":   self._fallback_count,
            "fallback_rate":    self._fallback_count / max(total, 1),
            "threshold":        self.fallback_threshold,
            "benchmark_mae":    self.benchmark_mae,
            "per_station_mae":  {
                st: self._get_rolling_mae(st)
                for st in self._errors
            },
        }

    def reset_state(self) -> None:
        """Reset rolling error windows (e.g., for new evaluation period)."""
        self._errors.clear()
        self._decisions.clear()
        self._fallback_count = 0
        self._primary_count  = 0
