"""Leakage-aware splitting and execution-aware signal backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedSplit:
    """Origin-index boundaries for train/validation/test with label embargoes."""

    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def purged_train_val_test_split(
    n_samples: int, val_fraction: float, test_fraction: float, horizon: int, min_train: int
) -> PurgedSplit:
    """Create chronological splits without labels crossing a later fold.

    A label at origin ``t`` consumes prices through ``t + horizon``.  The last
    ``horizon`` origins before each later fold are therefore embargoed.
    """
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")
    n_test = max(1, int(n_samples * test_fraction))
    n_val = max(1, int(n_samples * val_fraction))
    test_start = n_samples - n_test
    val_end = test_start - horizon
    val_start = val_end - n_val
    train_end = val_start - horizon
    if train_end < min_train or val_start <= train_end or val_end <= val_start:
        raise ValueError("not enough rows for requested sequence length, folds, and embargoes")
    return PurgedSplit(train_end, val_start, val_end, test_start, n_samples)


@dataclass(frozen=True)
class LinearCalibrator:
    intercept: float
    slope: float

    def transform(self, prediction: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(prediction, dtype=float)


def fit_linear_calibrator(prediction: np.ndarray, realized: np.ndarray) -> LinearCalibrator:
    """Fit a conservative affine forecast calibration on a development fold."""
    x = np.asarray(prediction, dtype=float)
    y = np.asarray(realized, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12:
        return LinearCalibrator(float(np.nanmean(y[mask])) if mask.any() else 0.0, 0.0)
    slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
    return LinearCalibrator(float(intercept), float(slope))


def positions_from_signal(signal: np.ndarray, leverage: float, threshold: float, max_position: float) -> np.ndarray:
    """Threshold, scale, and cap a risk-normalized trading signal."""
    signal = np.asarray(signal, dtype=float)
    position = np.where(np.abs(signal) >= threshold, leverage * signal, 0.0)
    return np.clip(position, -max_position, max_position)


def net_strategy_returns(
    positions: pd.Series,
    next_log_returns: pd.Series,
    carry_returns: pd.Series | None = None,
    transaction_cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Compute date-aligned gross, costs, carry, and net log-return approximation."""
    frame = pd.concat([positions.rename("position"), next_log_returns.rename("return")], axis=1).dropna()
    if carry_returns is None:
        frame["carry"] = 0.0
    else:
        frame["carry"] = carry_returns.reindex(frame.index).fillna(0.0)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["cost"] = frame["turnover"] * (transaction_cost_bps * 1e-4)
    frame["gross"] = frame["position"] * frame["return"]
    frame["net"] = frame["gross"] + frame["position"] * frame["carry"] - frame["cost"]
    return frame


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    std = returns.std(ddof=0)
    return 0.0 if std < 1e-12 else float(returns.mean() / std * np.sqrt(periods_per_year))
