"""Feature engineering and rolling (causal) normalization for FX return forecasting.

Every function here is a pure pandas transform (no Config, no I/O, no torch),
so each stage can be called and inspected independently — e.g. under a
debugger, or from a notebook — without pulling in the training pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def compute_log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns per pair; these are the model's raw (pre-standardization) input factors."""
    log_returns = np.log(panel).diff().dropna(how="any")
    logger.info("Log returns: %s", log_returns.shape)
    return log_returns


def compute_engineered_features(
    panel: pd.DataFrame,
    log_returns: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    momentum_window: int,
    vol_window: int,
) -> pd.DataFrame:
    """Per-pair engineered features, computed for every pair:

    - intraday_vol: (high - low) / low, the day's realized trading range.
    - momentum: trailing `momentum_window`-day cumulative log return (trend signal).
    - rvol: trailing `vol_window`-day realized volatility (rolling std of daily returns),
      distinct from the longer window used later to rolling-normalize all features.
    - skew: trailing `vol_window`-day rolling skewness of daily returns — asymmetry of
      the return distribution (negative skew means a fatter downside/crash tail, the
      well-documented shape of FX/equity returns).
    - kurt: trailing `vol_window`-day rolling excess kurtosis of daily returns — tail
      fatness relative to a normal distribution (pandas' `.kurt()` is already *excess*
      kurtosis, i.e. 0 for a normal distribution, not the raw convention where normal=3).

    skew/kurt are scale-invariant (computing them on raw vs. standardized
    returns gives identical values), so they don't need their own explicit
    z-scoring here — like every other column here, they still get rolling-
    normalized downstream by `compute_rolling_normalized_features`.
    """
    common_index = panel.index.intersection(high.index).intersection(low.index)
    panel = panel.loc[common_index]
    high = high.loc[common_index]
    low = low.loc[common_index]
    log_returns = log_returns.reindex(common_index)

    intraday_vol = (high - low) / low
    intraday_vol.columns = [f"{c}_intraday_vol" for c in intraday_vol.columns]

    log_price = np.log(panel)
    momentum = log_price.diff(momentum_window)
    momentum.columns = [f"{c}_momentum{momentum_window}" for c in momentum.columns]

    rvol = log_returns.rolling(window=vol_window).std()
    rvol.columns = [f"{c}_rvol{vol_window}" for c in rvol.columns]

    skew = log_returns.rolling(window=vol_window).skew()
    skew.columns = [f"{c}_skew{vol_window}" for c in skew.columns]

    kurt = log_returns.rolling(window=vol_window).kurt()
    kurt.columns = [f"{c}_kurt{vol_window}" for c in kurt.columns]

    engineered = pd.concat([intraday_vol, momentum, rvol, skew, kurt], axis=1).dropna(how="any")
    logger.info("Engineered features: %s", engineered.shape)
    return engineered


def compute_carry_feature(pairs: list[str], rates: pd.DataFrame) -> pd.DataFrame:
    """Interest-rate carry per pair: base currency's rate minus quote currency's rate.

    `rates` is a wide DataFrame, one column per currency code (e.g. "EUR",
    "USD"), already forward-filled to the daily price index (source rates are
    monthly; carry only changes when a central bank actually moves, so
    forward-filling introduces no lookahead). For pair "EURUSD" the base
    currency is EUR and the quote is USD, so carry = rate(EUR) - rate(USD) —
    positive when the base currency yields more than the quote currency,
    which under uncovered interest rate parity predicts base-currency
    depreciation (a negative expected forward return), the classic FX
    "carry" risk premium/signal.
    """
    carry = pd.DataFrame(
        {pair: rates[pair[:3]] - rates[pair[3:]] for pair in pairs}, index=rates.index
    ).dropna(how="any")
    logger.info("Carry: %s", carry.shape)
    return carry


@dataclass(frozen=True)
class CMAWindow:
    """One (short, long) EWMA pair used to build a crossover ("CMA") feature."""

    short: int
    long: int
    weight: float

    def __post_init__(self) -> None:
        if self.short >= self.long:
            raise ValueError(f"short window ({self.short}) must be < long window ({self.long})")

    @property
    def name(self) -> str:
        return f"cma_{self.short}_{self.long}"


# Three EWMA crossover windows spanning the horizons that matter for FX
# trend-following: a fast pair that reacts within days, the classic MACD
# pair (~2.5 vs ~5 weeks), and the classic golden/death-cross pair (~10
# weeks vs ~40 weeks) for the underlying multi-month trend regime.
DEFAULT_CMA_WINDOWS: tuple[CMAWindow, ...] = (
    CMAWindow(5, 20,1.0),
    CMAWindow(12, 26,1.0),
    CMAWindow(52, 200,1.0)
)


class CrossingMovingAverages:
    """Computes a series of EWMA crossover ("CMA") signals on FX pair price levels.

    Each configured (short, long) window produces one feature per FX pair:
    the spread between a fast and a slow exponentially-weighted moving
    average of the price level, normalized by the slow EWMA — a classic
    trend-following signal in the spirit of MACD. Positive when the fast
    average is above the slow one (uptrend), negative when below
    (downtrend); normalizing by the slow EWMA makes the signal comparable
    across pairs quoted at very different price levels (e.g. EURUSD ~1.1 vs
    USDJPY ~150).

    Each EWMA is causal by construction (`ewm(..., adjust=False)` only ever
    looks backward), so no future information leaks into the features.
    """

    def __init__(self, windows: Sequence[CMAWindow] = DEFAULT_CMA_WINDOWS) -> None:
        self.windows = list(windows)

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """`panel`: wide DataFrame of price levels, one column per FX pair.

        Returns a wide DataFrame with one column per (pair, window)
        combination, named "{pair}_{window.name}", e.g. "EURUSD_cma_12_26".
        """
        features = {}
        total_weight = np.sum([w.weight for w in self.windows])
        for symbol in panel.columns:
            price = panel[symbol]
            for window in self.windows:
                ewma_short = price.ewm(span=window.short, adjust=False).mean()
                ewma_long = price.ewm(span=window.long, adjust=False).mean()
                features[f"{symbol}_{window.name}"] = (ewma_short - ewma_long) / ewma_long * window.weight/total_weight

        cma = pd.DataFrame(features, index=panel.index).dropna(how="any")
        logger.info("CMA crossover features: %s", cma.shape)
        return cma


def build_symbol_frame(
    symbol: str,
    log_returns: pd.DataFrame,
    engineered: pd.DataFrame,
    momentum_window: int,
    vol_window: int,
    carry: pd.Series | None = None,
    cma: pd.DataFrame | None = None,
    extra_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One symbol's own raw (pre-normalization) feature block: return, intraday_vol,
    momentum, rvol, skew, kurt[, carry][, cma...]."""
    columns = {
        "return": log_returns[symbol],
        "intraday_vol": engineered[f"{symbol}_intraday_vol"],
        f"momentum{momentum_window}": engineered[f"{symbol}_momentum{momentum_window}"],
        f"rvol{vol_window}": engineered[f"{symbol}_rvol{vol_window}"],
        f"skew{vol_window}": engineered[f"{symbol}_skew{vol_window}"],
        f"kurt{vol_window}": engineered[f"{symbol}_kurt{vol_window}"],
    }
    if carry is not None:
        columns["carry"] = carry
    frame = pd.DataFrame(columns).dropna(how="any")
    if cma is not None:
        frame = pd.concat([frame, cma], axis=1).dropna(how="any")
    if extra_features is not None:
        frame = pd.concat([frame, extra_features], axis=1).dropna(how="any")
    return frame


def compute_rolling_normalized_features(raw_features: pd.DataFrame, window: int) -> pd.DataFrame:
    """Standardize every feature column by its own trailing rolling mean/std.

    Causal by construction: the value at time t only uses data from
    [t - window + 1, t], so no future information leaks into the normalization,
    and it adapts to changing volatility regimes instead of using one fixed
    train-set statistic for the whole series.

    Must be called per symbol (on that symbol's own raw feature block) rather
    than on a wide multi-symbol frame: computing the rolling mean/std jointly
    would let one symbol's missing dates truncate another's rolling window.
    Wide input frames are built by concatenating each symbol's *already
    normalized* output, not by normalizing after concatenation.
    """
    rolling_mean = raw_features.rolling(window=window).mean()
    rolling_std = raw_features.rolling(window=window).std().replace(0, np.nan)
    normalized = ((raw_features - rolling_mean) / rolling_std).dropna(how="any")
    return normalized


def compute_target_zscore(
    panel: pd.DataFrame, log_returns: pd.DataFrame, symbol: str, horizon: int, window: int
) -> pd.Series:
    """The `horizon`-day cumulative return for `symbol`, standardized (z-scored) by its own
    trailing rolling volatility — the "normalized aggregated return".

    The horizon-day return is standardized against a std dev given by the
    trailing rolling std of `symbol`'s *daily* return (causal, same window as
    the feature normalization), scaled by sqrt(horizon) for a horizon-day
    return under a random-walk variance assumption. Unlike
    `compute_target_quantile`, this is left as a continuous, unbounded
    z-score (not passed through the normal CDF) — roughly 0 for a typical
    no-move horizon, positive/negative in std-dev units for unusually large
    up/down moves.
    """
    log_price = np.log(panel[symbol])
    cum_return = log_price.shift(-horizon) - log_price

    rolling_std = log_returns[symbol].rolling(window=window).std()
    scale = (rolling_std * np.sqrt(horizon)).replace(0, np.nan)

    z = (cum_return / scale).dropna()
    return pd.Series(z, index=z.index, name=f"cum_return_{horizon}d_zscore")


def compute_target_quantile(
    panel: pd.DataFrame, log_returns: pd.DataFrame, symbol: str, horizon: int, window: int
) -> pd.Series:
    """`compute_target_zscore`, passed through the standard normal CDF to express it as a
    quantile in (0, 1) — 0.5 is the median (no move), values -> 1 are extreme positive
    returns, values -> 0 are extreme negative returns.
    """
    z = compute_target_zscore(panel, log_returns, symbol, horizon, window)
    quantile = pd.Series(norm.cdf(z.to_numpy()), index=z.index, name=f"cum_return_{horizon}d_quantile")
    return quantile


# 5-class direction labeling of `compute_target_quantile`'s continuous
# quantile: a 10/20/40/20/10 split, so the two "very_*" tail classes are each
# the most extreme 10% of moves, "bearish"/"bullish" the next 20% on each
# side, and "neutral" the middle 40% — equivalent to z-score bands of
# roughly ±1.28σ (very_*) and ±0.52σ (mild) under `compute_target_quantile`'s
# own normality assumption. Order matches class index 0-4.
CLASS_NAMES: tuple[str, ...] = ("very_bearish", "bearish", "neutral", "bullish", "very_bullish")
CLASS_QUANTILE_EDGES: tuple[float, ...] = (0.0, 0.1, 0.3, 0.7, 0.9, 1.0)


def compute_target_class(
    panel: pd.DataFrame, log_returns: pd.DataFrame, symbol: str, horizon: int, window: int
) -> pd.Series:
    """Buckets `compute_target_quantile`'s continuous return quantile into 5 discrete
    direction classes (see `CLASS_NAMES`), returned as integer labels 0-4.

    0=very_bearish, 1=bearish, 2=neutral, 3=bullish, 4=very_bullish (cut
    points: `CLASS_QUANTILE_EDGES`).
    """
    quantile = compute_target_quantile(panel, log_returns, symbol, horizon, window)
    labels = pd.cut(quantile, bins=CLASS_QUANTILE_EDGES, labels=False, include_lowest=True)
    return pd.Series(labels, index=quantile.index, name=f"cum_return_{horizon}d_class").astype("int64")


# 3-class direction labeling of `compute_target_quantile`'s continuous
# quantile: the same idea as CLASS_NAMES but collapsed to the classic
# bearish/neutral/bullish triad — bottom/top 30% each for the two tails,
# middle 40% neutral (equivalent to merging CLASS_QUANTILE_EDGES's two tail
# pairs on each side). Order matches class index 0-2; DIRECTION_VALUES gives
# each class's conceptual signed label (-1/0/+1) for anywhere that wants the
# number rather than the index used for nn.CrossEntropyLoss.
DIRECTION_CLASS_NAMES: tuple[str, ...] = ("bearish", "neutral", "bullish")
DIRECTION_VALUES: tuple[int, ...] = (-1, 0, 1)
DIRECTION_QUANTILE_EDGES: tuple[float, ...] = (0.0, 0.3, 0.7, 1.0)


def compute_target_direction(
    panel: pd.DataFrame, log_returns: pd.DataFrame, symbol: str, horizon: int, window: int
) -> pd.Series:
    """Buckets `compute_target_quantile`'s continuous return quantile into 3 discrete
    direction classes (see `DIRECTION_CLASS_NAMES`/`DIRECTION_VALUES`), returned as
    integer labels 0-2 for `nn.CrossEntropyLoss` (0=bearish/-1, 1=neutral/0,
    2=bullish/+1; cut points: `DIRECTION_QUANTILE_EDGES`).
    """
    quantile = compute_target_quantile(panel, log_returns, symbol, horizon, window)
    labels = pd.cut(quantile, bins=DIRECTION_QUANTILE_EDGES, labels=False, include_lowest=True)
    return pd.Series(labels, index=quantile.index, name=f"cum_return_{horizon}d_direction").astype("int64")
