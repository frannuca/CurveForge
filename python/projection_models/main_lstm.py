"""Train the LSTM + attention forecaster on a *pooled* panel of FX pairs.

Each FX pair gets its own dataset: its own log-return, intraday-volatility,
momentum, realized-volatility, interest-rate carry, and EWMA crossover
("CMA") columns, normalized against its own rolling history, paired with its
own N-day-ahead target — one instrument's own history in, one instrument's
own future target out. ``Config.target_mode`` picks what that target is:
``"zscore"`` (default) is the continuous, standardized ("z-scored")
cumulative return itself (see
`fx_forecasting.features.compute_target_zscore`); ``"class"`` buckets that
same z-score into 5 discrete direction labels, very_bearish..very_bullish
(see `fx_forecasting.features.compute_target_class`). Every symbol's
dataset shares the identical feature schema (same columns, same
normalization scheme), so once each symbol's (features, target) dataset is
built independently, they're concatenated into one pooled training set: a
single shared-weight model trains across every symbol at once, never told
which symbol a given sample came from, pushing it toward the shared
distributional dynamics of FX returns rather than memorizing one pair's
idiosyncrasies.

Deliberately written as small, independent, top-level functions (rather than one
monolithic pipeline) so each stage can be run and inspected separately under a
debugger: set a breakpoint after any call in ``main()`` and inspect the returned
DataFrame / tensors / model directly. Every stage logs the shape of what it
produced, so running with ``--debug`` (tiny data, 1 epoch) plus a look at the
console output is usually enough to understand what the model is seeing without
even opening a debugger.

Example:
    uv run python main_lstm.py --debug
    uv run python main_lstm.py --epochs 30 --hidden-size 64 --seq-len 60 --horizon 2
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from fx_forecasting.data.db import get_metric_series, get_time_series
from fx_forecasting.data.fx_downloader import MAJOR_FX_PAIRS,SINGLE_FX_PAIRS
from scipy.stats import norm

from fx_forecasting.features import (
    CLASS_NAMES,
    CLASS_QUANTILE_EDGES,
    DEFAULT_CMA_WINDOWS,
    CMAWindow,
    CrossingMovingAverages,
    build_symbol_frame,
    compute_carry_feature,
    compute_engineered_features,
    compute_log_returns,
    compute_rolling_normalized_features,
    compute_target_class,
    compute_target_zscore,
)
from fx_forecasting.models.lstm_forecaster import LSTMAttentionForecaster
from fx_forecasting.trading import (
    LinearCalibrator,
    annualized_sharpe,
    fit_linear_calibrator,
    net_strategy_returns,
    positions_from_signal,
    purged_train_val_test_split,
)

logger = logging.getLogger(__name__)

# Set to cooperatively stop an in-progress `train()` call early (checked once
# per epoch) — e.g. a "Stop" button in a UI driving `main()` in a background
# thread. `train()` clears it on every call, so it never leaks into the next run.
CANCEL_EVENT = threading.Event()


@dataclass
class Config:
    pairs: list[str] = field(default_factory=lambda: list(MAJOR_FX_PAIRS.keys()))
    years: int = 20
    # When set, the model is trained to forecast only this one pair — `pairs` still
    # defines the input universe (cross-asset xret_* features, carry, CMA, etc.), but
    # `build_pooled_datasets` builds/pools a single symbol's dataset instead of one
    # per pair. Must be one of `pairs`. `None` (default) keeps today's behavior:
    # every pair in `pairs` is pooled, each forecasting its own target.
    target_symbol: str | None = None

    seq_len: int = 60
    horizon: int = 2  # N days ahead for the single cumulative-return target
    target_mode: str = "zscore"  # "zscore" (continuous normalized return, default) or "class" (5-class direction label)
    momentum_window: int = 10  # trailing window (days) for the momentum feature
    vol_window: int = 10  # trailing window (days) for the realized-volatility feature
    cma_windows: list[CMAWindow] = field(default_factory=lambda: list(DEFAULT_CMA_WINDOWS))

    # Small/heavily-regularized by default: correlation diagnostics show these
    # features carry weak signal (|corr| ~0.02-0.2), and a larger model
    # (e.g. hidden_size=64, num_layers=2) was observed to overfit within a
    # single epoch — train_mse kept dropping while val_mse rose immediately.
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.3
    weight_decay: float = 1e-4

    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    lr_factor: float = 0.5  # multiply LR by this when val loss plateaus
    lr_patience: int = 3  # epochs with no val loss improvement before reducing LR
    min_lr: float = 1e-6
    early_stop_patience: int = 10  # stop training if val loss hasn't improved in this many epochs
    outlier_weight: float = 0.0  # start with calibrated likelihood; tune only inside walk-forward development folds
    variance_penalty_weight: float = 0.0  # kept for backwards-compatible experiments; disabled by default
    val_fraction: float = 0.2
    test_fraction: float = 0.1
    predict_uncertainty: bool = True
    use_spectral_features: bool = False  # off by default — validate via purged walk-forward before defaulting on
    spectral_embedding_dim: int = 8
    spectral_freq_bins: int = 16
    # If set, reduces the len(pairs) cross-asset xret_* input columns to this many learned,
    # PCA-like linear factors before the encoder (and spectral embedding, if also enabled)
    # ever sees them — see fx_forecasting.models.cross_asset_projection. Must be strictly
    # between 0 and len(pairs); None (default) leaves the raw xret_* columns untouched.
    cross_asset_pca_dim: int | None = None
    transaction_cost_bps: float = 1.0
    signal_threshold: float = 0.10
    signal_leverage: float = 0.50
    max_position: float = 1.0

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    plot_path: str = "artifacts/lstm_predictions.png"
    in_sample_plot_path: str = "artifacts/lstm_predictions_in_sample.png"
    test_plot_path: str = "artifacts/lstm_predictions_test.png"
    pnl_plot_path: str = "artifacts/lstm_pnl.png"
    test_pnl_plot_path: str = "artifacts/lstm_pnl_test.png"
    positions_pnl_plot_path: str = "artifacts/lstm_positions_pnl.png"
    test_positions_pnl_plot_path: str = "artifacts/lstm_positions_pnl_test.png"
    model_path: str = "artifacts/lstm_model.pt"
    features_csv_path: str = "artifacts/features.csv"
    write_features_csv: bool = True  # set False to skip the CSV write (e.g. many fast hyperparameter-search trials)
    infer: bool = False  # skip training; load `model_path` and run inference only


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_price_panel(cfg: Config) -> pd.DataFrame:
    """Load each pair's close price from Postgres and align them into one wide DataFrame."""
    end = date.today()
    start = end - timedelta(days=365 * cfg.years)

    panel = get_time_series(cfg.pairs, start, end)
    missing = [p for p in cfg.pairs if panel[p].isna().all()]
    if missing:
        raise ValueError(f"No data in DB for pairs {missing} between {start} and {end}")

    panel = panel.dropna(how="any")
    logger.info("Price panel: %s (from %s to %s)", panel.shape, start, end)
    return panel


def load_metric_panel(cfg: Config, metric_name: str) -> pd.DataFrame:
    """Load a stored OHLCV metric (e.g. 'high', 'low') for each pair from Postgres."""
    end = date.today()
    start = end - timedelta(days=365 * cfg.years)

    panel = get_metric_series(cfg.pairs, metric_name, start, end)
    panel = panel.dropna(how="any")
    logger.info("%s panel: %s (from %s to %s)", metric_name, panel.shape, start, end)
    return panel


def load_rate_panel(cfg: Config, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load each currency's short-term interest rate from Postgres, forward-filled onto `price_index`.

    Rates are monthly at the source (see rates_downloader.py); forward-filling
    onto the daily price index is causal — a fill only ever repeats the last
    *known* rate forward, so no future rate ever leaks backward.
    """
    end = date.today()
    start = end - timedelta(days=365 * cfg.years)

    currencies = sorted({c for pair in cfg.pairs for c in (pair[:3], pair[3:])})
    rates = get_time_series(currencies, start, end, source="fred", field="rate")
    missing = [c for c in currencies if rates[c].isna().all()]
    if missing:
        raise ValueError(
            f"No rate data in DB for currencies {missing}; run "
            f"`uv run python -m fx_forecasting.data.rates_downloader --upload` first."
        )

    rates_daily = rates.reindex(rates.index.union(price_index)).sort_index().ffill().reindex(price_index)
    logger.info("Rate panel (currencies, forward-filled to price index): %s", rates_daily.shape)
    return rates_daily


@dataclass
class RawPanels:
    """Raw, hyperparameter-independent inputs fetched from Postgres: price levels,
    high/low, and interest rates. Depends only on `cfg.pairs`/`cfg.years`, not on any
    of the feature/model hyperparameters (seq_len, momentum/vol windows, CMA windows,
    hidden_size, ...), so it can be fetched once and reused across many
    `build_pooled_datasets` calls — e.g. the many trials of a hyperparameter search
    (see `optimize_hyperparams.py`), where re-querying the DB per trial would dominate
    runtime.
    """

    panel: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    rates: pd.DataFrame


def load_raw_panels(cfg: Config) -> RawPanels:
    """Fetches everything `build_pooled_datasets` needs from Postgres, once."""
    panel = load_price_panel(cfg)
    high = load_metric_panel(cfg, "high")
    low = load_metric_panel(cfg, "low")
    rates = load_rate_panel(cfg, panel.index)
    return RawPanels(panel=panel, high=high, low=low, rates=rates)


class FXSequenceDataset(Dataset):
    """Sliding windows of rolling-normalized features paired with a target.

    Everything downstream (model input, loss, plots) lives in this
    normalized/labeled space; the model never sees a raw return value.
    `target`'s own dtype selects what it holds: an integer array is a 5-class
    direction label (`target_mode="class"`, see
    `fx_forecasting.features.CLASS_NAMES`), stored as `torch.long` for
    `nn.CrossEntropyLoss`; a float array is a continuous return z-score
    (`target_mode="zscore"`, see
    `fx_forecasting.features.compute_target_zscore`), stored as `torch.float32`.
    """

    def __init__(
        self, features: np.ndarray, target: np.ndarray, seq_len: int, dates: pd.Index, origin_start: int, origin_end: int
    ) -> None:
        self.features = torch.as_tensor(np.asarray(features).copy(), dtype=torch.float32)
        target_array = np.asarray(target)
        target_dtype = torch.long if np.issubdtype(target_array.dtype, np.integer) else torch.float32
        self.targets = torch.as_tensor(target_array.copy(), dtype=target_dtype)
        self.seq_len = seq_len
        self.dates = pd.DatetimeIndex(dates)

        # Row i in `features`/`target` corresponds to a window ending at i
        # (inclusive), i.e. sample i uses features[i - seq_len + 1 : i + 1].
        self.valid_indices = list(range(max(seq_len - 1, origin_start), origin_end))
        if not self.valid_indices:
            raise ValueError("split contains no complete sequences")

    @property
    def sample_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.valid_indices]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.valid_indices[idx]
        start = end - self.seq_len + 1
        x = self.features[start : end + 1]
        y = self.targets[end]  # scalar; nn.CrossEntropyLoss/MSE both expect shape (batch,) after collation
        return x, y


def build_datasets(
    features: pd.DataFrame, target: pd.Series, cfg: Config
) -> tuple[FXSequenceDataset, FXSequenceDataset, FXSequenceDataset]:
    """Purged chronological train/validation/test sets for one symbol.

    The label at a forecast origin uses the next ``cfg.horizon`` prices, so a
    horizon-sized embargo separates every adjacent fold.
    """
    aligned_index = features.index.intersection(target.index)
    features = features.loc[aligned_index]
    target = target.loc[aligned_index]

    split = purged_train_val_test_split(
        len(aligned_index), cfg.val_fraction, cfg.test_fraction, cfg.horizon, cfg.seq_len
    )
    values = features.to_numpy(dtype=np.float32)
    targets = target.to_numpy()
    train_ds = FXSequenceDataset(values, targets, cfg.seq_len, aligned_index, cfg.seq_len - 1, split.train_end)
    val_ds = FXSequenceDataset(values, targets, cfg.seq_len, aligned_index, split.val_start, split.val_end)
    test_ds = FXSequenceDataset(values, targets, cfg.seq_len, aligned_index, split.test_start, split.test_end)
    return train_ds, val_ds, test_ds


def save_features_csv(
    features_by_symbol: dict[str, pd.DataFrame], targets_by_symbol: dict[str, pd.Series], path: str
) -> None:
    """Writes every symbol's normalized model-input features (plus its target) to one long-format CSV.

    Columns: symbol, date, <feature columns>, target — one row per
    (symbol, date), i.e. exactly the values fed into `FXSequenceDataset`
    before windowing into sequences. `target` is NaN for the last `horizon`
    days of each symbol, where no forward-looking target exists yet. In
    `target_mode="class"`, an extra `target_label` column (e.g. "bullish")
    is added; `target_mode="zscore"` has no discrete label to show.
    """
    rows = []
    for symbol, features in features_by_symbol.items():
        raw_target = targets_by_symbol[symbol]
        is_class_target = pd.api.types.is_integer_dtype(raw_target)  # check before reindex NaN-upcasts to float

        row = features.copy()
        row["target"] = raw_target.reindex(row.index)
        if is_class_target:
            row["target_label"] = row["target"].map(lambda c: CLASS_NAMES[int(c)] if pd.notna(c) else None)
        row.insert(0, "symbol", symbol)
        row.index.name = "date"
        rows.append(row.reset_index())

    combined = pd.concat(rows, ignore_index=True)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path_obj, index=False)
    logger.info("Saved input features to %s: %s", path_obj, combined.shape)


def build_pooled_datasets(
    cfg: Config, raw: RawPanels | None = None,
) -> tuple[
    ConcatDataset,
    ConcatDataset,
    ConcatDataset,
    dict[str, FXSequenceDataset],
    dict[str, FXSequenceDataset],
    dict[str, FXSequenceDataset],
    dict[str, pd.DataFrame],
    int,
]:
    """Builds one (train, val) dataset per symbol — that symbol's own normalized
    features (return, intraday_vol, momentum, rvol) paired with that same
    symbol's own N-day-ahead target — then pools every symbol's pieces into
    one shared train set and one shared val set.

    Metrics and z-score normalization are computed per symbol, independently —
    each symbol's rolling mean/std only ever sees that symbol's own history, so
    one pair's missing dates or volatility regime can't leak into another's
    scale. Every symbol's dataset uses the identical feature schema (same
    column set, same normalization scheme), so a single shared-weight model
    can be trained across all of them: each training sample's input `x` is
    one symbol's own recent return/vol/momentum history, and the model is
    never told which symbol it came from. Each symbol keeps its own causal
    time-ordered train/val split (no lookahead across the split boundary).
    Pooling happens afterwards, by concatenating every symbol's train pieces
    into one set and every symbol's val pieces into another. Also returns the
    per-symbol train, val *and* test datasets (unpooled) so predictions can
    be inspected and plotted per instrument — in-sample (train), development
    out-of-sample (val), and a final untouched holdout (test) — plus a
    per-symbol `market_data_by_symbol` DataFrame (indexed by date, covering
    every split at once — the caller reindexes to whichever split's dates it
    needs, e.g. via `FXSequenceDataset.sample_dates`) with:

    - `next_log_return`: the actual next-day executable log return (not a
      z-score) — the real "return" leg for the execution-aware backtest
      (`log_and_plot_strategy_pnl`, `fx_forecasting.trading.net_strategy_returns`).
    - `daily_carry`: that day's interest-rate carry, converted from an
      annualized percentage to a daily rate — the backtest's carry leg.
    - `horizon_zscore`: the same `horizon`-day return z-score `target`
      itself is (or is discretized from, under `target_mode="class"`) —
      used by `log_hit_rate_summary` so the "class" hit rate can be judged
      against the real continuous outcome rather than requiring an exact
      class match (see `compute_hit_rate`).

    Train/val/test are chronological and *purged*: `fx_forecasting.trading.
    purged_train_val_test_split` embargoes `horizon` origins around each
    split boundary, since a label at origin `t` looks `horizon` days ahead —
    without the embargo, the last training origins before validation (and
    the last validation origins before test) would have labels drawn partly
    from the next split's price history. The test split is meant to be
    evaluated only once, after all tuning is done on train/val.

    `target_symbol` (if set) restricts this to a *single* pair's dataset:
    `pairs` still supplies the full input universe (cross-asset `xret_*`
    returns, carry, CMA, cross-sectional features all still cover every pair
    in `pairs`), but only `target_symbol` is actually forecast — useful when
    the goal is one specific pair's forecast informed by the whole universe's
    context, rather than a shared model pooled across every pair's own target.

    `raw`: pre-fetched `RawPanels` to reuse instead of hitting Postgres again
    (see `load_raw_panels`) — e.g. across many hyperparameter-search trials
    that only vary feature/model hyperparameters, not `pairs`/`years`.
    Fetched fresh if not given.
    """
    if raw is None:
        raw = load_raw_panels(cfg)
    panel, high, low, rates = raw.panel, raw.high, raw.low, raw.rates

    if cfg.target_symbol is not None and cfg.target_symbol not in cfg.pairs:
        raise ValueError(f"target_symbol={cfg.target_symbol!r} must be one of pairs={cfg.pairs}")
    symbols_to_build = [cfg.target_symbol] if cfg.target_symbol is not None else list(cfg.pairs)

    log_returns = compute_log_returns(panel)
    engineered = compute_engineered_features(panel, log_returns, high, low, cfg.momentum_window, cfg.vol_window)
    carry = compute_carry_feature(cfg.pairs, rates)
    cma = CrossingMovingAverages(cfg.cma_windows).compute(panel)
    cross_mean_return = log_returns.mean(axis=1).rename("cross_mean_return")
    cross_return_rank = log_returns.rank(axis=1, pct=True)
    cross_carry_rank = carry.rank(axis=1, pct=True)

    train_datasets = []
    train_datasets_by_symbol: dict[str, FXSequenceDataset] = {}
    val_datasets: dict[str, FXSequenceDataset] = {}
    test_datasets: dict[str, FXSequenceDataset] = {}
    normalized_by_symbol: dict[str, pd.DataFrame] = {}
    target_by_symbol: dict[str, pd.Series] = {}
    market_data_by_symbol: dict[str, pd.DataFrame] = {}
    num_factors: int | None = None

    for symbol in symbols_to_build:
        symbol_cma = cma[[f"{symbol}_{w.name}" for w in cfg.cma_windows]]
        symbol_cma.columns = [w.name for w in cfg.cma_windows]

        # One column per pair's own daily log return (including `symbol` itself,
        # which duplicates the `return` column below) — gives the model direct
        # access to every other pair's actual return, not just the coarse
        # cross-sectional summaries above, so it can learn asset-specific
        # co-movement/lead-lag structure. Every symbol's dataset gets the
        # identical set of `xret_{pair}` slots in the same order (`pairs`, not
        # `symbols_to_build`), so slot identity ("xret_EURUSD" always means
        # EURUSD's own return) is consistent regardless of which symbol is
        # "self" or whether `target_symbol` narrows `symbols_to_build` to one.
        # Raw (unbounded) return-scale series, so these go through the same
        # rolling z-score as `return` itself below, no special-casing needed.
        extra_features = pd.DataFrame(
            {"cross_mean_return": cross_mean_return}
            | {f"xret_{pair}": log_returns[pair] for pair in sorted(cfg.pairs)}
        )
        frame = build_symbol_frame(
            symbol,
            log_returns,
            engineered,
            cfg.momentum_window,
            cfg.vol_window,
            carry=carry[symbol],
            cma=symbol_cma,
            extra_features=extra_features,
        )
        normalized = compute_rolling_normalized_features(frame, cfg.seq_len)

        # The two rank features are already bounded/comparable by construction
        # (percentile rank across pairs at each date) and, with few pairs or a
        # carry differential that changes rarely, can stay literally constant
        # for stretches far longer than `seq_len` — a rolling window entirely
        # inside such a stretch has std=0, which `compute_rolling_normalized_features`
        # turns into NaN and then drops via `dropna(how="any")`, silently
        # zeroing out the whole symbol. Added post-normalization instead
        # (recentered to ~[-0.5, 0.5]).
        normalized["cross_return_rank"] = (cross_return_rank[symbol] - 0.5).reindex(normalized.index)
        normalized["cross_carry_rank"] = (cross_carry_rank[symbol] - 0.5).reindex(normalized.index)

        if cfg.target_symbol is None:
            # Static identity indicators make the shared model pair-aware without
            # contaminating the time-series normalisation of dynamic features.
            # Skipped when forecasting a single `target_symbol`: with only one
            # symbol ever "self", this one-hot would just be a constant column.
            for pair in cfg.pairs:
                normalized[f"pair_{pair}"] = float(pair == symbol)

        # Move the xret_* block to the trailing len(cfg.pairs) columns, regardless of
        # where they landed above — gives the model (see `cross_asset_pca_dim`,
        # `CrossAssetFactorProjection`) a stable, predictable slice to reduce, without
        # it needing any column-name awareness of its own.
        xret_columns = [f"xret_{pair}" for pair in sorted(cfg.pairs)]
        other_columns = [c for c in normalized.columns if c not in xret_columns]
        normalized = normalized[other_columns + xret_columns]
        num_factors = normalized.shape[1]

        if cfg.target_mode == "class":
            target = compute_target_class(panel, log_returns, symbol, cfg.horizon, cfg.seq_len)
        else:
            target = compute_target_zscore(panel, log_returns, symbol, cfg.horizon, cfg.seq_len)
        normalized_by_symbol[symbol] = normalized
        target_by_symbol[symbol] = target

        train_ds, val_ds, test_ds = build_datasets(normalized, target, cfg)
        train_datasets.append(train_ds)
        train_datasets_by_symbol[symbol] = train_ds
        val_datasets[symbol] = val_ds
        test_datasets[symbol] = test_ds
        aligned_index = normalized.index.intersection(target.index)
        market_data_by_symbol[symbol] = pd.DataFrame(
            {
                "next_log_return": log_returns[symbol].shift(-1).reindex(aligned_index),
                "daily_carry": (carry[symbol] / 100.0 / 252.0).reindex(aligned_index),
                "horizon_zscore": compute_target_zscore(panel, log_returns, symbol, cfg.horizon, cfg.seq_len).reindex(aligned_index),
            },
            index=aligned_index,
        )

        logger.info(
            "%s: %d factors, train=%d, val=%d, test=%d", symbol, num_factors, len(train_ds), len(val_ds), len(test_ds)
        )

    if cfg.write_features_csv:
        save_features_csv(normalized_by_symbol, target_by_symbol, cfg.features_csv_path)

    pooled_train = ConcatDataset(train_datasets)
    pooled_val = ConcatDataset(list(val_datasets.values()))
    pooled_test = ConcatDataset(list(test_datasets.values()))
    logger.info(
        "Pooled across %d symbols: %d factors, train=%d, val=%d, test=%d",
        len(symbols_to_build), num_factors, len(pooled_train), len(pooled_val), len(pooled_test),
    )
    return (
        pooled_train,
        pooled_val,
        pooled_test,
        train_datasets_by_symbol,
        val_datasets,
        test_datasets,
        market_data_by_symbol,
        num_factors,
    )


def build_model(cfg: Config, num_factors: int) -> LSTMAttentionForecaster:
    output_size = len(CLASS_NAMES) if cfg.target_mode == "class" else 1
    num_cross_asset_factors = len(cfg.pairs)
    if cfg.cross_asset_pca_dim is not None and not 0 < cfg.cross_asset_pca_dim < num_cross_asset_factors:
        raise ValueError(
            f"cross_asset_pca_dim={cfg.cross_asset_pca_dim} must be strictly between 0 and "
            f"len(pairs)={num_cross_asset_factors}"
        )
    model = LSTMAttentionForecaster(
        num_factors=num_factors,
        output_size=output_size,
        forecast_horizon=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        predict_uncertainty=cfg.predict_uncertainty,
        use_spectral_features=cfg.use_spectral_features,
        spectral_embedding_dim=cfg.spectral_embedding_dim,
        spectral_freq_bins=cfg.spectral_freq_bins,
        num_cross_asset_factors=num_cross_asset_factors,
        cross_asset_pca_dim=cfg.cross_asset_pca_dim,
    ).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    target_desc = f"{len(CLASS_NAMES)}-class direction" if cfg.target_mode == "class" else "z-score regression"
    logger.info(
        "Model: %d factors (per symbol, shared weights across symbols), %d-day %s target, %d params",
        num_factors, cfg.horizon, target_desc, n_params,
    )
    return model


def save_model(model: LSTMAttentionForecaster, cfg: Config, num_factors: int, path: str) -> None:
    """Saves weights plus the architecture/config needed to reconstruct the model for inference."""
    checkpoint = {
        "model_state": model.state_dict(),
        "num_factors": num_factors,
        "output_size": model.output_size,
        "target_mode": cfg.target_mode,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "predict_uncertainty": cfg.predict_uncertainty,
        "use_spectral_features": cfg.use_spectral_features,
        "spectral_embedding_dim": cfg.spectral_embedding_dim,
        "spectral_freq_bins": cfg.spectral_freq_bins,
        "cross_asset_pca_dim": cfg.cross_asset_pca_dim,
        "pairs": cfg.pairs,
        "target_symbol": cfg.target_symbol,
        "horizon": cfg.horizon,
        "seq_len": cfg.seq_len,
        "momentum_window": cfg.momentum_window,
        "vol_window": cfg.vol_window,
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path_obj)
    logger.info("Saved model checkpoint to %s", path_obj)


def load_model(path: str, device: str) -> tuple[LSTMAttentionForecaster, dict]:
    """Loads a checkpoint saved by `save_model` and rebuilds the model architecture around it."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LSTMAttentionForecaster(
        num_factors=checkpoint["num_factors"],
        output_size=checkpoint["output_size"],
        forecast_horizon=1,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        predict_uncertainty=checkpoint.get("predict_uncertainty", False),
        use_spectral_features=checkpoint.get("use_spectral_features", False),
        spectral_embedding_dim=checkpoint.get("spectral_embedding_dim", 8),
        spectral_freq_bins=checkpoint.get("spectral_freq_bins", 16),
        num_cross_asset_factors=len(checkpoint["pairs"]),
        cross_asset_pca_dim=checkpoint.get("cross_asset_pca_dim"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logger.info(
        "Loaded model checkpoint from %s (pairs=%s, horizon=%s, seq_len=%s, target_mode=%s)",
        path, checkpoint["pairs"], checkpoint["horizon"], checkpoint["seq_len"], checkpoint["target_mode"],
    )
    return model, checkpoint


def describe_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    """One-shot printout of a training batch, meant to be eyeballed or breakpointed on."""
    logger.info("Batch x: shape=%s dtype=%s mean=%.4f std=%.4f", tuple(x.shape), x.dtype, x.mean().item(), x.std().item())
    # y holds integer class indices; torch.mean/std require a floating dtype.
    logger.info(
        "Batch y: shape=%s dtype=%s mean=%.4f std=%.4f", tuple(y.shape), y.dtype, y.float().mean().item(), y.float().std().item()
    )


def class_distance_weights(outlier_weight: float, num_classes: int) -> torch.Tensor:
    """Per-class weights for CrossEntropyLoss, scaled by distance from the neutral (middle) class.

    weight(class) = 1 + outlier_weight * |class - neutral| / neutral, ranging
    from 1 at the neutral class (an unremarkable, near-median day) up to
    1 + outlier_weight at the most extreme classes (very_bearish /
    very_bullish — large realized moves). This pushes training to prioritize
    correctly calling large moves over minimizing error on the far more
    common near-neutral days, which is what actually matters for a trading
    strategy. `outlier_weight=0` recovers plain (unweighted) cross-entropy.
    """
    neutral = (num_classes - 1) / 2.0
    distances = torch.arange(num_classes, dtype=torch.float32) - neutral
    return 1.0 + outlier_weight * (distances.abs() / neutral)


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, outlier_weight: float) -> torch.Tensor:
    """MSE weighted so targets far from neutral (z=0) count more.

    The z-score target is mapped through the standard normal CDF to get a
    bounded (0,1) "quantile" purely for weighting purposes (the regression
    target itself stays the raw unbounded z-score): weight(z) = 1 +
    outlier_weight * |2*Phi(z) - 1|, ranging from 1 near the median (z≈0) up
    to 1 + outlier_weight at the extremes — the continuous-space analogue of
    `class_distance_weights`. `outlier_weight=0` recovers plain MSE.
    """
    phi = 0.5 * (1.0 + torch.erf(target / math.sqrt(2.0)))
    weight = 1.0 + outlier_weight * (2.0 * phi - 1.0).abs()
    return (weight * (pred - target) ** 2).mean()


def variance_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalizes the batch-level gap between predicted and actual standard deviation.

    Plain MSE has no incentive to keep any prediction spread once it's found
    that hedging near the target's mean (z≈0) minimizes average error under a
    noisy/weak-signal target — the classic regression-to-the-mean collapse.
    This term directly punishes that: `(std(pred) - std(target))**2`, added
    to the MSE loss with weight `Config.variance_penalty_weight`, forces
    predictions to keep a spread comparable to the real target's, at the
    cost of some calibration. `unbiased=False` avoids NaN on a batch of size 1.
    """
    return (pred.std(unbiased=False) - target.std(unbiased=False)) ** 2


def gaussian_nll_loss(pred: torch.Tensor, log_variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log likelihood for a mean forecast and conditional uncertainty."""
    return 0.5 * (torch.exp(-log_variance) * (pred - target).square() + log_variance).mean()


def run_epoch(
    model: LSTMAttentionForecaster,
    loader: DataLoader,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Runs one pass over `loader`. Trains if `optimizer` is given, else evaluates.

    Returns (weighted loss, plain hit rate) — see `compute_hit_rate` for what
    "hit" means in each `cfg.target_mode`. Loss is weighted per
    `class_distance_weights` (`target_mode="class"`) or `weighted_mse_loss`
    plus `variance_penalty` (`target_mode="zscore"`) — outlier-focused and
    collapse-resistant respectively, together the actual optimization
    target; hit rate stays a plain, unweighted diagnostic for comparability
    across runs.
    """
    is_train = optimizer is not None
    model.train(is_train)

    if cfg.target_mode == "class":
        class_weights = class_distance_weights(cfg.outlier_weight, model.output_size).to(cfg.device)
        ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    total_loss, total_hits, total_count = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        with torch.set_grad_enabled(is_train):
            output = model(x, return_uncertainty=cfg.predict_uncertainty and cfg.target_mode == "zscore")
            if cfg.target_mode == "class":
                loss = ce_loss_fn(output, y)
                hits = output.argmax(dim=-1) == y
            else:
                if cfg.predict_uncertainty:
                    mean, log_variance = output
                    pred = mean.squeeze(-1)
                    loss = gaussian_nll_loss(pred, log_variance.squeeze(-1), y)
                else:
                    pred = output.squeeze(-1)
                    loss = weighted_mse_loss(pred, y, cfg.outlier_weight)
                    if cfg.variance_penalty_weight:
                        loss = loss + cfg.variance_penalty_weight * variance_penalty(pred, y)
                hits = pred.sign() == y.sign()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_hits += hits.sum().item()
        total_count += batch_size

    n = len(loader.dataset)
    return total_loss / n, total_hits / total_count


def collect_predictions(
    model: LSTMAttentionForecaster, dataset: FXSequenceDataset, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over `dataset` in time order; returns (actual, predicted), shape (n,).

    `predicted` is the argmax class (`target_mode="class"`) or the raw
    predicted z-score (`target_mode="zscore"`), not raw logits.
    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model.eval()
    actual_batches, pred_batches = [], []
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(cfg.device)).cpu()
            actual_batches.append(y)
            pred_batches.append(output.argmax(dim=-1) if cfg.target_mode == "class" else output.squeeze(-1))

    actual = torch.cat(actual_batches).numpy()
    predicted = torch.cat(pred_batches).numpy()
    return actual, predicted


def collect_signal(model: LSTMAttentionForecaster, dataset: FXSequenceDataset, cfg: Config) -> np.ndarray:
    """Risk-adjusted trading signal s_t for every sample in `dataset`, time-ordered — the
    quantity `fx_forecasting.trading.positions_from_signal` sizes a position from.

    `cfg.predict_uncertainty=False`: the model's predicted return z-score is
    already risk-adjusted (its target is standardized by trailing realized
    volatility), so it's used directly as the signal.

    `cfg.predict_uncertainty=True`: divides the model's own predicted mean by
    its own predicted std (`exp(0.5 * log_variance)`), i.e. s_t = mu_hat/sigma_hat
    — a low-confidence prediction (wide predicted uncertainty) shrinks the
    signal even before thresholding/leverage are applied, rather than sizing
    a position purely off the point estimate.
    """
    if cfg.target_mode != "zscore":
        raise ValueError("collect_signal only supports target_mode='zscore'")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model.eval()
    signal_batches = []
    with torch.no_grad():
        for x, _y in loader:
            if cfg.predict_uncertainty:
                mean, log_variance = model(x.to(cfg.device), return_uncertainty=True)
                sigma = torch.exp(0.5 * log_variance).clamp_min(1e-3)
                signal = (mean / sigma).squeeze(-1)
            else:
                signal = model(x.to(cfg.device)).squeeze(-1)
            signal_batches.append(signal.cpu())
    return torch.cat(signal_batches).numpy()


def fit_signal_calibrators(
    model: LSTMAttentionForecaster, train_datasets: dict[str, FXSequenceDataset], cfg: Config
) -> dict[str, LinearCalibrator]:
    """Per symbol, an affine calibration (`fx_forecasting.trading.fit_linear_calibrator`)
    mapping the model's own raw training-set signal (`collect_signal`) onto what actually
    realized on the training set — corrects systematic over/under-scaling before the signal is
    used for position sizing (a raw model output minimizing a training loss has no guarantee of
    being correctly *scaled* for sizing, only correctly *ordered*). Fit only on the training
    fold, then reused unchanged for both validation and the final test holdout in
    `log_and_plot_strategy_pnl`, so no information from either split ever leaks into the
    calibration itself. No-op (empty dict) outside `target_mode="zscore"`.
    """
    if cfg.target_mode != "zscore":
        return {}
    calibrators: dict[str, LinearCalibrator] = {}
    for symbol, dataset in train_datasets.items():
        actual, _predicted = collect_predictions(model, dataset, cfg)
        signal = collect_signal(model, dataset, cfg)
        calibrators[symbol] = fit_linear_calibrator(signal, actual)
        logger.info(
            "  %-8s signal calibration: realized ~= %.4f + %.4f * signal", symbol,
            calibrators[symbol].intercept, calibrators[symbol].slope,
        )
    return calibrators


# Neutral band, in z-score units, that `compute_hit_rate` counts as a "hit"
# for a predicted neutral class — the same interior quantile band
# (`CLASS_QUANTILE_EDGES[2:4]` = 0.3..0.7) that `compute_target_class` itself
# uses to carve out the neutral class, just expressed in z-score space via
# the inverse normal CDF instead of quantile space.
_NEUTRAL_ZSCORE_LOW = float(norm.ppf(CLASS_QUANTILE_EDGES[2]))
_NEUTRAL_ZSCORE_HIGH = float(norm.ppf(CLASS_QUANTILE_EDGES[3]))


def compute_hit_rate(predicted: np.ndarray, actual_zscore: np.ndarray, target_mode: str) -> float:
    """Fraction of samples where the prediction "called it right".

    `target_mode="zscore"`: same-sign match (both predicted and actual
    z-score fall on the same side of zero) — the natural direction-call
    analogue for a continuous target.

    `target_mode="class"`: rather than requiring an exact predicted-class
    match (too strict — e.g. predicting "bullish" when the actual outcome is
    "very_bullish" is still a correct directional call), a hit is instead:
    a bearish prediction (`very_bearish`/`bearish`) with `actual_zscore < 0`,
    a bullish prediction (`bullish`/`very_bullish`) with `actual_zscore > 0`,
    or a neutral prediction with `actual_zscore` inside the neutral band
    (`_NEUTRAL_ZSCORE_LOW`..`_NEUTRAL_ZSCORE_HIGH`, matching the same band
    `compute_target_class` itself uses for the neutral class). `predicted`
    is still the discrete class index (argmax) in this mode — only the
    "actual" side is judged continuously, via `actual_zscore` (see
    `build_pooled_datasets`'s `market_data_by_symbol["horizon_zscore"]`), not
    the discretized class label.

    This is the metric a trading strategy actually cares about — distinct
    from (and often more informative than) the loss used to train/select the
    model. See `log_naive_baseline` for the real no-skill bar to compare
    against (not a flat 50%/20%, since classes are imbalanced by design).
    """
    if target_mode == "class":
        neutral_class = CLASS_NAMES.index("neutral")
        bearish_hit = (predicted < neutral_class) & (actual_zscore < 0)
        bullish_hit = (predicted > neutral_class) & (actual_zscore > 0)
        neutral_hit = (predicted == neutral_class) & (actual_zscore >= _NEUTRAL_ZSCORE_LOW) & (
            actual_zscore <= _NEUTRAL_ZSCORE_HIGH
        )
        return float(np.mean(bearish_hit | bullish_hit | neutral_hit))
    return float(np.mean(np.sign(predicted) == np.sign(actual_zscore)))


def log_hit_rate_summary(
    model: LSTMAttentionForecaster,
    datasets: dict[str, FXSequenceDataset],
    market_data_by_symbol: dict[str, pd.DataFrame],
    cfg: Config,
    label: str = "validation",
) -> None:
    """Logs the hit rate (see `compute_hit_rate`), per symbol and pooled across all of them.

    `datasets` can be any per-symbol `FXSequenceDataset` dict from
    `build_pooled_datasets` (validation or the final test holdout); `label`
    is just for the log line. Compare against `log_naive_baseline`'s
    baseline accuracy for the real no-skill bar.
    """
    if cfg.target_mode == "class":
        metric = "bearish/bullish sign match, neutral band match — see compute_hit_rate"
    else:
        metric = "sign(predicted) == sign(actual)"
    logger.info("Hit rate (%s, %s):", label, metric)
    all_predicted, all_zscore = [], []
    for symbol, dataset in datasets.items():
        actual, predicted = collect_predictions(model, dataset, cfg)
        if cfg.target_mode == "class":
            actual_zscore = market_data_by_symbol[symbol]["horizon_zscore"].reindex(dataset.sample_dates).to_numpy()
        else:
            actual_zscore = actual
        hit_rate = compute_hit_rate(predicted, actual_zscore, cfg.target_mode)
        logger.info("  %-8s hit_rate=%5.1f%%  (n=%d)", symbol, hit_rate * 100, len(actual))
        all_predicted.append(predicted)
        all_zscore.append(actual_zscore)

    pooled_hit_rate = compute_hit_rate(np.concatenate(all_predicted), np.concatenate(all_zscore), cfg.target_mode)
    logger.info(
        "  %-8s hit_rate=%5.1f%%  (n=%d, pooled across %d symbols)",
        "ALL", pooled_hit_rate * 100, sum(len(a) for a in all_predicted), len(datasets),
    )


def smooth_weights(weights: np.ndarray, window: int) -> np.ndarray:
    """Smooths per-period position weights with a trailing moving average of `window` periods.

    The target is an N-day-ahead (N=`horizon`) cumulative return, and the
    validation set advances one day at a time, so consecutive raw
    predictions are for heavily overlapping horizons — day-to-day changes in
    the raw prediction are dominated by noise rather than genuine new
    information. Averaging the position weight over that same `horizon`
    window turns it into a slower-turning position size. Causal
    (`min_periods=1`, no lookahead) and keeps the same length as `weights`.
    """
    return pd.Series(weights).rolling(window=window, min_periods=1).mean().to_numpy()


def compute_positions_and_pnl(
    model: LSTMAttentionForecaster,
    dataset: FXSequenceDataset,
    symbol: str,
    market: pd.DataFrame,
    calibrators: dict[str, LinearCalibrator],
    cfg: Config,
) -> tuple[pd.Series, pd.DataFrame]:
    """The full signal -> calibration -> smoothing -> position-sizing -> net-PnL pipeline for
    one symbol (shared by `log_and_plot_strategy_pnl` and `plot_positions_and_pnl`) — see
    `log_and_plot_strategy_pnl`'s docstring for what each stage does and why.

    Returns `(positions, result)`: `positions` is the raw position-size series (indexed by
    `dataset.sample_dates`); `result` is `net_strategy_returns`'s per-day
    turnover/cost/gross/net DataFrame, indexed by whichever of those dates actually had a
    next-day return to trade against (a subset of `positions`' own index — see
    `net_strategy_returns`'s `dropna()`).
    """
    raw_signal = collect_signal(model, dataset, cfg)
    calibrated_signal = calibrators[symbol].transform(raw_signal) if symbol in calibrators else raw_signal
    signal = smooth_weights(calibrated_signal, cfg.horizon)
    positions = pd.Series(
        positions_from_signal(signal, cfg.signal_leverage, cfg.signal_threshold, cfg.max_position),
        index=dataset.sample_dates,
    )
    result = net_strategy_returns(positions, market["next_log_return"], market["daily_carry"], cfg.transaction_cost_bps)
    return positions, result


def log_and_plot_strategy_pnl(
    model: LSTMAttentionForecaster,
    datasets: dict[str, FXSequenceDataset],
    market_data_by_symbol: dict[str, pd.DataFrame],
    calibrators: dict[str, LinearCalibrator],
    cfg: Config,
    path: str,
    sample_label: str = "out-of-sample (validation)",
) -> None:
    """Execution-aware backtest: an actual, date-aligned, cost-and-carry-aware net PnL, not a
    z-score-vs-z-score diagnostic.

    Position sizing (`fx_forecasting.trading.positions_from_signal`):
    `w_t = clip(leverage * s_t, -max_position, max_position) * 1(|s_t| > threshold)`, where
    `s_t` is the model's own risk-adjusted signal (`collect_signal`), affine-calibrated against
    what actually realized on the *training* fold only (`calibrators`, see
    `fit_signal_calibrators` — never re-fit here, so nothing from this split leaks into the
    calibration), then smoothed over `cfg.horizon` days (`smooth_weights`) to damp the turnover
    the target's own overlapping-horizon noise would otherwise cause. Below `signal_threshold`
    the position is flat (0) rather than a small noisy bet — trading only when the model claims
    enough edge to plausibly clear costs.

    PnL (`fx_forecasting.trading.net_strategy_returns`): `gross_t = w_{t-1} * r_t` using the
    *actual* next-day executable log return (`market_data_by_symbol[symbol]["next_log_return"]`,
    not a z-scored proxy), plus `w_{t-1} * carry_t`, minus `cost * |w_t - w_{t-1}|` — so turnover
    and financing both show up as a real drag rather than being ignored. Per-symbol results are
    joined by date (`pd.concat(..., axis=1)`), not truncated to the shortest symbol's length, so
    the pooled portfolio only ever averages days where a symbol actually has a position.

    Only meaningful for `target_mode="zscore"` (a continuous signal to size a position with);
    skipped for `target_mode="class"`.

    Also writes a `date, symbol, net_cumulative` sidecar CSV next to `path`
    (same name, `.csv` extension), pooled series included under
    `symbol="ALL"` — `dashboard.py` reads this for an interactive
    (zoomable, hoverable) version instead of just embedding this static PNG.
    """
    if cfg.target_mode != "zscore":
        logger.info("Skipping strategy PnL backtest: requires target_mode='zscore' (got %r).", cfg.target_mode)
        return

    symbols = list(datasets.keys())
    net_by_symbol: dict[str, pd.DataFrame] = {}

    logger.info(
        "Execution-aware backtest (%s): leverage=%.2f threshold=%.2f max_position=%.2f cost=%.1fbps:",
        sample_label, cfg.signal_leverage, cfg.signal_threshold, cfg.max_position, cfg.transaction_cost_bps,
    )
    for symbol in symbols:
        _positions, result = compute_positions_and_pnl(
            model, datasets[symbol], symbol, market_data_by_symbol[symbol], calibrators, cfg
        )
        net_by_symbol[symbol] = result
        sharpe = annualized_sharpe(result["net"])
        logger.info(
            "  %-8s net_total=%+.4f  net_mean=%+.6f  ann_sharpe=%6.2f  mean_turnover=%.3f  (n=%d)",
            symbol, result["net"].sum(), result["net"].mean(), sharpe, result["turnover"].mean(), len(result),
        )

    # Date-aligned equal-weight pool: an outer join across symbols' own dates (each symbol's
    # window can differ slightly), missing days contribute 0 rather than truncating every
    # symbol down to the shortest one's length.
    pooled_net = pd.concat({s: r["net"] for s, r in net_by_symbol.items()}, axis=1, sort=True).fillna(0.0).mean(axis=1)
    pooled_sharpe = annualized_sharpe(pooled_net)
    logger.info(
        "  %-8s net_total=%+.4f  net_mean=%+.6f  ann_sharpe=%6.2f  (n=%d, date-aligned equal-weighted across %d symbols)",
        "ALL", pooled_net.sum(), pooled_net.mean(), pooled_sharpe, len(pooled_net), len(symbols),
    )

    fig, axes = plt.subplots(len(symbols) + 1, 1, figsize=(12, 3 * (len(symbols) + 1)), sharex=False, squeeze=False)

    for i, symbol in enumerate(symbols):
        ax = axes[i, 0]
        result = net_by_symbol[symbol]
        ax.plot(result.index, result["net"].cumsum(), linewidth=1.2)
        ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.set_ylabel(f"{symbol}\nnet cum. PnL")

    ax = axes[-1, 0]
    ax.plot(pooled_net.index, pooled_net.cumsum(), label="pooled net PnL (equal-weighted, cost-aware)", linewidth=1.5, color="tab:blue")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_ylabel("Pooled net cum. PnL")
    ax.legend(loc="upper left", fontsize=8)

    axes[-1, 0].set_xlabel(f"{sample_label} date")
    fig.suptitle(f"Execution-aware backtest ({sample_label}): net PnL after transaction costs and carry")
    fig.tight_layout()

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_obj, dpi=150)
    plt.close(fig)
    logger.info("Saved %s strategy PnL plot to %s", sample_label, path_obj)

    csv_rows = [
        pd.DataFrame({"date": r.index, "symbol": symbol, "net_cumulative": r["net"].cumsum().to_numpy()})
        for symbol, r in net_by_symbol.items()
    ]
    csv_rows.append(pd.DataFrame({"date": pooled_net.index, "symbol": "ALL", "net_cumulative": pooled_net.cumsum().to_numpy()}))
    csv_path = path_obj.with_suffix(".csv")
    pd.concat(csv_rows, ignore_index=True).to_csv(csv_path, index=False)
    logger.info("Saved %s strategy PnL data to %s", sample_label, csv_path)


def plot_positions_and_pnl(
    model: LSTMAttentionForecaster,
    datasets: dict[str, FXSequenceDataset],
    market_data_by_symbol: dict[str, pd.DataFrame],
    calibrators: dict[str, LinearCalibrator],
    cfg: Config,
    path: str,
    sample_label: str = "out-of-sample (validation)",
) -> None:
    """One panel per symbol (all of `datasets` — just the target symbol under
    `cfg.target_symbol`, every pooled symbol otherwise): the position size and that symbol's
    own actual daily return on the left axis, cumulative PnL with and without transaction
    costs on a secondary (right) axis.

    Position and daily return share the left axis (both live on comparable, small
    magnitudes) so their relationship is directly visible — does the model's position track
    the sign of what actually happens next. The two PnL curves isolate exactly what
    transaction costs are doing to the strategy: "PnL (no transaction costs)" is
    `net + cost` (i.e. `gross + position * carry`, `net_strategy_returns`'s own `net` before
    its `cost` term is subtracted back out — carry is kept in both curves, since it isn't a
    transaction cost), so the gap between the two curves at any point is exactly the
    cumulative cost paid for turnover up to then.

    Shares `compute_positions_and_pnl` with `log_and_plot_strategy_pnl` — same signal,
    calibration, smoothing, and position sizing, so this is a detail view of the same
    backtest, not a separate one. Only meaningful for `target_mode="zscore"`; skipped for
    `target_mode="class"`.

    Also writes a `date, symbol, position, daily_return, pnl_with_costs, pnl_without_costs`
    sidecar CSV next to `path` (same name, `.csv` extension) — `dashboard.py` reads this for
    an interactive version instead of just embedding this static PNG.
    """
    if cfg.target_mode != "zscore":
        logger.info("Skipping positions/PnL plot: requires target_mode='zscore' (got %r).", cfg.target_mode)
        return

    symbols = list(datasets.keys())
    fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 3.5 * len(symbols)), sharex=False, squeeze=False)
    csv_rows = []

    for i, symbol in enumerate(symbols):
        positions, result = compute_positions_and_pnl(
            model, datasets[symbol], symbol, market_data_by_symbol[symbol], calibrators, cfg
        )
        aligned_positions = positions.reindex(result.index)
        daily_return = market_data_by_symbol[symbol]["next_log_return"].reindex(result.index)
        pnl_with_costs = result["net"].cumsum()
        pnl_without_costs = (result["net"] + result["cost"]).cumsum()

        ax = axes[i, 0]
        ax.plot(result.index, aligned_positions, label="position", color="tab:blue", drawstyle="steps-post", linewidth=1.1)
        ax.plot(result.index, daily_return, label="daily return", color="tab:grey", linewidth=0.8, alpha=0.8)
        ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.set_ylabel(f"{symbol}\nposition / daily return")

        ax2 = ax.twinx()
        ax2.plot(result.index, pnl_with_costs, label="PnL (with transaction costs)", color="tab:green", linewidth=1.4)
        ax2.plot(
            result.index, pnl_without_costs, label="PnL (no transaction costs)",
            color="tab:orange", linewidth=1.4, linestyle="--",
        )
        ax2.set_ylabel("cum. PnL")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        csv_rows.append(
            pd.DataFrame(
                {
                    "date": result.index,
                    "symbol": symbol,
                    "position": aligned_positions.to_numpy(),
                    "daily_return": daily_return.to_numpy(),
                    "pnl_with_costs": pnl_with_costs.to_numpy(),
                    "pnl_without_costs": pnl_without_costs.to_numpy(),
                }
            )
        )

    axes[-1, 0].set_xlabel(f"{sample_label} date")
    fig.suptitle(f"Positions & PnL ({sample_label}): position vs. daily return; cumulative PnL with/without transaction costs")
    fig.tight_layout()

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_obj, dpi=150)
    plt.close(fig)
    logger.info("Saved %s positions/PnL plot to %s", sample_label, path_obj)

    csv_path = path_obj.with_suffix(".csv")
    pd.concat(csv_rows, ignore_index=True).to_csv(csv_path, index=False)
    logger.info("Saved %s positions/PnL data to %s", sample_label, csv_path)


def collect_targets(dataset: Dataset) -> np.ndarray:
    """All targets in `dataset`, in DataLoader iteration order (for baseline comparisons)."""
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
    targets = [y for _, y in loader]
    return torch.cat(targets).squeeze(-1).numpy()


def log_naive_baseline(train_ds: ConcatDataset, val_ds: ConcatDataset, cfg: Config, output_size: int) -> None:
    """Logs the score of a zero-skill baseline, so the epoch logs below have a bar to clear.

    `target_mode="class"`: weighted cross-entropy / accuracy of always
    predicting the train-set's empirical class-frequency distribution /
    majority class (see `class_distance_weights`) — classes are
    intentionally imbalanced (10/20/40/20/10 split), so this isn't a flat
    1/num_classes chance.

    `target_mode="zscore"`: weighted MSE + variance penalty (see
    `weighted_mse_loss`, `variance_penalty`) / directional hit rate of always
    predicting the train-set's mean z-score. Since that's a constant
    prediction, `variance_penalty` evaluates to `target.std()**2` here —
    showing exactly how hard the penalty pushes against this degenerate
    "collapse to the mean" solution.

    Both branches derive the baseline only from the train-set's empirical
    target distribution (never looking at features), and use the same
    outlier/variance weighting as training, so they're directly comparable
    to the epoch logs below.
    """
    train_targets = collect_targets(train_ds)
    val_targets = collect_targets(val_ds)

    if cfg.target_mode == "class":
        train_targets = train_targets.astype(int)
        val_targets = val_targets.astype(int)

        class_counts = np.bincount(train_targets, minlength=output_size)
        class_probs = np.clip(class_counts / class_counts.sum(), 1e-8, 1.0)
        majority_class = int(class_counts.argmax())
        weights = class_distance_weights(cfg.outlier_weight, output_size).numpy()

        def weighted_ce(targets: np.ndarray) -> float:
            return float(np.mean(weights[targets] * -np.log(class_probs[targets])))

        def majority_accuracy(targets: np.ndarray) -> float:
            return float(np.mean(targets == majority_class))

        logger.info(
            "Naive baseline class distribution (train set): %s",
            {CLASS_NAMES[i]: f"{p:.1%}" for i, p in enumerate(class_probs)},
        )
        logger.info(
            "Naive baseline (predict train-set class distribution; majority class=%s): "
            "train_loss=%.5f train_acc=%5.1f%%  val_loss=%.5f val_acc=%5.1f%%",
            CLASS_NAMES[majority_class],
            weighted_ce(train_targets), majority_accuracy(train_targets) * 100,
            weighted_ce(val_targets), majority_accuracy(val_targets) * 100,
        )
        return

    baseline_pred = float(train_targets.mean())

    def weighted_loss(targets: np.ndarray) -> float:
        t = torch.as_tensor(targets, dtype=torch.float32)
        p = torch.full_like(t, baseline_pred)
        # A constant predictor has pred.std()==0, so this term evaluates to
        # target.std()**2 — showing exactly how much variance_penalty
        # punishes the degenerate "predict the mean everywhere" solution
        # this baseline represents, matching what run_epoch actually optimizes.
        loss = weighted_mse_loss(p, t, cfg.outlier_weight)
        loss = loss + cfg.variance_penalty_weight * variance_penalty(p, t)
        return loss.item()

    def directional_accuracy(targets: np.ndarray) -> float:
        return float(np.mean(np.sign(targets) == np.sign(baseline_pred)))

    logger.info(
        "Naive baseline (always predict train-set mean z-score=%.4f): "
        "train_loss=%.5f train_acc=%5.1f%%  val_loss=%.5f val_acc=%5.1f%%",
        baseline_pred,
        weighted_loss(train_targets), directional_accuracy(train_targets) * 100,
        weighted_loss(val_targets), directional_accuracy(val_targets) * 100,
    )


def plot_predictions_grid(
    model: LSTMAttentionForecaster,
    datasets: dict[str, FXSequenceDataset],
    cfg: Config,
    path: str,
    sample_label: str = "out-of-sample (validation)",
) -> None:
    """One panel per symbol: actual vs. predicted N-day target over time.

    `datasets` can be either the out-of-sample (validation) or in-sample
    (training) per-symbol datasets from `build_pooled_datasets` — comparing
    the two is the real test of whether the pooled/shared model learned
    generic FX distributional dynamics or just memorized the training set
    (in-sample looking much better than out-of-sample is the overfitting tell).

    Also writes a `date, symbol, actual, predicted` sidecar CSV next to `path`
    (same name, `.csv` extension) with the exact data the plot was built
    from, real dates via `FXSequenceDataset.sample_dates` — `dashboard.py`
    reads this to render an interactive (zoomable, hoverable) version instead
    of just embedding this static PNG.
    """
    symbols = list(datasets.keys())
    fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 3 * len(symbols)), sharex=False, squeeze=False)
    csv_rows = []

    for i, symbol in enumerate(symbols):
        actual, predicted = collect_predictions(model, datasets[symbol], cfg)
        csv_rows.append(
            pd.DataFrame(
                {"date": datasets[symbol].sample_dates, "symbol": symbol, "actual": actual, "predicted": predicted}
            )
        )
        ax = axes[i, 0]
        ax.plot(actual, label="actual", linewidth=1.0, marker=".", markersize=3)
        ax.plot(predicted, label="predicted", linewidth=1.0, alpha=0.8, marker=".", markersize=3)

        if cfg.target_mode == "class":
            num_classes = len(CLASS_NAMES)
            ax.axhline((num_classes - 1) / 2.0, color="grey", linewidth=0.5)
            ax.set_ylim(-0.5, num_classes - 0.5)
            ax.set_yticks(range(num_classes))
            ax.set_yticklabels(CLASS_NAMES, fontsize=7)
            ax.set_ylabel(symbol)
        else:
            ax.axhline(0.0, color="grey", linewidth=0.5)
            ax.set_ylabel(f"{symbol}\n(z-score)")

        ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel(f"{sample_label} sample (time-ordered, per symbol)")
    target_desc = "direction class" if cfg.target_mode == "class" else "return z-score"
    fig.suptitle(f"Pooled model ({sample_label}): actual vs. predicted {cfg.horizon}-day {target_desc}")
    fig.tight_layout()

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_obj, dpi=150)
    plt.close(fig)
    logger.info("Saved %s prediction plot to %s", sample_label, path_obj)

    csv_path = path_obj.with_suffix(".csv")
    pd.concat(csv_rows, ignore_index=True).to_csv(csv_path, index=False)
    logger.info("Saved %s prediction data to %s", sample_label, csv_path)


def train(model: LSTMAttentionForecaster, train_ds: ConcatDataset, val_ds: ConcatDataset, cfg: Config) -> float:
    """Trains `model` in place, then restores it to its best-val_loss epoch's weights.

    Given how weak the underlying signal is, the model reliably starts
    overfitting (train_loss keeps dropping while val_loss rises) well before
    `cfg.epochs` is reached, so the last epoch's weights are usually *not*
    what you want to ship — the best-val_loss checkpoint is tracked
    throughout and restored at the end. Also stops early once val_loss hasn't
    improved for `cfg.early_stop_patience` epochs, to avoid wasting compute
    once the model is clearly just memorizing the training set. Also stops
    (with the same best-checkpoint restore) if `CANCEL_EVENT` is set from
    outside — e.g. a "Stop" button driving `main()` in a background thread.

    Returns the best val_loss reached (the same value the restored
    checkpoint was selected on) — used by `optimize_hyperparams.py` as its
    optimization objective.
    """
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    describe_batch(*next(iter(train_loader)))
    log_naive_baseline(train_ds, val_ds, cfg, model.output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Adaptive LR: halves (by default) whenever val loss stops improving for
    # `lr_patience` epochs, instead of decaying on a fixed schedule.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    CANCEL_EVENT.clear()

    for epoch in range(1, cfg.epochs + 1):
        if CANCEL_EVENT.is_set():
            logger.info("Training cancelled by user at epoch %d/%d", epoch, cfg.epochs)
            break

        train_loss, train_acc = run_epoch(model, train_loader, cfg, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, cfg, optimizer=None)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "epoch %02d/%d  train_loss=%.5f train_acc=%5.1f%%  val_loss=%.5f val_acc=%5.1f%%  lr=%.2e",
            epoch, cfg.epochs, train_loss, train_acc * 100, val_loss, val_acc * 100, current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (no val_loss improvement for %d epochs)",
                    epoch, cfg.early_stop_patience,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best checkpoint: epoch %d/%d, val_loss=%.5f", best_epoch, cfg.epochs, best_val_loss)

    return best_val_loss


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser — factored out of `parse_args` so other tools (e.g. a UI) can
    introspect the full set of available parameters without duplicating this list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=list(SINGLE_FX_PAIRS.keys()))
    parser.add_argument(
        "--target-symbol",
        default=None,
        help="If set, forecast only this one pair (must be one of --pairs); --pairs still "
        "supplies the input universe (cross-asset returns, carry, CMA, ...). Default: pool "
        "every --pairs symbol, each forecasting its own target.",
    )
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5, help="N days ahead for the cumulative-return target.")
    parser.add_argument(
        "--target-mode",
        choices=["zscore", "class"],
        default="zscore",
        help="Regression target: continuous return z-score (default) or 5-class direction label.",
    )
    parser.add_argument("--momentum-window", type=int, default=15)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 penalty passed to the Adam optimizer.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR multiplier applied on val loss plateau.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Epochs to wait before reducing LR.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Stop training if val_loss hasn't improved in this many epochs.",
    )
    parser.add_argument(
        "--outlier-weight",
        type=float,
        default=10.0,
        help="Extra loss weight (linear in distance from the neutral class) for classes far from neutral. 0 = plain cross-entropy.",
    )
    parser.add_argument(
        "--variance-penalty-weight",
        type=float,
        default=10.0,
        help="target_mode='zscore' only: weight on (pred.std()-target.std())**2, to counteract collapse to a constant near-zero prediction. 0 disables.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of each symbol's history used for validation (early stopping, tuning).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of each symbol's history held out as a final, untouched test set (evaluated once, after tuning).",
    )
    parser.add_argument(
        "--no-predict-uncertainty",
        dest="predict_uncertainty",
        action="store_false",
        default=True,
        help="target_mode='zscore' only: disable the mean+log-variance head and Gaussian NLL loss, falling back to plain (weighted) MSE.",
    )
    parser.add_argument(
        "--use-spectral-features",
        action="store_true",
        help="Append a learned FFT-magnitude-spectrum embedding of the input window as one extra "
        "attendable key for the decoder's attention (see fx_forecasting.models.spectral_embedding).",
    )
    parser.add_argument(
        "--spectral-embedding-dim", type=int, default=8,
        help="Width of the spectral embedding (only used if --use-spectral-features).",
    )
    parser.add_argument(
        "--spectral-freq-bins", type=int, default=16,
        help="Number of pooled FFT frequency bins fed into the spectral embedding (only used if --use-spectral-features).",
    )
    parser.add_argument(
        "--cross-asset-pca-dim", type=int, default=None,
        help="If set, reduce the len(--pairs) cross-asset xret_* input columns to this many "
        "learned, PCA-like linear factors before the encoder sees them (must be strictly "
        "between 0 and len(--pairs); see fx_forecasting.models.cross_asset_projection).",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=1.0,
        help="Strategy backtest: round-trip transaction cost, in basis points of turnover (|position change|).",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.10,
        help="Strategy backtest: minimum |risk-adjusted signal| to take a position at all; smaller signals trade flat.",
    )
    parser.add_argument(
        "--signal-leverage",
        type=float,
        default=0.50,
        help="Strategy backtest: position size = leverage * signal, before capping at --max-position.",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=1.0,
        help="Strategy backtest: maximum absolute position size (post-leverage, pre-cap signal is clipped to this).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot-path", default="artifacts/lstm_predictions.png",
        help="Where to save the out-of-sample (validation) predictions plot.",
    )
    parser.add_argument(
        "--in-sample-plot-path",
        default="artifacts/lstm_predictions_in_sample.png",
        help="Where to save the in-sample (training) predictions plot.",
    )
    parser.add_argument(
        "--test-plot-path",
        default="artifacts/lstm_predictions_test.png",
        help="Where to save the final holdout (test) predictions plot.",
    )
    parser.add_argument(
        "--pnl-plot-path",
        default="artifacts/lstm_pnl.png",
        help="Where to save the validation strategy backtest cumulative-PnL plot (target_mode='zscore' only).",
    )
    parser.add_argument(
        "--test-pnl-plot-path",
        default="artifacts/lstm_pnl_test.png",
        help="Where to save the final holdout (test) strategy backtest cumulative-PnL plot (target_mode='zscore' only).",
    )
    parser.add_argument(
        "--positions-pnl-plot-path",
        default="artifacts/lstm_positions_pnl.png",
        help="Where to save the validation position/return/PnL-with-and-without-costs detail plot (target_mode='zscore' only).",
    )
    parser.add_argument(
        "--test-positions-pnl-plot-path",
        default="artifacts/lstm_positions_pnl_test.png",
        help="Where to save the final holdout (test) position/return/PnL-with-and-without-costs detail plot (target_mode='zscore' only).",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/lstm_model.pt",
        help="Where to save the trained model, or load it from when --infer is given.",
    )
    parser.add_argument(
        "--features-csv",
        dest="features_csv_path",
        default="artifacts/features.csv",
        help="Where to save the normalized per-symbol input features (plus target) as one long-format CSV.",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Skip training; load --model-path and only run inference/plotting.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shrink data/model drastically for a fast, steppable run (few pairs, 1 epoch).",
    )
    parser.add_argument(
        "--params-csv",
        metavar="CSV",
        default=None,
        help=(
            "Load the best trial from an optimize_hyperparams.py results CSV and apply it on "
            "top of the flags above — overrides --seq-len, --horizon, --momentum-window, "
            "--vol-window, --hidden-size, --num-layers, --dropout, --lr, --weight-decay, "
            "--outlier-weight, --variance-penalty-weight, --batch-size, and cma_windows, "
            "whichever of those the search actually covered (others are left as given)."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.debug:
        args.pairs = args.pairs[:2]
        args.years = min(args.years, 3)
        args.seq_len = min(args.seq_len, 20)
        args.hidden_size = 8
        args.num_layers = 1
        args.epochs = 1
        args.batch_size = 8

    cfg = Config(
        pairs=args.pairs,
        target_symbol=args.target_symbol,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        target_mode=args.target_mode,
        momentum_window=args.momentum_window,
        vol_window=args.vol_window,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        outlier_weight=args.outlier_weight,
        variance_penalty_weight=args.variance_penalty_weight,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        predict_uncertainty=args.predict_uncertainty,
        use_spectral_features=args.use_spectral_features,
        spectral_embedding_dim=args.spectral_embedding_dim,
        spectral_freq_bins=args.spectral_freq_bins,
        cross_asset_pca_dim=args.cross_asset_pca_dim,
        transaction_cost_bps=args.transaction_cost_bps,
        signal_threshold=args.signal_threshold,
        signal_leverage=args.signal_leverage,
        max_position=args.max_position,
        seed=args.seed,
        plot_path=args.plot_path,
        in_sample_plot_path=args.in_sample_plot_path,
        test_plot_path=args.test_plot_path,
        pnl_plot_path=args.pnl_plot_path,
        test_pnl_plot_path=args.test_pnl_plot_path,
        positions_pnl_plot_path=args.positions_pnl_plot_path,
        test_positions_pnl_plot_path=args.test_positions_pnl_plot_path,
        model_path=args.model_path,
        features_csv_path=args.features_csv_path,
        infer=args.infer,
    )

    if args.params_csv:
        from fx_forecasting.hparam_search import load_and_apply_best_trial

        cfg, applied = load_and_apply_best_trial(cfg, args.params_csv)
        logger.info("Loaded hyperparameters from %s: %s", args.params_csv, applied)

    return cfg


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args(argv)
    logger.info("Config: %s", cfg)
    set_seed(cfg.seed)

    (
        train_ds,
        val_ds,
        test_ds,
        train_datasets_by_symbol,
        val_datasets_by_symbol,
        test_datasets_by_symbol,
        market_data_by_symbol,
        num_factors,
    ) = build_pooled_datasets(cfg)

    if cfg.infer:
        model, checkpoint = load_model(cfg.model_path, cfg.device)
        if checkpoint["num_factors"] != num_factors:
            raise ValueError(
                f"Checkpoint expects {checkpoint['num_factors']} factors but --pairs "
                f"{cfg.pairs} currently produce {num_factors}; use the same --pairs as training."
            )
        if checkpoint["target_mode"] != cfg.target_mode:
            raise ValueError(
                f"Checkpoint was trained with target_mode={checkpoint['target_mode']!r} but "
                f"--target-mode={cfg.target_mode!r} was given; use the same --target-mode as training."
            )
        if checkpoint.get("target_symbol") != cfg.target_symbol:
            raise ValueError(
                f"Checkpoint was trained with target_symbol={checkpoint.get('target_symbol')!r} but "
                f"--target-symbol={cfg.target_symbol!r} was given; use the same --target-symbol as training "
                f"(a mismatch wouldn't be caught by the factor-count check above, since the input universe "
                f"can stay identical while the forecast target changes)."
            )
    else:
        model = build_model(cfg, num_factors)
        train(model, train_ds, val_ds, cfg)
        save_model(model, cfg, num_factors, cfg.model_path)

    calibrators = fit_signal_calibrators(model, train_datasets_by_symbol, cfg)

    plot_predictions_grid(model, val_datasets_by_symbol, cfg, cfg.plot_path, sample_label="out-of-sample (validation)")
    plot_predictions_grid(
        model, train_datasets_by_symbol, cfg, cfg.in_sample_plot_path, sample_label="in-sample (training)"
    )
    log_hit_rate_summary(model, val_datasets_by_symbol, market_data_by_symbol, cfg, label="validation")
    log_and_plot_strategy_pnl(
        model, val_datasets_by_symbol, market_data_by_symbol, calibrators, cfg, cfg.pnl_plot_path,
        sample_label="out-of-sample (validation)",
    )
    plot_positions_and_pnl(
        model, val_datasets_by_symbol, market_data_by_symbol, calibrators, cfg, cfg.positions_pnl_plot_path,
        sample_label="out-of-sample (validation)",
    )

    logger.info("=" * 70)
    logger.info(
        "FINAL HOLDOUT (test set): evaluate once — never used for early stopping, tuning, or model selection."
    )
    plot_predictions_grid(model, test_datasets_by_symbol, cfg, cfg.test_plot_path, sample_label="held-out test")
    log_hit_rate_summary(model, test_datasets_by_symbol, market_data_by_symbol, cfg, label="TEST (final holdout)")
    log_and_plot_strategy_pnl(
        model, test_datasets_by_symbol, market_data_by_symbol, calibrators, cfg, cfg.test_pnl_plot_path,
        sample_label="held-out test",
    )
    plot_positions_and_pnl(
        model, test_datasets_by_symbol, market_data_by_symbol, calibrators, cfg, cfg.test_positions_pnl_plot_path,
        sample_label="held-out test",
    )


if __name__ == "__main__":
    main()
