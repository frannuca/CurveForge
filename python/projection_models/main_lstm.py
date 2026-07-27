"""Train the LSTM + attention forecaster on a single target FX pair.

The model's recurrent input is a pretrained, orthogonally-reduced cross-asset return factor
sequence plus the target symbol's own (normalized) log return series: every pair in
``--pairs``' own daily log return (including the target's own) is compressed from K raw
series down to D orthogonal factors by a linear autoencoder pretrained separately (see
``pretrain_autoencoder.py``) and loaded here — not trained fresh alongside the rest of the
network — then concatenated, uncompressed, with the target's own return sequence so the LSTM
has undiluted access to the series it's actually forecasting on top of the shared cross-asset
structure. Everything else — momentum, realized volatility, skew, kurtosis, carry, intraday
volatility, and an optional spectral (FFT) summary of the combined factor sequence — bypasses
the LSTM entirely and is injected only at the final prediction layer, alongside the decoder's
attention context. (Volume change was considered too, but FX spot volume is uniformly 0 from
this project's data source — forex is OTC with no consolidated tape — so it carries no signal
and is not included.)

``Config.target_mode`` picks what the target is: ``"zscore"`` (default) is the continuous,
standardized ("z-scored") cumulative return itself (see
`fx_forecasting.features.compute_target_zscore`); ``"class"`` buckets that same z-score into
5 discrete direction labels, very_bearish..very_bullish (see
`fx_forecasting.features.compute_target_class`).

Deliberately written as small, independent, top-level functions (rather than one
monolithic pipeline) so each stage can be run and inspected separately under a
debugger: set a breakpoint after any call in ``main()`` and inspect the returned
DataFrame / tensors / model directly. Every stage logs the shape of what it
produced, so running with ``--debug`` (tiny data, 1 epoch) plus a look at the
console output is usually enough to understand what the model is seeing without
even opening a debugger.

Example:
    uv run python pretrain_autoencoder.py --debug
    uv run python main_lstm.py --debug --target-symbol EURUSD \\
        --autoencoder-path artifacts/cross_asset_autoencoder.pt
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
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm

from fx_forecasting.data.db import get_metric_series, get_time_series
from fx_forecasting.data.fx_downloader import MAJOR_FX_PAIRS
from fx_forecasting.features import (
    CLASS_NAMES,
    CLASS_QUANTILE_EDGES,
    build_symbol_frame,
    compute_carry_feature,
    compute_cross_sectional_target_class,
    compute_cross_sectional_target_zscore,
    compute_engineered_features,
    compute_log_returns,
    compute_rolling_normalized_features,
    compute_target_class,
    compute_target_zscore,
    compute_volatility_regime,
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
    # Mandatory: the one pair actually forecast. `pairs` still supplies the full
    # cross-asset input universe (every pair's own log return feeds the autoencoder), but
    # only `target_symbol`'s own target/side-features/backtest apply. Enforced both by the
    # CLI (`--target-symbol` is `required=True`) and here, in case Config is constructed
    # directly (e.g. from optimize_hyperparams.py or a notebook).
    target_symbol: str | None = None

    seq_len: int = 60
    horizon: int = 2  # N days ahead for the single cumulative-return target
    target_mode: str = "zscore"  # "zscore" (continuous normalized return, default) or "class" (5-class direction label)
    # If True, the target is target_symbol's horizon-day return *relative to* the
    # cross-sectional median of every other --pairs symbol (did it outperform the rest of
    # the universe), rather than its own absolute direction — see
    # fx_forecasting.features.compute_cross_sectional_target_zscore. Input features are
    # unchanged either way; only what the model is trained to predict (and what
    # log_hit_rate_summary judges a hit against) changes.
    cross_sectional_target: bool = False
    momentum_window: int = 10  # trailing window (days) for the momentum feature
    vol_window: int = 10  # trailing window (days) for the realized-volatility feature

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
    # Kept ON by default for this architecture (unlike outlier_weight): a weak-signal target
    # otherwise collapses to predicting ~its own unconditional mean regardless of input (see
    # `variance_penalty`) — still worth validating via purged walk-forward, not a substitute
    # for that discipline, just not something to start from disabled here.
    variance_penalty_weight: float = 10.0
    val_fraction: float = 0.2
    test_fraction: float = 0.1
    predict_uncertainty: bool = True
    use_spectral_features: bool = False  # off by default — validate via purged walk-forward before defaulting on
    spectral_embedding_dim: int = 8
    spectral_freq_bins: int = 16
    # Required: path to a checkpoint written by pretrain_autoencoder.py. Its own encoder
    # weights are loaded into the model's cross_asset_encoder (see build_model); this script
    # never trains that reduction fresh.
    autoencoder_path: str | None = None
    # Kumar, Raghunathan, Jones, Ma & Liang (2022) — fine-tuning a pretrained representation
    # can distort it and generalize worse under distribution shift, which daily FX returns
    # are not exempt from. Frozen by default; --fine-tune-autoencoder opts in.
    freeze_autoencoder: bool = True
    transaction_cost_bps: float = 1.0
    signal_threshold: float = 0.10
    signal_leverage: float = 0.50
    max_position: float = 1.0
    # If True, the backtest (compute_positions_and_pnl) zeroes the position on any day the
    # cross-sectional (peer-median) realized-vol regime is in its highest bucket — see
    # fx_forecasting.features.compute_volatility_regime. Diagnostic motivation: hit rate was
    # found to be consistently worse specifically in high-market-vol periods, replicated
    # across validation and test and across two different vol-regime definitions (see
    # regime_analysis.py) — this tests whether avoiding those periods actually improves
    # realized Sharpe/PnL, not just the hit-rate proxy. Does not affect training, only the
    # backtest; the model itself is unchanged either way.
    regime_gate: bool = False
    regime_vol_window: int = 20  # trailing realized-vol window feeding the regime label
    regime_window: int = 252  # minimum history (days) before a regime label is assigned (expanding comparison)
    regime_n_regimes: int = 3  # gate fires on the *last* (highest-vol) of this many equal-frequency buckets

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
    """Load a stored OHLCV metric (e.g. 'high', 'low', 'volume') for each pair from Postgres."""
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
    of the feature/model hyperparameters (seq_len, momentum/vol windows,
    hidden_size, ...), so it can be fetched once and reused across many
    `build_target_dataset` calls — e.g. the many trials of a hyperparameter search
    (see `optimize_hyperparams.py`), where re-querying the DB per trial would dominate
    runtime.
    """

    panel: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    rates: pd.DataFrame


def load_raw_panels(cfg: Config) -> RawPanels:
    """Fetches everything `build_target_dataset` needs from Postgres, once."""
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


def build_target_dataset(
    cfg: Config, raw: RawPanels | None = None,
) -> tuple[
    FXSequenceDataset,
    FXSequenceDataset,
    FXSequenceDataset,
    dict[str, FXSequenceDataset],
    dict[str, FXSequenceDataset],
    dict[str, FXSequenceDataset],
    dict[str, pd.DataFrame],
    int,
    int,
    int,
    list[str],
]:
    """Builds `cfg.target_symbol`'s train/val/test datasets from three column blocks:

    - The *side-feature* block (leading columns): `target_symbol`-only, non-sequential
      covariates — momentum, realized vol, skew, kurtosis, carry, and intraday volatility —
      rolling-normalized, but never fed to the LSTM; only their value
      at the forecast origin (the window's last timestep) is used, injected directly at the
      model's final prediction layer alongside the attention context (and, if enabled, the
      spectral summary) — see `fx_forecasting.models.lstm_forecaster`. (Day-over-day volume
      change was considered too, but FX spot volume from this project's data source
      (Yahoo/yfinance) is uniformly 0 — forex is OTC with no consolidated tape — so it
      carries no signal and isn't included.)
    - The *target-series* block (middle columns): `target_symbol`'s own rolling-normalized
      log return, as a genuine sequence — concatenated with the D orthogonal cross-asset
      factors (uncompressed) before the LSTM, so the recurrence has undiluted access to the
      series it's actually forecasting on top of the shared cross-asset structure the K->D
      reduction otherwise dilutes it into.
    - The *cross-asset* block (trailing K columns): one column per pair in `cfg.pairs`
      (`xret_{pair}`, including `target_symbol` itself), rolling-normalized log returns, fed
      through the pretrained orthogonal factor encoder
      (`fx_forecasting.models.orthogonal_autoencoder`, loaded by `build_model`). Always the
      *trailing* `len(cfg.pairs)` columns — `LSTMAttentionForecaster.encode` slices on this
      exact convention.

    All three blocks share the same causal rolling z-score (`compute_rolling_normalized_features`,
    window `cfg.seq_len`) and the same purged train/val/test split
    (`fx_forecasting.trading.purged_train_val_test_split`) as every other stage in this
    pipeline — see `build_datasets` for why the split is embargoed.

    The *target* itself (not an input column) is `target_symbol`'s own absolute horizon-day
    return by default, or — if `cfg.cross_sectional_target` — its return *relative to* the
    cross-sectional median of every other `cfg.pairs` symbol over the same horizon (did
    `target_symbol` outperform the rest of the universe, rather than did it go up at all; see
    `fx_forecasting.features.compute_cross_sectional_target_zscore`). Input features are
    identical either way — only the target construction, and what
    `market_data_by_symbol[symbol]["horizon_zscore"]` judges a `target_mode="class"` hit
    against, changes. The execution backtest (`log_and_plot_strategy_pnl`) is unaffected
    either way: it still sizes an absolute position in `target_symbol` off whatever signal the
    model produces and realizes PnL against `target_symbol`'s own next-day return — under
    `cross_sectional_target=True` that means using a relative-outperformance-trained signal as
    a directional timing cue on the symbol itself, not a market-neutral pairs trade against
    the peer basket.

    Returns `(train_ds, val_ds, test_ds, train_datasets_by_symbol, val_datasets_by_symbol,
    test_datasets_by_symbol, market_data_by_symbol, num_factors, num_side_features,
    num_target_series, feature_names)` — `feature_names` is `normalized`'s own column order
    (side features, then target series, then `xret_*`), the same convention
    `LSTMAttentionForecaster.encode` slices on positionally; exposed here so callers that need
    to know which raw column a given tensor position corresponds to (e.g. feature-importance
    reporting for a non-sequence model trained on the same tabular data — see
    `gbt_baseline.py`) don't have to re-derive the column-ordering convention themselves. The
    `*_by_symbol` dicts always have exactly one entry (`{cfg.target_symbol: ...}`), kept as
    dicts so `plot_predictions_grid`/
    `log_hit_rate_summary`/`log_and_plot_strategy_pnl`/`plot_positions_and_pnl` (which already
    only ever expected one symbol under the old optional `target_symbol` mode) need no
    changes. `market_data_by_symbol[symbol]` carries `next_log_return` (actual next-day
    executable return — the backtest's return leg), `daily_carry` (the backtest's carry leg),
    and `horizon_zscore` (the continuous return z-score `log_hit_rate_summary` judges a
    `target_mode="class"` hit against).

    `raw`: pre-fetched `RawPanels` to reuse instead of hitting Postgres again
    (see `load_raw_panels`) — e.g. across many hyperparameter-search trials
    that only vary feature/model hyperparameters, not `pairs`/`years`.
    Fetched fresh if not given.
    """
    if not cfg.target_symbol:
        raise ValueError("Config.target_symbol is required (this architecture always forecasts exactly one pair).")
    if cfg.target_symbol not in cfg.pairs:
        raise ValueError(f"target_symbol={cfg.target_symbol!r} must be one of pairs={cfg.pairs}")
    symbol = cfg.target_symbol

    if raw is None:
        raw = load_raw_panels(cfg)
    panel, high, low, rates = raw.panel, raw.high, raw.low, raw.rates

    log_returns = compute_log_returns(panel)
    engineered = compute_engineered_features(panel, log_returns, high, low, cfg.momentum_window, cfg.vol_window)
    carry = compute_carry_feature(cfg.pairs, rates)

    # build_symbol_frame's own first column is always the symbol's raw "return" — split off
    # into its own target-series block (below) rather than treated as an FFN-only side
    # feature, since it now feeds the LSTM directly as a sequence.
    side_frame_full = build_symbol_frame(
        symbol, log_returns, engineered, cfg.momentum_window, cfg.vol_window, carry=carry[symbol],
    )
    target_series_frame = side_frame_full[["return"]].rename(columns={"return": "target_log_return"})
    target_series_columns = list(target_series_frame.columns)
    side_frame = side_frame_full.drop(columns=["return"])
    side_columns = list(side_frame.columns)

    # Cross-asset block: one column per pair (including self), raw log returns.
    xret_columns = [f"xret_{pair}" for pair in sorted(cfg.pairs)]
    xret_frame = pd.DataFrame({col: log_returns[pair] for col, pair in zip(xret_columns, sorted(cfg.pairs))})

    raw_frame = pd.concat(
        [side_frame, target_series_frame, xret_frame], axis=1, sort=False
    ).dropna(how="any")
    normalized = compute_rolling_normalized_features(raw_frame, cfg.seq_len)
    # Guarantee column order regardless of how pd.concat/dropna ordered things above — side
    # features first, then the target series, then xret_* last (trailing K) — the model's
    # encode() slices on this exact convention.
    normalized = normalized[side_columns + target_series_columns + xret_columns]
    num_side_features = len(side_columns)
    num_target_series = len(target_series_columns)
    num_factors = normalized.shape[1]

    peer_symbols = [p for p in cfg.pairs if p != symbol]
    if cfg.cross_sectional_target:
        if not peer_symbols:
            raise ValueError("cross_sectional_target=True needs at least one other pair in cfg.pairs besides target_symbol")
        if cfg.target_mode == "class":
            target = compute_cross_sectional_target_class(panel, log_returns, symbol, peer_symbols, cfg.horizon, cfg.seq_len)
        else:
            target = compute_cross_sectional_target_zscore(panel, log_returns, symbol, peer_symbols, cfg.horizon, cfg.seq_len)
        horizon_zscore = compute_cross_sectional_target_zscore(panel, log_returns, symbol, peer_symbols, cfg.horizon, cfg.seq_len)
    else:
        if cfg.target_mode == "class":
            target = compute_target_class(panel, log_returns, symbol, cfg.horizon, cfg.seq_len)
        else:
            target = compute_target_zscore(panel, log_returns, symbol, cfg.horizon, cfg.seq_len)
        horizon_zscore = compute_target_zscore(panel, log_returns, symbol, cfg.horizon, cfg.seq_len)

    if cfg.write_features_csv:
        save_features_csv({symbol: normalized}, {symbol: target}, cfg.features_csv_path)

    train_ds, val_ds, test_ds = build_datasets(normalized, target, cfg)

    aligned_index = normalized.index.intersection(target.index)
    market_data_by_symbol = {
        symbol: pd.DataFrame(
            {
                "next_log_return": log_returns[symbol].shift(-1).reindex(aligned_index),
                "daily_carry": (carry[symbol] / 100.0 / 252.0).reindex(aligned_index),
                "horizon_zscore": horizon_zscore.reindex(aligned_index),
            },
            index=aligned_index,
        )
    }
    if cfg.regime_gate:
        if not peer_symbols:
            raise ValueError("regime_gate=True needs at least one other pair in cfg.pairs besides target_symbol")
        market_return = log_returns[peer_symbols].median(axis=1)
        market_vol_regime = compute_volatility_regime(
            market_return, cfg.regime_vol_window, cfg.regime_window, cfg.regime_n_regimes
        )
        market_data_by_symbol[symbol]["market_vol_regime"] = market_vol_regime.reindex(aligned_index)

    logger.info(
        "%s: %d factors (%d cross-asset + %d target series + %d side), train=%d, val=%d, test=%d",
        symbol, num_factors, len(cfg.pairs), num_target_series, num_side_features,
        len(train_ds), len(val_ds), len(test_ds),
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        {symbol: train_ds},
        {symbol: val_ds},
        {symbol: test_ds},
        market_data_by_symbol,
        num_factors,
        num_side_features,
        num_target_series,
        side_columns + target_series_columns + xret_columns,
    )


def build_model(
    cfg: Config, num_factors: int, num_side_features: int, num_target_series: int
) -> LSTMAttentionForecaster:
    """Constructs the model and loads the pretrained cross-asset factor encoder into it.

    `cfg.autoencoder_path` must point to a checkpoint written by `pretrain_autoencoder.py`
    for this exact `cfg.pairs`/`cfg.seq_len` — the pretrained weights are only valid against
    data normalized the identical way, and the encoder's own input width is fixed to
    `len(cfg.pairs)`; a mismatch on either is rejected with a clear error rather than
    silently miscalibrating every downstream forecast.
    """
    output_size = len(CLASS_NAMES) if cfg.target_mode == "class" else 1
    num_cross_asset_factors = len(cfg.pairs)

    if not cfg.autoencoder_path:
        raise ValueError("--autoencoder-path is required — run pretrain_autoencoder.py first.")
    checkpoint = torch.load(cfg.autoencoder_path, map_location=cfg.device, weights_only=False)
    if sorted(checkpoint["pairs"]) != sorted(cfg.pairs):
        raise ValueError(
            f"Autoencoder checkpoint {cfg.autoencoder_path!r} was pretrained on "
            f"pairs={checkpoint['pairs']} but --pairs={cfg.pairs} was given; retrain the "
            f"autoencoder for this pair universe."
        )
    if checkpoint["seq_len"] != cfg.seq_len:
        raise ValueError(
            f"Autoencoder checkpoint {cfg.autoencoder_path!r} was pretrained with "
            f"seq_len={checkpoint['seq_len']} (the rolling-normalization window) but "
            f"--seq-len={cfg.seq_len} was given; the pretrained weights are only valid "
            f"against data normalized the same way."
        )
    factor_dim = checkpoint["factor_dim"]

    model = LSTMAttentionForecaster(
        num_factors=num_factors,
        output_size=output_size,
        num_cross_asset_factors=num_cross_asset_factors,
        factor_dim=factor_dim,
        num_side_features=num_side_features,
        num_target_series=num_target_series,
        forecast_horizon=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        predict_uncertainty=cfg.predict_uncertainty,
        use_spectral_features=cfg.use_spectral_features,
        spectral_embedding_dim=cfg.spectral_embedding_dim,
        spectral_freq_bins=cfg.spectral_freq_bins,
    ).to(cfg.device)
    model.cross_asset_encoder.load_state_dict(checkpoint["encoder_state"])
    if cfg.freeze_autoencoder:
        for p in model.cross_asset_encoder.parameters():
            p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_desc = f"{len(CLASS_NAMES)}-class direction" if cfg.target_mode == "class" else "z-score regression"
    logger.info(
        "Model: %d cross-asset factors -> %d orthogonal (%s) + %d target series, %d side features, "
        "%d-day %s target, %d params (%d trainable)",
        num_cross_asset_factors, factor_dim, "frozen" if cfg.freeze_autoencoder else "fine-tuned",
        num_target_series, num_side_features, cfg.horizon, target_desc, n_params, n_trainable,
    )
    return model


def save_model(model: LSTMAttentionForecaster, cfg: Config, num_factors: int, path: str) -> None:
    """Saves weights plus the architecture/config needed to reconstruct the model for inference.

    The cross-asset encoder's weights (frozen or fine-tuned) are already part of
    `model.state_dict()`, so `--infer` never needs the original `--autoencoder-path`
    checkpoint again.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "num_factors": num_factors,
        "num_cross_asset_factors": model.num_cross_asset_factors,
        "factor_dim": model.factor_dim,
        "num_side_features": model.num_side_features,
        "num_target_series": model.num_target_series,
        "output_size": model.output_size,
        "target_mode": cfg.target_mode,
        "cross_sectional_target": cfg.cross_sectional_target,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "predict_uncertainty": cfg.predict_uncertainty,
        "use_spectral_features": cfg.use_spectral_features,
        "spectral_embedding_dim": cfg.spectral_embedding_dim,
        "spectral_freq_bins": cfg.spectral_freq_bins,
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
        num_cross_asset_factors=checkpoint["num_cross_asset_factors"],
        factor_dim=checkpoint["factor_dim"],
        num_side_features=checkpoint.get("num_side_features", 0),
        num_target_series=checkpoint.get("num_target_series", 0),
        forecast_horizon=1,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        predict_uncertainty=checkpoint.get("predict_uncertainty", False),
        use_spectral_features=checkpoint.get("use_spectral_features", False),
        spectral_embedding_dim=checkpoint.get("spectral_embedding_dim", 8),
        spectral_freq_bins=checkpoint.get("spectral_freq_bins", 16),
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
                # Applies regardless of which loss path is active: Gaussian NLL alone lets the
                # model cheaply minimize loss by predicting a near-constant mean and inflating
                # its own uncertainty to explain away the resulting errors (Seitzer et al. 2022,
                # "On the Pitfalls of Heteroscedastic Uncertainty Estimation") rather than
                # learning genuine directional signal in the mean itself — the same mean-collapse
                # this penalty guards against under plain MSE.
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
    `build_target_dataset`'s `market_data_by_symbol["horizon_zscore"]`), not
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
    `build_target_dataset` (validation or the final test holdout); `label`
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
    """The full signal -> calibration -> smoothing -> position-sizing -> (optional regime
    gate) -> net-PnL pipeline for one symbol (shared by `log_and_plot_strategy_pnl` and
    `plot_positions_and_pnl`) — see `log_and_plot_strategy_pnl`'s docstring for what each
    stage does and why.

    If `cfg.regime_gate`, the position is zeroed on any day `market["market_vol_regime"]`
    (see `build_target_dataset`) is in its highest bucket — a day whose regime isn't known
    yet (NaN, before the regime detector's own warm-up period) trades normally rather than
    being gated, since there's nothing to gate *on* yet. The zeroed position still flows
    through `net_strategy_returns` normally, so exiting into a gated period is correctly
    charged its own transaction cost, not treated as free.

    Returns `(positions, result)`: `positions` is the (possibly gated) position-size series
    (indexed by `dataset.sample_dates`); `result` is `net_strategy_returns`'s per-day
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
    if cfg.regime_gate:
        regime = market["market_vol_regime"].reindex(positions.index)
        is_high_vol = regime == (cfg.regime_n_regimes - 1)  # NaN (regime not yet known) compares False -> not gated
        positions = positions * (~is_high_vol).astype(float)
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
    not a z-scored proxy — this is exactly the no-leakage requirement: the signal computed
    from information available through day t sizes a position whose PnL realizes against
    day t+1's return), plus `w_{t-1} * carry_t`, minus `cost * |w_t - w_{t-1}|` — so turnover
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
    """One panel per symbol (just the target symbol, always, under this architecture): the
    position size and that symbol's own actual daily return on the left axis, cumulative PnL
    with and without transaction costs on a secondary (right) axis.

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


def log_naive_baseline(train_ds: Dataset, val_ds: Dataset, cfg: Config, output_size: int) -> None:
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
    (training) per-symbol datasets from `build_target_dataset` — comparing
    the two is the real test of whether the model learned generic dynamics
    or just memorized the training set (in-sample looking much better than
    out-of-sample is the overfitting tell).

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


def train(model: LSTMAttentionForecaster, train_ds: Dataset, val_ds: Dataset, cfg: Config) -> float:
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
    parser.add_argument(
        "--pairs", nargs="+", default=list(MAJOR_FX_PAIRS.keys()),
        help="Cross-asset input universe (every pair's own log return feeds the pretrained "
        "factor autoencoder); must include --target-symbol and have at least 2 entries.",
    )
    parser.add_argument(
        "--target-symbol",
        required=True,
        help="The one pair to forecast (must be one of --pairs); --pairs still supplies the "
        "full cross-asset input universe (every pair's own log return feeds the pretrained "
        "factor autoencoder).",
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
    parser.add_argument(
        "--cross-sectional-target",
        action="store_true",
        help="Train against target_symbol's horizon-day return relative to the cross-"
        "sectional median of every other --pairs symbol (did it outperform the rest of the "
        "universe), instead of its own absolute direction. Input features are unchanged; "
        "only the target (and what hit rate is judged against) changes.",
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
        help="target_mode='zscore' only: weight on (pred.std()-target.std())**2, to counteract collapse to a constant near-zero prediction (the model always predicting ~0 regardless of input). 0 disables.",
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
        help="Compute a learned FFT-magnitude-spectrum summary of the cross-asset factor "
        "sequence and feed it into the final prediction layer alongside the other side "
        "features (see fx_forecasting.models.spectral_embedding).",
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
        "--autoencoder-path",
        required=True,
        help="Path to a checkpoint written by pretrain_autoencoder.py for this exact --pairs "
        "and --seq-len — its encoder is loaded as the model's cross-asset factor reduction.",
    )
    parser.add_argument(
        "--fine-tune-autoencoder",
        dest="freeze_autoencoder",
        action="store_false",
        default=True,
        help="Allow the pretrained cross-asset encoder's weights to update during training "
        "(default: frozen — see Config.freeze_autoencoder).",
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
    parser.add_argument(
        "--regime-gate",
        action="store_true",
        help="Strategy backtest: zero the position whenever the cross-sectional (peer-median) "
        "realized-vol regime is in its highest bucket (see regime_analysis.py for the "
        "diagnostic this is based on). Backtest-only — doesn't affect training or the model.",
    )
    parser.add_argument("--regime-vol-window", type=int, default=20, help="Trailing realized-vol window feeding --regime-gate's label.")
    parser.add_argument(
        "--regime-window", type=int, default=252,
        help="Minimum history (days) before --regime-gate's regime label is assigned (expanding comparison).",
    )
    parser.add_argument("--regime-n-regimes", type=int, default=3, help="Number of equal-frequency vol regime buckets for --regime-gate; gates on the highest.")
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
            "--outlier-weight, --variance-penalty-weight, --batch-size, "
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
        cross_sectional_target=args.cross_sectional_target,
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
        autoencoder_path=args.autoencoder_path,
        freeze_autoencoder=args.freeze_autoencoder,
        transaction_cost_bps=args.transaction_cost_bps,
        signal_threshold=args.signal_threshold,
        signal_leverage=args.signal_leverage,
        max_position=args.max_position,
        regime_gate=args.regime_gate,
        regime_vol_window=args.regime_vol_window,
        regime_window=args.regime_window,
        regime_n_regimes=args.regime_n_regimes,
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
        num_side_features,
        num_target_series,
        _feature_names,
    ) = build_target_dataset(cfg)

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
        if checkpoint.get("cross_sectional_target", False) != cfg.cross_sectional_target:
            raise ValueError(
                f"Checkpoint was trained with cross_sectional_target="
                f"{checkpoint.get('cross_sectional_target', False)!r} but "
                f"--cross-sectional-target={cfg.cross_sectional_target!r} was given; use the same flag as training "
                f"(the model's predictions mean different things depending on which target it was trained "
                f"against, and hit-rate/PnL would be judged inconsistently otherwise)."
            )
        if checkpoint.get("target_symbol") != cfg.target_symbol:
            raise ValueError(
                f"Checkpoint was trained with target_symbol={checkpoint.get('target_symbol')!r} but "
                f"--target-symbol={cfg.target_symbol!r} was given; use the same --target-symbol as training "
                f"(a mismatch wouldn't be caught by the factor-count check above, since the input universe "
                f"can stay identical while the forecast target changes)."
            )
    else:
        model = build_model(cfg, num_factors, num_side_features, num_target_series)
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
