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
from fx_forecasting.features import (
    CLASS_NAMES,
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

logger = logging.getLogger(__name__)

# Set to cooperatively stop an in-progress `train()` call early (checked once
# per epoch) — e.g. a "Stop" button in a UI driving `main()` in a background
# thread. `train()` clears it on every call, so it never leaks into the next run.
CANCEL_EVENT = threading.Event()


@dataclass
class Config:
    pairs: list[str] = field(default_factory=lambda: list(MAJOR_FX_PAIRS.keys()))
    years: int = 20

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
    outlier_weight: float = 2.0  # extra loss weight for classes far from neutral (see class_distance_weights); 0 = plain CE
    variance_penalty_weight: float = 1.0  # zscore mode only: penalizes pred.std() vs target.std(); counteracts collapse to a constant near-zero prediction; 0 disables
    val_fraction: float = 0.2

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    plot_path: str = "artifacts/lstm_predictions.png"
    in_sample_plot_path: str = "artifacts/lstm_predictions_in_sample.png"
    pnl_plot_path: str = "artifacts/lstm_pnl.png"
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

    def __init__(self, features: np.ndarray, target: np.ndarray, seq_len: int) -> None:
        self.features = torch.as_tensor(np.asarray(features).copy(), dtype=torch.float32)
        target_array = np.asarray(target)
        target_dtype = torch.long if np.issubdtype(target_array.dtype, np.integer) else torch.float32
        self.targets = torch.as_tensor(target_array.copy(), dtype=target_dtype)
        self.seq_len = seq_len

        # Row i in `features`/`target` corresponds to a window ending at i
        # (inclusive), i.e. sample i uses features[i - seq_len + 1 : i + 1].
        self.valid_indices = list(range(seq_len - 1, len(self.features)))

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
) -> tuple[FXSequenceDataset, FXSequenceDataset]:
    """Time-ordered train/val split for a single symbol's features/target. Fit-free (rolling) normalization."""
    aligned_index = features.index.intersection(target.index)
    features = features.loc[aligned_index]
    target = target.loc[aligned_index]

    n_val = int(len(aligned_index) * cfg.val_fraction)
    n_train = len(aligned_index) - n_val
    if n_train <= cfg.seq_len:
        raise ValueError(f"Not enough training rows ({n_train}) for seq_len={cfg.seq_len}")

    train_features_df = features.iloc[:n_train]
    val_features_df = features.iloc[n_train - cfg.seq_len + 1 :]  # keep lookback context for val

    train_target = target.iloc[:n_train]
    val_target = target.iloc[n_train - cfg.seq_len + 1 :]

    train_ds = FXSequenceDataset(
        train_features_df.to_numpy(dtype=np.float32), train_target.to_numpy(), cfg.seq_len
    )
    val_ds = FXSequenceDataset(val_features_df.to_numpy(dtype=np.float32), val_target.to_numpy(), cfg.seq_len)
    return train_ds, val_ds


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
    dict[str, FXSequenceDataset],
    dict[str, FXSequenceDataset],
    dict[str, np.ndarray],
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
    per-symbol train *and* val datasets (unpooled) so predictions can be
    inspected and plotted per instrument — in-sample (train) as well as
    out-of-sample (val) — plus a per-symbol *daily* (horizon=1) z-scored
    return array aligned 1:1 with each val dataset's iteration order — used
    by the strategy backtest (`log_and_plot_strategy_pnl`) as PnL's "return"
    leg, since the model's own target is an overlapping N-day-ahead return,
    not a genuinely daily one.

    `raw`: pre-fetched `RawPanels` to reuse instead of hitting Postgres again
    (see `load_raw_panels`) — e.g. across many hyperparameter-search trials
    that only vary feature/model hyperparameters, not `pairs`/`years`.
    Fetched fresh if not given.
    """
    if raw is None:
        raw = load_raw_panels(cfg)
    panel, high, low, rates = raw.panel, raw.high, raw.low, raw.rates

    log_returns = compute_log_returns(panel)
    engineered = compute_engineered_features(panel, log_returns, high, low, cfg.momentum_window, cfg.vol_window)
    carry = compute_carry_feature(cfg.pairs, rates)
    cma = CrossingMovingAverages(cfg.cma_windows).compute(panel)

    train_datasets = []
    train_datasets_by_symbol: dict[str, FXSequenceDataset] = {}
    val_datasets: dict[str, FXSequenceDataset] = {}
    normalized_by_symbol: dict[str, pd.DataFrame] = {}
    target_by_symbol: dict[str, pd.Series] = {}
    daily_return_by_symbol: dict[str, np.ndarray] = {}
    num_factors: int | None = None

    for symbol in cfg.pairs:
        symbol_cma = cma[[f"{symbol}_{w.name}" for w in cfg.cma_windows]]
        symbol_cma.columns = [w.name for w in cfg.cma_windows]

        frame = build_symbol_frame(
            symbol,
            log_returns,
            engineered,
            cfg.momentum_window,
            cfg.vol_window,
            carry=carry[symbol],
            cma=symbol_cma,
        )
        normalized = compute_rolling_normalized_features(frame, cfg.seq_len)
        num_factors = normalized.shape[1]

        if cfg.target_mode == "class":
            target = compute_target_class(panel, log_returns, symbol, cfg.horizon, cfg.seq_len)
        else:
            target = compute_target_zscore(panel, log_returns, symbol, cfg.horizon, cfg.seq_len)
        normalized_by_symbol[symbol] = normalized
        target_by_symbol[symbol] = target

        train_ds, val_ds = build_datasets(normalized, target, cfg)
        train_datasets.append(train_ds)
        train_datasets_by_symbol[symbol] = train_ds
        val_datasets[symbol] = val_ds

        # Daily (horizon=1) z-scored return for the same validation dates,
        # in the same order `collect_predictions` iterates the val dataset —
        # see the docstring above for why this (not `target`) is the PnL
        # return leg. `aligned_index`/`n_train` replicate `build_datasets`'
        # own split arithmetic; the val dates are exactly its tail `n_val` rows.
        aligned_index = normalized.index.intersection(target.index)
        n_train = len(aligned_index) - int(len(aligned_index) * cfg.val_fraction)
        val_dates = aligned_index[n_train:]
        daily_target = compute_target_zscore(panel, log_returns, symbol, 1, cfg.seq_len)
        daily_return_by_symbol[symbol] = daily_target.reindex(val_dates).to_numpy()

        logger.info(
            "%s: %d factors, train sequences=%d, val sequences=%d", symbol, num_factors, len(train_ds), len(val_ds)
        )

    if cfg.write_features_csv:
        save_features_csv(normalized_by_symbol, target_by_symbol, cfg.features_csv_path)

    pooled_train = ConcatDataset(train_datasets)
    pooled_val = ConcatDataset(list(val_datasets.values()))
    logger.info(
        "Pooled across %d symbols: %d factors (per symbol), train sequences=%d, val sequences=%d",
        len(cfg.pairs), num_factors, len(pooled_train), len(pooled_val),
    )
    return pooled_train, pooled_val, train_datasets_by_symbol, val_datasets, daily_return_by_symbol, num_factors


def build_model(cfg: Config, num_factors: int) -> LSTMAttentionForecaster:
    output_size = len(CLASS_NAMES) if cfg.target_mode == "class" else 1
    model = LSTMAttentionForecaster(
        num_factors=num_factors,
        output_size=output_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
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
        "pairs": cfg.pairs,
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
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
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
            output = model(x)
            if cfg.target_mode == "class":
                loss = ce_loss_fn(output, y)
                hits = output.argmax(dim=-1) == y
            else:
                pred = output.squeeze(-1)
                loss = weighted_mse_loss(pred, y, cfg.outlier_weight)
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


def compute_hit_rate(actual: np.ndarray, predicted: np.ndarray, target_mode: str) -> float:
    """Fraction of samples where the prediction "called it right".

    `target_mode="class"`: exact predicted-class match. `target_mode="zscore"`:
    same-sign match (both predicted and actual z-score fall on the same side
    of zero) — the natural direction-call analogue for a continuous target.
    This is the metric a trading strategy actually cares about — distinct
    from (and often more informative than) the loss used to train/select the
    model. See `log_naive_baseline` for the real no-skill bar to compare
    against (not a flat 50%/20%, since classes are imbalanced by design).
    """
    if target_mode == "class":
        return float(np.mean(predicted == actual))
    return float(np.mean(np.sign(predicted) == np.sign(actual)))


def log_hit_rate_summary(
    model: LSTMAttentionForecaster, val_datasets: dict[str, FXSequenceDataset], cfg: Config
) -> None:
    """Logs the hit rate (see `compute_hit_rate`), per symbol and pooled across all of them.

    Compare against `log_naive_baseline`'s baseline accuracy for the real
    no-skill bar.
    """
    metric = "predicted class == actual class" if cfg.target_mode == "class" else "sign(predicted) == sign(actual)"
    logger.info("Hit rate (%s):", metric)
    all_actual, all_predicted = [], []
    for symbol, dataset in val_datasets.items():
        actual, predicted = collect_predictions(model, dataset, cfg)
        hit_rate = compute_hit_rate(actual, predicted, cfg.target_mode)
        logger.info("  %-8s hit_rate=%5.1f%%  (n=%d)", symbol, hit_rate * 100, len(actual))
        all_actual.append(actual)
        all_predicted.append(predicted)

    pooled_hit_rate = compute_hit_rate(np.concatenate(all_actual), np.concatenate(all_predicted), cfg.target_mode)
    logger.info(
        "  %-8s hit_rate=%5.1f%%  (n=%d, pooled across %d symbols)",
        "ALL", pooled_hit_rate * 100, sum(len(a) for a in all_actual), len(val_datasets),
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


def compute_strategy_pnl(daily_return: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Per-period PnL of a signal-weighted strategy: position size = predicted z-score.

    Position (positive = long, negative = short, magnitude = conviction) is
    the predicted z-score — expected to already be smoothed via
    `smooth_weights` before being passed in here. `daily_return` must be a
    genuinely *daily* (horizon=1) z-scored return, not the model's own
    N-day-ahead target — that target is an overlapping window (it shares
    N-1 days with the next day's target), so using it as PnL's per-period
    "return" leg would double/triple/N-times count the same realized move
    and make `scale_to_target_vol`'s annualization (which assumes one
    independent row per trading day) meaningless. `pnl_t = predicted_t *
    daily_return_t` needs no further scaling, since both are already in the
    same normalized z-score units. A model that's both correctly directional
    and well-calibrated in magnitude earns positive expected PnL; pure noise
    earns ~0 in expectation.
    """
    return predicted * daily_return


def compute_benchmark_pnl(daily_return: np.ndarray) -> np.ndarray:
    """Per-period PnL of a naive equal-weight benchmark: unconditional position size of 1
    (always long), no prediction used — just the daily realized z-score itself as that
    period's return (see `compute_strategy_pnl` for why this must be daily, not the
    model's own N-day-ahead target). This is the baseline `compute_strategy_pnl`'s
    signal-weighted strategy needs to beat to justify using the model's predictions at all.
    """
    return daily_return.copy()


def scale_to_target_vol(pnl: np.ndarray, target_annual_vol: float = 0.20, periods_per_year: int = 252) -> np.ndarray:
    """Uniformly rescales `pnl` so its annualized volatility equals `target_annual_vol`.

    Puts return streams that are on different native scales (e.g. an
    always-long benchmark vs. a signal-weighted strategy whose position size
    varies with model confidence) onto the same risk footing, so their
    cumulative PnL shapes are directly comparable — differences reflect
    skill/shape, not just one strategy running "hotter" than the other.
    Assumes one row of `pnl` per trading day (252/year); leaves a
    near-all-zero series unscaled rather than dividing by ~0.
    """
    realized_annual_vol = pnl.std(ddof=0) * math.sqrt(periods_per_year)
    if realized_annual_vol < 1e-12:
        return pnl
    return pnl * (target_annual_vol / realized_annual_vol)


def log_and_plot_strategy_pnl(
    model: LSTMAttentionForecaster,
    val_datasets: dict[str, FXSequenceDataset],
    daily_return_by_symbol: dict[str, np.ndarray],
    cfg: Config,
) -> None:
    """Backtests investing the predicted z-score in each symbol on the validation set.

    Logs per-symbol PnL summary stats for the signal-weighted strategy, then
    plots: one panel per symbol (raw signal-strategy PnL) plus a final
    comparison panel overlaying two pooled, equal-weighted-across-symbols
    portfolios, both rescaled to the same `TARGET_ANNUAL_VOL` (see
    `scale_to_target_vol`) so their shapes are directly comparable —
    the naive always-long benchmark (`compute_benchmark_pnl`, no prediction
    used) vs. the z-score-weighted strategy (`compute_strategy_pnl`). PnL's
    "return" leg is `daily_return_by_symbol` (see `build_pooled_datasets`),
    a genuinely daily z-scored return, not the model's own overlapping
    N-day-ahead target. Only meaningful for `target_mode="zscore"` (a
    continuous position size); skipped for `target_mode="class"`, where the
    model has no continuous predicted magnitude to size a position with.
    """
    if cfg.target_mode != "zscore":
        logger.info("Skipping strategy PnL backtest: requires target_mode='zscore' (got %r).", cfg.target_mode)
        return

    TARGET_ANNUAL_VOL = 0.20

    symbols = list(val_datasets.keys())
    signal_pnl_by_symbol: dict[str, np.ndarray] = {}
    benchmark_pnl_by_symbol: dict[str, np.ndarray] = {}

    logger.info(
        "Strategy backtest (position = %d-day moving average of predicted z-score, realized return = daily z-score):",
        cfg.horizon,
    )
    for symbol in symbols:
        _target, predicted = collect_predictions(model, val_datasets[symbol], cfg)
        daily_return = daily_return_by_symbol[symbol]
        smoothed_predicted = smooth_weights(predicted, cfg.horizon)
        pnl = compute_strategy_pnl(daily_return, smoothed_predicted)
        signal_pnl_by_symbol[symbol] = pnl
        benchmark_pnl_by_symbol[symbol] = compute_benchmark_pnl(daily_return)
        sharpe = pnl.mean() / (pnl.std() + 1e-12)
        logger.info(
            "  %-8s total_pnl=%8.3f  mean_pnl=%+.5f  per-period Sharpe-like=%.3f  (n=%d)",
            symbol, pnl.sum(), pnl.mean(), sharpe, len(pnl),
        )

    # Equal-weighted pooling across symbols: average PnL at each time step,
    # truncated to the shortest symbol's validation length (symbols' val
    # sets differ slightly in length due to per-symbol dropna alignment, not
    # date-matched — a reasonable approximation, not exact backtesting infra).
    min_len = min(len(p) for p in signal_pnl_by_symbol.values())
    pooled_signal_pnl = np.mean([p[:min_len] for p in signal_pnl_by_symbol.values()], axis=0)
    pooled_benchmark_pnl = np.mean([p[:min_len] for p in benchmark_pnl_by_symbol.values()], axis=0)

    pooled_sharpe = pooled_signal_pnl.mean() / (pooled_signal_pnl.std() + 1e-12)
    logger.info(
        "  %-8s total_pnl=%8.3f  mean_pnl=%+.5f  per-period Sharpe-like=%.3f  (n=%d, equal-weighted across %d symbols)",
        "ALL", pooled_signal_pnl.sum(), pooled_signal_pnl.mean(), pooled_sharpe, min_len, len(symbols),
    )

    scaled_signal_pnl = scale_to_target_vol(pooled_signal_pnl, TARGET_ANNUAL_VOL)
    scaled_benchmark_pnl = scale_to_target_vol(pooled_benchmark_pnl, TARGET_ANNUAL_VOL)
    logger.info(
        "  vol-adjusted to %.0f%% annualized: signal total_pnl=%8.3f  equal-weighted-benchmark total_pnl=%8.3f",
        TARGET_ANNUAL_VOL * 100, scaled_signal_pnl.sum(), scaled_benchmark_pnl.sum(),
    )

    fig, axes = plt.subplots(len(symbols) + 1, 1, figsize=(12, 3 * (len(symbols) + 1)), sharex=False, squeeze=False)

    for i, symbol in enumerate(symbols):
        ax = axes[i, 0]
        ax.plot(np.cumsum(signal_pnl_by_symbol[symbol]), linewidth=1.2)
        ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.set_ylabel(f"{symbol}\ncum. PnL")

    ax = axes[-1, 0]
    ax.plot(
        np.cumsum(scaled_benchmark_pnl), label="equal-weighted portfolio (no signal)", linewidth=1.5, color="tab:grey"
    )
    ax.plot(np.cumsum(scaled_signal_pnl), label="z-score-weighted strategy", linewidth=1.5, color="tab:blue")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_ylabel(f"Pooled cum. PnL\n(vol-adj. {TARGET_ANNUAL_VOL:.0%})")
    ax.legend(loc="upper left", fontsize=8)

    axes[-1, 0].set_xlabel(
        "validation sample (time-ordered, per symbol; pooled panel truncated to shortest symbol)"
    )
    fig.suptitle(f"Backtest: cumulative PnL of investing the predicted {cfg.horizon}-day return z-score per symbol")
    fig.tight_layout()

    path = Path(cfg.pnl_plot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved strategy PnL plot to %s", path)


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
    """
    symbols = list(datasets.keys())
    fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 3 * len(symbols)), sharex=False, squeeze=False)

    for i, symbol in enumerate(symbols):
        actual, predicted = collect_predictions(model, datasets[symbol], cfg)
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
    parser.add_argument("--pairs", nargs="+", default=list(MAJOR_FX_PAIRS.keys()))
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=5, help="N days ahead for the cumulative-return target.")
    parser.add_argument(
        "--target-mode",
        choices=["zscore", "class"],
        default="zscore",
        help="Regression target: continuous return z-score (default) or 5-class direction label.",
    )
    parser.add_argument("--momentum-window", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 penalty passed to the Adam optimizer.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR multiplier applied on val loss plateau.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Epochs to wait before reducing LR.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
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
        default=1.0,
        help="target_mode='zscore' only: weight on (pred.std()-target.std())**2, to counteract collapse to a constant near-zero prediction. 0 disables.",
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
        "--pnl-plot-path",
        default="artifacts/lstm_pnl.png",
        help="Where to save the strategy backtest cumulative-PnL plot (target_mode='zscore' only).",
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
        seed=args.seed,
        plot_path=args.plot_path,
        in_sample_plot_path=args.in_sample_plot_path,
        pnl_plot_path=args.pnl_plot_path,
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

    train_ds, val_ds, train_datasets_by_symbol, val_datasets_by_symbol, daily_return_by_symbol, num_factors = (
        build_pooled_datasets(cfg)
    )

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
    else:
        model = build_model(cfg, num_factors)
        train(model, train_ds, val_ds, cfg)
        save_model(model, cfg, num_factors, cfg.model_path)

    plot_predictions_grid(model, val_datasets_by_symbol, cfg, cfg.plot_path, sample_label="out-of-sample (validation)")
    plot_predictions_grid(
        model, train_datasets_by_symbol, cfg, cfg.in_sample_plot_path, sample_label="in-sample (training)"
    )
    log_hit_rate_summary(model, val_datasets_by_symbol, cfg)
    log_and_plot_strategy_pnl(model, val_datasets_by_symbol, daily_return_by_symbol, cfg)


if __name__ == "__main__":
    main()
