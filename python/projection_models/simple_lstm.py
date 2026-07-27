"""Train a simple (no-attention) LSTM classifier on a single FX symbol.

Unlike `main_lstm.py` (every symbol pooled into one shared-weight model,
attention pooling over the encoder, a 5-class or continuous z-score target),
this is the minimal version: one symbol, a plain LSTM (final hidden state ->
linear -> logits, see `SimpleLSTMClassifier`), and a 3-class direction
target — bearish (-1), neutral (0), bullish (+1) — for the `horizon`-day-
ahead cumulative return (see `fx_forecasting.features.compute_target_direction`).
Reuses the same feature engineering (return, intraday_vol, momentum, rvol,
carry, CMA crossovers) and Postgres/FRED data loading as `main_lstm.py`.

Example:
    uv run python simple_lstm.py --symbol EURUSD --debug
    uv run python simple_lstm.py --symbol EURUSD --years 15 --epochs 100
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from fx_forecasting.features import (
    DEFAULT_CMA_WINDOWS,
    DIRECTION_CLASS_NAMES,
    CMAWindow,
    CrossingMovingAverages,
    build_symbol_frame,
    compute_carry_feature,
    compute_engineered_features,
    compute_log_returns,
    compute_rolling_normalized_features,
    compute_target_direction,
)
from fx_forecasting.models.simple_lstm_classifier import SimpleLSTMClassifier
from main_lstm import (
    FXSequenceDataset,
    build_datasets,
    class_distance_weights,
    load_raw_panels,
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    symbol: str = "EURUSD"
    years: int = 20

    seq_len: int = 40
    horizon: int = 5  # N days ahead for the cumulative-return direction target
    momentum_window: int = 5
    vol_window: int = 10
    cma_windows: list[CMAWindow] = field(default_factory=lambda: list(DEFAULT_CMA_WINDOWS))

    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-6
    early_stop_patience: int = 20
    outlier_weight: float = 5.0  # extra loss weight for bearish/bullish vs neutral; 0 = plain cross-entropy
    val_fraction: float = 0.2
    test_fraction: float = 0.1  # held out, untouched, for build_datasets' purged split (main_lstm.py); unused here otherwise

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    plot_path: str = "artifacts/simple_lstm_predictions.png"
    model_path: str = "artifacts/simple_lstm_model.pt"
    infer: bool = False  # skip training; load `model_path` and run inference only

    @property
    def pairs(self) -> list[str]:
        """`load_raw_panels` (from main_lstm.py, reused here) expects a `.pairs` list."""
        return [self.symbol]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(cfg: Config) -> tuple[FXSequenceDataset, FXSequenceDataset, FXSequenceDataset, int]:
    """Builds the (train, val, test) dataset for `cfg.symbol` — same feature engineering and
    purged chronological split as `main_lstm.py`'s pooled pipeline (see `build_datasets`), just
    for one symbol, with no pooling at all. `test` is returned but not otherwise used here — this
    script has no test-holdout evaluation of its own yet."""
    raw = load_raw_panels(cfg)
    log_returns = compute_log_returns(raw.panel)
    engineered = compute_engineered_features(
        raw.panel, log_returns, raw.high, raw.low, cfg.momentum_window, cfg.vol_window
    )
    carry = compute_carry_feature([cfg.symbol], raw.rates)
    cma = CrossingMovingAverages(cfg.cma_windows).compute(raw.panel)
    symbol_cma = cma[[f"{cfg.symbol}_{w.name}" for w in cfg.cma_windows]]
    symbol_cma.columns = [w.name for w in cfg.cma_windows]

    frame = build_symbol_frame(
        cfg.symbol,
        log_returns,
        engineered,
        cfg.momentum_window,
        cfg.vol_window,
        carry=carry[cfg.symbol],
        cma=symbol_cma,
    )
    normalized = compute_rolling_normalized_features(frame, cfg.seq_len)
    num_factors = normalized.shape[1]

    target = compute_target_direction(raw.panel, log_returns, cfg.symbol, cfg.horizon, cfg.seq_len)

    train_ds, val_ds, test_ds = build_datasets(normalized, target, cfg)
    logger.info(
        "%s: %d factors, train=%d, val=%d, test=%d", cfg.symbol, num_factors, len(train_ds), len(val_ds), len(test_ds)
    )
    return train_ds, val_ds, test_ds, num_factors


def build_model(cfg: Config, num_factors: int) -> SimpleLSTMClassifier:
    model = SimpleLSTMClassifier(
        num_factors=num_factors,
        num_classes=len(DIRECTION_CLASS_NAMES),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %d factors, %d-day / %d-class direction target, %d params",
        num_factors, cfg.horizon, len(DIRECTION_CLASS_NAMES), n_params,
    )
    return model


def save_model(model: SimpleLSTMClassifier, cfg: Config, num_factors: int, path: str) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "num_factors": num_factors,
        "num_classes": model.num_classes,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "symbol": cfg.symbol,
        "horizon": cfg.horizon,
        "seq_len": cfg.seq_len,
        "momentum_window": cfg.momentum_window,
        "vol_window": cfg.vol_window,
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path_obj)
    logger.info("Saved model checkpoint to %s", path_obj)


def load_model(path: str, device: str) -> tuple[SimpleLSTMClassifier, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SimpleLSTMClassifier(
        num_factors=checkpoint["num_factors"],
        num_classes=checkpoint["num_classes"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logger.info(
        "Loaded model checkpoint from %s (symbol=%s, horizon=%s, seq_len=%s)",
        path, checkpoint["symbol"], checkpoint["horizon"], checkpoint["seq_len"],
    )
    return model, checkpoint


def run_epoch(
    model: SimpleLSTMClassifier, loader: DataLoader, cfg: Config, optimizer: torch.optim.Optimizer | None
) -> tuple[float, float]:
    """Runs one pass over `loader`. Trains if `optimizer` is given, else evaluates.

    Returns (weighted cross-entropy loss, plain accuracy) — accuracy is the
    fraction of samples where the model's argmax prediction exactly matches
    the true class, a plain unweighted diagnostic distinct from the
    outlier-focused loss actually being optimized (see `class_distance_weights`).
    """
    is_train = optimizer is not None
    model.train(is_train)

    class_weights = class_distance_weights(cfg.outlier_weight, model.num_classes).to(cfg.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = loss_fn(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_count += batch_size

    n = len(loader.dataset)
    return total_loss / n, total_correct / total_count


def train(model: SimpleLSTMClassifier, train_ds: FXSequenceDataset, val_ds: FXSequenceDataset, cfg: Config) -> float:
    """Trains `model` in place, then restores it to its best-val_loss epoch's weights.

    Same overfitting-aware design as `main_lstm.py::train`: tracks the
    best-val_loss checkpoint throughout (rather than just using the last
    epoch's weights) and stops early once val_loss hasn't improved for
    `cfg.early_stop_patience` epochs. Returns the best val_loss reached.
    """
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
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


def collect_predictions(model: SimpleLSTMClassifier, dataset: FXSequenceDataset, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over `dataset` in time order; returns (actual, predicted) class labels."""
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model.eval()
    actual_batches, pred_batches = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(cfg.device)).cpu()
            actual_batches.append(y)
            pred_batches.append(logits.argmax(dim=-1))

    actual = torch.cat(actual_batches).numpy()
    predicted = torch.cat(pred_batches).numpy()
    return actual, predicted


def plot_predictions(model: SimpleLSTMClassifier, val_ds: FXSequenceDataset, cfg: Config) -> None:
    """Actual vs. predicted direction class over time, for the validation set."""
    actual, predicted = collect_predictions(model, val_ds, cfg)
    num_classes = len(DIRECTION_CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(actual, label="actual", linewidth=1.0, marker=".", markersize=3)
    ax.plot(predicted, label="predicted", linewidth=1.0, alpha=0.8, marker=".", markersize=3)
    ax.axhline((num_classes - 1) / 2.0, color="grey", linewidth=0.5)
    ax.set_ylim(-0.5, num_classes - 0.5)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(DIRECTION_CLASS_NAMES, fontsize=8)
    ax.set_xlabel("validation sample (time-ordered)")
    ax.set_ylabel(cfg.symbol)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{cfg.symbol}: actual vs. predicted {cfg.horizon}-day direction")
    fig.tight_layout()

    path = Path(cfg.plot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved prediction plot to %s", path)

    hit_rate = float(np.mean(actual == predicted))
    logger.info("Hit rate (predicted class == actual class): %.1f%% (n=%d)", hit_rate * 100, len(actual))


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser — factored out of `parse_args` so other tools (e.g. a UI) can
    introspect the full set of available parameters without duplicating this list."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="EURUSD", help="Single FX symbol to train on.")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=5, help="N days ahead for the direction target.")
    parser.add_argument("--momentum-window", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 penalty passed to the Adam optimizer.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR multiplier applied on val loss plateau.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Epochs to wait before reducing LR.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--early-stop-patience", type=int, default=20,
        help="Stop training if val_loss hasn't improved in this many epochs.",
    )
    parser.add_argument(
        "--outlier-weight", type=float, default=5.0,
        help="Extra loss weight for bearish/bullish vs. neutral (linear in distance from neutral). 0 = plain cross-entropy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-path", default="artifacts/simple_lstm_predictions.png")
    parser.add_argument(
        "--model-path", default="artifacts/simple_lstm_model.pt",
        help="Where to save the trained model, or load it from when --infer is given.",
    )
    parser.add_argument(
        "--infer", action="store_true", help="Skip training; load --model-path and only run inference/plotting."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Shrink data/model drastically for a fast, steppable run (1 epoch)."
    )
    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.debug:
        args.years = min(args.years, 3)
        args.seq_len = min(args.seq_len, 20)
        args.hidden_size = 8
        args.epochs = 1
        args.batch_size = 8

    return Config(
        symbol=args.symbol,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
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
        seed=args.seed,
        plot_path=args.plot_path,
        model_path=args.model_path,
        infer=args.infer,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args(argv)
    logger.info("Config: %s", cfg)
    set_seed(cfg.seed)

    train_ds, val_ds, _test_ds, num_factors = build_dataset(cfg)

    if cfg.infer:
        model, checkpoint = load_model(cfg.model_path, cfg.device)
        if checkpoint["num_factors"] != num_factors:
            raise ValueError(
                f"Checkpoint expects {checkpoint['num_factors']} factors but the current "
                f"feature set produces {num_factors}; use the same feature-window flags as training."
            )
    else:
        model = build_model(cfg, num_factors)
        train(model, train_ds, val_ds, cfg)
        save_model(model, cfg, num_factors, cfg.model_path)

    plot_predictions(model, val_ds, cfg)


if __name__ == "__main__":
    main()
