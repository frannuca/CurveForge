"""Pretrains the orthogonal cross-asset factor autoencoder consumed by `main_lstm.py`.

Compresses the daily log returns of `--pairs` (K series, target symbol included) down to
`--factor-dim` (D) orthogonal factors — a linear, orthonormal-row encoder
(`fx_forecasting.models.orthogonal_autoencoder.OrthogonalAutoencoder`) trained by gradient
descent to minimize reconstruction MSE, which (per Baldi & Hornik, 1989) recovers the same
subspace classical PCA would, with the individual factors themselves also mutually orthogonal
by construction (not just spanning the right subspace) thanks to the orthonormal-row
constraint. This is a plain cross-sectional reconstruction task — each day's K-dim return
vector is encoded/decoded independently, no sequence windowing involved.

Must be run once before `main_lstm.py --autoencoder-path ...` — the main model loads this
checkpoint's encoder weights rather than training the reduction itself. The checkpoint
records exactly which `--pairs`/`--seq-len` it was pretrained for; `main_lstm.py` refuses to
load a mismatched one, since the rolling-normalization window and pair universe must match
for the pretrained weights to mean anything.

Example:
    uv run python pretrain_autoencoder.py --pairs EURUSD USDJPY GBPUSD AUDUSD --factor-dim 2
    uv run python main_lstm.py --target-symbol EURUSD \\
        --autoencoder-path artifacts/cross_asset_autoencoder.pt
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from fx_forecasting.data.fx_downloader import MAJOR_FX_PAIRS
from fx_forecasting.features import compute_log_returns, compute_rolling_normalized_features
from fx_forecasting.models.orthogonal_autoencoder import OrthogonalAutoencoder
from fx_forecasting.trading import purged_train_val_test_split
from main_lstm import Config, load_raw_panels, set_seed

logger = logging.getLogger(__name__)

# Set to cooperatively stop an in-progress `train_autoencoder()` call early (checked once
# per epoch) — mirrors `main_lstm.CANCEL_EVENT`, so a dashboard "Stop" button works the
# same way for this script.
CANCEL_EVENT = threading.Event()


def build_factor_frame(cfg: Config):
    """Rolling-normalized daily log returns for every pair in `cfg.pairs`, sorted, one
    column per pair (`xret_{pair}`) — identical convention/window to the cross-asset block
    `main_lstm.build_target_dataset` builds, so this checkpoint's encoder sees the same
    kind of input at inference time it was pretrained on.
    """
    raw = load_raw_panels(cfg)
    log_returns = compute_log_returns(raw.panel)
    pairs = sorted(cfg.pairs)
    xret_frame = pd.DataFrame({f"xret_{pair}": log_returns[pair] for pair in pairs})
    normalized = compute_rolling_normalized_features(xret_frame, cfg.seq_len)
    return normalized[[f"xret_{pair}" for pair in pairs]]


def train_autoencoder(
    model: OrthogonalAutoencoder,
    train_values: np.ndarray,
    val_values: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    early_stop_patience: int,
    device: str,
) -> float:
    """Trains `model` in place via plain MSE reconstruction; restores it to its
    best-val_loss epoch's weights, mirroring `main_lstm.train`'s best-checkpoint-restore and
    early-stopping behavior (and its epoch log line format, so the dashboard's existing
    `train_progress` regex works unchanged for this script too).

    Returns the best val_loss reached.
    """
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(train_values.copy(), dtype=torch.float32)), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(val_values.copy(), dtype=torch.float32)), batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    CANCEL_EVENT.clear()

    for epoch in range(1, epochs + 1):
        if CANCEL_EVENT.is_set():
            logger.info("Pretraining cancelled by user at epoch %d/%d", epoch, epochs)
            break

        model.train()
        train_loss_total, train_n = 0.0, 0
        for (x,) in train_loader:
            x = x.to(device)
            x_hat = model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * x.size(0)
            train_n += x.size(0)
        train_loss = train_loss_total / train_n

        model.eval()
        val_loss_total, val_n = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_hat = model(x)
                loss = torch.nn.functional.mse_loss(x_hat, x)
                val_loss_total += loss.item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_loss_total / val_n

        logger.info(
            "epoch %02d/%d  train_mse=%.6f  val_mse=%.6f  orthogonality_error=%.2e",
            epoch, epochs, train_loss, val_loss, model.orthogonality_error(),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (no val_mse improvement for %d epochs)", epoch, early_stop_patience
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best checkpoint: epoch %d/%d, val_mse=%.6f", best_epoch, epochs, best_val_loss)

    return best_val_loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=list(MAJOR_FX_PAIRS.keys()))
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument(
        "--seq-len", type=int, default=60,
        help="Rolling-normalization window; must match the --seq-len main_lstm.py will use.",
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Used only to size the train/val/test split boundaries the same way "
        "main_lstm.py's purged split would (no forward-looking label exists in this "
        "reconstruction task, so there's no leakage risk from this value itself).",
    )
    parser.add_argument(
        "--factor-dim", type=int, default=4,
        help="D: number of orthogonal factors to reduce the K pairs down to (must be < len(--pairs)).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--autoencoder-path", default="artifacts/cross_asset_autoencoder.pt",
        help="Where to save the pretrained encoder checkpoint.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Shrink data/model drastically for a fast, steppable run."
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        args.pairs = args.pairs[:2] if len(args.pairs) > 2 else args.pairs
        args.years = min(args.years, 3)
        args.seq_len = min(args.seq_len, 20)
        args.epochs = 1
        args.batch_size = 8
        args.factor_dim = 1
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    if len(args.pairs) < 2:
        raise ValueError(f"--pairs needs at least 2 pairs to compress; got {args.pairs}")
    if not 0 < args.factor_dim < len(args.pairs):
        raise ValueError(f"--factor-dim ({args.factor_dim}) must be strictly between 0 and len(--pairs) ({len(args.pairs)})")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Config(pairs=args.pairs, years=args.years, seq_len=args.seq_len, seed=args.seed)
    normalized = build_factor_frame(cfg)
    pairs = sorted(args.pairs)

    split = purged_train_val_test_split(
        len(normalized), args.val_fraction, args.test_fraction, args.horizon, args.seq_len
    )
    values = normalized.to_numpy(dtype=np.float32)
    train_values = values[: split.train_end]
    val_values = values[split.val_start : split.val_end]
    test_values = values[split.test_start : split.test_end]
    logger.info(
        "Cross-asset factor data: %d pairs, %d rows (train=%d, val=%d, test=%d)",
        len(pairs), len(normalized), len(train_values), len(val_values), len(test_values),
    )

    model = OrthogonalAutoencoder(len(pairs), args.factor_dim).to(device)
    logger.info(
        "Autoencoder: %d cross-asset factors -> %d orthogonal factors (orthogonality_error=%.2e at init)",
        len(pairs), args.factor_dim, model.orthogonality_error(),
    )

    train_autoencoder(
        model, train_values, val_values, args.epochs, args.batch_size, args.lr, args.weight_decay,
        args.early_stop_patience, device,
    )

    model.eval()
    with torch.no_grad():
        test_x = torch.as_tensor(test_values.copy(), dtype=torch.float32).to(device)
        test_mse = torch.nn.functional.mse_loss(model(test_x), test_x).item()
    logger.info(
        "FINAL HOLDOUT (test set): test_mse=%.6f  orthogonality_error=%.2e", test_mse, model.orthogonality_error()
    )

    checkpoint = {
        "encoder_state": model.encoder.state_dict(),
        "num_cross_asset_factors": len(pairs),
        "factor_dim": args.factor_dim,
        "pairs": pairs,
        "seq_len": args.seq_len,
    }
    path_obj = Path(args.autoencoder_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path_obj)
    logger.info("Saved pretrained cross-asset autoencoder to %s", path_obj)


if __name__ == "__main__":
    main()
