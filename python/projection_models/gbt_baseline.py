"""Gradient-boosted-tree sanity check: is there *any* exploitable nonlinear signal in the
same engineered features `main_lstm.py` uses, before spending more effort on architecture?

Trains a histogram-based gradient-boosted tree (`sklearn.ensemble.HistGradientBoosting*` —
the same histogram-binning algorithm LightGBM popularized, ships fully self-contained with
scikit-learn's wheel, no external OpenMP/system library needed) on the exact same tabular
data `main_lstm.py::build_target_dataset` builds — same rolling-normalized features, same
purged train/val/test split, same `target_symbol` and `target_mode` — but flattened to one
row per forecast origin (a tree has no notion of a sequence; it only ever sees the single
feature vector available at time t, the row-level analogue of what the LSTM's
side-features-at-t and last-timestep factors also emphasize heavily via attention). Reports
hit rate via the exact same `compute_hit_rate` function `main_lstm.py` uses, so the numbers
are directly comparable.

Early stopping is done manually against `main_lstm.py`'s own purged validation split (via
`warm_start`, adding one tree at a time and tracking the best validation score) rather than
`HistGradientBoosting*`'s own built-in `early_stopping` option, which carves out its *own*
internal validation split at random from whatever training rows it's given — exactly the
kind of temporally-random split this project has been careful to avoid everywhere else
(a label at day t depends on prices through t+horizon, so a random split can leak information
across the boundary the purged split exists to embargo).

Motivation: naive linear features (momentum, carry, CMA crossovers — see the project's own
prior diagnostics) showed near-zero, sign-flipping-between-splits correlation with future
returns at every horizon tried, and the LSTM+attention+orthogonal-factor architecture
converges to an honest ~50% hit rate. Gradient-boosted trees are a well-established, cheap,
strong baseline for exactly this kind of tabular financial-return-prediction problem — Gu,
Kelly & Xiu (2020, "Empirical Asset Pricing via Machine Learning," Review of Financial
Studies) found trees competitive with or beating neural networks at moderate data scale, and
their nonlinear splits can pick up interaction effects (e.g. "momentum only works when
volatility is low") that a linear correlation check can't see and that a small, heavily
regularized LSTM might not learn either. A negative result here is real evidence the ceiling
is the data/signal, not the architecture; a positive result points at what kind of
nonlinearity the LSTM should be trying (and currently isn't) capturing.

Example:
    uv run python gbt_baseline.py --pairs EURUSD USDJPY GBPUSD USDCHF AUDUSD USDCAD NZDUSD \\
        --target-symbol EURUSD --years 15 --horizon 5
"""

from __future__ import annotations

import argparse
import copy
import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from main_lstm import Config, FXSequenceDataset, build_target_dataset, compute_hit_rate

logger = logging.getLogger(__name__)


def flatten_dataset(dataset: FXSequenceDataset) -> tuple[np.ndarray, np.ndarray]:
    """Row-level (non-windowed) features/target at each sample's forecast origin.

    A gradient-boosted tree has no notion of a sequence — it only ever sees the single
    feature vector available at time t, `dataset.features[origin]` (the same rolling-
    normalized row `FXSequenceDataset.__getitem__`'s window ends on), not the trailing
    `seq_len`-day window the LSTM consumes. `dataset.valid_indices` is the exact same set of
    forecast origins `main_lstm.py`'s own windowed dataset uses, so the resulting rows align
    to the identical dates/split as any `main_lstm.py` run over the same `Config`.
    """
    idx = dataset.valid_indices
    X = dataset.features[idx].numpy()
    y = dataset.targets[idx].numpy()
    return X, y


def fit_with_early_stopping(
    model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    max_iter: int, early_stopping_rounds: int, is_classification: bool,
):
    """Adds one tree at a time (`warm_start`) and tracks the best validation score, mirroring
    `main_lstm.train`'s own best-checkpoint-on-val-loss pattern — deliberately not
    `HistGradientBoosting*`'s built-in `early_stopping`, which would carve its own random
    (non-chronological) validation split out of `X_train`, undermining the purged split this
    project relies on everywhere else. Returns `(best_model, best_iteration, best_val_score)`.
    """
    best_val_score = float("inf")
    best_iteration = 0
    best_model = None
    epochs_without_improvement = 0

    for i in range(1, max_iter + 1):
        model.max_iter = i
        model.fit(X_train, y_train)

        if is_classification:
            val_proba = model.predict_proba(X_val)
            val_score = log_loss(y_val, val_proba, labels=model.classes_)
        else:
            val_pred = model.predict(X_val)
            val_score = float(np.mean((val_pred - y_val) ** 2))

        if val_score < best_val_score:
            best_val_score = val_score
            best_iteration = i
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_rounds:
                break

    return best_model, best_iteration, best_val_score


def log_feature_importance(
    model, X_val: np.ndarray, y_val: np.ndarray, feature_names: list[str], seed: int, top_n: int = 15
) -> None:
    """Permutation importance on the validation set — how much validation performance drops
    when a feature's values are shuffled, a direct measure of predictive contribution, unlike
    split-count/gain-based importance (which histogram-based GBMs in scikit-learn don't even
    expose) that can be biased toward high-cardinality features.
    """
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=seed, n_jobs=-1)
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    logger.info("Top %d feature importances (permutation, validation set):", min(top_n, len(order)))
    for rank, i in enumerate(order, start=1):
        logger.info("  %2d. %-20s %+.5f (+/- %.5f)", rank, feature_names[i], result.importances_mean[i], result.importances_std[i])


def train_and_evaluate(cfg: Config, params: dict, max_iter: int, early_stopping_rounds: int) -> None:
    (
        train_ds,
        val_ds,
        test_ds,
        _train_by_symbol,
        _val_by_symbol,
        _test_by_symbol,
        market_data_by_symbol,
        num_factors,
        num_side_features,
        num_target_series,
        feature_names,
    ) = build_target_dataset(cfg)

    X_train, y_train = flatten_dataset(train_ds)
    X_val, y_val = flatten_dataset(val_ds)
    X_test, y_test = flatten_dataset(test_ds)
    logger.info(
        "Flattened tabular data: %d features (%d cross-asset + %d target series + %d side), "
        "train=%d val=%d test=%d",
        num_factors, len(cfg.pairs), num_target_series, num_side_features,
        len(X_train), len(X_val), len(X_test),
    )

    is_classification = cfg.target_mode == "class"
    model = (HistGradientBoostingClassifier if is_classification else HistGradientBoostingRegressor)(
        early_stopping=False, warm_start=True, **params
    )
    model, best_iteration, best_val_score = fit_with_early_stopping(
        model, X_train, y_train, X_val, y_val, max_iter, early_stopping_rounds, is_classification
    )
    logger.info(
        "Best iteration: %d/%d (validation %s=%.5f)",
        best_iteration, max_iter, "log_loss" if is_classification else "mse", best_val_score,
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    if is_classification:
        symbol = cfg.target_symbol
        val_actual_zscore = market_data_by_symbol[symbol]["horizon_zscore"].reindex(val_ds.sample_dates).to_numpy()
        test_actual_zscore = market_data_by_symbol[symbol]["horizon_zscore"].reindex(test_ds.sample_dates).to_numpy()
        val_hit = compute_hit_rate(val_pred, val_actual_zscore, "class")
        test_hit = compute_hit_rate(test_pred, test_actual_zscore, "class")
    else:
        val_hit = compute_hit_rate(val_pred, y_val, "zscore")
        test_hit = compute_hit_rate(test_pred, y_test, "zscore")
        logger.info("Validation correlation(pred, actual): %+.4f", np.corrcoef(val_pred, y_val)[0, 1])
        logger.info("Test correlation(pred, actual):       %+.4f", np.corrcoef(test_pred, y_test)[0, 1])

    logger.info("Validation hit rate: %5.1f%%  (n=%d)", val_hit * 100, len(X_val))
    logger.info("TEST hit rate:       %5.1f%%  (n=%d)  <- final holdout, evaluate once", test_hit * 100, len(X_test))
    log_feature_importance(model, X_val, y_val, feature_names, cfg.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument(
        "--target-symbol", required=True,
        help="The one pair to forecast (must be one of --pairs); --pairs still supplies the "
        "full cross-asset input universe (every pair's own log return is a raw input column).",
    )
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=60, help="Rolling-normalization window (same role as in main_lstm.py).")
    parser.add_argument("--horizon", type=int, default=5, help="N days ahead for the cumulative-return target.")
    parser.add_argument(
        "--target-mode", choices=["zscore", "class"], default="zscore",
        help="Regression on the continuous return z-score (default), or 5-class direction label.",
    )
    parser.add_argument(
        "--cross-sectional-target", action="store_true",
        help="Predict target_symbol's return relative to the cross-sectional median of the "
        "other --pairs symbols, instead of its own absolute direction.",
    )
    parser.add_argument("--momentum-window", type=int, default=15)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=500, help="Max boosting iterations (early-stopped on validation).")
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=4, help="Shallow trees — this is a ~thousand-row tabular problem, not a big-data one.")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Shrink data for a fast, steppable run.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        args.pairs = args.pairs[:2] if len(args.pairs) > 2 else args.pairs
        args.years = min(args.years, 3)
        args.seq_len = min(args.seq_len, 20)
        args.max_iter = 30
        args.early_stopping_rounds = 10
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

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
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        write_features_csv=False,
        seed=args.seed,
    )
    logger.info("Config: %s", cfg)

    params = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "l2_regularization": args.l2_regularization,
        "random_state": args.seed,
    }
    train_and_evaluate(cfg, params, args.max_iter, args.early_stopping_rounds)


if __name__ == "__main__":
    main()
