"""Gradient-boosted-tree classifier: cross-asset FX log returns -> 5-class direction forecast.

The *only* input features are the rolling-normalized daily log returns of `--pairs` (one
column per pair, `--target-symbol` included) — no engineered momentum/vol/skew/carry
features, no cross-asset factor reduction, no recurrence. A tree sees one flat feature
vector per forecast origin, the same rolling-normalized row `main_lstm.py`'s own pipeline
would compute (`fx_forecasting.features.compute_rolling_normalized_features`, same causal,
per-column rolling z-score), just without any of that pipeline's non-return columns.

Target: `--target-symbol`'s `--horizon`-day cumulative return, bucketed into 5 classes —
very_bearish=-2, bearish=-1, neutral=0, bullish=1, very_bullish=2 — by the same 10/20/40/20/10
quantile split `main_lstm.py`'s own `compute_target_class` uses
(`fx_forecasting.features.CLASS_QUANTILE_EDGES`): the two "very_*" tail classes are each the
most extreme 10% of moves, "bearish"/"bullish" the next 20% each side, "neutral" the middle
40%. That split is a reasonable stand-in for "typical trading strategy" boundaries: it only
calls a *strong* (very_bearish/very_bullish) signal on genuine tail moves, calls a *moderate*
signal on the next tier, and — critically — stays neutral on the ambiguous middle 40% of days
rather than forcing a directional call on every single one, which is closer to how a real
discretionary or systematic strategy would size conviction than an even 20/20/20/20/20 split
would be.

Uses scikit-learn's `HistGradientBoostingClassifier` (see `gbt_baseline.py` for why: a
histogram-based GBM in the same family as LightGBM, ships fully self-contained with
scikit-learn's own wheel, no external OpenMP/system library needed on this machine, unlike
xgboost). Same purged train/val/test split (`fx_forecasting.trading.
purged_train_val_test_split`) and the same manual, purged-validation-based early stopping
(not scikit-learn's own built-in `early_stopping`, which would carve its own random,
non-chronological split out of the training rows) as `gbt_baseline.py` — reuses that
module's `fit_with_early_stopping` directly rather than duplicating it.

Example:
    uv run python gbt_classifier.py --pairs EURUSD USDJPY GBPUSD USDCHF AUDUSD USDCAD NZDUSD \\
        --target-symbol EURUSD --years 15 --horizon 5
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from fx_forecasting.features import (
    compute_log_returns,
    compute_rolling_normalized_features,
    compute_target_class,
    compute_target_zscore,
)
from fx_forecasting.trading import purged_train_val_test_split
from gbt_baseline import fit_with_early_stopping
from main_lstm import Config, compute_hit_rate, load_price_panel

logger = logging.getLogger(__name__)

# very_bearish=-2 .. very_bullish=2 — the signed encoding requested for this model, as
# opposed to main_lstm.py's own 0-4 index convention (fx_forecasting.features.CLASS_NAMES),
# where index 2 is neutral. Both describe the same 5 classes/boundaries; only the label
# values differ, so predictions are shifted +2 wherever main_lstm.py's own 0-4-based
# compute_hit_rate is reused, rather than duplicating that logic for a shifted label space.
TRADING_CLASS_NAMES: dict[int, str] = {-2: "very_bearish", -1: "bearish", 0: "neutral", 1: "bullish", 2: "very_bullish"}


def build_features_and_target(cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns `(features, target_class, target_zscore)`, all aligned to the same index.

    `features`: rolling-normalized `xret_{pair}` log returns, one column per `cfg.pairs`
    symbol (target included) — the *only* thing this model ever sees.
    `target_class`: `cfg.target_symbol`'s horizon-day return class, in the -2..2 encoding.
    `target_zscore`: the same target's continuous z-score (`compute_target_zscore`) — not a
    model input, kept only so `compute_hit_rate`'s `target_mode="class"` path has the
    continuous "actual" value it needs to judge a neutral-band hit against (see
    `main_lstm.py::log_hit_rate_summary` for the same pattern).
    """
    panel = load_price_panel(cfg)
    log_returns = compute_log_returns(panel)

    xret_columns = [f"xret_{pair}" for pair in sorted(cfg.pairs)]
    xret_frame = pd.DataFrame({col: log_returns[pair] for col, pair in zip(xret_columns, sorted(cfg.pairs))})
    normalized = compute_rolling_normalized_features(xret_frame, cfg.seq_len)

    target_0_4 = compute_target_class(panel, log_returns, cfg.target_symbol, cfg.horizon, cfg.seq_len)
    target_class = (target_0_4 - 2).rename("target_class")
    target_zscore = compute_target_zscore(panel, log_returns, cfg.target_symbol, cfg.horizon, cfg.seq_len)

    aligned_index = normalized.index.intersection(target_class.index).intersection(target_zscore.index)
    return normalized.loc[aligned_index], target_class.loc[aligned_index], target_zscore.loc[aligned_index]


def split_flat(
    features: pd.DataFrame, target_class: pd.Series, target_zscore: pd.Series, cfg: Config
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Purged chronological train/val/test split (embargoed by `cfg.horizon`, same
    discipline as everywhere else in this project — see
    `fx_forecasting.trading.purged_train_val_test_split`), applied to flat (non-windowed)
    rows — a tree has no notion of a sequence, so there's no `FXSequenceDataset` here.

    Returns `{"train": (X, y_class, y_zscore, dates), "val": ..., "test": ...}`.
    """
    split = purged_train_val_test_split(len(features), cfg.val_fraction, cfg.test_fraction, cfg.horizon, cfg.seq_len)
    X = features.to_numpy(dtype=np.float32)
    y_class = target_class.to_numpy()
    y_zscore = target_zscore.to_numpy()
    dates = features.index

    bounds = {
        "train": slice(0, split.train_end),
        "val": slice(split.val_start, split.val_end),
        "test": slice(split.test_start, split.test_end),
    }
    return {name: (X[s], y_class[s], y_zscore[s], dates[s]) for name, s in bounds.items()}


def log_class_distribution(label: str, y_class: np.ndarray) -> None:
    counts = {TRADING_CLASS_NAMES[c]: float(np.mean(y_class == c)) for c in sorted(TRADING_CLASS_NAMES)}
    logger.info("Class distribution (%s): %s", label, {k: f"{v:.1%}" for k, v in counts.items()})


def log_feature_importance(model, X_val: np.ndarray, y_val: np.ndarray, feature_names: list[str], seed: int, top_n: int = 15) -> None:
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=seed, n_jobs=-1)
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    logger.info("Top %d feature importances (permutation, validation set):", min(top_n, len(order)))
    for rank, i in enumerate(order, start=1):
        logger.info("  %2d. %-16s %+.5f (+/- %.5f)", rank, feature_names[i], result.importances_mean[i], result.importances_std[i])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", required=True, help="Cross-asset universe; every pair's own log return is a feature column.")
    parser.add_argument("--target-symbol", required=True, help="The one pair to forecast (must be one of --pairs).")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=60, help="Rolling-normalization window for the log-return features.")
    parser.add_argument("--horizon", type=int, default=5, help="N days ahead for the cumulative-return target.")
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

    cfg = Config(pairs=args.pairs, target_symbol=args.target_symbol, years=args.years, seq_len=args.seq_len, horizon=args.horizon,
                 val_fraction=args.val_fraction, test_fraction=args.test_fraction, write_features_csv=False, seed=args.seed)
    logger.info("Config: pairs=%s target_symbol=%s years=%d seq_len=%d horizon=%d", cfg.pairs, cfg.target_symbol, cfg.years, cfg.seq_len, cfg.horizon)

    features, target_class, target_zscore = build_features_and_target(cfg)
    feature_names = list(features.columns)
    splits = split_flat(features, target_class, target_zscore, cfg)
    X_train, y_train, _, _ = splits["train"]
    X_val, y_val, yz_val, val_dates = splits["val"]
    X_test, y_test, yz_test, test_dates = splits["test"]
    logger.info(
        "Flattened tabular data: %d features (log returns only), train=%d val=%d test=%d",
        len(feature_names), len(X_train), len(X_val), len(X_test),
    )
    log_class_distribution("train", y_train)

    model = HistGradientBoostingClassifier(
        early_stopping=False, warm_start=True, max_depth=args.max_depth, learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization, random_state=args.seed,
    )
    model, best_iteration, best_val_score = fit_with_early_stopping(
        model, X_train, y_train, X_val, y_val, args.max_iter, args.early_stopping_rounds, is_classification=True
    )
    logger.info("Best iteration: %d/%d (validation log_loss=%.5f)", best_iteration, args.max_iter, best_val_score)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    log_class_distribution("validation predictions", val_pred)
    log_class_distribution("test predictions", test_pred)

    # compute_hit_rate's target_mode="class" path assumes main_lstm.py's own 0-4 index
    # convention (neutral = index 2) — shift both sides of the -2..2 encoding back to 0-4
    # to reuse it rather than duplicating that logic for a shifted label space.
    val_hit = compute_hit_rate(val_pred + 2, yz_val, "class")
    test_hit = compute_hit_rate(test_pred + 2, yz_test, "class")
    logger.info("Validation hit rate: %5.1f%%  (n=%d)", val_hit * 100, len(X_val))
    logger.info("TEST hit rate:       %5.1f%%  (n=%d)  <- final holdout, evaluate once", test_hit * 100, len(X_test))

    log_feature_importance(model, X_val, y_val, feature_names, cfg.seed)


if __name__ == "__main__":
    main()
