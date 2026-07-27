"""Regime-conditional diagnostic: is `target_symbol`'s predictability (hit rate,
correlation) concentrated in a particular volatility regime, rather than uniform across all
market conditions — before building any regime-*gated* trading logic around that hypothesis.

Trains the same gradient-boosted tree `gbt_baseline.py` does (same features, same purged
split, same target), then breaks its validation/test predictions down by two causal
volatility-regime labels (see `fx_forecasting.features.compute_volatility_regime`):

- `target_symbol`'s own realized-vol regime (idiosyncratic — is *this pair* unusually
  volatile right now).
- A market-wide regime: the realized vol of the cross-sectional median return across every
  other `--pairs` symbol (a simple risk-on/risk-off proxy — is the *whole universe*
  unusually volatile right now, not just this one pair).

Both are reported for validation *and* test separately, deliberately — the whole point is to
check whether any apparent regime-conditional edge on validation actually replicates
out-of-sample on test, the same discipline this project applies everywhere else. A pattern
that only shows up in one split is noise, not a regime effect worth building a gate around.

Example:
    uv run python regime_analysis.py --pairs EURUSD USDJPY GBPUSD USDCHF AUDUSD USDCAD NZDUSD \\
        --target-symbol EURUSD --years 15 --horizon 5
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from fx_forecasting.features import compute_log_returns, compute_volatility_regime
from gbt_baseline import build_parser as gbt_build_parser
from gbt_baseline import fit_with_early_stopping, flatten_dataset
from main_lstm import Config, build_target_dataset, compute_hit_rate, load_raw_panels

logger = logging.getLogger(__name__)


def regime_breakdown(
    predicted: np.ndarray,
    actual: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: list[str],
    target_mode: str,
    actual_zscore: np.ndarray | None,
) -> None:
    """Logs hit rate (+ correlation for `target_mode='zscore'`) separately for each regime
    bucket. `regime_labels` may contain NaN (rows before the regime detector's own warm-up
    period) — those rows simply never match any `regime_id` and are silently excluded from
    every bucket, not counted anywhere.
    """
    hit_rate_actual = actual_zscore if actual_zscore is not None else actual
    for regime_id, name in enumerate(regime_names):
        mask = regime_labels == regime_id
        n = int(mask.sum())
        if n < 5:
            logger.info("  %-10s n=%-4d (too few samples to report)", name, n)
            continue
        hit = compute_hit_rate(predicted[mask], hit_rate_actual[mask], target_mode)
        if target_mode == "zscore":
            corr = float(np.corrcoef(predicted[mask], actual[mask])[0, 1])
            logger.info("  %-10s n=%-4d  hit_rate=%5.1f%%  corr=%+.4f", name, n, hit * 100, corr)
        else:
            logger.info("  %-10s n=%-4d  hit_rate=%5.1f%%", name, n, hit * 100)


def regime_names_for(n_regimes: int) -> list[str]:
    if n_regimes == 3:
        return ["low_vol", "mid_vol", "high_vol"]
    return [f"regime_{i}" for i in range(n_regimes)]


def build_parser():
    """Extends `gbt_baseline.py`'s own parser with regime-specific flags — same model,
    features, and split; this script only adds the regime breakdown on top."""
    parser = gbt_build_parser()
    parser.add_argument(
        "--regime-vol-window", type=int, default=20,
        help="Trailing realized-vol window feeding the regime label (same role as --vol-window).",
    )
    parser.add_argument(
        "--regime-window", type=int, default=252,
        help="Minimum history (days) required before a regime label is assigned — an expanding, "
        "not fixed-size, comparison (see fx_forecasting.features.compute_volatility_regime).",
    )
    parser.add_argument("--n-regimes", type=int, default=3, help="Number of equal-frequency volatility regime buckets.")
    return parser


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        args.pairs = args.pairs[:2] if len(args.pairs) > 2 else args.pairs
        args.years = min(args.years, 3)
        args.seq_len = min(args.seq_len, 20)
        args.max_iter = 30
        args.early_stopping_rounds = 10
        args.regime_window = min(args.regime_window, 60)
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

    (
        train_ds, val_ds, test_ds,
        _train_by_symbol, _val_by_symbol, _test_by_symbol,
        market_data_by_symbol, num_factors, num_side_features, num_target_series, _feature_names,
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
    params = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "l2_regularization": args.l2_regularization,
        "random_state": args.seed,
    }
    model = (HistGradientBoostingClassifier if is_classification else HistGradientBoostingRegressor)(
        early_stopping=False, warm_start=True, **params
    )
    model, best_iteration, best_val_score = fit_with_early_stopping(
        model, X_train, y_train, X_val, y_val, args.max_iter, args.early_stopping_rounds, is_classification
    )
    logger.info(
        "Best iteration: %d/%d (validation %s=%.5f)",
        best_iteration, args.max_iter, "log_loss" if is_classification else "mse", best_val_score,
    )
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    symbol = cfg.target_symbol
    peer_symbols = [p for p in cfg.pairs if p != symbol]
    raw = load_raw_panels(cfg)
    log_returns = compute_log_returns(raw.panel)

    own_vol_regime = compute_volatility_regime(
        log_returns[symbol], args.regime_vol_window, args.regime_window, args.n_regimes
    )
    market_return = log_returns[peer_symbols].median(axis=1)
    market_vol_regime = compute_volatility_regime(
        market_return, args.regime_vol_window, args.regime_window, args.n_regimes
    )
    regime_names = regime_names_for(args.n_regimes)

    for split_label, dataset, pred, actual in (("VALIDATION", val_ds, val_pred, y_val), ("TEST", test_ds, test_pred, y_test)):
        actual_zscore = (
            market_data_by_symbol[symbol]["horizon_zscore"].reindex(dataset.sample_dates).to_numpy()
            if is_classification else None
        )

        logger.info("=== %s: overall (no regime split) ===", split_label)
        overall_hit = compute_hit_rate(pred, actual_zscore if actual_zscore is not None else actual, cfg.target_mode)
        if cfg.target_mode == "zscore":
            logger.info("  n=%-4d  hit_rate=%5.1f%%  corr=%+.4f", len(pred), overall_hit * 100, np.corrcoef(pred, actual)[0, 1])
        else:
            logger.info("  n=%-4d  hit_rate=%5.1f%%", len(pred), overall_hit * 100)

        logger.info("=== %s: breakdown by %s's own volatility regime ===", split_label, symbol)
        own_labels = own_vol_regime.reindex(dataset.sample_dates).to_numpy()
        regime_breakdown(pred, actual, own_labels, regime_names, cfg.target_mode, actual_zscore)

        logger.info("=== %s: breakdown by market-wide (peer median) volatility regime ===", split_label)
        market_labels = market_vol_regime.reindex(dataset.sample_dates).to_numpy()
        regime_breakdown(pred, actual, market_labels, regime_names, cfg.target_mode, actual_zscore)


if __name__ == "__main__":
    main()
