"""Hyperparameter search for the LSTM forecaster via differential evolution.

Searches a configurable list of `Config` parameters (CMA windows, seq_len,
LSTM architecture, learning rate, regularization, ...) to minimize
validation loss on a fixed data window, using scipy's `differential_evolution`.
Each trial trains a small/fast model (few epochs, aggressive early stopping)
reusing a single cached fetch of the raw price/rate panels (`RawPanels`), so
DE's many objective evaluations don't each pay for a fresh DB round-trip.
Every trial (not just the best) is appended to a CSV as it completes, so a
long search survives interruption and can be inspected/reloaded afterwards.

Example:
    uv run python optimize_hyperparams.py --pairs EURUSD USDJPY --years 10 \\
        --maxiter 20 --popsize 12 --trial-epochs 8

    uv run python optimize_hyperparams.py --load artifacts/hparam_search.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import logging
import threading
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from fx_forecasting.hparam_search import (
    DEFAULT_PARAM_SPECS,
    ParamSpec,
    apply_params,
    decode_params,
    format_as_cli_args,
    load_best_trial,
)
from main_lstm import (
    Config,
    RawPanels,
    build_model,
    build_pooled_datasets,
    load_raw_panels,
    set_seed,
    train,
)

logger = logging.getLogger(__name__)

FAILED_TRIAL_PENALTY = 1e6

# Set to cooperatively stop an in-progress search early (checked once per
# trial, at the start of the DE objective) — e.g. a "Stop" button driving
# `main()` in a background thread. `main()` clears it before starting.
CANCEL_EVENT = threading.Event()


def init_results_csv(path: str, specs: list[ParamSpec]) -> None:
    """Creates `path` with just the header row, if it doesn't already exist.

    Written once from the main process *before* any trial runs, so trials
    (which may run in separate worker processes — see `run_trial`) only ever
    append and never race each other over "am I the first row?".
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if path_obj.exists():
        return
    fieldnames = ["trial", "elapsed_s", "loss", "status"] + [s.name for s in specs]
    with path_obj.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_trial_row(
    path: str, specs: list[ParamSpec], decoded: dict[str, float | int], loss: float, elapsed_s: float, status: str
) -> int:
    """Appends one trial's row to `path`, opening it fresh each call rather than holding a
    long-lived file handle — a file object can't be pickled/shared with a worker process,
    so each call (wherever it runs) does its own open-append-close. Returns a "trial"
    number approximated from the file's current row count: exact under `--workers 1`
    (the common case), only cosmetically approximate (not used for correctness anywhere)
    if multiple worker processes happen to append concurrently.
    """
    fieldnames = ["trial", "elapsed_s", "loss", "status"] + [s.name for s in specs]
    with open(path, newline="") as f:
        trial_number = sum(1 for _ in f)  # header + existing rows
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({"trial": trial_number, "elapsed_s": round(elapsed_s, 3), "loss": loss, "status": status, **decoded})
    return trial_number


def run_trial(x: np.ndarray, base_cfg: Config, specs: list[ParamSpec], raw: RawPanels, results_csv: str) -> float:
    """The DE objective body: decode `x` -> fast train -> append a row to `results_csv` ->
    return the resulting val_loss for DE to minimize.

    Deliberately a top-level (module-level) function, not a closure returned
    from a factory — `scipy.optimize.differential_evolution`'s `workers`
    option runs the objective in separate worker processes via
    `multiprocessing`, which needs to pickle the objective (as a
    `functools.partial` around this function — see `main`) to ship it there;
    closures and open file handles (the previous design's `TrialLogger`)
    can't be pickled, which is why `--workers` > 1 used to crash.

    Failures (e.g. a sampled combination that leaves too few training rows)
    are caught and scored as `FAILED_TRIAL_PENALTY` rather than crashing the
    whole search — DE just learns to avoid that region.
    """
    decoded = decode_params(specs, x)
    start = time.monotonic()
    try:
        cfg = apply_params(base_cfg, decoded)
        set_seed(cfg.seed)
        train_ds, val_ds, _train_datasets, _val_datasets, _daily_returns, num_factors = build_pooled_datasets(
            cfg, raw=raw
        )
        model = build_model(cfg, num_factors)
        loss = train(model, train_ds, val_ds, cfg)
        status = "ok"
    except Exception:
        logger.exception("Trial failed with params=%s", decoded)
        loss = FAILED_TRIAL_PENALTY
        status = "failed"

    elapsed = time.monotonic() - start
    trial_number = append_trial_row(results_csv, specs, decoded, loss, elapsed, status)
    logger.info(
        "trial %d: loss=%.5f status=%s elapsed=%.1fs params=%s", trial_number, loss, status, elapsed, decoded
    )
    return loss


def make_cancel_callback():
    """DE `callback`, called once per generation *in the main process* (regardless of
    `workers`), unlike the per-trial `CANCEL_EVENT` check that used to live in the
    objective — that only ever ran in whichever process evaluated a given trial, so
    with `workers` > 1 (separate worker processes, each with its own independent
    `CANCEL_EVENT` instance) a Stop request from the main process was never actually
    visible to them. This callback is the one cancellation path guaranteed to see the
    same `CANCEL_EVENT` the dashboard's Stop button sets.
    """

    def callback(intermediate_result=None) -> bool:  # noqa: ARG001 — signature required by scipy
        return CANCEL_EVENT.is_set()

    return callback


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser — factored out of `parse_args` so other tools (e.g. a UI) can
    introspect the full set of available parameters without duplicating this list."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=None, help="Defaults to Config's default pairs.")
    parser.add_argument("--years", type=int, default=10, help="Data window to optimize over.")
    parser.add_argument("--target-mode", choices=["zscore", "class"], default="zscore")
    parser.add_argument(
        "--params", nargs="+", default=None,
        help="Subset of DEFAULT_PARAM_SPECS names to search (default: all of them).",
    )
    parser.add_argument("--trial-epochs", type=int, default=8, help="Max epochs per trial — keep this small for speed.")
    parser.add_argument("--trial-early-stop-patience", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=15, help="DE generations.")
    parser.add_argument("--popsize", type=int, default=10, help="DE population-size multiplier.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel trial workers (scipy DE `workers`); 1 = sequential (default, safest).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-csv", default="artifacts/hparam_search.csv")
    parser.add_argument(
        "--load", metavar="CSV",
        help="Skip optimizing; just print the best trial from an existing results CSV and exit.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    specs = list(DEFAULT_PARAM_SPECS)
    if args.params:
        chosen = set(args.params)
        specs = [s for s in specs if s.name in chosen]
        missing = chosen - {s.name for s in specs}
        if missing:
            raise ValueError(f"Unknown --params: {sorted(missing)}; choices are {[s.name for s in DEFAULT_PARAM_SPECS]}")

    if args.load:
        best = load_best_trial(args.load)
        logger.info("Best trial in %s: loss=%s (trial #%s)", args.load, best["loss"], best["trial"])
        logger.info("As CLI args: %s", format_as_cli_args(best, specs))
        return

    base_cfg = Config(
        pairs=args.pairs or Config().pairs,
        years=args.years,
        target_mode=args.target_mode,
        epochs=args.trial_epochs,
        early_stop_patience=args.trial_early_stop_patience,
        write_features_csv=False,
        seed=args.seed,
    )

    logger.info(
        "Fetching raw panels once for pairs=%s years=%d (reused across all trials)...", base_cfg.pairs, base_cfg.years
    )
    raw = load_raw_panels(base_cfg)

    init_results_csv(args.results_csv, specs)
    # A `functools.partial` around a top-level function stays picklable (needed for
    # `workers` > 1's multiprocessing — see `run_trial`'s docstring), unlike a closure.
    objective = functools.partial(run_trial, base_cfg=base_cfg, specs=specs, raw=raw, results_csv=args.results_csv)
    CANCEL_EVENT.clear()

    bounds = [(s.low, s.high) for s in specs]
    logger.info(
        "Starting differential evolution over %d params (maxiter=%d, popsize=%d, workers=%d) -> %s",
        len(specs), args.maxiter, args.popsize, args.workers, args.results_csv,
    )
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=False,
        workers=args.workers,
        updating="deferred" if args.workers != 1 else "immediate",
        callback=make_cancel_callback(),
    )
    if CANCEL_EVENT.is_set():
        logger.info("Search stopped by user after partial progress; reporting the best result found so far.")

    best_decoded = decode_params(specs, result.x)
    logger.info("Best DE result: loss=%.5f params=%s", result.fun, best_decoded)
    logger.info("As CLI args: %s", format_as_cli_args(best_decoded, specs))
    logger.info("Full trial history saved to %s (reload with --load %s)", args.results_csv, args.results_csv)


if __name__ == "__main__":
    main()
