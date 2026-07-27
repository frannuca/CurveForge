"""Shared parameter-spec/decode logic for hyperparameter search — no dependency on
`main_lstm.py` or `optimize_hyperparams.py`, so both can import from here without a
circular import (`optimize_hyperparams.py` already imports from `main_lstm.py`, so
`main_lstm.py` can't import back from it directly).

`apply_params`/`load_and_apply_best_trial` work on any dataclass-like `Config` purely
via `hasattr`/`dataclasses.replace`, without ever importing the `Config` class itself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from main_lstm import Config


def best_plot_path_for(results_csv: str) -> str:
    """Derives the "best trial so far" out-of-sample prediction-plot path from a
    results CSV path — shared by `optimize_hyperparams.py` (writes it whenever a new
    best trial is found) and `dashboard.py` (polls for it while a search is running),
    so the naming convention only lives in one place.
    """
    path = Path(results_csv)
    return str(path.with_name(path.stem + "_best_predictions.png"))


@dataclass(frozen=True)
class ParamSpec:
    """One searchable parameter: name, DE bounds, and how to decode a real value into it."""

    name: str
    low: float
    high: float
    kind: str = "float"  # "float" | "int" | "log_float"

    def decode(self, x: float) -> float | int:
        if self.kind == "int":
            return int(round(x))
        if self.kind == "log_float":
            return float(10.0**x)  # bounds are given in log10 space, e.g. (-4, -1) -> 1e-4..1e-1
        return float(x)


# The default searchable parameter list. Edit freely, or narrow it at the CLI
# with --params. vol_window's floor of 2 is deliberate: a rolling std with
# window=1 is always NaN, which collapses every downstream feature to 0 rows.
DEFAULT_PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("seq_len", 15, 90, "int"),
    ParamSpec("horizon", 1, 20, "int"),
    ParamSpec("momentum_window", 3, 30, "int"),
    ParamSpec("vol_window", 2, 30, "int"),
    ParamSpec("hidden_size", 8, 128, "int"),
    ParamSpec("num_layers", 1, 3, "int"),
    ParamSpec("dropout", 0.0, 0.5, "float"),
    ParamSpec("lr", -4.0, -1.0, "log_float"),  # 1e-4 .. 1e-1
    ParamSpec("weight_decay", -8.0, -2.0, "log_float"),  # 1e-8 .. 1e-2
    ParamSpec("outlier_weight", 0.0, 15.0, "float"),
    ParamSpec("variance_penalty_weight", 0.0, 5.0, "float"),
    ParamSpec("batch_size", 16, 128, "int"),
)


def decode_params(specs: list[ParamSpec], x) -> dict[str, float | int]:
    """Decodes DE's real-valued vector `x` into a {param_name: value} dict via `specs`."""
    return {spec.name: spec.decode(xi) for spec, xi in zip(specs, x)}


def apply_params(base_cfg: Config, decoded: dict[str, float | int]) -> Config:
    """Returns a copy of `base_cfg` with every decoded scalar param overridden, via
    `dataclasses.replace` whenever `base_cfg` actually has a matching field name."""
    overrides = {k: v for k, v in decoded.items() if hasattr(base_cfg, k)}
    return replace(base_cfg, **overrides)


def load_best_trial(csv_path: str) -> dict[str, str]:
    """Reads a trials CSV written by `optimize_hyperparams.append_trial_row` and returns
    the lowest-loss `status="ok"` row (raw strings, as read from the CSV)."""
    with open(csv_path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["status"] == "ok"]
    if not rows:
        raise ValueError(f"No successful trials found in {csv_path}")
    return min(rows, key=lambda r: float(r["loss"]))


def load_and_apply_best_trial(
    base_cfg: Config, csv_path: str, specs: list[ParamSpec] | None = None
) -> tuple[Config, dict[str, Any]]:
    """Loads the best trial from `csv_path` and applies it on top of `base_cfg`.

    Only overrides params that are actually present as columns in the CSV
    (a search run with `--params` narrowed to a subset writes only those
    columns), so this works regardless of which subset the original search
    covered. CSV values are already the search's *decoded* (final, typed)
    values — this only needs to parse them back from text, via each spec's
    `kind` (not re-run `ParamSpec.decode`, which maps DE's raw bounded float
    onto a typed value, not a re-parse of an already-typed value).

    Returns (new_cfg, decoded) so the caller can log exactly what was applied.
    """
    specs = specs or list(DEFAULT_PARAM_SPECS)
    best_row = load_best_trial(csv_path)

    decoded: dict[str, Any] = {}
    for spec in specs:
        if spec.name not in best_row:
            continue
        raw = best_row[spec.name]
        decoded[spec.name] = int(round(float(raw))) if spec.kind == "int" else float(raw)

    return apply_params(base_cfg, decoded), decoded


def format_as_cli_args(best: dict, specs: list[ParamSpec]) -> str:
    """Formats a best-trial row as `--flag value` args ready to paste into `main_lstm.py`."""
    parts = [f"--{spec.name.replace('_', '-')} {best[spec.name]}" for spec in specs]
    return " ".join(parts)
