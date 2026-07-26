# FX Forecasting

LSTM forecasters for FX pairs. Four entry points:

- `main_lstm.py` — the main model: every symbol pooled into one
  shared-weight LSTM + attention model, `target_mode="zscore"` (continuous)
  or `"class"` (5-class direction) target. Train, evaluate, or run inference.
- `simple_lstm.py` — a minimal alternative: one symbol at a time, a plain
  LSTM (no attention, no pooling across symbols), 3-class direction target
  (bearish/neutral/bullish) — see the "Simple single-symbol model" section
  below.
- `optimize_hyperparams.py` — search for good hyperparameters via
  differential evolution (for `main_lstm.py`), then feed the result straight
  back into it.
- `dashboard.py` — a Dash web UI exposing training/evaluation and the
  hyperparameter search (see the "Web dashboard" section below).

See `fx_forecasting/models/README.md` for the `main_lstm.py` model
architecture, and the module docstrings in `main_lstm.py`/`simple_lstm.py`/
`optimize_hyperparams.py` for more detail than this file covers.

## Setup

```bash
uv sync
cp .env.example .env   # fill in POSTGRES_* and FRED_API_KEY
```

All of these read FX price data from Postgres and interest-rate data from
FRED (for the carry feature) — populate these once before training:

```bash
uv run python -m fx_forecasting.data.fx_downloader --upload
uv run python -m fx_forecasting.data.rates_downloader --upload
```

## Training a model — `main_lstm.py`

```bash
# Quick sanity check: tiny data, 1 epoch, steppable under a debugger
uv run python main_lstm.py --debug

# A real run
uv run python main_lstm.py --pairs EURUSD USDJPY GBPUSD --years 15 --epochs 100

# Load the results of a hyperparameter search on top of your other flags
uv run python main_lstm.py --years 15 --epochs 200 --params-csv artifacts/hparam_search.csv

# Inference only, using a previously trained checkpoint
uv run python main_lstm.py --pairs EURUSD USDJPY --infer --model-path artifacts/lstm_model.pt
```

Each run writes to `artifacts/` by default:

| File | Contents |
|---|---|
| `lstm_model.pt` | model checkpoint (weights + architecture/config needed to reload it) |
| `lstm_predictions.png` | actual vs. predicted target, one panel per symbol |
| `lstm_pnl.png` | strategy backtest — cumulative PnL of investing the predicted z-score (`target_mode="zscore"` only) |
| `features.csv` | every symbol's normalized input features + target, long-format |

Key flags (`--help` for the full list):

| Flag | Purpose |
|---|---|
| `--pairs`, `--years` | which symbols and how much history |
| `--target-mode {zscore,class}` | regression on the continuous return z-score (default), or 5-class direction label |
| `--seq-len`, `--horizon` | LSTM lookback window; N-day-ahead target horizon |
| `--momentum-window`, `--vol-window` | engineered-feature windows |
| `--hidden-size`, `--num-layers`, `--dropout` | model architecture |
| `--lr`, `--weight-decay`, `--epochs`, `--early-stop-patience` | training |
| `--outlier-weight`, `--variance-penalty-weight` | loss shaping (weight extreme moves more; resist collapsing to a constant near-zero prediction) |
| `--infer` | skip training, just load `--model-path` and evaluate/plot |
| `--params-csv` | apply the best trial from an `optimize_hyperparams.py` run (see below) |
| `--debug` | shrink everything for a fast, steppable sanity check |

## Simple single-symbol model — `simple_lstm.py`

A deliberately minimal alternative to `main_lstm.py`, for when you want to
train and inspect a single symbol in isolation rather than the pooled
multi-symbol model:

- **One symbol** (`--symbol`), not a pooled panel — no `ConcatDataset`
  machinery at all.
- **Plain LSTM** (`SimpleLSTMClassifier`, in
  `fx_forecasting/models/simple_lstm_classifier.py`): the LSTM's own final
  hidden state feeds directly into a linear layer — no attention pooling.
- **3-class target**: bearish (-1) / neutral (0) / bullish (+1) for the
  `--horizon`-day-ahead cumulative return (30/40/30 quantile split — see
  `fx_forecasting.features.compute_target_direction`), rather than
  `main_lstm.py`'s 5-class or continuous z-score target.

It still reuses `main_lstm.py`'s feature engineering (return, intraday_vol,
momentum, rvol, carry, CMA crossovers) and Postgres/FRED data loading, so the
[Setup](#setup) steps above still apply.

```bash
# Quick sanity check
uv run python simple_lstm.py --symbol EURUSD --debug

# A real run
uv run python simple_lstm.py --symbol EURUSD --years 15 --epochs 200

# Inference only, using a previously trained checkpoint
uv run python simple_lstm.py --symbol EURUSD --infer --model-path artifacts/simple_lstm_model.pt
```

Writes `artifacts/simple_lstm_model.pt` and
`artifacts/simple_lstm_predictions.png` (actual vs. predicted direction over
the validation set) by default; `--help` for the full flag list, which
mirrors `main_lstm.py`'s naming (`--seq-len`, `--hidden-size`,
`--outlier-weight`, `--early-stop-patience`, ...) wherever the same concept
applies.

## Searching for hyperparameters — `optimize_hyperparams.py`

Runs `scipy.optimize.differential_evolution` over a configurable parameter
list (CMA windows, `seq_len`/`horizon`, LSTM architecture, LR, regularization,
...), training a small/fast model per trial to minimize validation loss.
Every trial (not just the best) is appended to a CSV as it completes.

```bash
# Search over every default parameter
uv run python optimize_hyperparams.py --pairs EURUSD USDJPY --years 10 \
    --maxiter 20 --popsize 12 --trial-epochs 8 \
    --results-csv artifacts/hparam_search.csv

# Narrow the search to a subset of parameters (others stay at Config defaults)
uv run python optimize_hyperparams.py --params hidden_size dropout lr \
    --maxiter 15 --popsize 10 --results-csv artifacts/hparam_search.csv

# Inspect an existing results CSV without running a new search
uv run python optimize_hyperparams.py --load artifacts/hparam_search.csv
```

Searchable parameter names (for `--params`): `seq_len`, `horizon`,
`momentum_window`, `vol_window`, `cma1_short`, `cma1_long`, `cma2_short`,
`cma2_long`, `cma3_short`, `cma3_long` (the 3 CMA `(short, long)` EWMA
pairs), `hidden_size`, `num_layers`, `dropout`, `lr`, `weight_decay`,
`outlier_weight`, `variance_penalty_weight`, `batch_size`. Edit
`DEFAULT_PARAM_SPECS` in `fx_forecasting/hparam_search.py` to change bounds
or add new ones.

Key flags:

| Flag | Purpose |
|---|---|
| `--pairs`, `--years` | data window to optimize over (kept small/fast on purpose) |
| `--params` | subset of parameters to search; default is all of them |
| `--trial-epochs`, `--trial-early-stop-patience` | per-trial training budget — keep small for speed |
| `--maxiter`, `--popsize`, `--workers` | DE search budget / parallelism |
| `--results-csv` | where trials are appended (crash-resilient: flushed after every trial) |
| `--load` | skip searching; just print the best trial from an existing CSV |

### Using the result

```bash
uv run python optimize_hyperparams.py --pairs EURUSD USDJPY --years 10 \
    --maxiter 20 --results-csv artifacts/hparam_search.csv

uv run python main_lstm.py --pairs EURUSD USDJPY --years 20 --epochs 200 \
    --params-csv artifacts/hparam_search.csv
```

`--params-csv` loads the lowest-loss row from the results CSV and applies it
on top of `main_lstm.py`'s other flags — it only overrides whichever
parameters the search actually covered (e.g. a search run with `--params
hidden_size dropout` leaves everything else, including CMA windows, at
whatever `main_lstm.py`'s own flags/defaults say). This is deliberately
separate from the fast trial run: you typically want a full `--years` window
and a real `--epochs` budget for the final model, neither of which the search
itself needs.

## Web dashboard — `dashboard.py`

A Dash UI over `main_lstm.py` and `optimize_hyperparams.py` (not
`simple_lstm.py`, which has no dashboard tab yet), so you don't need the
terminal for routine runs:

```bash
uv run python dashboard.py
```

Then open <http://127.0.0.1:8050/>. Two tabs, "Train / Evaluate" and
"Hyperparameter Search", each auto-generate their form directly from that
script's own `argparse` parser (`main_lstm.build_parser()` /
`optimize_hyperparams.build_parser()`) — every CLI flag documented above has
a matching field, and the two stay in sync automatically as the underlying
scripts change; nothing about individual parameters is hardcoded in the
dashboard itself. Leave a field blank to fall back to that flag's own
script default.

Runs execute in a background thread (training/search can take a while) so
the page stays responsive; a log panel and a progress bar update live while
a run is in progress, and:

- **Train / Evaluate** shows the out-of-sample and in-sample predictions
  plots and the strategy PnL plot once training finishes.
- **Hyperparameter Search** shows the top 10 trials (by loss) from the
  results CSV, plus an out-of-sample predictions plot for the best trial
  found *so far* — both refresh live as the search runs, not just once it
  completes, so you can watch the model actually improve in real time
  (whenever a trial beats every prior one, `optimize_hyperparams.py`
  re-saves that plot; works correctly with `--workers` > 1 too — see
  `maybe_save_best_prediction_plot`'s docstring for why that needed care).

This is a local single-user tool (job state lives in the server process, not
per-browser-session) — don't expose it beyond localhost.
