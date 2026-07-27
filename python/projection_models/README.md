# FX Forecasting

LSTM forecasters for FX pairs. Entry points:

- `pretrain_autoencoder.py` — trains the orthogonal cross-asset factor
  autoencoder `main_lstm.py` depends on: compresses the daily log returns of
  a whole FX universe (K pairs) down to D orthogonal factors, a literal
  gradient-trained equivalent of PCA. Must be run once before `main_lstm.py`
  — see "Pretraining the cross-asset factor autoencoder" below.
- `main_lstm.py` — the main model: forecasts one target symbol
  (`--target-symbol`, mandatory) with a single, attention-free LSTM (no
  encoder/decoder split) whose recurrent input is the raw cross-asset return
  sequence itself, generating its own forecast by continuing that same
  recurrence; the pretrained orthogonal factor snapshot, momentum, vol, skew,
  kurtosis, carry, intraday volatility, and an optional spectral/FFT summary
  all feed the final output layer directly, not the recurrence. `--target-mode`
  picks between two alternative output heads over that same trunk: a 5-class
  direction target (`class`, default — `very_bearish` .. `very_bullish`) or a
  directly predicted continuous return z-score (`zscore`). Train, evaluate, or
  run inference.
- `simple_lstm.py` — a minimal alternative: one symbol at a time, a plain
  LSTM (no cross-asset autoencoder), 3-class direction target
  (bearish/neutral/bullish) — see the "Simple single-symbol model" section
  below.
- `optimize_hyperparams.py` — search for good hyperparameters via
  differential evolution (for `main_lstm.py`), then feed the result straight
  back into it.
- `dashboard.py` — a Dash web UI exposing pretraining, training/evaluation,
  and the hyperparameter search (see the "Web dashboard" section below).

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

## Pretraining the cross-asset factor autoencoder — `pretrain_autoencoder.py`

`main_lstm.py`'s final prediction layer (not the LSTM itself — see below) uses a pretrained,
orthogonal reduction of the raw, highly collinear `xret_<pair>` log returns (broad USD
strength/weakness shows up in most FX pairs at once). Train that reduction first:

```bash
# Quick sanity check
uv run python pretrain_autoencoder.py --debug

# A real run: compress 7 majors down to 3 orthogonal factors
uv run python pretrain_autoencoder.py --pairs EURUSD USDJPY GBPUSD USDCHF AUDUSD USDCAD NZDUSD \
    --years 20 --factor-dim 3 --autoencoder-path artifacts/cross_asset_autoencoder.pt
```

The encoder (`fx_forecasting/models/orthogonal_autoencoder.py`) is a linear
map with rows constrained to stay exactly orthonormal throughout training
(`torch.nn.utils.parametrizations.orthogonal`), trained to minimize plain
reconstruction MSE — per Baldi & Hornik (1989), a linear autoencoder trained
this way recovers the same subspace PCA would, and the orthonormality
constraint makes the individual learned factors themselves mutually
orthogonal too, not just subspace-equivalent — a real, gradient-trained
analogue of PCA rather than an arbitrary learned linear map. `--factor-dim`
(D) must be strictly less than `len(--pairs)` (K).

The checkpoint records exactly which `--pairs` (sorted) and `--seq-len`
(the rolling-normalization window) it was pretrained for; `main_lstm.py`
refuses to load one that doesn't match — the pretrained weights are only
meaningful against data normalized the identical way.

## Training a model — `main_lstm.py`

```bash
# Quick sanity check: tiny data, 1 epoch, steppable under a debugger
uv run python main_lstm.py --debug --target-symbol EURUSD --autoencoder-path artifacts/debug_autoencoder.pt

# A real run
uv run python main_lstm.py --pairs EURUSD USDJPY GBPUSD --target-symbol EURUSD \
    --years 15 --epochs 100 --autoencoder-path artifacts/cross_asset_autoencoder.pt

# Load the results of a hyperparameter search on top of your other flags
uv run python main_lstm.py --target-symbol EURUSD --autoencoder-path artifacts/cross_asset_autoencoder.pt \
    --years 15 --epochs 200 --params-csv artifacts/hparam_search.csv

# Inference only, using a previously trained checkpoint (the autoencoder's weights are
# already baked into this checkpoint — --autoencoder-path is not needed again)
uv run python main_lstm.py --pairs EURUSD USDJPY --target-symbol EURUSD \
    --infer --model-path artifacts/lstm_model.pt
```

`--target-symbol` is mandatory: this architecture always forecasts exactly
one pair. `--pairs` still supplies the full cross-asset input
universe — every pair's own daily log return (`xret_<pair>`, target symbol
included) feeds a single, attention-free LSTM *directly*, as a raw
(rolling-normalized), full multi-channel sequence: the LSTM consumes the real
history, then continues its own recurrence (same weights, no second module,
no attention) to generate the forecast itself, so it learns its own temporal
representation of the whole return history rather than being handed a
pre-compressed one or attending back over it. The *same* cross-asset block, at
the forecast origin's own last timestep only, is *also* reduced through the
pretrained factor autoencoder (`--autoencoder-path`, required; see
"Pretraining the cross-asset factor autoencoder" above) — a static,
structurally-orthogonal snapshot of "where the whole universe stands right
now" — and, along with `target_symbol`'s own momentum, realized vol, skew,
kurtosis, carry, and intraday volatility, bypasses the LSTM entirely and is
concatenated directly into the final output layer, alongside an optional
spectral (FFT) summary of the LSTM's own input sequence
(`--use-spectral-features`). The pretrained encoder is frozen by default
(`--fine-tune-autoencoder` opts into updating it during training — usually
not necessary, and riskier under FX's non-stationary return distributions
per Kumar et al. 2022).

`--target-mode` picks what that output layer produces, as two mutually exclusive
alternatives over the identical trunk above:

- `class` (default): 5 discrete direction classes (`very_bearish` .. `very_bullish` — see
  `fx_forecasting.features.compute_target_class`), trained under a class-distance-weighted
  `nn.CrossEntropyLoss` (`--outlier-weight`). A continuous trading signal is still derived
  from the predicted class *distribution* — not just its most likely class — via
  `main_lstm.collect_signal`: the expected value of a fixed per-class score
  (very_bearish=-2 .. very_bullish=2) under the predicted probabilities, so position sizing
  naturally scales with how conviction-weighted (lopsided) the distribution is, not just
  which single class wins.
- `zscore`: the continuous horizon return z-score itself, predicted directly, trained under a
  magnitude-weighted MSE loss (also `--outlier-weight`, weighting large-|z| targets more).
  The model's own output *is* the trading signal — no distribution to take an expectation
  over.

Neither mode dominates the other a priori — compare them like any other hyperparameter,
under the same purged walk-forward + frozen test holdout discipline described below.

Each symbol's history is split chronologically into train / validation / a
final **test holdout** (`--val-fraction`, `--test-fraction`), with a
`horizon`-sized embargo purged around each split boundary
(`fx_forecasting.trading.purged_train_val_test_split`) — a label at day `t`
looks `horizon` days ahead, so without the embargo the last training/
validation rows before a later split would have labels drawn partly from
that split's own price history. The test split is meant to be looked at once
per model: it's evaluated at the end of every run (clearly marked "FINAL
HOLDOUT" in the logs) but never used for early stopping, hyperparameter
search, or model selection — treat repeated tuning against it as backtest
overfitting.

Each run writes to `artifacts/` by default:

| File | Contents |
|---|---|
| `lstm_model.pt` | model checkpoint (weights + architecture/config needed to reload it) |
| `lstm_predictions.png` / `..._in_sample.png` / `..._test.png` | actual vs. predicted direction (class or continuous z-score, per `--target-mode`), one panel per symbol — validation / training / final holdout |
| `lstm_pnl.png` / `lstm_pnl_test.png` | execution-aware strategy backtest — net cumulative PnL after transaction costs and carry — validation / final holdout |
| `lstm_positions_pnl.png` / `..._test.png` | detail view, one panel per symbol: position size + that symbol's actual daily return, plus cumulative PnL with vs. without transaction costs on a second axis — the gap between the two PnL curves is exactly the cost paid for turnover |
| `features.csv` | every symbol's normalized input features + target, long-format |

Every `*_predictions*.png` and `*_pnl*.png` above also gets a same-named
`.csv` sidecar (`date, symbol, actual, predicted` / `date, symbol,
net_cumulative`) with the exact data the plot was built from —
`dashboard.py` reads these to render an interactive version instead of the
static PNG (see [Web dashboard](#web-dashboard--dashboardpy)).

The strategy backtest (`fx_forecasting/trading.py`) is not a z-score-vs-z-score diagnostic:
position size comes from the model's own risk-graded signal (`main_lstm.collect_signal`,
the expected value of the predicted class distribution), affine-calibrated against what
actually realized on the *training* fold only (never validation/test), thresholded and
leverage-scaled (`--signal-threshold`, `--signal-leverage`, `--max-position`), then evaluated
against the *actual* next-day log return plus carry, net of a per-unit-turnover transaction
cost (`--transaction-cost-bps`) — a flat position below the threshold trades nothing rather
than sizing a position off low-confidence noise.

Key flags (`--help` for the full list):

| Flag | Purpose |
|---|---|
| `--pairs`, `--years` | cross-asset input universe and how much history |
| `--target-symbol` | **required**: the one pair to forecast (must be in `--pairs`) |
| `--autoencoder-path` | **required**: checkpoint from `pretrain_autoencoder.py` for this exact `--pairs`/`--seq-len` |
| `--fine-tune-autoencoder` | allow the pretrained cross-asset encoder to keep updating during training (default: frozen) |
| `--target-mode` | `class` (default, 5-way direction) or `zscore` (direct continuous regression) — two alternative output heads over the same trunk |
| `--cross-sectional-target` | train against return relative to the cross-sectional median of the other `--pairs` symbols, instead of absolute direction |
| `--seq-len`, `--horizon` | LSTM lookback window (must match the autoencoder checkpoint's own `--seq-len`); N-day-ahead target horizon |
| `--val-fraction`, `--test-fraction` | size of the validation fold and the final untouched test holdout |
| `--momentum-window`, `--vol-window` | engineered-feature windows (FFN-side features, target symbol only) |
| `--hidden-size`, `--num-layers`, `--dropout` | LSTM architecture |
| `--use-spectral-features`, `--spectral-embedding-dim`, `--spectral-freq-bins` | append a learned FFT-magnitude-spectrum embedding of the LSTM's own input sequence directly into the final prediction layer, alongside the other side features (off by default — see `fx_forecasting/models/spectral_embedding.py`) |
| `--lr`, `--weight-decay`, `--epochs`, `--early-stop-patience` | training |
| `--outlier-weight` | loss shaping: extra cross-entropy weight (linear in distance from the neutral class) for classes far from neutral — default 10, prioritizes correctly calling large moves over the far more common near-neutral days |
| `--regime-gate`, `--regime-vol-window`, `--regime-window`, `--regime-n-regimes` | backtest-only: zero the position whenever the cross-sectional (peer-median) realized-vol regime is in its highest bucket (see `fx_forecasting.features.compute_volatility_regime`) — motivated by a diagnostic finding that hit rate was consistently worse in high-market-vol periods |
| `--transaction-cost-bps`, `--signal-threshold`, `--signal-leverage`, `--max-position` | execution-aware backtest: turnover cost, minimum signal to trade at all, position scale, and cap |
| `--infer` | skip training, just load `--model-path` and evaluate/plot (the autoencoder's weights are already baked into that checkpoint) |
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
  `main_lstm.py`'s 5-class target.

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
list (`seq_len`/`horizon`, momentum/vol windows, LSTM architecture, LR,
regularization, ...), training a small/fast model per trial to minimize validation loss.
Every trial (not just the best) is appended to a CSV as it completes.

```bash
# Search over every default parameter
uv run python optimize_hyperparams.py --pairs EURUSD USDJPY --target-symbol EURUSD \
    --autoencoder-path artifacts/cross_asset_autoencoder.pt --years 10 \
    --maxiter 20 --popsize 12 --trial-epochs 8 \
    --results-csv artifacts/hparam_search.csv

# Narrow the search to a subset of parameters (others stay at Config defaults)
uv run python optimize_hyperparams.py --target-symbol EURUSD \
    --autoencoder-path artifacts/cross_asset_autoencoder.pt \
    --params hidden_size dropout lr \
    --maxiter 15 --popsize 10 --results-csv artifacts/hparam_search.csv

# Inspect an existing results CSV without running a new search
uv run python optimize_hyperparams.py --load artifacts/hparam_search.csv
```

Searchable parameter names (for `--params`): `seq_len`, `horizon`,
`momentum_window`, `vol_window`, `hidden_size`, `num_layers`, `dropout`,
`lr`, `weight_decay`, `outlier_weight`, `batch_size`. Edit
`DEFAULT_PARAM_SPECS` in `fx_forecasting/hparam_search.py` to change bounds
or add new ones. Note: including `seq_len` in the search means most sampled
values won't match `--autoencoder-path`'s own pretrained `seq_len` — those
trials fail fast (caught, scored as a large penalty) rather than crashing the
search, but it wastes evaluations; leave `seq_len` out of `--params` (the
default list includes it) unless you're prepared for that, or pretrain
several autoencoder checkpoints across the `seq_len` values you want to try.

Key flags:

| Flag | Purpose |
|---|---|
| `--pairs`, `--years` | cross-asset input universe and data window to optimize over (kept small/fast on purpose) |
| `--target-symbol` | **required**: search for a model forecasting only this one pair (must be in `--pairs`) |
| `--autoencoder-path` | **required**: pretrained checkpoint from `pretrain_autoencoder.py` (see above) |
| `--target-mode` | `class` (default) or `zscore` — search over whichever output head you plan to actually train (see `main_lstm.py`'s own `--target-mode`) |
| `--params` | subset of parameters to search; default is all of them |
| `--trial-epochs`, `--trial-early-stop-patience` | per-trial training budget — keep small for speed |
| `--maxiter`, `--popsize`, `--workers` | DE search budget / parallelism |
| `--results-csv` | where trials are appended (crash-resilient: flushed after every trial) |
| `--load` | skip searching; just print the best trial from an existing CSV |

### Using the result

```bash
uv run python optimize_hyperparams.py --pairs EURUSD USDJPY --target-symbol EURUSD \
    --autoencoder-path artifacts/cross_asset_autoencoder.pt --years 10 \
    --maxiter 20 --results-csv artifacts/hparam_search.csv

uv run python main_lstm.py --pairs EURUSD USDJPY --target-symbol EURUSD \
    --autoencoder-path artifacts/cross_asset_autoencoder.pt --years 20 --epochs 200 \
    --params-csv artifacts/hparam_search.csv
```

`--params-csv` loads the lowest-loss row from the results CSV and applies it
on top of `main_lstm.py`'s other flags — it only overrides whichever
parameters the search actually covered (e.g. a search run with `--params
hidden_size dropout` leaves everything else at whatever `main_lstm.py`'s own
flags/defaults say). This is deliberately
separate from the fast trial run: you typically want a full `--years` window
and a real `--epochs` budget for the final model, neither of which the search
itself needs.

## Web dashboard — `dashboard.py`

A Dash UI over `pretrain_autoencoder.py`, `main_lstm.py`, and
`optimize_hyperparams.py` (not `simple_lstm.py` or the GBT scripts, which
have no dashboard tab yet), so you don't need the terminal for routine runs:

```bash
uv run python dashboard.py
```

Then open <http://127.0.0.1:8050/>. Three tabs, "Pretrain Autoencoder",
"Train / Evaluate", and "Hyperparameter Search", each auto-generate their
form directly from that script's own `argparse` parser
(`pretrain_autoencoder.build_parser()` / `main_lstm.build_parser()` /
`optimize_hyperparams.build_parser()`) — every CLI flag documented above has
a matching field, and all three stay in sync automatically as the underlying
scripts change; nothing about individual parameters is hardcoded in the
dashboard itself. Leave a field blank to fall back to that flag's own
script default.

Runs execute in a background thread (training/search can take a while) so
the page stays responsive; a log panel and a progress bar update live while
a run is in progress. The log panel auto-scrolls to follow new lines as long
as you're already at the bottom; scroll up to read earlier output and it
stops following until you scroll back down. And:

- **Pretrain Autoencoder** runs `pretrain_autoencoder.py` and reports where
  the resulting checkpoint was saved once it finishes — feed that path into
  the "Train / Evaluate" tab's `--autoencoder-path` field. No plot here
  (reconstruction quality is logged as train/val/test MSE, not plotted); the
  real check is how the resulting `main_lstm.py` model performs.
- **Train / Evaluate** shows the out-of-sample and in-sample predictions
  plots and the strategy PnL plot once training finishes.
- **Hyperparameter Search** shows the top 10 trials (by loss) from the
  results CSV, plus an out-of-sample predictions plot for the best trial
  found *so far* — both refresh live as the search runs, not just once it
  completes, so you can watch the model actually improve in real time
  (whenever a trial beats every prior one, `optimize_hyperparams.py`
  re-saves that plot; works correctly with `--workers` > 1 too — see
  `maybe_save_best_prediction_plot`'s docstring for why that needed care).

Every plot in the dashboard is an interactive Plotly chart (`dcc.Graph`), not
a static image — scroll/box-zoom, pan, and hover over a point to read its
exact date and value, the same way for predictions and PnL plots alike. This
reads the `.csv` sidecar `main_lstm.py` writes next to each plot's PNG (see
`render_interactive_plot` in `dashboard.py`), not the PNG itself; the PNG is
still written to `artifacts/` for non-dashboard use (reports, `--help`-only
CLI runs, ...).

This is a local single-user tool (job state lives in the server process, not
per-browser-session) — don't expose it beyond localhost.
