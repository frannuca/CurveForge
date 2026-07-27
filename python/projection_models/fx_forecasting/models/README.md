# LSTM sequence forecaster — architecture guide

This explains `lstm_forecaster.py` piece by piece, so you can follow the code
while reading. See also `pretrain_autoencoder.py` and
`fx_forecasting/models/orthogonal_autoencoder.py` for the pretrained
cross-asset factor reduction this model depends on.

## What it does, in one sentence

A single `nn.LSTM` (no encoder/decoder split, no attention) consumes the *raw*
(rolling-normalized upstream, uncompressed) cross-asset return sequence — every pair in the
input universe's own daily log return (K series, target included) — then continues its own
recurrence for `forecast_horizon` further steps to directly generate the forecast sequence
itself. A pretrained, orthogonally-reduced snapshot of that same sequence (a linear
autoencoder trained separately by `pretrain_autoencoder.py` and loaded frozen, by default,
into this model), the target symbol's own side features (momentum, realized vol, skew,
kurtosis, carry, intraday volatility), and an optional spectral (FFT) summary all bypass the
LSTM entirely and are injected only at the final feed-forward output layer (Lim, Arık, Loeff &
Pfister, 2021, "Temporal Fusion Transformers": non-sequential covariates are best fed directly
to the output stage rather than forced through the recurrent bottleneck).

That output layer produces one of two mutually exclusive things, chosen by `target_mode`:
`output_size` discrete direction-class logits (`target_mode="class"`, the default — trained
under a weighted `nn.CrossEntropyLoss`, see `main_lstm.py::class_distance_weights`), or a
single continuous value (`target_mode="zscore"` — the horizon return z-score itself, trained
under a magnitude-weighted MSE, see `main_lstm.py::weighted_mse_loss`). Both alternatives
share the exact same trunk; only the final `nn.Linear`'s output width and what loss trains it
differ — see `main_lstm.py`'s `--target-mode` flag.

`main_lstm.py` currently always uses `forecast_horizon=1` (a single N-day-ahead target, not a
genuine multi-step sequence), in which case the model's output collapses to `(batch,
output_size)` (class mode) or `(batch,)` (zscore mode) — but the architecture itself supports
`forecast_horizon > 1` for genuine multi-step-ahead forecasting.

## The shape of the problem

- Input `x`: `(batch, seq_len, num_factors)`, where `num_factors = num_side_features +
  num_cross_asset_factors` — two column blocks in that order: the leading
  `num_side_features` columns are the target symbol's own non-sequential covariates
  (momentum, rvol, skew, kurt, carry, intraday volatility); the *trailing*
  `num_cross_asset_factors` (K) columns are the per-pair `xret_*` cross-asset log returns
  (including the target's own) — see `main_lstm.py::build_target_dataset`. The side features
  never enter the LSTM; only the trailing K columns do.
- Output, `target_mode="class"`: `(batch, forecast_horizon, output_size)` if
  `forecast_horizon > 1`, else `(batch, output_size)` — `output_size` is `len(CLASS_NAMES)`
  (5) in `main_lstm.py::build_model`. No output activation: `nn.CrossEntropyLoss` applies
  `log_softmax` internally, so pre-squashing the logits here would double-apply a nonlinearity
  and distort the loss.
- Output, `target_mode="zscore"`: `(batch, forecast_horizon)` if `forecast_horizon > 1`, else
  `(batch,)` — again no output activation, since it's an unbounded regression target.
  `output_size` is ignored in this mode (the output head is fixed at width 1).

## Cross-asset sequence -> LSTM (twice); orthogonal snapshot + side features + spectral -> FFN

```
 x (batch, seq_len, num_factors)
        │
        │  split: base = x[..., :-K] (side features), xret = x[..., -K:] (cross-asset)
        ▼
 xret (batch, seq_len, K)
        │
        ├───────────────────────────────────────────────────┐
        ▼                                                     ▼
 input_norm (LayerNorm)                          cross_asset_encoder(xret[:, -1, :])
        │                                          (pretrained, orthonormal rows)
        ▼                                                     │
 ┌───────────────┐                                            │
 │  self.lstm     │  phase 1: consume the real history          │
 │  (nn.LSTM)     │  -> final (h_n, c_n) only                    │
 └───────────────┘                                             │
        │  (h_n, c_n)                                          │
        ▼                                                       │
 ┌───────────────┐  phase 2: SAME weights, `forecast_horizon`    │
 │  self.lstm     │  more steps, each fed the same learned         │
 │  (again)       │  continuation_token — no real per-step input     │
 └───────────────┘                                                   │
        │  forecast_outputs[:, t, :]  (this step's forecast)         │
        ▼                                                            │
 for t in range(forecast_horizon):                                   │
   pre_output = [dropout(forecast_outputs[:, t, :]), spectral_vec?,  │
                 factors_at_t ◀───────────────────────────────────────┘
                 side_features_at_t]           (side_features_at_t = base[:, -1, :])
   output_features = pre_output_norm(cat(pre_output))
   step_out = classifier(output_features)      -> this step's class logits / z-score
```

### The LSTM's own input (`self.input_norm`, `self.lstm`)

```python
self.input_norm = nn.LayerNorm(num_cross_asset_factors)
self.lstm = nn.LSTM(
    input_size=num_cross_asset_factors,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    dropout=dropout if num_layers > 1 else 0.0,
)
self.continuation_token = nn.Parameter(torch.zeros(num_cross_asset_factors))
```

The LSTM's *entire* input, in both phases, is width K: the real, LayerNorm'd cross-asset
return sequence in phase 1, the learned `continuation_token` (broadcast identically to every
step) in phase 2. There is no second `nn.LSTM` instance and no attention mechanism — `forward`
calls `self.lstm(...)` twice, sharing the exact same weights both times:

```python
_, (h_n, c_n) = self.lstm(lstm_input)                                   # phase 1: history
continuation = self.continuation_token.view(1, 1, -1).expand(batch_size, forecast_horizon, -1)
forecast_outputs, _ = self.lstm(continuation, (h_n, c_n))               # phase 2: generate
```

This is the plain, attention-free sequence-to-sequence pattern (Sutskever, Vinyals & Le, 2014,
"Sequence to Sequence Learning with Neural Networks" §2: the decoder is conditioned only on
the encoder's final state, not on attending back over it), collapsed further into a *single*
shared LSTM instance for both phases: there's nothing for a second recurrent module or an
attention mechanism to add when the "decoder" phase has no real per-step input to attend
*from* in the first place — every generation step receives the identical continuation token,
so the entire forecast comes purely from the LSTM's own hidden-state trajectory continuing to
evolve after the real input ends. `forecast_outputs[:, t, :]` (the hidden state at generation
step `t`) *is* that step's forecast.

### The orthogonal snapshot (`self.cross_asset_encoder`) — feeds the output layer, not the LSTM

```python
self.cross_asset_encoder = OrthogonalAutoencoder(num_cross_asset_factors, factor_dim).encoder
...
factors_at_t = self.cross_asset_encoder(xret[:, -1, :])
```

A linear map (`K -> D`, `D < K`) with rows constrained to stay exactly orthonormal
throughout training (`torch.nn.utils.parametrizations.orthogonal` — Lezcano-Casado &
Martínez-Rubio, 2019). Per Baldi & Hornik (1989), a linear autoencoder trained by
reconstruction MSE recovers the same subspace PCA would; the orthonormality constraint
makes the individual learned factors themselves mutually orthogonal too — a real,
gradient-trained analogue of PCA (see `fx_forecasting/models/orthogonal_autoencoder.py`),
not an arbitrary learned linear map. Pretrained by `pretrain_autoencoder.py`; this class
only owns the architecture — `main_lstm.py::build_model` loads the pretrained weights
afterward and, by default, freezes them (Kumar et al. 2022: fine-tuning a pretrained
representation risks distorting it, a real concern under FX's non-stationary return
distributions).

Applied only to the forecast origin's own *last* timestep (`xret[:, -1, :]`, not the
LayerNorm'd `input_norm` output the LSTM sees) — a static, structurally-orthogonal
"where does the whole cross-asset universe stand right now" summary, computed once and reused
unchanged at every forecast step, complementary to (not a replacement for) the raw sequence
the LSTM consumes. It reads from the *raw* `xret`, not `input_norm`'s output, because
`pretrain_autoencoder.py` calibrated it against data normalized only by the same upstream
causal rolling z-score `build_target_dataset` applies, not this module's own additional
LSTM-input LayerNorm.

### Side features and spectral summary (output layer only)

`side_features_at_t = base[:, -1, :]` — the forecast origin's own (last-timestep) side
feature values, a static per-window vector, not a sequence. If `use_spectral_features=True`,
`self.spectral` (a `SpectralEmbedding`) runs on the LSTM's own (LayerNorm'd) input sequence,
producing another static vector. All three (side features, orthogonal snapshot, spectral
summary) are concatenated directly into every forecast step's output-layer input — never fed
into the LSTM's recurrence itself.

### Output layer (`self.classifier`)

```python
pre_output_size = hidden_size + (spectral_embedding_dim if use_spectral_features else 0) + factor_dim + num_side_features
self.pre_output_norm = nn.LayerNorm(pre_output_size)
self.classifier = nn.Sequential(
    nn.Linear(pre_output_size, pre_output_size),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(pre_output_size, self.output_size),   # output_size == 1 when target_mode="zscore"
)

for t in range(self.forecast_horizon):
    pre_output = [self.dropout(forecast_outputs[:, t, :])]
    if spectral_vec is not None:
        pre_output.append(spectral_vec)
    pre_output.append(factors_at_t)
    pre_output.append(side_features_at_t)
    output_features = self.pre_output_norm(torch.cat(pre_output, dim=-1))
    step_logits.append(self.classifier(output_features))
```

Concatenates this step's own (dropout-regularized) LSTM output with the spectral summary (if
enabled), the orthogonal cross-asset snapshot, and the side features at the forecast origin,
LayerNorms the result (`pre_output_norm` — so no one block's raw scale dominates purely from
its natural units), then projects to `output_size` outputs — `output_size` discrete class
logits under `target_mode="class"`, or a single continuous value under `target_mode="zscore"`.
The same name (`self.classifier`) is kept for both modes since it's the same module either
way — only its final layer's output width and what loss trains it differ.

## Where this is wired up

- `pretrain_autoencoder.py` — trains `OrthogonalAutoencoder(num_cross_asset_factors,
  factor_dim)` on the input universe's cross-asset log returns, saves its
  `.encoder` weights (plus `pairs`/`seq_len`) to `--autoencoder-path`.
- `main_lstm.py::build_target_dataset` — builds `x`'s two column blocks: the
  target symbol's side features (leading columns) and every pair's own `xret_*` log
  return (trailing K columns); and the target itself, either the discretized direction class
  or the continuous z-score, depending on `cfg.target_mode`.
- `main_lstm.py::build_model` — constructs the model:
  `LSTMSeqClassifier(num_factors=<total feature count>,
  output_size=len(CLASS_NAMES) if cfg.target_mode == "class" else 1,
  num_cross_asset_factors=len(cfg.pairs), factor_dim=<from checkpoint>,
  num_side_features=<computed>, hidden_size=cfg.hidden_size, target_mode=cfg.target_mode, ...)`,
  then loads the pretrained `--autoencoder-path` checkpoint's weights into
  `model.cross_asset_encoder` and (by default) freezes them.
- `main_lstm.py::run_epoch` — calls `model(x)` and computes either a weighted
  `nn.CrossEntropyLoss` (`class_distance_weights`) against argmax-based hit accuracy
  (`target_mode="class"`), or a magnitude-weighted MSE (`weighted_mse_loss`) against
  sign-match hit accuracy (`target_mode="zscore"`).
- `main_lstm.py::collect_predictions` — same `model(x)` call; `target_mode="class"` applies
  argmax + remaps to the signed `CLASS_VALUES` representation, `target_mode="zscore"` uses the
  raw continuous output directly. Used to build the actual-vs-predicted plots and hit-rate
  summary either way.
- `main_lstm.py::collect_signal` — the continuous trading signal that actually drives position
  sizing in the execution-aware backtest. `target_mode="class"`: the expected value of the
  predicted class *distribution* (softmax over the logits, not just argmax) under a fixed
  per-class score (very_bearish=-2 .. very_bullish=2) — see that function's own docstring.
  `target_mode="zscore"`: simply the model's own raw output, which already lives directly in
  that space.

## Quick mental model

Think of it as: *"Read the last 60 days of the whole FX universe's raw returns with one LSTM —
then keep running that same LSTM forward, with no new real input, to generate the forecast
itself out of its own hidden-state momentum. Separately, take a quick orthogonal-factor
snapshot of where the universe stands right now (pretrained autoencoder, last day only).
Combine that snapshot with the target symbol's own momentum/vol/carry/etc. at today (side
features) and, optionally, a frequency-domain summary of the whole input window (spectral),
and turn all of that plus the LSTM's generated step into either a class or a raw z-score."*
With `forecast_horizon=1` (this project's current usage) the generation phase only ever
produces one step.
