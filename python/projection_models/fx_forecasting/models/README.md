# LSTM + Attention forecaster — architecture guide

This explains `lstm_forecaster.py` piece by piece, so you can follow the code
while reading. See also `pretrain_autoencoder.py` and
`fx_forecasting/models/orthogonal_autoencoder.py` for the pretrained
cross-asset factor reduction this model depends on.

## What it does, in one sentence

The model forecasts one target FX pair. Its recurrent input is a pretrained,
orthogonally-reduced cross-asset return factor sequence — every pair in the
input universe's own daily log return (K series, target included) compressed
down to D orthogonal factors by a linear autoencoder trained separately
(`pretrain_autoencoder.py`) and loaded frozen (by default) into this model —
concatenated with the target symbol's own (normalized) log return sequence,
uncompressed, so the recurrence has undiluted access to the series it's
actually forecasting on top of the shared cross-asset structure. Everything
else — momentum, realized vol, skew, kurtosis, carry, intraday volatility,
and an optional spectral (FFT) summary of the combined sequence — bypasses
the LSTM/attention recurrence entirely and is injected only at the final
prediction layer (Lim, Arık, Loeff & Pfister, 2021, "Temporal Fusion
Transformers": non-sequential covariates are best fed directly to the output
stage rather than forced through the same recurrent bottleneck as the
genuine time-varying signal).

An encoder-decoder LSTM with attention then forecasts `forecast_horizon`
steps ahead **recursively** over that factor sequence — the classic
sequence-to-sequence-with-attention design (Sutskever et al., 2014; Bahdanau
et al., 2015), adapted for continuous or per-class-logit regression instead
of token generation. The model itself is task-agnostic; `main_lstm.py`'s
`Config.target_mode` decides what each step's `output_size` values mean:

- `target_mode="zscore"` (**default**): `output_size=1`, a single continuous
  value regressing the standardized ("z-scored") `horizon`-day cumulative
  return — see `fx_forecasting/features.py::compute_target_zscore`. Roughly 0
  for a typical no-move horizon, positive/negative in std-dev units for
  unusually large up/down moves. Trained with a weighted MSE loss.
- `target_mode="class"`: `output_size=5`, per-class logits over 5 discrete
  direction classes (`very_bearish` .. `very_bullish`) — see
  `fx_forecasting/features.py::compute_target_class`. Trained with a weighted
  `nn.CrossEntropyLoss`.

`main_lstm.py` currently always uses `forecast_horizon=1` (a single N-day-ahead
target, not a multi-step sequence), in which case the model's output collapses
to `(batch, output_size)` and the recursive decoder loop below just runs once
— but the architecture itself supports `forecast_horizon > 1` for genuine
multi-step-ahead forecasting.

## The shape of the problem

- Input `x`: `(batch, seq_len, num_factors)`, where `num_factors =
  num_side_features + num_target_series + num_cross_asset_factors` — three
  column blocks in that order: the leading `num_side_features` columns are
  the target symbol's own non-sequential covariates (momentum, rvol, skew,
  kurt, carry, intraday volatility); the middle `num_target_series` columns
  are the target symbol's own (normalized) log return sequence; the
  *trailing* `num_cross_asset_factors` (K) columns are the per-pair `xret_*`
  cross-asset log returns (including the target's own) — see
  `main_lstm.py::build_target_dataset`.
- Output: `(batch, forecast_horizon, output_size)` if `forecast_horizon > 1`,
  else `(batch, output_size)` — `output_size` is `1` (zscore mode) or `5`
  (class mode) in `main_lstm.py::build_model`. No output activation in
  either case: for zscore mode the target is an unbounded continuous value,
  so there's nothing to squash; for class mode, `nn.CrossEntropyLoss` applies
  `log_softmax` internally, so pre-squashing the logits (e.g. with a sigmoid
  or ReLU) here would double-apply a nonlinearity and distort the loss.

## Cross-asset factor reduction -> encoder -> recursive attention decoder

Before the LSTM ever runs, `encode()` splits `x` into its three blocks, reduces the
cross-asset block through the pretrained orthogonal encoder, and concatenates that with the
target's own return series:

```
 x (batch, seq_len, num_factors)
        │
        │  split: base = x[..., :-(K+T)] (side features),
        │         target_series = x[..., -(K+T):-K], xret = x[..., -K:]
        ▼
 xret (batch, seq_len, K) ──▶ cross_asset_encoder (pretrained, orthonormal rows)
        │                            │
        │                    factors (batch, seq_len, D)
        │                            │
        │      target_series (batch, seq_len, T) ──▶ cat([factors, target_series])
        │                            │
        │                    factor_norm (LayerNorm, width D+T)
        │                            ▼
        │                   ┌──────────────┐   encoder_outputs (batch, seq_len, hidden)
        │                   │  ENCODER LSTM │ ─────────────────────────────────────────┐
        │                   │  (nn.LSTM)    │   final (h, c) seed the decoder            │
        │                   └──────────────┘        │                                  │
        │                                            ▼                                  │
        │                                   ┌─────────────────┐                         │
        │                          ┌───────▶│   DECODER LOOP  │◀────────────────────────┘
        │                          │        │ (runs `forecast_horizon` times)           │
        │                          │        │                                            │
        │                          │        │  1. attention_pool(dec_h, encoder_outputs)   │
        │                          │        │       -> context vector                      │
        │                          │        │  2. LSTMCell(prev_embed + context)             │
        │                          │        │       -> new dec_h, dec_c                       │
        │                          │        │  3. output_layer(pre_output_norm([dec_h, context,│
        │                          │        │       spectral_vec?, side_features_at_t]))        │
        │                          │        │       -> this step's raw output                  │
        │                          └────────┤  4. embed this step's value (or ground truth,       │
        │                                   │     with teacher forcing) for the next iteration     │
        │                                   └─────────────────────────────────────────────────────┘
        └──▶ side_features_at_t = base[:, -1, :] (forecast origin's own values) ───────────────────┘
                (feeds step 3 directly, at every decoding step, unchanged)
```

### Cross-asset factor encoder + target series (`self.cross_asset_encoder`)

```python
self.cross_asset_encoder = OrthogonalAutoencoder(num_cross_asset_factors, factor_dim).encoder
lstm_input_size = factor_dim + num_target_series
self.factor_norm = nn.LayerNorm(lstm_input_size)
```

A linear map (`K -> D`, `D < K`) with rows constrained to stay exactly
orthonormal throughout training (`torch.nn.utils.parametrizations.orthogonal`
— Lezcano-Casado & Martínez-Rubio, 2019). Per Baldi & Hornik (1989), a linear
autoencoder trained by reconstruction MSE recovers the same subspace PCA
would; the orthonormality constraint makes the individual learned factors
themselves mutually orthogonal too — a real, gradient-trained analogue of
PCA (see `fx_forecasting/models/orthogonal_autoencoder.py`), not an arbitrary
learned linear map. Pretrained by `pretrain_autoencoder.py`; this class only
owns the architecture — `main_lstm.py::build_model` loads the pretrained
weights afterward and, by default, freezes them (Kumar et al. 2022: fine-
tuning a pretrained representation risks distorting it, a real concern under
FX's non-stationary return distributions).

Since `D < K`, this compression can dilute the target symbol's own
idiosyncratic dynamics into the shared cross-asset structure. `num_target_series`
columns of the target's own (normalized) log return are concatenated onto the D
factors, uncompressed, before the LSTM ever sees either — giving the
recurrence privileged, undiluted access to the series it's actually
forecasting, on top of the shared factor structure. `factor_norm` LayerNorms
the combined `(D + num_target_series)`-dim sequence before the LSTM, since
the orthogonal map preserves total variance but can redistribute it unevenly
across the D output dimensions depending on the data's own covariance
structure.

### Encoder (`self.encoder`)

```python
self.encoder = nn.LSTM(
    input_size=factor_dim + num_target_series,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    dropout=dropout if num_layers > 1 else 0.0,
)
```

A standard multi-layer LSTM — but its *entire* input is now the
`(D + num_target_series)`-dim, LayerNorm'd sequence, never the raw side
features. It produces `encoder_outputs` (the hidden state at **every**
timestep, shape `(batch, seq_len, hidden_size)` — what the decoder attends
over) plus the final layer's last hidden/cell state, which seeds the decoder
(`encode()` returns `encoder_outputs, h_n[-1], c_n[-1], spectral_vec,
side_features_at_t`). Inherently causal: at internal step `t`, the LSTM has
only processed inputs `0..t`.

### Side features and spectral summary (final prediction layer only)

`side_features_at_t = base[:, -1, :]` — the forecast origin's own (last-
timestep) side-feature values, a static per-window vector, not a sequence.
If `use_spectral_features=True`, `self.spectral` (a `SpectralEmbedding`) also
runs on the same combined `(D + num_target_series)`-dim sequence the LSTM
does, producing another static vector. Both are concatenated directly into
the decoder's per-step output features (see the decoder loop below) — never
fed into the recurrence or the attention mechanism itself.

### Attention (`AttentionPool`)

At each decoding step, the decoder doesn't just rely on the encoder's final
summary — it looks back at *every* encoder timestep and learns which ones
matter most for the current step, queried by its own evolving hidden state:

```python
def forward(self, query: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
    query_proj = self.query_proj(query).unsqueeze(1)    # decoder state -> (batch, 1, hidden)
    keys_proj = self.key_proj(keys)                     # encoder states -> (batch, seq_len, hidden)
    scores = self.energy_proj(torch.tanh(query_proj + keys_proj)).squeeze(-1)  # (batch, seq_len)
    weights = torch.softmax(scores, dim=-1)             # attention weights, sum to 1
    context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # weighted avg of encoder states
    return context, weights
```

- `query` = the decoder's current hidden state (`dec_h`) — changes every
  decoding step, unlike a fixed learned parameter.
- `keys` = `encoder_outputs`, every encoder timestep.
- `scores`: for each of the `seq_len` timesteps, "how relevant is this day to
  the current step's prediction" — additive/Bahdanau-style: project both
  query and keys into the same space, add, squash with `tanh`, project to a
  scalar (`energy_proj`).
- `context`: the weighted average of encoder states — the "relevant history"
  for this specific decoding step.

### Decoder loop (in `forward`)

```python
prev_embed = self.start_token.unsqueeze(0).expand(batch_size, -1)
for t in range(self.forecast_horizon):
    context, _ = self.attention_pool(dec_h, encoder_outputs)          # (1) look back at input history
    decoder_input = torch.cat([prev_embed, context], dim=-1)          # (2) combine with last value
    dec_h, dec_c = self.decoder_cell(decoder_input, (dec_h, dec_c))   #     advance the decoder LSTM state

    pre_output = [self.dropout(dec_h), context]
    if spectral_vec is not None:
        pre_output.append(spectral_vec)
    pre_output.append(side_features_at_t)
    output_features = self.pre_output_norm(torch.cat(pre_output, dim=-1))
    step_out = self.output_layer(output_features)
    step_outputs.append(step_out)                                     # (3) this step's raw output

    use_teacher_forcing = targets is not None and self.training and torch.rand(()).item() < teacher_forcing_ratio
    next_value = targets[:, t] if use_teacher_forcing else (step_out.argmax(-1) if self.is_classification else step_out.squeeze(-1))
    prev_embed = self.value_embedding(next_value)                      # (4) feed it back for the next step
```

1. **Attend**: given where the decoder currently is (`dec_h`), figure out
   which input days matter (`context`).
2. **Step the LSTM**: feed `[previous value's embedding, attention context]`
   into an `LSTMCell` to get the new decoder state.
3. **Predict**: concatenate the (dropout-regularized) decoder state, the
   attention context, the spectral summary (if enabled), and the side
   features at the forecast origin, LayerNorm the result (`pre_output_norm`
   — so no one block's raw scale dominates purely from its natural units),
   then project to `output_size` raw values — this step's output.
4. **Feed forward**: embed the value just produced (or, during training,
   possibly the ground truth instead — see teacher forcing below) so the
   *next* iteration's decoder input reflects "what happened at step t". On
   the very first iteration there's no previous value yet, so `prev_embed`
   starts as the learned `start_token`.

`value_embedding`'s type depends on what's being fed back: `nn.Embedding(output_size,
hidden_size)` for a discrete class ID (`target_mode="class"`), or `nn.Linear(1,
hidden_size)` for a continuous scalar (`target_mode="zscore"`) — the standard
way to embed a discrete vs. continuous decoder output back into the next
step's input.

### Teacher forcing

During training, instead of always feeding the model's own (possibly wrong)
prediction back into itself at step 4, there's a `teacher_forcing_ratio`
probability of feeding the **true** target value instead (Bengio et al.,
2015, "Scheduled Sampling"): early in training especially, this keeps errors
from one step from compounding into every later step, since the decoder
practices predicting from a *correct* history rather than its own mistakes.
Only matters when `forecast_horizon > 1` — with `forecast_horizon=1`
(`main_lstm.py`'s current usage) there's no "next step" for it to affect.

## Where this is wired up

- `pretrain_autoencoder.py` — trains `OrthogonalAutoencoder(num_cross_asset_factors,
  factor_dim)` on the input universe's cross-asset log returns, saves its
  `.encoder` weights (plus `pairs`/`seq_len`) to `--autoencoder-path`.
- `main_lstm.py::build_target_dataset` — builds `x`'s three column blocks: the
  target symbol's side features (leading columns), the target symbol's own
  log return sequence (middle column), and every pair's own `xret_*` log
  return (trailing K columns).
- `main_lstm.py::build_model` — constructs the model:
  `LSTMAttentionForecaster(num_factors=<total feature count>,
  output_size=<1 or len(CLASS_NAMES)>, num_cross_asset_factors=len(cfg.pairs),
  factor_dim=<from checkpoint>, num_side_features=<computed>,
  num_target_series=<computed>, hidden_size=cfg.hidden_size, ...)`,
  then loads the pretrained `--autoencoder-path` checkpoint's weights into
  `model.cross_asset_encoder` and (by default) freezes them.
- `main_lstm.py::run_epoch` — calls `model(x)` (no `targets`, so no teacher
  forcing — harmless at `forecast_horizon=1` since there's no next step to
  feed forward into anyway) and branches on `cfg.target_mode`: weighted MSE
  (+ variance penalty, active by default — see `variance_penalty`) + squeezed
  scalar predictions (zscore), or weighted `nn.CrossEntropyLoss` + `argmax`
  predictions (class).
- `main_lstm.py::collect_predictions` — same `model(x)` call, mode-aware
  post-processing, used to build the actual-vs-predicted plots and hit-rate
  summary.

## Quick mental model

Think of it as: *"Reduce the last 60 days of the whole FX universe's returns
to a handful of orthogonal factors (pretrained autoencoder). Read those
factors (encoder). Then, for each future step you need to forecast, look
back at whichever of those days seem most relevant right now (attention),
combine that with the target symbol's own momentum/vol/carry/etc. at today
(side features), turn that into a number or class (decoder), and carry that
forward into the next step."* With `forecast_horizon=1` (this project's
current usage) that loop only ever runs once.
