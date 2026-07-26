# LSTM + Attention forecaster — architecture guide

This explains `lstm_forecaster.py` piece by piece, with line numbers, so you can
follow the code while reading. All line numbers refer to
`fx_forecasting/models/lstm_forecaster.py` as of this writing.

## What it does, in one sentence

Given a window of the last `seq_len` days of (normalized) per-symbol FX
features, an encoder-decoder LSTM with attention forecasts `forecast_horizon`
steps ahead **recursively** — the classic sequence-to-sequence-with-attention
design (Sutskever et al., 2014; Bahdanau et al., 2015), adapted for
continuous or per-class-logit regression instead of token generation. The
model itself is task-agnostic; `main_lstm.py`'s `Config.target_mode` decides
what each step's `output_size` values mean:

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

- Input `x`: `(batch, seq_len, num_factors)` — e.g. `seq_len=60` days of
  history, `num_factors=8` (one symbol's own return, intraday_vol, momentum,
  rvol, carry, and 3 EWMA-crossover features — see
  `main_lstm.py::build_pooled_datasets`).
- Output: `(batch, forecast_horizon, output_size)` if `forecast_horizon > 1`,
  else `(batch, output_size)` — `output_size` is `1` (zscore mode) or `5`
  (class mode) in `main_lstm.py::build_model`. No output activation in
  either case: for zscore mode the target is an unbounded continuous value,
  so there's nothing to squash; for class mode, `nn.CrossEntropyLoss` applies
  `log_softmax` internally, so pre-squashing the logits (e.g. with a sigmoid
  or ReLU) here would double-apply a nonlinearity and distort the loss.

## Encoder -> recursive attention decoder

Unlike a simple encode-then-pool design, this *is* a sequence-to-sequence
decoder: the decoder unrolls step by step for `forecast_horizon` iterations,
attending over every encoder timestep afresh at each step and feeding its own
prediction back in as the next step's input.

```
 x (batch, seq_len, num_factors)
        │
        ▼
 ┌──────────────┐   encoder_outputs (batch, seq_len, hidden)
 │  ENCODER LSTM │ ─────────────────────────────────────────┐
 │  (nn.LSTM)    │   final (h, c) seed the decoder            │
 └──────────────┘        │                                  │
                          ▼                                  │
                 ┌─────────────────┐                         │
        ┌───────▶│   DECODER LOOP  │◀────────────────────────┘
        │        │ (runs `forecast_horizon` times)           │
        │        │                                            │
        │        │  1. attention_pool(dec_h, encoder_outputs)   │
        │        │       -> context vector                      │
        │        │  2. LSTMCell(prev_embed + context)             │
        │        │       -> new dec_h, dec_c                       │
        │        │  3. output_layer(dec_h + context)                │
        │        │       -> this step's raw output                  │
        └────────┤  4. embed this step's value (or ground truth,       │
                 │     with teacher forcing) for the next iteration     │
                 └─────────────────────────────────────────────────────┘
```

### Encoder (`self.encoder`)

```python
self.encoder = nn.LSTM(
    input_size=num_factors,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    dropout=dropout if num_layers > 1 else 0.0,
)
```

A standard multi-layer LSTM. It reads the whole `(batch, seq_len, num_factors)`
input and produces `encoder_outputs` (the hidden state at **every** timestep,
shape `(batch, seq_len, hidden_size)` — what the decoder attends over) plus
the final layer's last hidden/cell state, which seeds the decoder
(`encode()` returns `encoder_outputs, h_n[-1], c_n[-1]`). Inherently causal:
at internal step `t`, the LSTM has only processed inputs `0..t`.

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

    step_out = self.output_layer(torch.cat([self.dropout(dec_h), context], dim=-1))
    step_outputs.append(step_out)                                     # (3) this step's raw output

    use_teacher_forcing = targets is not None and self.training and torch.rand(()).item() < teacher_forcing_ratio
    next_value = targets[:, t] if use_teacher_forcing else (step_out.argmax(-1) if self.is_classification else step_out.squeeze(-1))
    prev_embed = self.value_embedding(next_value)                      # (4) feed it back for the next step
```

1. **Attend**: given where the decoder currently is (`dec_h`), figure out
   which input days matter (`context`).
2. **Step the LSTM**: feed `[previous value's embedding, attention context]`
   into an `LSTMCell` to get the new decoder state.
3. **Predict**: combine the (dropout-regularized) decoder state with the
   attention context, project to `output_size` raw values — this step's output.
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

- `main_lstm.py::build_model` — constructs the model:
  `LSTMAttentionForecaster(num_factors=<per-symbol feature count>, output_size=<1 or len(CLASS_NAMES)>, hidden_size=cfg.hidden_size, ...)`,
  choosing `output_size` from `cfg.target_mode` and leaving `forecast_horizon`
  at its default of `1`.
- `main_lstm.py::run_epoch` — calls `model(x)` (no `targets`, so no teacher
  forcing — harmless at `forecast_horizon=1` since there's no next step to
  feed forward into anyway) and branches on `cfg.target_mode`: weighted
  `nn.MSELoss`-style loss + squeezed scalar predictions (zscore), or weighted
  `nn.CrossEntropyLoss` + `argmax` predictions (class).
- `main_lstm.py::collect_predictions` — same `model(x)` call, mode-aware
  post-processing, used to build the actual-vs-predicted plots and hit-rate
  summary.

## Quick mental model

Think of it as: *"Read the last 60 days of one symbol's market data
(encoder). Then, for each future step you need to forecast, look back at
whichever of those 60 days seem most relevant right now (attention), turn
that into a number or class (decoder), and carry that forward into the next
step."* With `forecast_horizon=1` (this project's current usage) that loop
only ever runs once.
