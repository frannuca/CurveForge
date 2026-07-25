# LSTM + Attention forecaster — architecture guide

This explains `lstm_forecaster.py` piece by piece, with line numbers, so you can
follow the code while reading. All line numbers refer to
`fx_forecasting/models/lstm_forecaster.py` as of this writing.

## What it does, in one sentence

Given a window of the last `seq_len` days of (normalized) per-symbol FX
features, produce `output_size` raw values — the model itself is task-agnostic;
`main_lstm.py`'s `Config.target_mode` decides what those values mean:

- `target_mode="zscore"` (**default**): `output_size=1`, a single continuous
  value regressing the standardized ("z-scored") `horizon`-day cumulative
  return — see `fx_forecasting/features.py::compute_target_zscore`. Roughly 0
  for a typical no-move horizon, positive/negative in std-dev units for
  unusually large up/down moves. Trained with a weighted MSE loss.
- `target_mode="class"`: `output_size=5`, per-class logits over 5 discrete
  direction classes (`very_bearish` .. `very_bullish`) — see
  `fx_forecasting/features.py::compute_target_class`. Trained with a weighted
  `nn.CrossEntropyLoss`.

## The shape of the problem

- Input `x`: `(batch, seq_len, num_factors)` — e.g. `seq_len=60` days of
  history, `num_factors=8` (one symbol's own return, intraday_vol, momentum,
  rvol, carry, and 3 EWMA-crossover features — see
  `main_lstm.py::build_pooled_datasets`).
- Output: `(batch, output_size)` — `output_size=1` (zscore mode) or `5`
  (class mode), set in `main_lstm.py::build_model`. No output activation in
  either case: for zscore mode the target is an unbounded continuous value,
  so there's nothing to squash; for class mode, `nn.CrossEntropyLoss` applies
  `log_softmax` internally, so pre-squashing the logits (e.g. with a sigmoid
  or ReLU) here would double-apply a nonlinearity and distort the loss.

## Encoder -> attention pooling -> projection

This is a simple encode-then-pool design, not a sequence-to-sequence decoder:
the whole input window is compressed into one context vector and projected
directly to the output, with no per-step autoregressive loop.

```
 x (batch, seq_len, num_factors)
        │
        ▼
 ┌──────────────┐   states (batch, seq_len, hidden)
 │  ENCODER LSTM │ ──────────────────────────────────┐
 │  (nn.LSTM)    │                                    │
 └──────────────┘                                     ▼
                                          ┌─────────────────────┐
                                          │   ATTENTION POOL     │
                                          │  learned query vector │
                                          │  attends over `states` │
                                          └─────────────────────┘
                                                     │
                                          context (batch, hidden)
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │       Linear          │
                                          └─────────────────────┘
                                                     │
                                                     ▼
                                    raw output (batch, output_size)
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
input and returns `states`, the hidden state at **every** timestep, shape
`(batch, seq_len, hidden_size)`. This is inherently causal: at internal step
`t`, the LSTM has only processed inputs `0..t`, so no future leakage is
possible by construction.

### Attention pooling (`AttentionPool`)

Rather than decoding step by step, the full sequence of encoder states is
compressed into one context vector in a single shot, using additive attention
against a **learned query parameter** (not a decoder hidden state — there is
no decoder):

```python
def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
    keys_proj = self.key_proj(states)                              # (batch, seq_len, hidden)
    scores = self.energy_proj(torch.tanh(keys_proj + self.query)).squeeze(-1)  # (batch, seq_len)
    weights = torch.softmax(scores, dim=-1)                         # attention weights, sum to 1
    context = torch.bmm(weights.unsqueeze(1), states).squeeze(1)    # weighted avg of encoder states
    return context, weights
```

- `self.query`: a single learned vector, shared across all inputs — it
  encodes "what kind of pattern in the history is worth attending to",
  learned during training rather than supplied per-sample.
- `scores`: for each of the `seq_len` timesteps, a scalar "how relevant is
  this day", computed by projecting the state, adding the query, squashing
  with `tanh`, then projecting to a single number (`energy_proj`) — the same
  additive/Bahdanau-style scoring as before, just with a fixed query instead
  of a moving decoder state.
- `weights`: softmax over those scores — a probability distribution over the
  `seq_len` input days.
- `context`: the weighted average of the encoder states — a single
  `(batch, hidden_size)` summary of "the whole window, weighted by relevance".

### Projection head

```python
self.output_layer = nn.Linear(hidden_size, output_size)
...
return self.output_layer(self.dropout(context))
```

The pooled context is dropout-regularized and linearly projected to
`output_size` raw values — deliberately no output activation, since the
right nonlinearity (if any) depends entirely on which `target_mode` the
caller is using, not on the model itself.

## Where this is wired up

- `main_lstm.py::build_model` — constructs the model:
  `LSTMAttentionForecaster(num_factors=<per-symbol feature count>, output_size=<1 or len(CLASS_NAMES)>, hidden_size=cfg.hidden_size, ...)`,
  choosing `output_size` from `cfg.target_mode`.
- `main_lstm.py::run_epoch` — calls `model(x)` to get raw outputs, then
  branches on `cfg.target_mode`: weighted `nn.MSELoss`-style loss + squeezed
  scalar predictions (zscore), or weighted `nn.CrossEntropyLoss` + `argmax`
  predictions (class). No separate teacher-forcing training path either way,
  since there's no autoregressive decoding.
- `main_lstm.py::collect_predictions` — same `model(x)` call, mode-aware
  post-processing, used to build the actual-vs-predicted plots and hit-rate
  summary.

## Quick mental model

Think of it as: *"Read the last 60 days of one symbol's market data
(encoder). Decide which of those days matter most (attention pooling). Turn
that single summary into raw output values (projection head) — a z-score or
a set of class scores, depending on what's being trained."* One shot, no
step-by-step decoding.
