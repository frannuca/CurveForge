"""LSTM sequence generator + feed-forward output head — no encoder-decoder split, no attention.

A single `nn.LSTM` consumes the raw (rolling-normalized upstream) cross-asset log-return
history — its *only* input — then continues its own recurrence for `forecast_horizon` further
steps to directly generate the forecast sequence itself: since there's no real future return
to feed it at those steps (that's what's being forecast), a single learned "continuation"
vector stands in at each of them, letting the LSTM's own hidden-state dynamics carry the
forecast forward from where the real history left off. This is the plain, attention-free
sequence-to-sequence pattern (Sutskever, Vinyals & Le, 2014, "Sequence to Sequence Learning
with Neural Networks" §2: the decoder is conditioned only on the encoder's final state, not on
attending back over it) — collapsed further here into a *single* shared LSTM instance for both
phases (no separate encoder/decoder weight sets), since there's nothing for a second recurrent
module or an attention mechanism to add when the "decoder" has no real per-step input to
attend *from* in the first place.

A pretrained, orthogonally-reduced snapshot of the same cross-asset history (last input
timestep only), the target symbol's own side features (last input timestep only), and an
optional spectral (FFT) summary of the input sequence are computed once and reused at every
forecast step — concatenated with that step's own LSTM output and passed through a small
feed-forward network to produce that step's output (Lim, Arık, Loeff & Pfister, 2021,
"Temporal Fusion Transformers": non-sequential/static covariates are best given directly to
the output stage rather than forced through the recurrence).

`target_mode` picks what that output head produces, as two mutually exclusive alternatives
over the exact same trunk (everything up to `pre_output_norm` is identical either way):
`"class"` (default) — `output_size` discrete direction-class logits, trained under
`nn.CrossEntropyLoss`; or `"zscore"` — a single continuous value per step (the horizon return
z-score itself), trained under a regression loss (e.g. MSE) — see `main_lstm.py::run_epoch`
for how each mode's loss/metrics differ. Classification lets the trading signal
(`main_lstm.collect_signal`) recover a conviction-weighted expected value from the *entire*
predicted class distribution; direct z-score regression instead lets the model express any
real-valued conviction directly, at the cost of the discrete-class-imbalance handling
(`class_distance_weights`) classification gets for free. Both are kept as options because
neither dominates the other a priori — chosen by `--target-mode` per run, compared like any
other hyperparameter under the same purged walk-forward + frozen test holdout discipline as
everything else in this pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fx_forecasting.models.orthogonal_autoencoder import OrthogonalAutoencoder
from fx_forecasting.models.spectral_embedding import SpectralEmbedding


class LSTMSeqClassifier(nn.Module):
    """A single LSTM generates `forecast_horizon` forecast steps directly from the cross-asset
    log-return history; a feed-forward network turns each step into either class logits or a
    continuous z-score prediction, depending on `target_mode`.

    Input: ``x`` of shape (batch, seq_len, num_factors), two column blocks in this exact
    order (`main_lstm.py::build_target_dataset`'s convention): leading `num_side_features`
    columns of target-symbol-only non-sequential covariates (momentum, rvol, skew, kurt,
    carry, intraday volatility), then the *trailing* `num_cross_asset_factors` (K) columns of
    per-pair `xret_*` cross-asset log returns (including the target's own) — the LSTM's
    *only* input; the side features never enter the recurrence.

    Output, `target_mode="class"` (default): raw per-step class logits, shape
    (batch, forecast_horizon, output_size) when ``forecast_horizon > 1``, or
    (batch, output_size) when it's 1 — no output activation: `nn.CrossEntropyLoss` applies
    `log_softmax` internally during training (explicit `torch.softmax` is applied separately
    at inference where a probability distribution is actually needed — see
    `main_lstm.collect_signal`), so pre-squashing the logits here would double-apply a
    nonlinearity and distort the loss.

    Output, `target_mode="zscore"`: a single continuous value per step, shape
    (batch, forecast_horizon) when ``forecast_horizon > 1``, or (batch,) when it's 1 — again no
    output activation, since it's a regression target with no natural bound to squash toward.
    `output_size` is ignored in this mode (the output head is fixed at width 1).

    `self.lstm` is called *twice*, sharing the same weights both times:

    1. Over the real `seq_len`-step history (`xret`, LayerNorm'd via `input_norm`), producing
       a final `(h_n, c_n)` — the only thing carried from this phase to the next.
    2. Over `forecast_horizon` further steps, each fed the *same* learned
       `continuation_token` (there is no real input for a future step — this is what's being
       forecast), starting from that `(h_n, c_n)`. The hidden state at each of these steps
       *is* that step's forecast, generated purely from the LSTM's own recurrence continuing
       to evolve — no attention over the history, no second LSTM instance.

    `cross_asset_encoder`: the *same* trailing K columns, at the forecast origin's own last
    timestep only, are projected to `factor_dim` (D < K) orthogonal factors by an
    `OrthogonalAutoencoder`'s encoder (see `fx_forecasting.models.orthogonal_autoencoder`) —
    pretrained by `pretrain_autoencoder.py`, its weights loaded and optionally frozen by
    `main_lstm.build_model`, *not* trained fresh here. A static, structurally-orthogonal
    snapshot of "where the whole cross-asset universe stands right now," reused unchanged at
    every forecast step's classifier input, alongside the side features and (if enabled) the
    spectral summary.
    """

    def __init__(
        self,
        num_factors: int,
        output_size: int,
        num_cross_asset_factors: int,
        factor_dim: int,
        num_side_features: int = 0,
        forecast_horizon: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_spectral_features: bool = False,
        spectral_embedding_dim: int = 8,
        spectral_freq_bins: int = 16,
        target_mode: str = "class",
    ) -> None:
        super().__init__()
        if target_mode not in ("class", "zscore"):
            raise ValueError(f"target_mode must be 'class' or 'zscore', got {target_mode!r}")
        if target_mode == "class" and output_size < 2:
            raise ValueError("output_size must be >= 2 (classification requires at least 2 classes)")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")
        if num_cross_asset_factors <= 0:
            raise ValueError("num_cross_asset_factors must be > 0")
        if num_factors < num_cross_asset_factors + num_side_features:
            raise ValueError(
                f"num_factors ({num_factors}) must be >= num_cross_asset_factors "
                f"({num_cross_asset_factors}) + num_side_features ({num_side_features})"
            )

        self.num_factors = num_factors
        self.target_mode = target_mode
        # A "zscore" head is a fixed single scalar per step regardless of what output_size
        # was passed — output_size only ever means "number of classes" in "class" mode.
        self.output_size = output_size if target_mode == "class" else 1
        self.num_cross_asset_factors = num_cross_asset_factors
        self.factor_dim = factor_dim
        self.num_side_features = num_side_features
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_spectral_features = use_spectral_features

        # Pretrained (see pretrain_autoencoder.py), orthogonally-constrained K -> D snapshot
        # of the forecast origin's own cross-asset state — feeds the classifier (see
        # forward), not the LSTM.
        self.cross_asset_encoder = OrthogonalAutoencoder(num_cross_asset_factors, factor_dim).encoder

        # The LSTM's entire input, in both phases, is width K: real cross-asset returns in
        # phase 1, the learned continuation_token (broadcast every step) in phase 2.
        self.input_norm = nn.LayerNorm(num_cross_asset_factors)
        self.lstm = nn.LSTM(
            input_size=num_cross_asset_factors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.continuation_token = nn.Parameter(torch.zeros(num_cross_asset_factors))

        # Operates on the LSTM's own (LayerNorm'd) history input, and — like the orthogonal
        # snapshot — feeds the classifier directly (see pre_output_norm/classifier below),
        # never the recurrence itself.
        self.spectral = (
            SpectralEmbedding(num_cross_asset_factors, spectral_freq_bins, spectral_embedding_dim)
            if use_spectral_features
            else None
        )

        self.dropout = nn.Dropout(dropout)

        # Feed-forward classifier head: this step's own LSTM hidden state, plus every
        # non-sequential covariate — spectral summary (if enabled), the orthogonal
        # cross-asset snapshot, and the side features at the forecast origin — LayerNorm'd
        # together immediately before the network so no one block's raw scale dominates the
        # others purely because of its natural units, not its actual predictive relevance.
        pre_output_size = (
            hidden_size
            + (spectral_embedding_dim if use_spectral_features else 0)
            + factor_dim
            + num_side_features
        )
        self.pre_output_norm = nn.LayerNorm(pre_output_size)
        self.classifier = nn.Sequential(
            nn.Linear(pre_output_size, pre_output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pre_output_size, self.output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: normalized multifactor input sequence, shape (batch, seq_len, num_factors).

        Returns:
            `target_mode="class"`: per-step class logits, shape
            (batch, forecast_horizon, output_size) if `forecast_horizon > 1`, else
            (batch, output_size). `target_mode="zscore"`: per-step continuous prediction,
            shape (batch, forecast_horizon) if `forecast_horizon > 1`, else (batch,).
        """
        batch_size = x.size(0)
        k = self.num_cross_asset_factors
        base, xret = x[..., :-k], x[..., -k:]
        lstm_input = self.input_norm(xret)

        # Phase 1: consume the real history.
        _, (h_n, c_n) = self.lstm(lstm_input)

        # Phase 2: continue the *same* LSTM for forecast_horizon more steps, generating the
        # forecast sequence purely from its own hidden-state evolution — no real input, no
        # second recurrent module, no attention over phase 1's outputs.
        continuation = self.continuation_token.view(1, 1, -1).expand(batch_size, self.forecast_horizon, -1)
        forecast_outputs, _ = self.lstm(continuation, (h_n, c_n))  # (batch, forecast_horizon, hidden_size)

        factors_at_t = self.cross_asset_encoder(xret[:, -1, :])
        side_features_at_t = base[:, -1, :]
        spectral_vec = self.spectral(lstm_input) if self.spectral is not None else None

        step_logits = []
        for t in range(self.forecast_horizon):
            pre_output = [self.dropout(forecast_outputs[:, t, :])]
            if spectral_vec is not None:
                pre_output.append(spectral_vec)
            pre_output.append(factors_at_t)
            pre_output.append(side_features_at_t)
            output_features = self.pre_output_norm(torch.cat(pre_output, dim=-1))
            step_logits.append(self.classifier(output_features))

        result = torch.stack(step_logits, dim=1)  # (batch, forecast_horizon, output_size)
        if self.target_mode == "zscore":
            result = result.squeeze(-1)  # (batch, forecast_horizon) — output_size is fixed at 1
        return result.squeeze(1) if self.forecast_horizon == 1 else result
