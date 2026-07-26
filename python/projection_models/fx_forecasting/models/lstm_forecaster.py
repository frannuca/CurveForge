"""Causal LSTM encoder-decoder with attention, forecasting N steps ahead recursively.

The encoder consumes a normalized, multifactor input sequence (e.g. several FX
pairs' standardized returns) and is causal by construction: at step ``t`` it
has only seen inputs up to ``t``. The decoder then unrolls autoregressively
for ``forecast_horizon`` steps — the classic sequence-to-sequence-with-
attention design (Sutskever et al., 2014, "Sequence to Sequence Learning
with Neural Networks"; Bahdanau et al., 2015, "Neural Machine Translation by
Jointly Learning to Align and Translate"), adapted here for continuous or
per-class-logit regression instead of token generation. At every decoding
step the model attends over *every* encoder timestep — queried by the
decoder's own current hidden state, so the attention weights are recomputed
fresh at each step, not fixed — then feeds its own prediction from step t
back in as input for step t+1 (with optional teacher forcing during
training), the standard way to extend a single-step model into a genuine
multi-step forecaster.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fx_forecasting.models.cross_asset_projection import CrossAssetFactorProjection
from fx_forecasting.models.spectral_embedding import SpectralEmbedding


class AttentionPool(nn.Module):
    """Additive (Bahdanau-style) attention: scores every encoder state against a query,
    returning their weighted sum as a context vector.

    Used as the decoder's attention: `query` is the decoder's current hidden
    state, which changes at every recursive decoding step (unlike a fixed
    learned parameter), so the model can attend to different parts of the
    input history depending on what it's already forecast so far.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            query: decoder hidden state, shape (batch, hidden_size).
            keys: encoder states, shape (batch, seq_len, hidden_size).

        Returns:
            context: attention-weighted summary, shape (batch, hidden_size).
            weights: attention weights over the sequence, shape (batch, seq_len).
        """
        query_proj = self.query_proj(query).unsqueeze(1)
        keys_proj = self.key_proj(keys)
        scores = self.energy_proj(torch.tanh(query_proj + keys_proj)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class LSTMAttentionForecaster(nn.Module):
    """Encoder-decoder LSTM with Bahdanau attention, forecasting `forecast_horizon`
    steps ahead recursively.

    Input: ``x`` of shape (batch, seq_len, num_factors) — normalized time series.
    Output: raw per-step values, shape (batch, forecast_horizon, output_size)
    when ``forecast_horizon > 1``, or (batch, output_size) when it's 1 (so
    existing single-step callers, e.g. ``main_lstm.py``, are unaffected) — no
    output activation, since the right nonlinearity (softmax for
    classification, none for regression) depends on the loss/target the
    caller has chosen.

    Feedback at each decoding step depends on what `output_size` represents:
    a per-step class ID (embedded via `nn.Embedding`) when `output_size > 1`
    (classification, logits over classes), or the raw scalar value (via a
    linear projection) when `output_size == 1` (continuous regression) — the
    standard way to feed a discrete vs. continuous decoder output back in as
    the next step's input.

    `use_spectral_features=True` additionally runs the (possibly cross-asset-reduced, see
    below) input through a `SpectralEmbedding` (a small learned projection of the input
    window's own FFT magnitude spectrum) and appends it, projected to `hidden_size`, as one
    extra attendable "key" after the encoder's own `seq_len` states — the decoder's
    attention can then choose to draw on this frequency-domain summary at each step,
    without the encoder's own LSTM recurrence ever seeing it — see
    `fx_forecasting.models.spectral_embedding`.

    `cross_asset_pca_dim=D` (requires `num_cross_asset_factors=K>D`) additionally runs the
    *trailing* `K` columns of `x` — by convention, `main_lstm.py::build_pooled_datasets`
    always places the per-pair `xret_*` cross-asset-return columns last — through a learned,
    PCA-like linear bottleneck (`CrossAssetFactorProjection`) before anything else touches
    `x`: the raw K (often highly collinear) per-pair returns become D learned combinations
    of them, applied identically at every timestep, shrinking the effective input width the
    encoder (and, if enabled, the spectral embedding) actually sees from
    `num_factors` to `num_factors - K + D` — see
    `fx_forecasting.models.cross_asset_projection`.
    """

    def __init__(
        self,
        num_factors: int,
        output_size: int,
        forecast_horizon: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_uncertainty: bool = False,
        use_spectral_features: bool = False,
        spectral_embedding_dim: int = 8,
        spectral_freq_bins: int = 16,
        num_cross_asset_factors: int = 0,
        cross_asset_pca_dim: int | None = None,
    ) -> None:
        super().__init__()
        if output_size < 1:
            raise ValueError("output_size must be >= 1")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")

        self.num_factors = num_factors
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_classification = output_size > 1
        self.predict_uncertainty = predict_uncertainty and not self.is_classification
        self.use_spectral_features = use_spectral_features
        self.num_cross_asset_factors = num_cross_asset_factors
        self.cross_asset_pca_dim = cross_asset_pca_dim

        # Optional learned linear bottleneck over the trailing `num_cross_asset_factors`
        # columns of `x` (the per-pair xret_* cross-asset returns) — reduces K collinear
        # raw returns to D learned combinations before anything else (encoder, spectral
        # embedding) sees them. See `encode`.
        self.cross_asset_proj = (
            CrossAssetFactorProjection(num_cross_asset_factors, cross_asset_pca_dim)
            if cross_asset_pca_dim is not None
            else None
        )
        effective_num_factors = (
            num_factors - num_cross_asset_factors + cross_asset_pca_dim
            if self.cross_asset_proj is not None
            else num_factors
        )

        # Optional frequency-domain summary of the input window (see
        # `SpectralEmbedding`), projected to hidden_size and appended as one extra
        # attendable key after the encoder runs (see `encode`) — the encoder's own
        # LSTM recurrence never sees it, only the decoder's attention does, so it
        # can learn to draw on periodic/cyclical structure at each decoding step
        # without that structure ever influencing how the encoder reads the raw
        # time-domain input. Operates on the same (possibly cross-asset-reduced)
        # representation the encoder does, not the original raw `x`.
        self.spectral = (
            SpectralEmbedding(effective_num_factors, spectral_freq_bins, spectral_embedding_dim)
            if use_spectral_features
            else None
        )
        self.spectral_key_proj = (
            nn.Linear(spectral_embedding_dim, hidden_size) if use_spectral_features else None
        )

        self.encoder = nn.LSTM(
            input_size=effective_num_factors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention_pool = AttentionPool(hidden_size)

        # Embeds the previous step's output back into the decoder's input space.
        if self.is_classification:
            self.value_embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.value_embedding = nn.Linear(1, hidden_size)
        self.start_token = nn.Parameter(torch.zeros(hidden_size))

        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size * 2, output_size)
        self.log_variance_layer = nn.Linear(hidden_size * 2, 1) if self.predict_uncertainty else None

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.cross_asset_proj is not None:
            k = self.num_cross_asset_factors
            base, xret = x[..., :-k], x[..., -k:]
            x = torch.cat([base, self.cross_asset_proj(xret)], dim=-1)
        encoder_outputs, (h_n, c_n) = self.encoder(x)
        if self.spectral is not None:
            spectral_key = self.spectral_key_proj(self.spectral(x)).unsqueeze(1)  # (batch, 1, hidden_size)
            encoder_outputs = torch.cat([encoder_outputs, spectral_key], dim=1)  # one extra attendable key
        return encoder_outputs, h_n[-1], c_n[-1]

    def forward(
        self,
        x: Tensor,
        targets: Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        return_uncertainty: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            x: normalized multifactor input sequence, shape (batch, seq_len, num_factors).
            targets: ground truth for teacher forcing during training, shape
                (batch, forecast_horizon) — class indices if classification,
                continuous values if regression. Ignored in eval mode or when
                None (the model then always feeds back its own prediction).
            teacher_forcing_ratio: probability of feeding the true previous
                step's value (rather than the model's own prediction) at each
                decoding step beyond the first. Only applies while training
                and when `targets` is given.

        Returns:
            Per-step outputs, shape (batch, forecast_horizon, output_size) if
            `forecast_horizon > 1`, else (batch, output_size).
        """
        batch_size = x.size(0)
        encoder_outputs, dec_h, dec_c = self.encode(x)
        prev_embed = self.start_token.unsqueeze(0).expand(batch_size, -1)

        step_outputs = []
        step_log_variances = []
        for t in range(self.forecast_horizon):
            context, _ = self.attention_pool(dec_h, encoder_outputs)
            decoder_input = torch.cat([prev_embed, context], dim=-1)
            dec_h, dec_c = self.decoder_cell(decoder_input, (dec_h, dec_c))

            output_features = torch.cat([self.dropout(dec_h), context], dim=-1)
            step_out = self.output_layer(output_features)
            step_outputs.append(step_out)
            if self.log_variance_layer is not None:
                # Clamping prevents numerical overflow in the Gaussian NLL while
                # still allowing the model to express a wide range of uncertainty.
                step_log_variances.append(self.log_variance_layer(output_features).clamp(-8.0, 5.0))

            use_teacher_forcing = (
                targets is not None
                and self.training
                and torch.rand(()).item() < teacher_forcing_ratio
            )
            if self.is_classification:
                next_value = targets[:, t] if use_teacher_forcing else step_out.argmax(dim=-1)
                prev_embed = self.value_embedding(next_value)
            else:
                next_value = targets[:, t] if use_teacher_forcing else step_out.squeeze(-1)
                prev_embed = self.value_embedding(next_value.unsqueeze(-1))

        result = torch.stack(step_outputs, dim=1)  # (batch, forecast_horizon, output_size)
        result = result.squeeze(1) if self.forecast_horizon == 1 else result
        if not return_uncertainty:
            return result
        if not step_log_variances:
            raise ValueError("return_uncertainty=True requires predict_uncertainty=True regression model")
        log_variance = torch.stack(step_log_variances, dim=1)
        log_variance = log_variance.squeeze(1) if self.forecast_horizon == 1 else log_variance
        return result, log_variance
