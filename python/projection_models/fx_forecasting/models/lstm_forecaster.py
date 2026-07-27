"""Causal LSTM encoder-decoder with attention, forecasting N steps ahead recursively.

The encoder consumes a *pretrained, orthogonally-reduced* cross-asset return factor
sequence (see `cross_asset_encoder` below) and is causal by construction: at step ``t`` it
has only seen inputs up to ``t``. The decoder then unrolls autoregressively for
``forecast_horizon`` steps — the classic sequence-to-sequence-with-attention design
(Sutskever et al., 2014, "Sequence to Sequence Learning with Neural Networks"; Bahdanau et
al., 2015, "Neural Machine Translation by Jointly Learning to Align and Translate"), adapted
here for continuous or per-class-logit regression instead of token generation. At every
decoding step the model attends over *every* encoder timestep — queried by the decoder's own
current hidden state, so the attention weights are recomputed fresh at each step, not fixed —
then feeds its own prediction from step t back in as input for step t+1 (with optional
teacher forcing during training), the standard way to extend a single-step model into a
genuine multi-step forecaster.

Everything that is *not* part of the clean return-factor sequence — momentum, realized
vol/skew/kurtosis, carry, CMA crossovers, intraday volatility, and the
optional spectral (FFT) summary of the factor sequence itself — bypasses the recurrence
entirely and is injected only at the final prediction layer (Lim, Arık, Loeff & Pfister,
2021, "Temporal Fusion Transformers": non-sequential covariates are best given directly to
the output stage rather than forced through the same recurrent bottleneck as the genuine
time-varying signal).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fx_forecasting.models.orthogonal_autoencoder import OrthogonalAutoencoder
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
    steps ahead recursively, over a pretrained-and-reduced cross-asset factor sequence.

    Input: ``x`` of shape (batch, seq_len, num_factors), three column blocks in this exact
    order (`main_lstm.py::build_target_dataset`'s convention): leading target-symbol-only
    side features (momentum, rvol, skew, kurt, carry, CMA, intraday volatility), then
    `num_target_series` columns of the target symbol's own (normalized) log return sequence,
    then the *trailing* `num_cross_asset_factors` (K) columns of per-pair `xret_*`
    cross-asset log returns. Output: raw per-step values, shape
    (batch, forecast_horizon, output_size) when ``forecast_horizon > 1``, or
    (batch, output_size) when it's 1 — no output activation, since the right nonlinearity
    (softmax for classification, none for regression) depends on the loss/target the caller
    has chosen.

    Feedback at each decoding step depends on what `output_size` represents:
    a per-step class ID (embedded via `nn.Embedding`) when `output_size > 1`
    (classification, logits over classes), or the raw scalar value (via a
    linear projection) when `output_size == 1` (continuous regression) — the
    standard way to feed a discrete vs. continuous decoder output back in as
    the next step's input.

    `cross_asset_encoder`: the trailing `num_cross_asset_factors` (K) columns of `x` are
    projected to `factor_dim` (D < K) orthogonal factors by an `OrthogonalAutoencoder`'s
    encoder (see `fx_forecasting.models.orthogonal_autoencoder`) — pretrained by
    `pretrain_autoencoder.py`, its weights loaded and optionally frozen by
    `main_lstm.build_model`, *not* trained fresh here. Since `factor_dim < num_cross_asset_factors`,
    this compression can dilute the target symbol's own idiosyncratic dynamics into the
    shared cross-asset structure; `num_target_series` columns of the target's own (normalized)
    log return sequence are concatenated onto the D factors, uncompressed, before the LSTM
    ever sees either — giving the recurrence privileged, undiluted access to the series it's
    actually forecasting, on top of the shared factor structure. `factor_norm` LayerNorms the
    combined (D + num_target_series)-dim sequence before the LSTM, since the orthogonal map
    redistributes variance across the D outputs unevenly depending on the data's own
    covariance structure and could otherwise saturate the LSTM's gates.

    `use_spectral_features=True` additionally runs the same combined sequence through a
    `SpectralEmbedding` (a small learned projection of its own FFT magnitude spectrum) —
    like every other side feature, this is concatenated directly into the final prediction
    layer's input, not fed back into the recurrence or attention.

    `num_side_features`: width of the leading (non-sequential) columns of `x`; only their
    *last* timestep (the forecast origin's own current side-feature values) is used, also
    injected only at the final prediction layer.
    """

    def __init__(
        self,
        num_factors: int,
        output_size: int,
        num_cross_asset_factors: int,
        factor_dim: int,
        num_side_features: int = 0,
        num_target_series: int = 0,
        forecast_horizon: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_uncertainty: bool = False,
        use_spectral_features: bool = False,
        spectral_embedding_dim: int = 8,
        spectral_freq_bins: int = 16,
    ) -> None:
        super().__init__()
        if output_size < 1:
            raise ValueError("output_size must be >= 1")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")
        if num_cross_asset_factors <= 0:
            raise ValueError("num_cross_asset_factors must be > 0")
        if num_target_series < 0:
            raise ValueError("num_target_series must be >= 0")
        if num_factors < num_cross_asset_factors + num_target_series + num_side_features:
            raise ValueError(
                f"num_factors ({num_factors}) must be >= num_cross_asset_factors "
                f"({num_cross_asset_factors}) + num_target_series ({num_target_series}) "
                f"+ num_side_features ({num_side_features})"
            )

        self.num_factors = num_factors
        self.output_size = output_size
        self.num_cross_asset_factors = num_cross_asset_factors
        self.factor_dim = factor_dim
        self.num_side_features = num_side_features
        self.num_target_series = num_target_series
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_classification = output_size > 1
        self.predict_uncertainty = predict_uncertainty and not self.is_classification
        self.use_spectral_features = use_spectral_features

        # Pretrained (see pretrain_autoencoder.py), orthogonally-constrained K -> D reduction
        # of the trailing K cross-asset xret_* columns — weights are loaded (and, unless
        # fine-tuning is requested, frozen) by main_lstm.build_model after construction; this
        # class only owns the architecture, never the pretrained state itself.
        self.cross_asset_encoder = OrthogonalAutoencoder(num_cross_asset_factors, factor_dim).encoder
        lstm_input_size = factor_dim + num_target_series
        self.factor_norm = nn.LayerNorm(lstm_input_size)

        self.encoder = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention_pool = AttentionPool(hidden_size)

        # Operates on the same factor sequence the encoder does (not the raw side features),
        # and — unlike the earlier "extra attention key" design — feeds the final prediction
        # layer directly (see pre_output_norm/output_layer below), never the attention/LSTM.
        self.spectral = (
            SpectralEmbedding(lstm_input_size, spectral_freq_bins, spectral_embedding_dim)
            if use_spectral_features
            else None
        )

        # Embeds the previous step's output back into the decoder's input space.
        if self.is_classification:
            self.value_embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.value_embedding = nn.Linear(1, hidden_size)
        self.start_token = nn.Parameter(torch.zeros(hidden_size))

        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Final feed-forward head: decoder hidden state + attention context (both width
        # hidden_size) plus every non-sequential covariate — spectral summary (if enabled)
        # and the side features at the forecast origin — LayerNorm'd together immediately
        # before this last linear layer so no one block's raw scale dominates the others
        # purely because of its natural units, not its actual predictive relevance.
        pre_output_size = (
            hidden_size * 2
            + (spectral_embedding_dim if use_spectral_features else 0)
            + num_side_features
        )
        self.pre_output_norm = nn.LayerNorm(pre_output_size)
        self.output_layer = nn.Linear(pre_output_size, output_size)
        self.log_variance_layer = nn.Linear(pre_output_size, 1) if self.predict_uncertainty else None

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Returns `(encoder_outputs, dec_h0, dec_c0, spectral_vec_or_None, side_features_at_t)`.

        `side_features_at_t` is the forecast origin's own (last-timestep) side-feature
        vector — a static, per-window covariate, not a sequence — reused unchanged at every
        decoding step in `forward`.
        """
        k = self.num_cross_asset_factors
        t = self.num_target_series
        base = x[..., : -(k + t)] if t > 0 else x[..., :-k]
        xret = x[..., -k:]
        factors = self.cross_asset_encoder(xret)
        if t > 0:
            target_series = x[..., -(k + t) : -k]
            lstm_input = self.factor_norm(torch.cat([factors, target_series], dim=-1))
        else:
            lstm_input = self.factor_norm(factors)
        encoder_outputs, (h_n, c_n) = self.encoder(lstm_input)
        spectral_vec = self.spectral(lstm_input) if self.spectral is not None else None
        side_features_at_t = base[:, -1, :]
        return encoder_outputs, h_n[-1], c_n[-1], spectral_vec, side_features_at_t

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
        encoder_outputs, dec_h, dec_c, spectral_vec, side_features_at_t = self.encode(x)
        prev_embed = self.start_token.unsqueeze(0).expand(batch_size, -1)

        step_outputs = []
        step_log_variances = []
        for t in range(self.forecast_horizon):
            context, _ = self.attention_pool(dec_h, encoder_outputs)
            decoder_input = torch.cat([prev_embed, context], dim=-1)
            dec_h, dec_c = self.decoder_cell(decoder_input, (dec_h, dec_c))

            pre_output = [self.dropout(dec_h), context]
            if spectral_vec is not None:
                pre_output.append(spectral_vec)
            pre_output.append(side_features_at_t)
            output_features = self.pre_output_norm(torch.cat(pre_output, dim=-1))

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
