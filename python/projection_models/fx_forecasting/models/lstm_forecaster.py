"""Causal LSTM + attention-pooling head for FX forecasting, task-agnostic.

The encoder consumes a normalized, multifactor input sequence (e.g. several FX
pairs' standardized returns) and is causal by construction: at step ``t`` it
has only seen inputs up to ``t``. Attention pooling then compresses the full
sequence of encoder states into a single context vector by learning which
timesteps matter most, and a linear head projects that context directly into
``output_size`` raw values — no autoregressive decoding, no output
activation. The caller decides what those values mean: per-class logits for
``nn.CrossEntropyLoss`` (classification, ``output_size=num_classes``), or a
single continuous value for ``nn.MSELoss``-style regression
(``output_size=1``) — see ``main_lstm.py``'s ``Config.target_mode``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AttentionPool(nn.Module):
    """Additive attention pooling: compresses a sequence of states into one context vector.

    Unlike Bahdanau attention (queried by a decoder's hidden state at each
    step), the query here is a single learned parameter, since there's no
    decoder — the whole sequence is summarized once.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            states: encoder hidden states, shape (batch, seq_len, hidden_size).

        Returns:
            context: attention-weighted summary, shape (batch, hidden_size).
            weights: attention weights over the sequence, shape (batch, seq_len).
        """
        keys_proj = self.key_proj(states)
        scores = self.energy_proj(torch.tanh(keys_proj + self.query)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return context, weights


class LSTMAttentionForecaster(nn.Module):
    """Causal LSTM + attention-pooling head, producing `output_size` raw values per sample.

    Input: ``x`` of shape (batch, seq_len, num_factors) — normalized time series.
    Output: raw values, shape (batch, output_size) — no activation applied,
    since the right nonlinearity (softmax for classification, none for
    regression) depends on the loss/target the caller has chosen.

    Pipeline: LSTM encoder -> per-step states (batch, seq_len, hidden_size)
    -> attention pooling -> context (batch, hidden_size) -> Linear ->
    (batch, output_size).
    """

    def __init__(
        self,
        num_factors: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_size < 1:
            raise ValueError("output_size must be >= 1")

        self.num_factors = num_factors
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=num_factors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention_pool = AttentionPool(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: normalized multifactor input sequence, shape (batch, seq_len, num_factors).

        Returns:
            raw output values, shape (batch, output_size).
        """
        states, _ = self.encoder(x)  # (batch, seq_len, hidden_size)
        context, _ = self.attention_pool(states)  # (batch, hidden_size)
        return self.output_layer(self.dropout(context))
