"""A minimal LSTM classifier for single-symbol FX direction forecasting.

Unlike `LSTMAttentionForecaster` (pooled across symbols, attention over
every encoder timestep), this is deliberately the simplest sensible LSTM
classifier: one symbol's own feature history in, the LSTM's own final
hidden state taken directly as the sequence summary (no attention pooling),
one linear layer to per-class logits. See `simple_lstm.py` for the training
script and `fx_forecasting.features.compute_target_direction` for the
3-class (bearish/neutral/bullish) target this is meant to predict.
"""

from __future__ import annotations

from torch import Tensor, nn


class SimpleLSTMClassifier(nn.Module):
    """LSTM encoder -> final hidden state -> linear -> per-class logits.

    Input: `x` of shape (batch, seq_len, num_factors) — one symbol's own
    normalized feature history.
    Output: raw per-class logits, shape (batch, num_classes) — no output
    activation, since `nn.CrossEntropyLoss` applies `log_softmax` internally.
    """

    def __init__(
        self,
        num_factors: int,
        num_classes: int = 3,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.num_factors = num_factors
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_factors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: normalized multifactor input sequence, shape (batch, seq_len, num_factors).

        Returns:
            per-class logits, shape (batch, num_classes).
        """
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # final layer's hidden state at the last timestep, (batch, hidden_size)
        return self.output_layer(self.dropout(last_hidden))
