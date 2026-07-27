"""A learned frequency-domain summary of an input window, concatenated directly into the
final prediction layer's input alongside the other non-sequential side features (see
`LSTMAttentionForecaster.encode`/`forward`) — never fed into the LSTM or attention itself.

Financial time series often carry periodic/cyclical structure (weekly seasonality,
trend-cycle shape) that a purely time-domain recurrent model can only pick up
implicitly, given enough data. Computing the window's own magnitude spectrum and
projecting it through a small learned layer makes that structure available to the
final prediction explicitly, rather than something the LSTM+attention have to infer
on their own (Lim, Arık, Loeff & Pfister, 2021, "Temporal Fusion Transformers":
non-sequential/precomputed covariates are best given directly to the output stage
rather than forced through the same recurrent bottleneck as the genuine
time-varying signal).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpectralEmbedding(nn.Module):
    """Real FFT magnitude spectrum of an input window's *cumulative sum*, pooled to a fixed
    number of frequency bins and projected to a fixed-size embedding.

    `freq_bins` (not `seq_len`) is what the embedding's input width depends on, via
    `adaptive_avg_pool1d` — so this module works unchanged across different
    `Config.seq_len` values (e.g. across an `optimize_hyperparams.py` search that
    varies `seq_len`), without needing to know the sequence length at construction
    time.

    FFT of the cumulative sum, not of `x` itself: `x`'s channels are mostly return-like
    (differenced) quantities, which are close to serially uncorrelated by construction —
    their power spectrum is close to flat/white, with little for the FFT to find. Summing
    along time first turns each channel back into the discrete analogue of a "level" path
    (return -> price, the same relationship `compute_target_zscore` exploits in reverse via
    `log_price.shift(-horizon) - log_price`): a random walk's power spectrum concentrates at
    low frequencies (~1/f^2), so trend/cyclical structure shows up far more clearly here than
    in the raw differenced series.

    Deliberately magnitude-only (not phase): the magnitude/power spectrum is the
    standard, robust descriptor of how much periodic energy sits at each frequency;
    phase is far more sensitive to exact window alignment and is left out of this
    first version.
    """

    def __init__(self, num_factors: int, freq_bins: int = 16, embedding_dim: int = 8) -> None:
        super().__init__()
        self.freq_bins = freq_bins
        self.embedding_dim = embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(num_factors * freq_bins, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: normalized multifactor input window, shape (batch, seq_len, num_factors).

        Returns:
            Spectral embedding, shape (batch, embedding_dim) — one vector per window.
        """
        levels = torch.cumsum(x, dim=1)  # return-like channels -> level/price-like paths
        spectrum = torch.fft.rfft(levels, dim=1)  # (batch, seq_len // 2 + 1, num_factors), complex
        magnitude = torch.log1p(spectrum.abs())  # log-scaled power, stable dynamic range
        magnitude = magnitude.permute(0, 2, 1)  # (batch, num_factors, freq)
        pooled = F.adaptive_avg_pool1d(magnitude, self.freq_bins)  # (batch, num_factors, freq_bins)
        return self.proj(pooled.flatten(1))
