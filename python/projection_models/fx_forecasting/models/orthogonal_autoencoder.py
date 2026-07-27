"""A linear, orthogonally-constrained autoencoder for cross-asset return factor extraction —
the real-PCA-equivalent way to compress K correlated FX pairs' returns into D orthogonal
factors, trained by gradient descent rather than eigendecomposition of a covariance matrix.

Baldi & Hornik (1989, "Neural networks and principal component analysis") show a linear
autoencoder trained to minimize reconstruction MSE recovers the same subspace PCA does. On
its own that only guarantees the right *subspace*, not that the individual learned
dimensions are mutually orthogonal (any invertible transform of the code composed with the
inverse in the decoder leaves reconstruction loss unchanged). Constraining the encoder's
weight matrix to have orthonormal rows (`torch.nn.utils.parametrizations.orthogonal` —
Lezcano-Casado & Martínez-Rubio, 2019, "Cheap Orthogonal Constraints in Neural Networks")
closes that gap: the constraint is structural (enforced by the parametrization on every
forward pass), not a soft penalty to balance against the reconstruction loss.

Pretrained by `pretrain_autoencoder.py`; only `.encoder`'s weights are loaded into
`LSTMAttentionForecaster.cross_asset_encoder` afterwards — the decoder here exists solely to
define this module's own reconstruction training objective.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn.functional import linear
from torch.nn.utils.parametrizations import orthogonal


class OrthogonalAutoencoder(nn.Module):
    """`encoder`: K -> D linear map with orthonormal rows. `forward` reconstructs K from D
    via the *tied* transpose of that same map (no separate decoder parameters) — since
    `encoder.weight` (shape D x K) has orthonormal rows, `x_hat = z @ W = x @ (W^T W)` is
    exactly the orthogonal projection of `x` onto the D-dim row-space of `W`, the same
    reconstruction PCA itself performs when keeping only its top-D components.
    """

    def __init__(self, num_cross_asset_factors: int, factor_dim: int) -> None:
        super().__init__()
        if not 0 < factor_dim < num_cross_asset_factors:
            raise ValueError(
                f"factor_dim ({factor_dim}) must be strictly between 0 and "
                f"num_cross_asset_factors ({num_cross_asset_factors})"
            )
        self.num_cross_asset_factors = num_cross_asset_factors
        self.factor_dim = factor_dim
        self.encoder = nn.Linear(num_cross_asset_factors, factor_dim, bias=False)
        orthogonal(self.encoder, name="weight")

    def encode(self, x: Tensor) -> Tensor:
        """x: (..., num_cross_asset_factors) -> (..., factor_dim)."""
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruction round-trip, for the pretraining objective only: (..., K) -> (..., K)."""
        z = self.encode(x)
        return linear(z, self.encoder.weight.t())

    def orthogonality_error(self) -> float:
        """`||W W^T - I||_F`, purely diagnostic — should already be ~0 by construction (the
        parametrization enforces it structurally), useful only as a sanity check after
        loading a checkpoint saved under a different PyTorch/parametrization version."""
        w = self.encoder.weight
        eye = w.new_zeros(w.shape[0], w.shape[0])
        eye.fill_diagonal_(1.0)
        return (w @ w.t() - eye).norm().item()
