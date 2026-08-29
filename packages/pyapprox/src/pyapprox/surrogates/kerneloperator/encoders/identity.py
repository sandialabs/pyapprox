"""Identity function encoder — latent coordinates are the samples."""

from __future__ import annotations

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class IdentityFunctionEncoder(Generic[Array]):
    """Encoder that does not reduce: ``latent_dim == full_dim``.

    Useful where an encoder is structurally required but no reduction is
    wanted, so a caller need not special-case its absence.

    Parameters
    ----------
    full_dim : int
        Dimension of the space, which is also the latent dimension.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, full_dim: int, bkd: Backend[Array]) -> None:
        self._full_dim = full_dim
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def full_dim(self) -> int:
        return self._full_dim

    def latent_dim(self) -> int:
        return self._full_dim

    def encode(self, samples: Array) -> Array:
        return samples

    def decode(self, latents: Array) -> Array:
        return latents

    def decode_std(self, std_latents: Array) -> Array:
        """Exact here: an identity decoder propagates std unchanged."""
        return std_latents
