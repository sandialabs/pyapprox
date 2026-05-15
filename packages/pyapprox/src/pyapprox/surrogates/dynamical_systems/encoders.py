"""State-space encoders for dynamical systems learning.

Provides IdentityEncoder and LinearEncoder for mapping between full
and reduced state spaces.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class IdentityEncoder(Generic[Array]):
    """Identity encoder: encode(x) = x, decode(z) = z.

    Used when no dimensionality reduction is needed.

    Parameters
    ----------
    nstates : int
        Number of state variables.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, nstates: int, bkd: Backend[Array]):
        validate_backend(bkd)
        self._nstates = nstates
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def full_dim(self) -> int:
        return self._nstates

    def latent_dim(self) -> int:
        return self._nstates

    def encode(self, states: Array) -> Array:
        """Identity encoding. Returns states unchanged.

        Parameters
        ----------
        states : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nstates, nsamples)
        """
        return states

    def decode(self, latents: Array) -> Array:
        """Identity decoding. Returns latents unchanged.

        Parameters
        ----------
        latents : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nstates, nsamples)
        """
        return latents

    def encode_jacobian(self) -> Array:
        """Return identity matrix.

        Returns
        -------
        Array
            Shape: (nstates, nstates)
        """
        return self._bkd.eye(self._nstates)


class LinearEncoder(Generic[Array]):
    """Linear encoder via a user-supplied projection matrix.

    encode(x) = P @ x, decode(z) = P^+ @ z (pseudoinverse).

    Parameters
    ----------
    projection_matrix : Array
        Projection matrix P. Shape: (latent_dim, full_dim).
        Rows should be orthonormal for best numerical behavior.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, projection_matrix: Array, bkd: Backend[Array]):
        validate_backend(bkd)
        if projection_matrix.ndim != 2:
            raise ValueError(
                f"projection_matrix must be 2D, got {projection_matrix.ndim}D"
            )
        self._P = projection_matrix
        self._bkd = bkd
        self._latent_dim = projection_matrix.shape[0]
        self._full_dim = projection_matrix.shape[1]
        self._P_pinv = bkd.pinv(projection_matrix)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def full_dim(self) -> int:
        return self._full_dim

    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, states: Array) -> Array:
        """Project states to latent space: P @ states.

        Parameters
        ----------
        states : Array
            Full states. Shape: (full_dim, nsamples)

        Returns
        -------
        Array
            Latent states. Shape: (latent_dim, nsamples)
        """
        return self._P @ states

    def decode(self, latents: Array) -> Array:
        """Reconstruct from latent space: P^+ @ latents.

        Parameters
        ----------
        latents : Array
            Latent states. Shape: (latent_dim, nsamples)

        Returns
        -------
        Array
            Reconstructed states. Shape: (full_dim, nsamples)
        """
        return self._P_pinv @ latents

    def encode_jacobian(self) -> Array:
        """Return the projection matrix P (constant Jacobian).

        Returns
        -------
        Array
            Shape: (latent_dim, full_dim)
        """
        return self._P
