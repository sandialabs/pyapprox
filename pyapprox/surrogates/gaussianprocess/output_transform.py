"""
Output affine transformation for Gaussian Process regression.

This module provides a protocol and concrete implementations for
affine output transformations: y_original = scale * y_scaled + shift.

Used to normalize GP training outputs for numerical stability while
providing predictions and statistics in the original output space.
"""

from typing import Generic, Optional, Protocol, runtime_checkable
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class OutputAffineTransformProtocol(Protocol[Array]):
    """Protocol for affine output transformations.

    An affine transformation maps between scaled (internal) and original
    (user) spaces:
        y_original = scale * y_scaled + shift
        y_scaled = (y_original - shift) / scale

    Methods
    -------
    scale() -> Array
        Multiplicative factor, shape (nqoi,).
    shift() -> Array
        Additive offset, shape (nqoi,).
    transform(y) -> Array
        Map from scaled to original space.
    inverse_transform(y) -> Array
        Map from original to scaled space.
    """

    def scale(self) -> Array:
        """Return the multiplicative scale factor, shape (nqoi,)."""
        ...

    def shift(self) -> Array:
        """Return the additive shift, shape (nqoi,)."""
        ...

    def transform(self, y: Array) -> Array:
        """Transform from scaled to original space.

        Parameters
        ----------
        y : Array
            Scaled values, shape (nqoi, n_samples).

        Returns
        -------
        Array
            Original-space values, shape (nqoi, n_samples).
        """
        ...

    def inverse_transform(self, y: Array) -> Array:
        """Transform from original to scaled space.

        Parameters
        ----------
        y : Array
            Original-space values, shape (nqoi, n_samples).

        Returns
        -------
        Array
            Scaled values, shape (nqoi, n_samples).
        """
        ...


class OutputStandardScaler(Generic[Array]):
    """Standard scaling: y_scaled = (y - mean) / std.

    Parameters
    ----------
    mean : Array
        Per-output mean, shape (nqoi,).
    std : Array
        Per-output standard deviation, shape (nqoi,).
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(
        self, mean: Array, std: Array, bkd: Backend[Array]
    ) -> None:
        self._mean = mean
        self._std = std
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def scale(self) -> Array:
        """Return std, shape (nqoi,)."""
        return self._std

    def shift(self) -> Array:
        """Return mean, shape (nqoi,)."""
        return self._mean

    def transform(self, y: Array) -> Array:
        """Transform from scaled to original: std * y + mean.

        Parameters
        ----------
        y : Array
            Scaled values, shape (nqoi, n_samples).

        Returns
        -------
        Array
            Original-space values, shape (nqoi, n_samples).
        """
        return self._std[:, None] * y + self._mean[:, None]

    def inverse_transform(self, y: Array) -> Array:
        """Transform from original to scaled: (y - mean) / std.

        Parameters
        ----------
        y : Array
            Original-space values, shape (nqoi, n_samples).

        Returns
        -------
        Array
            Scaled values, shape (nqoi, n_samples).
        """
        return (y - self._mean[:, None]) / self._std[:, None]

    @staticmethod
    def from_data(
        y: Array, bkd: Backend[Array]
    ) -> "OutputStandardScaler[Array]":
        """Create scaler from training data.

        Computes per-output mean and standard deviation. Outputs with
        zero variance (constant) get std=1.0 to avoid division by zero.

        Parameters
        ----------
        y : Array
            Training outputs, shape (nqoi, n_train).
        bkd : Backend[Array]
            Backend for numerical operations.

        Returns
        -------
        OutputStandardScaler[Array]
            Fitted scaler.
        """
        mean = bkd.mean(y, axis=1)  # (nqoi,)
        std = bkd.std(y, axis=1)  # (nqoi,)
        std = bkd.where(std == 0.0, bkd.ones_like(std), std)
        return OutputStandardScaler(mean, std, bkd)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OutputStandardScaler(mean={self._mean}, std={self._std})"
        )


class IdentityOutputTransform(Generic[Array]):
    """Identity (no-op) output transform.

    Useful when code expects a transform object but no scaling is needed.

    Parameters
    ----------
    nqoi : int
        Number of output dimensions.
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(self, nqoi: int, bkd: Backend[Array]) -> None:
        self._nqoi = nqoi
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def scale(self) -> Array:
        """Return ones, shape (nqoi,)."""
        return self._bkd.ones(self._nqoi)

    def shift(self) -> Array:
        """Return zeros, shape (nqoi,)."""
        return self._bkd.zeros(self._nqoi)

    def transform(self, y: Array) -> Array:
        """Return y unchanged."""
        return y

    def inverse_transform(self, y: Array) -> Array:
        """Return y unchanged."""
        return y

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IdentityOutputTransform(nqoi={self._nqoi})"
