"""
Input affine transformation for Gaussian Process regression.

This module provides a protocol and concrete implementations for
affine input transformations: z_scaled = (z - shift) / scale.

Used to normalize GP training inputs for numerical stability while
providing predictions and statistics in the original input space.

Design Principle: Use IdentityInputTransform to avoid conditionals.
If no transform is provided, use IdentityInputTransform so all code
paths use the transform unconditionally.
"""

from typing import Generic, Protocol, runtime_checkable
from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class InputAffineTransformProtocol(Protocol[Array]):
    """Protocol for affine input transformations.

    An affine transformation maps between original and scaled spaces:
        z_scaled = (z - shift) / scale
        z_original = scale * z_scaled + shift

    Methods
    -------
    scale() -> Array
        Per-dimension scale factor (sigma_z), shape (nvars,).
    shift() -> Array
        Per-dimension shift (mu_z), shape (nvars,).
    nvars() -> int
        Number of input dimensions.
    transform(z) -> Array
        Map from original to scaled space.
    inverse_transform(z_scaled) -> Array
        Map from scaled to original space.
    jacobian_factor() -> Array
        Returns 1/scale for scaling first derivatives, shape (nvars,).
    hessian_factor() -> Array
        Returns outer product of 1/scale for second derivatives,
        shape (nvars, nvars).
    """

    def scale(self) -> Array:
        """Return per-dimension scale factor, shape (nvars,)."""
        ...

    def shift(self) -> Array:
        """Return per-dimension shift, shape (nvars,)."""
        ...

    def nvars(self) -> int:
        """Return number of input dimensions."""
        ...

    def transform(self, z: Array) -> Array:
        """Transform from original to scaled space.

        Parameters
        ----------
        z : Array
            Original-space values, shape (nvars, nsamples).

        Returns
        -------
        Array
            Scaled values, shape (nvars, nsamples).
        """
        ...

    def inverse_transform(self, z_scaled: Array) -> Array:
        """Transform from scaled to original space.

        Parameters
        ----------
        z_scaled : Array
            Scaled values, shape (nvars, nsamples).

        Returns
        -------
        Array
            Original-space values, shape (nvars, nsamples).
        """
        ...

    def jacobian_factor(self) -> Array:
        """Return 1/scale for scaling first derivatives, shape (nvars,)."""
        ...

    def hessian_factor(self) -> Array:
        """Return outer product of 1/scale, shape (nvars, nvars)."""
        ...


class IdentityInputTransform(Generic[Array]):
    """Identity (no-op) input transform.

    Useful when code expects a transform object but no scaling is needed.

    Parameters
    ----------
    nvars : int
        Number of input dimensions.
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(self, nvars: int, bkd: Backend[Array]) -> None:
        self._nvars = nvars
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def scale(self) -> Array:
        """Return ones, shape (nvars,)."""
        return self._bkd.ones(self._nvars)

    def shift(self) -> Array:
        """Return zeros, shape (nvars,)."""
        return self._bkd.zeros(self._nvars)

    def nvars(self) -> int:
        """Return number of input dimensions."""
        return self._nvars

    def transform(self, z: Array) -> Array:
        """Return z unchanged."""
        return z

    def inverse_transform(self, z_scaled: Array) -> Array:
        """Return z_scaled unchanged."""
        return z_scaled

    def jacobian_factor(self) -> Array:
        """Return ones, shape (nvars,)."""
        return self._bkd.ones(self._nvars)

    def hessian_factor(self) -> Array:
        """Return ones matrix, shape (nvars, nvars)."""
        return self._bkd.ones((self._nvars, self._nvars))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IdentityInputTransform(nvars={self._nvars})"


class InputStandardScaler(Generic[Array]):
    """Standardize inputs: z_scaled = (z - mean) / std.

    Parameters
    ----------
    mean : Array
        Per-dimension mean, shape (nvars,).
    std : Array
        Per-dimension standard deviation, shape (nvars,).
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(
        self, mean: Array, std: Array, bkd: Backend[Array]
    ) -> None:
        self._mean = mean
        self._std = std
        self._bkd = bkd
        self._nvars = mean.shape[0]

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def scale(self) -> Array:
        """Return std, shape (nvars,)."""
        return self._std

    def shift(self) -> Array:
        """Return mean, shape (nvars,)."""
        return self._mean

    def nvars(self) -> int:
        """Return number of input dimensions."""
        return self._nvars

    def transform(self, z: Array) -> Array:
        """Transform from original to scaled: (z - mean) / std.

        Parameters
        ----------
        z : Array
            Original-space values, shape (nvars, nsamples).

        Returns
        -------
        Array
            Scaled values, shape (nvars, nsamples).
        """
        return (z - self._mean[:, None]) / self._std[:, None]

    def inverse_transform(self, z_scaled: Array) -> Array:
        """Transform from scaled to original: std * z_scaled + mean.

        Parameters
        ----------
        z_scaled : Array
            Scaled values, shape (nvars, nsamples).

        Returns
        -------
        Array
            Original-space values, shape (nvars, nsamples).
        """
        return self._std[:, None] * z_scaled + self._mean[:, None]

    def jacobian_factor(self) -> Array:
        """Return 1/std for each dimension, shape (nvars,)."""
        return 1.0 / self._std

    def hessian_factor(self) -> Array:
        """Return outer product of 1/std, shape (nvars, nvars)."""
        inv_std = 1.0 / self._std
        return self._bkd.outer(inv_std, inv_std)

    @staticmethod
    def from_data(
        z: Array, bkd: Backend[Array]
    ) -> "InputStandardScaler[Array]":
        """Create scaler from training data.

        Computes per-dimension mean and standard deviation. Dimensions
        with zero variance (constant) get std=1.0 to avoid division by zero.

        Parameters
        ----------
        z : Array
            Training inputs, shape (nvars, nsamples).
        bkd : Backend[Array]
            Backend for numerical operations.

        Returns
        -------
        InputStandardScaler[Array]
            Fitted scaler.
        """
        mean = bkd.mean(z, axis=1)  # (nvars,)
        std = bkd.std(z, axis=1)  # (nvars,)
        std = bkd.where(std == 0.0, bkd.ones_like(std), std)
        return InputStandardScaler(mean, std, bkd)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"InputStandardScaler(mean={self._mean}, std={self._std})"
        )


class InputBoundsScaler(Generic[Array]):
    """Scale inputs from [lb, ub] to a target range.

    Parameters
    ----------
    lower_bounds : Array
        Per-dimension lower bounds, shape (nvars,).
    upper_bounds : Array
        Per-dimension upper bounds, shape (nvars,).
    bkd : Backend[Array]
        Backend for numerical operations.
    target_range : tuple[float, float]
        Target range for scaled values. Default is (0.0, 1.0).
    """

    def __init__(
        self,
        lower_bounds: Array,
        upper_bounds: Array,
        bkd: Backend[Array],
        target_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._bkd = bkd
        self._nvars = lower_bounds.shape[0]
        self._target_min, self._target_max = target_range

        # Compute scale and shift for: z_scaled = (z - shift) / scale
        # Maps [lb, ub] -> [target_min, target_max]
        data_range = upper_bounds - lower_bounds
        target_range_size = self._target_max - self._target_min

        self._scale = data_range / target_range_size
        self._shift = lower_bounds - self._target_min * self._scale

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def scale(self) -> Array:
        """Return per-dimension scale factor, shape (nvars,)."""
        return self._scale

    def shift(self) -> Array:
        """Return per-dimension shift, shape (nvars,)."""
        return self._shift

    def nvars(self) -> int:
        """Return number of input dimensions."""
        return self._nvars

    def transform(self, z: Array) -> Array:
        """Transform from original to scaled space.

        Parameters
        ----------
        z : Array
            Original-space values, shape (nvars, nsamples).

        Returns
        -------
        Array
            Scaled values, shape (nvars, nsamples).
        """
        return (z - self._shift[:, None]) / self._scale[:, None]

    def inverse_transform(self, z_scaled: Array) -> Array:
        """Transform from scaled to original space.

        Parameters
        ----------
        z_scaled : Array
            Scaled values, shape (nvars, nsamples).

        Returns
        -------
        Array
            Original-space values, shape (nvars, nsamples).
        """
        return self._scale[:, None] * z_scaled + self._shift[:, None]

    def jacobian_factor(self) -> Array:
        """Return 1/scale for each dimension, shape (nvars,)."""
        return 1.0 / self._scale

    def hessian_factor(self) -> Array:
        """Return outer product of 1/scale, shape (nvars, nvars)."""
        inv_scale = 1.0 / self._scale
        return self._bkd.outer(inv_scale, inv_scale)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"InputBoundsScaler(scale={self._scale}, shift={self._shift})"
        )
