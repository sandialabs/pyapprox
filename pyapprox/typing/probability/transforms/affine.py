"""
Affine probability transforms.

Provides affine (linear) transforms for location-scale families:
    y = (x - loc) / scale
    x = loc + scale * y
"""

from typing import Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


class AffineTransform(Generic[Array]):
    """
    Affine transform for location-scale families.

    Maps between original space and standardized (canonical) space:
        y = (x - loc) / scale  (to canonical)
        x = loc + scale * y    (from canonical)

    The Jacobian is constant: dy/dx = 1/scale, dx/dy = scale.

    Parameters
    ----------
    loc : Array
        Location parameters. Shape: (nvars,)
    scale : Array
        Scale parameters. Shape: (nvars,)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> loc = np.array([0.0, 1.0])
    >>> scale = np.array([1.0, 2.0])
    >>> transform = AffineTransform(loc, scale, bkd)
    >>> x = np.array([[0.0, 1.0], [1.0, 3.0]])
    >>> y = transform.map_to_canonical(x)  # [[0, 1], [0, 1]]
    """

    def __init__(
        self,
        loc: Array,
        scale: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._loc = loc
        self._scale = scale

        if loc.ndim == 2:
            self._loc = bkd.flatten(loc)
        if scale.ndim == 2:
            self._scale = bkd.flatten(scale)

        self._nvars = self._loc.shape[0]

        if self._scale.shape[0] != self._nvars:
            raise ValueError(
                f"loc and scale must have same length: "
                f"{self._nvars} vs {self._scale.shape[0]}"
            )

        # Pre-compute for efficiency
        self._inv_scale = 1.0 / self._scale

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def loc(self) -> Array:
        """Return location parameters."""
        return self._loc

    def scale(self) -> Array:
        """Return scale parameters."""
        return self._scale

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform samples to canonical (standardized) space.

        y = (x - loc) / scale

        Parameters
        ----------
        samples : Array
            Samples in original space. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Samples in canonical space. Shape: (nvars, nsamples)
        """
        if samples.ndim == 1:
            samples = self._bkd.reshape(samples, (self._nvars, 1))

        # Broadcast: (nvars, 1) operations
        return (samples - self._loc[:, None]) * self._inv_scale[:, None]

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform samples from canonical space to original space.

        x = loc + scale * y

        Parameters
        ----------
        canonical_samples : Array
            Samples in canonical space. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Samples in original space. Shape: (nvars, nsamples)
        """
        if canonical_samples.ndim == 1:
            canonical_samples = self._bkd.reshape(
                canonical_samples, (self._nvars, 1)
            )

        return self._loc[:, None] + self._scale[:, None] * canonical_samples

    def map_to_canonical_with_jacobian(
        self, samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform to canonical space with Jacobian diagonal.

        The Jacobian of y = (x - loc) / scale is:
            dy/dx = 1/scale (diagonal)

        Parameters
        ----------
        samples : Array
            Samples in original space. Shape: (nvars, nsamples)

        Returns
        -------
        Tuple[Array, Array]
            canonical_samples : Shape: (nvars, nsamples)
            jacobian_diag : Shape: (nvars, nsamples)
        """
        canonical = self.map_to_canonical(samples)
        nsamples = canonical.shape[1]

        # Jacobian is constant: 1/scale, broadcast to nsamples
        jacobian_diag = self._bkd.tile(
            self._inv_scale[:, None], (1, nsamples)
        )

        return canonical, jacobian_diag

    def map_from_canonical_with_jacobian(
        self, canonical_samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform from canonical space with Jacobian diagonal.

        The Jacobian of x = loc + scale * y is:
            dx/dy = scale (diagonal)

        Parameters
        ----------
        canonical_samples : Array
            Samples in canonical space. Shape: (nvars, nsamples)

        Returns
        -------
        Tuple[Array, Array]
            samples : Shape: (nvars, nsamples)
            jacobian_diag : Shape: (nvars, nsamples)
        """
        samples = self.map_from_canonical(canonical_samples)
        nsamples = samples.shape[1]

        # Jacobian is constant: scale, broadcast to nsamples
        jacobian_diag = self._bkd.tile(
            self._scale[:, None], (1, nsamples)
        )

        return samples, jacobian_diag

    def log_det_jacobian_to_canonical(self, samples: Array) -> Array:
        """
        Compute log absolute determinant of Jacobian (to canonical).

        log|det(dy/dx)| = sum_i log(1/scale_i) = -sum_i log(scale_i)

        This is constant for affine transforms.

        Parameters
        ----------
        samples : Array
            Samples (unused, included for interface consistency).

        Returns
        -------
        Array
            Log determinant. Shape: (nsamples,)
        """
        if samples.ndim == 1:
            nsamples = 1
        else:
            nsamples = samples.shape[1]

        log_det = -float(self._bkd.sum(self._bkd.log(self._scale)))
        return self._bkd.full((nsamples,), log_det)

    def log_det_jacobian_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Compute log absolute determinant of Jacobian (from canonical).

        log|det(dx/dy)| = sum_i log(scale_i)

        Parameters
        ----------
        canonical_samples : Array
            Canonical samples (unused, included for interface consistency).

        Returns
        -------
        Array
            Log determinant. Shape: (nsamples,)
        """
        if canonical_samples.ndim == 1:
            nsamples = 1
        else:
            nsamples = canonical_samples.shape[1]

        log_det = float(self._bkd.sum(self._bkd.log(self._scale)))
        return self._bkd.full((nsamples,), log_det)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AffineTransform(nvars={self._nvars})"
