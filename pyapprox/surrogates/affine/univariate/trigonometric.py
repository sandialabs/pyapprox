"""Trigonometric polynomial basis on bounded domains.

Provides basis functions [1, cos(kx), sin(kx)] on a domain mapped
to the canonical interval [-pi, pi].
"""

import math
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class TrigonometricPolynomial1D(Generic[Array]):
    r"""1D trigonometric polynomial basis on [a, b] mapped to [-pi, pi].

    Basis functions:
        [1, cos(x), cos(2x), ..., cos(Kx), sin(x), sin(2x), ..., sin(Kx)]

    where nterms = 2K + 1 (must be odd).

    Satisfies Basis1DProtocol.

    Parameters
    ----------
    bounds : Array
        Domain bounds [a, b].
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bounds: Array, bkd: Backend[Array]):
        self._bkd = bkd
        self._bounds = bounds
        self._half_indices = None
        # Affine transform: maps [a, b] -> [-pi, pi]
        self._loc = (bounds[0] + bounds[1]) / 2.0
        self._scale = (bounds[1] - bounds[0]) / (2.0 * math.pi)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms.

        Parameters
        ----------
        nterms : int
            Must be odd (2K + 1).
        """
        if nterms % 2 != 1:
            raise ValueError("nterms must be an odd number")
        K = (nterms - 1) // 2
        self._half_indices = self._bkd.arange(1, K + 1)[None, :]

    def nterms(self) -> int:
        """Return the number of basis terms."""
        if self._half_indices is None:
            return 0
        return self._half_indices.shape[1] * 2 + 1

    def _map_to_canonical(self, samples: Array) -> Array:
        """Map samples from physical domain [a, b] to canonical [-pi, pi]."""
        return (samples - self._loc) / self._scale

    def __call__(self, samples: Array) -> Array:
        """Evaluate all basis functions at given samples.

        Parameters
        ----------
        samples : Array, shape (1, nsamples)
            Sample points in the physical domain.

        Returns
        -------
        Array, shape (nsamples, nterms)
            Basis function values.
        """
        can_samples = self._map_to_canonical(samples)
        return self._bkd.hstack(
            (
                self._bkd.ones((can_samples.shape[1], 1)),
                self._bkd.cos(can_samples.T * self._half_indices),
                self._bkd.sin(can_samples.T * self._half_indices),
            )
        )

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate derivatives of all basis functions.

        Parameters
        ----------
        samples : Array, shape (1, nsamples)
            Sample points in the physical domain.

        Returns
        -------
        Array, shape (nsamples, nterms)
            Derivatives of basis functions w.r.t. the physical variable.
        """
        can_samples = self._map_to_canonical(samples)
        k = self._half_indices  # shape (1, K)
        # d/dx = (1/scale) * d/d(can)
        scale_inv = 1.0 / self._scale
        return self._bkd.hstack(
            (
                self._bkd.zeros((can_samples.shape[1], 1)),
                -k * self._bkd.sin(can_samples.T * k) * scale_inv,
                k * self._bkd.cos(can_samples.T * k) * scale_inv,
            )
        )

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of all basis functions.

        Parameters
        ----------
        samples : Array, shape (1, nsamples)
            Sample points in the physical domain.

        Returns
        -------
        Array, shape (nsamples, nterms)
            Second derivatives of basis functions w.r.t. the physical variable.
        """
        can_samples = self._map_to_canonical(samples)
        k = self._half_indices  # shape (1, K)
        scale_inv_sq = 1.0 / self._scale**2
        return self._bkd.hstack(
            (
                self._bkd.zeros((can_samples.shape[1], 1)),
                -(k**2) * self._bkd.cos(can_samples.T * k) * scale_inv_sq,
                -(k**2) * self._bkd.sin(can_samples.T * k) * scale_inv_sq,
            )
        )
