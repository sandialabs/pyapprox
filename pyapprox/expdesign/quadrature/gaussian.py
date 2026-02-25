"""
Gaussian quadrature sampler for OED.

This module provides a wrapper around the surrogates quadrature infrastructure
to create tensor product Gauss-Hermite quadrature for standard normal distributions.
"""

from typing import Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.univariate import (
    HermitePolynomial1D,
    GaussQuadratureRule,
)
from pyapprox.surrogates.quadrature import TensorProductQuadratureRule


class GaussianQuadratureSampler(Generic[Array]):
    """
    Tensor product Gaussian quadrature sampler for standard normal.

    Wraps the surrogates quadrature infrastructure to provide Gauss-Hermite
    quadrature with a sampler interface suitable for OED.

    Implements QuadratureSamplerProtocol.

    Parameters
    ----------
    nvars : int
        Number of random variables.
    bkd : Backend[Array]
        Computational backend.
    npoints_1d : int, optional
        Number of quadrature points per dimension. Default is 5.

    Notes
    -----
    The total number of quadrature points is `npoints_1d ** nvars`, which
    grows exponentially with dimension. This sampler is most appropriate
    for low-dimensional problems (nvars <= 5).

    The quadrature exactly integrates polynomials of degree up to
    `2 * npoints_1d - 1` against the standard normal density.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> sampler = GaussianQuadratureSampler(2, bkd, npoints_1d=5)
    >>> samples, weights = sampler.sample(0)  # nsamples is ignored
    >>> samples.shape  # (2, 25)
    >>> weights.shape  # (25,)
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        npoints_1d: int = 5,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._npoints_1d = npoints_1d
        self._samples: Optional[Array] = None
        self._weights: Optional[Array] = None
        self._setup_quadrature()

    def _setup_quadrature(self) -> None:
        """Set up tensor product Gauss-Hermite quadrature using surrogates."""
        # Create Hermite polynomial for standard normal (rho=0, prob_meas=True)
        hermite = HermitePolynomial1D(self._bkd, rho=0.0, prob_meas=True)
        hermite.set_nterms(self._npoints_1d)

        # Create Gauss quadrature rule
        gauss_rule = GaussQuadratureRule(hermite)

        # Create univariate rules for all dimensions
        univariate_rules = [gauss_rule] * self._nvars
        npoints_1d_list = [self._npoints_1d] * self._nvars

        # Build tensor product quadrature
        tp_rule = TensorProductQuadratureRule(
            self._bkd,
            univariate_rules,
            npoints_1d_list,
        )

        # Get samples and weights
        self._samples, self._weights = tp_rule()

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """
        Return the number of random variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self._nvars

    def reset(self) -> None:
        """Reset does nothing for deterministic quadrature."""
        pass

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Return Gaussian quadrature samples.

        Note: nsamples is ignored; returns all quadrature points.

        Parameters
        ----------
        nsamples : int
            Ignored. Returns all npoints_1d^nvars points.

        Returns
        -------
        samples : Array
            Quadrature points. Shape: (nvars, npoints_total)
        weights : Array
            Quadrature weights. Shape: (npoints_total,)
        """
        if self._samples is None:
            self._setup_quadrature()
        return self._samples, self._weights

    def npoints(self) -> int:
        """
        Return the total number of quadrature points.

        Returns
        -------
        int
            Total number of points (npoints_1d ** nvars).
        """
        return self._npoints_1d ** self._nvars
