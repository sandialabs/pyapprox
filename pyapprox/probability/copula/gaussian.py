"""
Gaussian copula.

The Gaussian copula density is:
    c(u) = |Sigma|^{-1/2} * exp(-0.5 * z^T (Sigma^{-1} - I) z)
where z_i = Phi^{-1}(u_i) and Sigma is the correlation matrix.

All normal CDF/inverse CDF operations use erf/erfinv backend methods
for autograd compatibility (no scipy).
"""

import math
from typing import Generic

import numpy as np

from pyapprox.probability.copula.correlation.protocols import (
    CorrelationParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList

_SQRT2 = math.sqrt(2.0)


class GaussianCopula(Generic[Array]):
    """
    Gaussian copula parameterized by a pluggable correlation strategy.

    Parameters
    ----------
    correlation_param : CorrelationParameterizationProtocol[Array]
        Strategy for parameterizing and computing with the correlation
        matrix.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    TypeError
        If correlation_param does not satisfy
        CorrelationParameterizationProtocol.
    """

    def __init__(
        self,
        correlation_param: CorrelationParameterizationProtocol[Array],
        bkd: Backend[Array],
    ):
        if not isinstance(correlation_param, CorrelationParameterizationProtocol):
            raise TypeError(
                "correlation_param must satisfy "
                "CorrelationParameterizationProtocol, "
                f"got {type(correlation_param).__name__}"
            )
        self._corr_param = correlation_param
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the dimension of the copula."""
        return self._corr_param.nvars()

    def nparams(self) -> int:
        """Return the number of free parameters."""
        return self._corr_param.nparams()

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list for optimization."""
        return self._corr_param.hyp_list()

    def correlation_param(self) -> CorrelationParameterizationProtocol[Array]:
        """Return the correlation parameterization object."""
        return self._corr_param

    def _validate_input(self, u: Array) -> None:
        """Validate that input is 2D with correct shape."""
        if u.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nvars, nsamples), got {u.ndim}D"
            )
        if u.shape[0] != self.nvars():
            raise ValueError(f"Expected {self.nvars()} variables, got {u.shape[0]}")

    def _standard_normal_invcdf(self, u: Array) -> Array:
        """
        Compute Phi^{-1}(u) using erfinv (autograd-safe).

        Parameters
        ----------
        u : Array
            Values in (0, 1). Any shape.

        Returns
        -------
        Array
            Standard normal quantiles. Same shape as u.
        """
        return _SQRT2 * self._bkd.erfinv(2.0 * u - 1.0)

    def _standard_normal_cdf(self, z: Array) -> Array:
        """
        Compute Phi(z) using erf (autograd-safe).

        Parameters
        ----------
        z : Array
            Standard normal values. Any shape.

        Returns
        -------
        Array
            CDF values in (0, 1). Same shape as z.
        """
        return 0.5 * (1.0 + self._bkd.erf(z / _SQRT2))

    def logpdf(self, u: Array) -> Array:
        """
        Evaluate the log copula density.

        log c(u) = -0.5 * log|Sigma| - 0.5 * z^T (Sigma^{-1} - I) z

        Parameters
        ----------
        u : Array
            Points in (0,1)^d. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log copula density values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        self._validate_input(u)
        # Clip to avoid infinities at boundaries
        u_clipped = self._bkd.clip(u, 1e-10, 1.0 - 1e-10)
        # Transform to standard normal
        z = self._standard_normal_invcdf(u_clipped)
        # Compute log copula density
        log_det = self._corr_param.log_det()
        quad = self._corr_param.quad_form(z)
        result = -0.5 * log_det - 0.5 * quad
        return self._bkd.reshape(result, (1, -1))

    def sample(self, nsamples: int) -> Array:
        """
        Draw samples from the copula.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Samples in (0,1)^d. Shape: (nvars, nsamples)
        """
        eps = self._bkd.asarray(
            np.random.normal(0, 1, (self.nvars(), nsamples)).astype(np.float64)
        )
        z = self._corr_param.sample_transform(eps)
        return self._standard_normal_cdf(z)

    def kl_divergence(self, other: "GaussianCopula[Array]") -> Array:
        """
        Compute KL(self || other) for two Gaussian copulas.

        Reduces to KL between two zero-mean multivariate normals:
            KL = 0.5 * (tr(Sigma_q^{-1} Sigma_p) - d
                        + log|Sigma_q| - log|Sigma_p|)

        Parameters
        ----------
        other : GaussianCopula
            The other Gaussian copula.

        Returns
        -------
        Array
            KL divergence value (scalar Array, preserves autograd).
        """
        sigma_p = self._corr_param.correlation_matrix()
        sigma_q = other.correlation_param().correlation_matrix()
        sigma_q_inv = self._bkd.inv(sigma_q)

        d = self.nvars()
        trace_term = self._bkd.sum(sigma_q_inv * sigma_p)
        log_det_q = other.correlation_param().log_det()
        log_det_p = self._corr_param.log_det()

        return 0.5 * (trace_term - d + log_det_q - log_det_p)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianCopula(nvars={self.nvars()}, nparams={self.nparams()})"
