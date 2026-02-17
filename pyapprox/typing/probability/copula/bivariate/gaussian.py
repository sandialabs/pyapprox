"""
Bivariate Gaussian copula.

The bivariate Gaussian copula with correlation rho has density:
    c(u1, u2) = (1-rho^2)^{-1/2} exp(-(rho^2(z1^2+z2^2) - 2*rho*z1*z2)
                / (2*(1-rho^2)))
where z_i = Phi^{-1}(u_i).

The parameter is stored as arctanh(rho) for unbounded optimization.
All normal CDF/inverse CDF operations use erf/erfinv for autograd
compatibility (no scipy).
"""

import math
from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)


_SQRT2 = math.sqrt(2.0)
_TWO_OVER_PI = 2.0 / math.pi


class BivariateGaussianCopula(Generic[Array]):
    """
    Bivariate Gaussian copula parameterized by correlation rho.

    The correlation is stored internally as arctanh(rho) for
    unbounded optimization. The copula parameter rho is recovered
    as tanh(arctanh_rho).

    Parameters
    ----------
    rho : float
        Correlation parameter in (-1, 1).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, rho: float, bkd: Backend[Array]):
        self._bkd = bkd
        arctanh_rho = math.atanh(rho)
        self._hyp = HyperParameter(
            name="arctanh_rho",
            nparams=1,
            values=bkd.asarray([arctanh_rho]),
            bounds=(-10.0, 10.0),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._hyp])

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (always 2 for bivariate)."""
        return 2

    def nparams(self) -> int:
        """Return the number of free parameters."""
        return 1

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list for optimization."""
        return self._hyp_list

    def _rho(self) -> Array:
        """Recover rho = tanh(stored arctanh_rho)."""
        return self._bkd.tanh(self._hyp.get_values()[0])

    def _validate_input(self, u: Array) -> None:
        """Validate that input is 2D with shape (2, nsamples)."""
        if u.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (2, nsamples), "
                f"got {u.ndim}D"
            )
        if u.shape[0] != 2:
            raise ValueError(
                f"Expected 2 variables, got {u.shape[0]}"
            )

    def _validate_h_input(self, u1: Array, u2: Array) -> None:
        """Validate h-function inputs."""
        if u1.ndim != 2 or u1.shape[0] != 1:
            raise ValueError(
                f"u1 must have shape (1, nsamples), got {u1.shape}"
            )
        if u2.ndim != 2 or u2.shape[0] != 1:
            raise ValueError(
                f"u2 must have shape (1, nsamples), got {u2.shape}"
            )

    def _standard_normal_invcdf(self, u: Array) -> Array:
        """Compute Phi^{-1}(u) using erfinv (autograd-safe)."""
        return _SQRT2 * self._bkd.erfinv(2.0 * u - 1.0)

    def _standard_normal_cdf(self, z: Array) -> Array:
        """Compute Phi(z) using erf (autograd-safe)."""
        return 0.5 * (1.0 + self._bkd.erf(z / _SQRT2))

    def logpdf(self, u: Array) -> Array:
        """
        Evaluate the log copula density.

        log c(u1,u2) = -0.5*log(1-rho^2)
                       - (rho^2*(z1^2+z2^2) - 2*rho*z1*z2) / (2*(1-rho^2))

        Parameters
        ----------
        u : Array
            Points in (0,1)^2. Shape: (2, nsamples)

        Returns
        -------
        Array
            Log copula density values. Shape: (1, nsamples)
        """
        self._validate_input(u)
        u_clipped = self._bkd.clip(u, 1e-10, 1.0 - 1e-10)
        z = self._standard_normal_invcdf(u_clipped)
        z1 = z[0:1, :]
        z2 = z[1:2, :]

        rho = self._rho()
        rho2 = rho * rho
        one_minus_rho2 = 1.0 - rho2

        log_term = -0.5 * self._bkd.log(one_minus_rho2)
        quad_term = -(rho2 * (z1 * z1 + z2 * z2) - 2.0 * rho * z1 * z2) / (
            2.0 * one_minus_rho2
        )
        return log_term + quad_term

    def h_function(self, u1: Array, u2: Array) -> Array:
        """
        Conditional CDF: h(u1|u2) = Phi((z1 - rho*z2) / sqrt(1-rho^2)).

        Parameters
        ----------
        u1 : Array
            First variable values. Shape: (1, nsamples)
        u2 : Array
            Conditioning variable values. Shape: (1, nsamples)

        Returns
        -------
        Array
            Conditional CDF values in (0, 1). Shape: (1, nsamples)
        """
        self._validate_h_input(u1, u2)
        u1_c = self._bkd.clip(u1, 1e-10, 1.0 - 1e-10)
        u2_c = self._bkd.clip(u2, 1e-10, 1.0 - 1e-10)
        z1 = self._standard_normal_invcdf(u1_c)
        z2 = self._standard_normal_invcdf(u2_c)

        rho = self._rho()
        arg = (z1 - rho * z2) / self._bkd.sqrt(1.0 - rho * rho)
        return self._standard_normal_cdf(arg)

    def h_inverse(self, v: Array, u2: Array) -> Array:
        """
        Inverse of h-function: u1 = Phi(sqrt(1-rho^2)*Phi^{-1}(v) + rho*z2).

        Parameters
        ----------
        v : Array
            Target values in (0, 1). Shape: (1, nsamples)
        u2 : Array
            Conditioning variable values. Shape: (1, nsamples)

        Returns
        -------
        Array
            Recovered u1 values. Shape: (1, nsamples)
        """
        self._validate_h_input(v, u2)
        v_c = self._bkd.clip(v, 1e-10, 1.0 - 1e-10)
        u2_c = self._bkd.clip(u2, 1e-10, 1.0 - 1e-10)
        z_v = self._standard_normal_invcdf(v_c)
        z2 = self._standard_normal_invcdf(u2_c)

        rho = self._rho()
        z1 = self._bkd.sqrt(1.0 - rho * rho) * z_v + rho * z2
        return self._standard_normal_cdf(z1)

    def sample(self, nsamples: int) -> Array:
        """
        Draw samples from the copula.

        Uses the h-function inverse: generate (w1, w2) ~ U(0,1)
        independently, then u2 = h_inverse(w2, u1).

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Samples in (0,1)^2. Shape: (2, nsamples)
        """
        w = self._bkd.asarray(
            np.random.uniform(0, 1, (2, nsamples)).astype(np.float64)
        )
        u1 = w[0:1, :]
        w2 = w[1:2, :]
        u2 = self.h_inverse(w2, u1)
        return self._bkd.concatenate([u1, u2], axis=0)

    def kendall_tau(self) -> Array:
        """
        Compute Kendall's tau = 2/pi * arcsin(rho).

        Returns
        -------
        Array
            Kendall's tau as a scalar Array.
        """
        rho = self._rho()
        return _TWO_OVER_PI * self._bkd.arcsin(rho)

    def __repr__(self) -> str:
        """Return string representation."""
        rho_val = float(self._bkd.to_numpy(self._rho()))
        return f"BivariateGaussianCopula(rho={rho_val:.4f})"
