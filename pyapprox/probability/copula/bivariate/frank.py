"""
Frank copula.

The Frank copula with parameter theta != 0 has CDF:
    C(u1, u2) = -1/theta * log(1 + (e^{-theta u1}-1)(e^{-theta u2}-1)
                / (e^{-theta}-1))

The parameter theta can be any nonzero real number (positive for
positive dependence, negative for negative dependence). It is stored
directly as a HyperParameter with wide bounds.
"""

from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)

# TODO: Fix typing issues

# Gauss-Legendre quadrature for Debye function
# Use np to avoid cicular imports with pyapprox gauss quadrature module
_GL_NODES_32, _GL_WEIGHTS_32 = np.polynomial.legendre.leggauss(32)


class FrankCopula(Generic[Array]):
    """
    Frank copula with parameter theta != 0.

    No tail dependence (symmetric copula).

    Parameters
    ----------
    theta : float
        Dependence parameter, must be != 0.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, theta: float, bkd: Backend[Array]):
        if theta == 0.0:
            raise ValueError("theta must be != 0 for Frank copula")
        self._bkd = bkd
        self._hyp = HyperParameter(
            name="theta",
            nparams=1,
            values=bkd.asarray([theta]),
            bounds=(-50.0, 50.0),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._hyp])
        # Precompute GL nodes/weights for Debye function
        # TODO: do not use as type, this will break if we want to use float32
        # let backend do correct conversion
        self._gl_nodes = bkd.asarray(_GL_NODES_32.astype(np.float64))
        self._gl_weights = bkd.asarray(_GL_WEIGHTS_32.astype(np.float64))

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of free parameters."""
        return 1

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list for optimization."""
        return self._hyp_list

    def _theta(self) -> Array:
        """Get the copula parameter."""
        return self._hyp.get_values()[0]

    def _validate_input(self, u: Array) -> None:
        """Validate that input is 2D with shape (2, nsamples)."""
        if u.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (2, nsamples), got {u.ndim}D"
            )
        if u.shape[0] != 2:
            raise ValueError(f"Expected 2 variables, got {u.shape[0]}")

    def _validate_h_input(self, u1: Array, u2: Array) -> None:
        """Validate h-function inputs."""
        if u1.ndim != 2 or u1.shape[0] != 1:
            raise ValueError(f"u1 must have shape (1, nsamples), got {u1.shape}")
        if u2.ndim != 2 or u2.shape[0] != 1:
            raise ValueError(f"u2 must have shape (1, nsamples), got {u2.shape}")

    def logpdf(self, u: Array) -> Array:
        """
        Evaluate the log copula density.

        c(u1,u2) = -theta*(e^{-theta}-1)*e^{-theta*(u1+u2)}
                   / ((e^{-theta}-1) + (e^{-theta*u1}-1)(e^{-theta*u2}-1))^2

        Parameters
        ----------
        u : Array
            Points in (0,1)^2. Shape: (2, nsamples)

        Returns
        -------
        Array
            Log copula density values. Shape: (1, nsamples)
        """
        # TODO: Do we need clip it degrades derivatives
        self._validate_input(u)
        u_c = self._bkd.clip(u, 1e-10, 1.0 - 1e-10)
        u1 = u_c[0:1, :]
        u2 = u_c[1:2, :]

        theta = self._theta()
        e1 = self._bkd.exp(-theta * u1)
        e2 = self._bkd.exp(-theta * u2)
        et = self._bkd.exp(-theta)

        numer = -theta * (et - 1.0) * e1 * e2
        denom = (et - 1.0) + (e1 - 1.0) * (e2 - 1.0)

        return self._bkd.log(self._bkd.abs(numer)) - 2.0 * self._bkd.log(
            self._bkd.abs(denom)
        )

    def h_function(self, u1: Array, u2: Array) -> Array:
        """
        Conditional CDF: h(u1|u2) = dC/du2.

        h(u1|u2) = e^{-theta u2} (e^{-theta u1} - 1)
                   / ((e^{-theta} - 1) + (e^{-theta u1} - 1)(e^{-theta u2} - 1))

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
        # TODO: Do we need clip it degrades derivatives
        self._validate_h_input(u1, u2)
        u1_c = self._bkd.clip(u1, 1e-10, 1.0 - 1e-10)
        u2_c = self._bkd.clip(u2, 1e-10, 1.0 - 1e-10)

        theta = self._theta()
        e1 = self._bkd.exp(-theta * u1_c)
        e2 = self._bkd.exp(-theta * u2_c)
        et = self._bkd.exp(-theta)

        numer = e2 * (e1 - 1.0)
        denom = (et - 1.0) + (e1 - 1.0) * (e2 - 1.0)

        result = numer / denom
        return self._bkd.clip(result, 1e-10, 1.0 - 1e-10)

    def h_inverse(self, v: Array, u2: Array) -> Array:
        """
        Inverse of h-function (analytical closed form).

        u1 = -1/theta * log(1 + v*(e^{-theta} - 1)
                            / (e^{-theta*u2}*(1-v) + v))

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
        # TODO: Do we need clip it degrades derivatives
        self._validate_h_input(v, u2)
        v_c = self._bkd.clip(v, 1e-10, 1.0 - 1e-10)
        u2_c = self._bkd.clip(u2, 1e-10, 1.0 - 1e-10)

        theta = self._theta()
        et = self._bkd.exp(-theta)
        e2 = self._bkd.exp(-theta * u2_c)

        a = 1.0 + v_c * (et - 1.0) / (e2 * (1.0 - v_c) + v_c)
        a = self._bkd.clip(a, 1e-30, None)

        result = -1.0 / theta * self._bkd.log(a)
        return self._bkd.clip(result, 1e-10, 1.0 - 1e-10)

    def sample(self, nsamples: int) -> Array:
        """
        Draw samples from the copula via h-function inverse.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Samples in (0,1)^2. Shape: (2, nsamples)
        """
        # TODO: do not use as type, this will break if we want to use float32
        # use 0., 1. so floats are created and let abckend do correct conversion
        w = self._bkd.asarray(np.random.uniform(0, 1, (2, nsamples)).astype(np.float64))
        u1 = w[0:1, :]
        w2 = w[1:2, :]
        u2 = self.h_inverse(w2, u1)
        return self._bkd.concatenate([u1, u2], axis=0)

    def kendall_tau(self) -> Array:
        """
        Compute Kendall's tau = 1 - 4/theta * (1 - D_1(theta)).

        D_1(theta) is the first Debye function:
            D_1(theta) = (1/theta) * integral_0^theta t/(e^t - 1) dt

        Computed via Gauss-Legendre quadrature.

        Returns
        -------
        Array
            Kendall's tau as a scalar Array.
        """
        theta = self._theta()

        # Map GL nodes from [-1, 1] to [0, theta] (signed)
        # For negative theta this integrates from 0 to theta < 0
        half_t = theta / 2.0
        t = half_t * (self._gl_nodes + 1.0)  # (32,)

        # D_1 integrand: t / (e^t - 1)
        # Near t=0: limit is 1. For |t| < 1e-10 use Taylor: t/(e^t-1) ~ 1 - t/2
        # Away from 0, evaluate directly
        exp_t = self._bkd.exp(t)
        # Avoid division by zero at t=0 by adding small epsilon to denominator
        integrand = t / (exp_t - 1.0 + 1e-30)

        # Weighted sum: integral of t/(e^t-1) from 0 to theta
        integral = self._bkd.sum(self._gl_weights * integrand) * half_t
        D1 = integral / theta

        return 1.0 - 4.0 / theta * (1.0 - D1)

    def __repr__(self) -> str:
        """Return string representation."""
        theta_val = self._bkd.to_float(self._theta())
        return f"FrankCopula(theta={theta_val:.4f})"
