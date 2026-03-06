"""
Gumbel copula.

The Gumbel copula with parameter theta >= 1 has CDF:
    C(u1, u2) = exp(-A^{1/theta})
where A = (-log u1)^theta + (-log u2)^theta.

The parameter is stored as log(theta - 1) so that
theta = 1 + exp(stored) >= 1. The h-function inverse uses
bisection since no closed-form exists.
"""

import math
from typing import Generic, cast

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)


class GumbelCopula(Generic[Array]):
    """
    Gumbel copula with parameter theta >= 1.

    Exhibits upper tail dependence: lambda_U = 2 - 2^{1/theta}.
    theta = 1 corresponds to independence.

    Parameters
    ----------
    theta : float
        Dependence parameter, must be >= 1.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, theta: float, bkd: Backend[Array]):
        if theta < 1.0:
            raise ValueError(f"theta must be >= 1, got {theta}")
        self._bkd = bkd
        # Store log(theta - 1) so theta = 1 + exp(stored) >= 1
        theta_minus_1 = theta - 1.0
        log_tm1 = math.log(theta_minus_1) if theta_minus_1 > 0 else -30.0
        self._hyp = HyperParameter(
            name="log_theta_minus_1",
            nparams=1,
            values=bkd.asarray([log_tm1]),
            bounds=(-50.0, 10.0),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._hyp])

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
        """Recover theta = 1 + exp(stored log(theta-1))."""
        return 1.0 + self._bkd.exp(self._hyp.get_values()[0])

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

        log c(u1,u2) = log C(u1,u2) - log(u1) - log(u2)
                       + (theta-1)*(log(-log u1) + log(-log u2))
                       + (2/theta - 2)*log(A)
                       + log(1 + (theta-1)*A^{-1/theta})
                       - A^{1/theta}   [absorbed in log C]

        Using the standard formula:
        c = C/(u1*u2) * A^{2/theta-2} * ((-log u1)(-log u2))^{theta-1}
            * (1 + (theta-1)*A^{-1/theta})

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

        neg_log_u1 = -self._bkd.log(u1)
        neg_log_u2 = -self._bkd.log(u2)

        a1 = neg_log_u1**theta
        a2 = neg_log_u2**theta
        A = a1 + a2

        A_inv_theta = A ** (1.0 / theta)

        # log C = -A^{1/theta}
        log_C = -A_inv_theta

        # Full log density
        result = (
            log_C
            - self._bkd.log(u1)
            - self._bkd.log(u2)
            + (theta - 1.0) * (self._bkd.log(neg_log_u1) + self._bkd.log(neg_log_u2))
            + (2.0 / theta - 2.0) * self._bkd.log(A)
            + self._bkd.log(1.0 + (theta - 1.0) * A ** (-1.0 / theta))
        )

        return result

    def h_function(self, u1: Array, u2: Array) -> Array:
        """
        Conditional CDF: h(u1|u2) = dC/du2.

        h(u1|u2) = C(u1,u2) / u2 * (-log u2)^{theta-1} * A^{1/theta - 1}

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

        neg_log_u1 = -self._bkd.log(u1_c)
        neg_log_u2 = -self._bkd.log(u2_c)

        a1 = neg_log_u1**theta
        a2 = neg_log_u2**theta
        A = a1 + a2

        # C = exp(-A^{1/theta})
        C = self._bkd.exp(-(A ** (1.0 / theta)))

        result = C / u2_c * neg_log_u2 ** (theta - 1.0) * A ** (1.0 / theta - 1.0)
        return self._bkd.clip(result, 1e-10, 1.0 - 1e-10)

    def h_inverse(self, v: Array, u2: Array) -> Array:
        """
        Inverse of h-function via bisection.

        Finds u1 such that h(u1, u2) = v.

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

        # Bisection: find u1 in (0, 1) such that h(u1, u2) - v = 0
        lb = self._bkd.full(v_c.shape, 1e-10)
        ub = self._bkd.full(v_c.shape, 1.0 - 1e-10)

        for _ in range(60):
            mid = (lb + ub) / 2.0
            h_mid = self.h_function(mid, u2_c)
            # Where h(mid) > v, u1 should be smaller (h is increasing in u1)
            too_high = cast(Array, h_mid > v_c)
            lb = self._bkd.where(too_high, lb, mid)
            ub = self._bkd.where(too_high, mid, ub)

        return (lb + ub) / 2.0

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
        # TODO: do not use astype, this will break if we want to use float32
        # use 0., 1. so floats are created and let abckend do correct conversion
        w = self._bkd.asarray(np.random.uniform(0, 1, (2, nsamples)).astype(np.float64))
        u1 = w[0:1, :]
        w2 = w[1:2, :]
        u2 = self.h_inverse(w2, u1)
        return self._bkd.concatenate([u1, u2], axis=0)

    def kendall_tau(self) -> Array:
        """
        Compute Kendall's tau = 1 - 1/theta.

        Returns
        -------
        Array
            Kendall's tau as a scalar Array.
        """
        theta = self._theta()
        return 1.0 - 1.0 / theta

    def __repr__(self) -> str:
        """Return string representation."""
        theta_val = self._bkd.to_float(self._theta())
        return f"GumbelCopula(theta={theta_val:.4f})"
