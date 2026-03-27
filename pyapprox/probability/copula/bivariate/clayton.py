"""
Clayton copula.

The Clayton copula with parameter theta > 0 has CDF:
    C(u1, u2) = (u1^{-theta} + u2^{-theta} - 1)^{-1/theta}

The density is:
    c(u1, u2) = (1+theta) * (u1*u2)^{-(1+theta)}
                * (u1^{-theta} + u2^{-theta} - 1)^{-(2+1/theta)}

The parameter theta is stored as log(theta) via LogHyperParameter
for positivity-constrained optimization.
"""

from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameterList,
)
from pyapprox.util.hyperparameter.log_hyperparameter import (
    LogHyperParameter,
)


class ClaytonCopula(Generic[Array]):
    """
    Clayton copula with parameter theta > 0.

    Exhibits lower tail dependence: lambda_L = 2^{-1/theta}.

    Parameters
    ----------
    theta : float
        Dependence parameter, must be > 0.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, theta: float, bkd: Backend[Array]):
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}")
        self._bkd = bkd
        self._hyp = LogHyperParameter(
            name="theta",
            nparams=1,
            user_values=bkd.asarray([theta]),
            user_bounds=(1e-6, 100.0),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._hyp])

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of free parameters."""
        return 1

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the hyperparameter list for optimization."""
        return self._hyp_list

    def _theta(self) -> Array:
        """Recover theta = exp(stored log_theta)."""
        return self._hyp.exp_values()[0]

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

        log c = log(1+theta) - (1+theta)*(log u1 + log u2)
                - (2+1/theta)*log(u1^{-theta} + u2^{-theta} - 1)

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
        # TODO: Do we need clip it degrades derivatives
        u_c = self._bkd.clip(u, 1e-10, 1.0 - 1e-10)
        u1 = u_c[0:1, :]
        u2 = u_c[1:2, :]

        theta = self._theta()

        term1 = self._bkd.log(1.0 + theta)
        term2 = -(1.0 + theta) * (self._bkd.log(u1) + self._bkd.log(u2))
        A = u1 ** (-theta) + u2 ** (-theta) - 1.0
        A = self._bkd.clip(A, 1e-30, None)
        term3 = -(2.0 + 1.0 / theta) * self._bkd.log(A)

        return term1 + term2 + term3

    def h_function(self, u1: Array, u2: Array) -> Array:
        """
        Conditional CDF: h(u1|u2) = u2^{-(theta+1)} * A^{-(1+1/theta)}

        where A = u1^{-theta} + u2^{-theta} - 1.

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
        A = u1_c ** (-theta) + u2_c ** (-theta) - 1.0
        A = self._bkd.clip(A, 1e-30, None)

        result = u2_c ** (-(theta + 1.0)) * A ** (-(1.0 + 1.0 / theta))
        return self._bkd.clip(result, 1e-10, 1.0 - 1e-10)

    def h_inverse(self, v: Array, u2: Array) -> Array:
        """
        Inverse of h-function (analytical closed form).

        u1 = (( v^{-theta/(1+theta)} * u2^{-theta} - 1 ) * (-1) +
        u2^{-theta})^{-1/theta}

        Derived by solving h(u1, u2) = v for u1:
          v = u2^{-(theta+1)} * A^{-(1+1/theta)}
          => A = (v * u2^{theta+1})^{-theta/(theta+1)}
          => u1^{-theta} = A - u2^{-theta} + 1
          => u1 = (A - u2^{-theta} + 1)^{-1/theta}

        where A = u1^{-theta} + u2^{-theta} - 1.

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

        # A = (v * u2^{theta+1})^{-theta/(theta+1)}
        A = (v_c * u2_c ** (theta + 1.0)) ** (-theta / (theta + 1.0))

        # u1^{-theta} = A - u2^{-theta} + 1
        u1_neg_theta = A - u2_c ** (-theta) + 1.0
        u1_neg_theta = self._bkd.clip(u1_neg_theta, 1e-30, None)

        result = u1_neg_theta ** (-1.0 / theta)
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
        # TODO: do not use astype, this will break if we want to use float32
        # use 0., 1. so floats are created and let abckend do correct conversion
        w = self._bkd.asarray(np.random.uniform(0, 1, (2, nsamples)).astype(np.float64))
        u1 = w[0:1, :]
        w2 = w[1:2, :]
        u2 = self.h_inverse(w2, u1)
        return self._bkd.concatenate([u1, u2], axis=0)

    def kendall_tau(self) -> Array:
        """
        Compute Kendall's tau = theta / (theta + 2).

        Returns
        -------
        Array
            Kendall's tau as a scalar Array.
        """
        theta = self._theta()
        return theta / (theta + 2.0)

    def __repr__(self) -> str:
        """Return string representation."""
        theta_val = self._bkd.to_float(self._theta())
        return f"ClaytonCopula(theta={theta_val:.4f})"
