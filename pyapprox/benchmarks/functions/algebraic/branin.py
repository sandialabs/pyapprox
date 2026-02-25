"""Branin (Branin-Hoo) function for optimization benchmarks.

The Branin function is a classic benchmark for optimization algorithms,
featuring three global minima in a 2D search space.

Implements FunctionWithJacobianAndHVPProtocol directly (no inheritance).
"""

import math
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class BraninFunction(Generic[Array]):
    """Branin function: f(x) = a*(x2 - b*x1^2 + c*x1 - r)^2 + s*(1-t)*cos(x1) + s.

    Standard parameters:
        a = 1
        b = 5.1 / (4 * pi^2)
        c = 5 / pi
        r = 6
        s = 10
        t = 1 / (8 * pi)

    Standard domain: x1 in [-5, 10], x2 in [0, 15]
    Global minimum: f(x*) ≈ 0.397887
    Three global minimizers:
        (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)

    This function implements FunctionWithJacobianAndHVPProtocol.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    References
    ----------
    Branin, F.H. (1972). "Widely convergent method of finding multiple
    solutions of simultaneous nonlinear equations."
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        # Standard parameters
        self._a = 1.0
        self._b = 5.1 / (4 * math.pi**2)
        self._c = 5.0 / math.pi
        self._r = 6.0
        self._s = 10.0
        self._t = 1.0 / (8 * math.pi)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 2

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function at multiple samples.

        Parameters
        ----------
        samples : Array
            Input samples of shape (2, nsamples).

        Returns
        -------
        Array
            Function values of shape (1, nsamples).
        """
        x1 = samples[0:1, :]
        x2 = samples[1:2, :]
        bkd = self._bkd

        term = x2 - self._b * x1**2 + self._c * x1 - self._r
        result = self._a * term**2 + self._s * (1 - self._t) * bkd.cos(x1) + self._s
        return result

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (2, 1).

        Returns
        -------
        Array
            Jacobian matrix of shape (1, 2).

        Raises
        ------
        ValueError
            If sample is not 2D with shape (2, 1).
        """
        if sample.ndim != 2 or sample.shape != (2, 1):
            raise ValueError(f"sample must have shape (2, 1), got {sample.shape}")

        x1 = sample[0, 0]
        x2 = sample[1, 0]
        bkd = self._bkd

        # term = x2 - b*x1^2 + c*x1 - r
        term = x2 - self._b * x1**2 + self._c * x1 - self._r

        # df/dx1 = 2*a*term*(-2*b*x1 + c) - s*(1-t)*sin(x1)
        dterm_dx1 = -2 * self._b * x1 + self._c
        df_dx1 = 2 * self._a * term * dterm_dx1 - self._s * (1 - self._t) * bkd.sin(x1)

        # df/dx2 = 2*a*term
        df_dx2 = 2 * self._a * term

        return bkd.reshape(bkd.stack([df_dx1, df_dx2]), (1, 2))

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (2, 1).
        vec : Array
            Direction vector of shape (2, 1).

        Returns
        -------
        Array
            Hessian-vector product of shape (2, 1).

        Raises
        ------
        ValueError
            If sample or vec is not 2D with shape (2, 1).
        """
        if sample.ndim != 2 or sample.shape != (2, 1):
            raise ValueError(f"sample must have shape (2, 1), got {sample.shape}")
        if vec.ndim != 2 or vec.shape != (2, 1):
            raise ValueError(f"vec must have shape (2, 1), got {vec.shape}")

        x1 = sample[0, 0]
        x2 = sample[1, 0]
        v1 = vec[0, 0]
        v2 = vec[1, 0]
        bkd = self._bkd

        # term = x2 - b*x1^2 + c*x1 - r
        term = x2 - self._b * x1**2 + self._c * x1 - self._r
        dterm_dx1 = -2 * self._b * x1 + self._c

        # Second derivatives:
        # H11 = d^2f/dx1^2 = 2*a*(dterm_dx1)^2 + 2*a*term*(-2*b) - s*(1-t)*cos(x1)
        H11 = (
            2 * self._a * dterm_dx1**2
            + 2 * self._a * term * (-2 * self._b)
            - self._s * (1 - self._t) * bkd.cos(x1)
        )

        # H12 = H21 = d^2f/dx1dx2 = 2*a*dterm_dx1 * 1 = 2*a*(-2*b*x1 + c)
        H12 = 2 * self._a * dterm_dx1

        # H22 = d^2f/dx2^2 = 2*a
        H22 = 2 * self._a

        # H @ v
        hvp1 = H11 * v1 + H12 * v2
        hvp2 = H12 * v1 + H22 * v2

        return bkd.reshape(bkd.stack([hvp1, hvp2]), (2, 1))


# Global minimum value
BRANIN_GLOBAL_MINIMUM = 0.397887357729739

# Three global minimizers
BRANIN_MINIMIZERS = [
    (-math.pi, 12.275),
    (math.pi, 2.275),
    (9.42478, 2.475),
]
