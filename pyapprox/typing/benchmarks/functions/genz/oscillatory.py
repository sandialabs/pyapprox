"""Oscillatory Genz function for quadrature benchmarks.

f(x) = cos(2*pi*w[0] + c^T @ x)

Implements FunctionWithJacobianAndHVPProtocol.
"""

import math
from typing import Generic, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend


class OscillatoryFunction(Generic[Array]):
    """Oscillatory Genz function: f(x) = cos(2*pi*w[0] + c^T @ x).

    This function is used for testing integration algorithms. It has an
    oscillatory behavior controlled by the coefficients c.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    c : Sequence[float]
        Coefficients controlling oscillation in each dimension.
        Shape: (nvars,).
    w : Sequence[float]
        Weights controlling phase shift. Only w[0] is used.
        Shape: (nvars,).

    References
    ----------
    Genz, A. (1984). "Testing multidimensional integration routines."
    """

    def __init__(
        self,
        bkd: Backend[Array],
        c: Sequence[float],
        w: Sequence[float],
    ) -> None:
        if len(c) == 0:
            raise ValueError("c must have at least one element")
        if len(w) != len(c):
            raise ValueError("c and w must have the same length")
        self._bkd = bkd
        self._c = bkd.array(c)[:, None]  # (nvars, 1)
        self._w = bkd.array(w)[:, None]  # (nvars, 1)
        self._nvars = len(c)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function at multiple samples.

        Parameters
        ----------
        samples : Array
            Input samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Function values of shape (1, nsamples).
        """
        bkd = self._bkd
        # arg = 2*pi*w[0] + c^T @ x
        arg = 2.0 * math.pi * self._w[0, 0] + (self._c.T @ samples)
        return bkd.cos(arg)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix of shape (1, nvars).

        Raises
        ------
        ValueError
            If sample is not 2D with shape (nvars, 1).
        """
        if sample.ndim != 2 or sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample must have shape ({self._nvars}, 1), got {sample.shape}"
            )
        bkd = self._bkd
        arg = 2.0 * math.pi * self._w[0, 0] + bkd.sum(self._c * sample)
        # df/dx_i = -c_i * sin(arg)
        jac = -self._c[:, 0] * bkd.sin(arg)
        return bkd.reshape(jac, (1, self._nvars))

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1).

        Raises
        ------
        ValueError
            If sample or vec is not 2D with shape (nvars, 1).
        """
        if sample.ndim != 2 or sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample must have shape ({self._nvars}, 1), got {sample.shape}"
            )
        if vec.ndim != 2 or vec.shape != (self._nvars, 1):
            raise ValueError(
                f"vec must have shape ({self._nvars}, 1), got {vec.shape}"
            )
        bkd = self._bkd
        arg = 2.0 * math.pi * self._w[0, 0] + bkd.sum(self._c * sample)
        # H = -cos(arg) * c @ c^T
        # H @ v = -cos(arg) * c * (c^T @ v) = -cos(arg) * (c . v) * c
        c_dot_v = bkd.sum(self._c * vec)
        return -bkd.cos(arg) * c_dot_v * self._c

    def integrate(self) -> float:
        """Compute analytical integral over [0, 1]^nvars.

        Returns
        -------
        float
            The analytical integral value.
        """
        bkd = self._bkd
        return float(self._oscillatory_recursive_integrate(0, True))

    def _oscillatory_recursive_integrate(
        self, var_id: int, cosine: bool
    ) -> Array:
        """Recursively compute the integral.

        Uses the recurrence relation for integrating cosines.
        """
        bkd = self._bkd
        c_val = self._c[var_id, 0]
        C1 = bkd.sin(c_val) / c_val
        C2 = (1 - bkd.cos(c_val)) / c_val

        if var_id == self._nvars - 1:
            if cosine:
                return C1
            return C2

        if cosine:
            return (
                C1 * self._oscillatory_recursive_integrate(var_id + 1, True)
                - C2 * self._oscillatory_recursive_integrate(var_id + 1, False)
            )
        return (
            C2 * self._oscillatory_recursive_integrate(var_id + 1, True)
            + C1 * self._oscillatory_recursive_integrate(var_id + 1, False)
        )

    def _compute_integral(self) -> Array:
        """Compute the integral using the recursive formula."""
        bkd = self._bkd
        C1 = bkd.cos(2.0 * math.pi * self._w[0, 0])
        C2 = bkd.sin(2.0 * math.pi * self._w[0, 0])
        return (
            C1 * self._oscillatory_recursive_integrate(0, True)
            - C2 * self._oscillatory_recursive_integrate(0, False)
        )
