"""Corner Peak Genz function for quadrature benchmarks.

f(x) = (1 + c^T @ x)^{-(D+1)}

Implements FunctionWithJacobianAndHVPProtocol.
"""

from typing import Generic, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend


class CornerPeakFunction(Generic[Array]):
    """Corner Peak Genz function.

    f(x) = (1 + c^T @ x)^{-(D+1)}

    This function has a peak at the corner (0, 0, ..., 0) and decays
    toward the opposite corner.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    c : Sequence[float]
        Coefficients controlling decay rate. Shape: (nvars,).

    References
    ----------
    Genz, A. (1984). "Testing multidimensional integration routines."
    """

    def __init__(
        self,
        bkd: Backend[Array],
        c: Sequence[float],
    ) -> None:
        if len(c) == 0:
            raise ValueError("c must have at least one element")
        self._bkd = bkd
        self._c = bkd.array(c)[:, None]  # (nvars, 1)
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
        # s = 1 + c^T @ x
        s = 1.0 + self._c.T @ samples  # (1, nsamples)
        # f = s^{-(D+1)}
        return s ** (-(self._nvars + 1))

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
        s = 1.0 + bkd.sum(self._c * sample)
        # df/dx_i = -(D+1) * c_i * s^{-(D+2)}
        jac = -(self._nvars + 1) * self._c[:, 0] * s ** (-(self._nvars + 2))
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
        D = self._nvars
        s = 1.0 + bkd.sum(self._c * sample)
        # H = (D+1)(D+2) * s^{-(D+3)} * c @ c^T
        # H @ v = (D+1)(D+2) * s^{-(D+3)} * c * (c . v)
        c_dot_v = bkd.sum(self._c * vec)
        coef = (D + 1) * (D + 2) * s ** (-(D + 3))
        return coef * c_dot_v * self._c

    def integrate(self) -> float:
        """Compute analytical integral over [0, 1]^nvars.

        Returns
        -------
        float
            The analytical integral value.

        Raises
        ------
        ValueError
            If coefficients are too small for accurate computation.
        """
        bkd = self._bkd
        c_prod = bkd.prod(self._c)
        if float(c_prod) < 1e-14:
            raise ValueError(
                "coefficients too small for corner_peak integral to be "
                "computed accurately with recursion"
            )
        return float(self._corner_peak_integrate_recursive(0.0, self._nvars))

    def _corner_peak_integrate_recursive(
        self, integral: float, D: int
    ) -> Array:
        """Recursively compute the integral."""
        bkd = self._bkd
        if D == 0:
            return 1.0 / (1.0 + integral)
        c_val = self._c[D - 1, 0]
        return (
            1.0
            / (D * c_val)
            * (
                self._corner_peak_integrate_recursive(integral, D - 1)
                - self._corner_peak_integrate_recursive(
                    integral + c_val, D - 1
                )
            )
        )
