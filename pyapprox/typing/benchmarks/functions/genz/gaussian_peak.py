"""Gaussian Peak Genz function for quadrature benchmarks.

f(x) = exp(-sum_i c_i^2 * (x_i - w_i)^2)

Implements FunctionWithJacobianAndHVPProtocol.
"""

import math
from typing import Generic, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend


class GaussianPeakFunction(Generic[Array]):
    """Gaussian Peak Genz function.

    f(x) = exp(-sum_i c_i^2 * (x_i - w_i)^2)

    This function is a product of Gaussians centered at w.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    c : Sequence[float]
        Coefficients controlling the width of the Gaussian in each
        dimension. Larger c means narrower peak. Shape: (nvars,).
    w : Sequence[float]
        Peak center location. Shape: (nvars,).

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
        t = samples - self._w  # (nvars, nsamples)
        # -sum_i c_i^2 * t_i^2
        exponent = -bkd.sum(self._c**2 * t**2, axis=0, keepdims=True)
        return bkd.exp(exponent)

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
        t = sample - self._w  # (nvars, 1)
        exponent = -bkd.sum(self._c**2 * t**2)
        f = bkd.exp(exponent)
        # df/dx_i = f * (-2 * c_i^2 * t_i)
        jac = f * (-2.0 * self._c**2 * t)
        return jac.T  # (1, nvars)

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
        t = sample - self._w  # (nvars, 1)
        c2 = self._c**2  # (nvars, 1)
        exponent = -bkd.sum(c2 * t**2)
        f = bkd.exp(exponent)

        # h_i = -2 * c_i^2 * t_i
        h = -2.0 * c2 * t  # (nvars, 1)

        # dh_i/dx_i = -2 * c_i^2
        # (H @ v)_i = f * (dh_i/dx_i * v_i + h_i * (h . v))
        #           = f * (-2 * c_i^2 * v_i + h_i * (h . v))
        h_dot_v = bkd.sum(h * vec)
        hvp = f * (-2.0 * c2 * vec + h * h_dot_v)
        return hvp

    def integrate(self) -> float:
        """Compute analytical integral over [0, 1]^nvars.

        Uses the error function (erf) for the analytical solution.

        Returns
        -------
        float
            The analytical integral value.
        """
        bkd = self._bkd
        c = self._c[:, 0]
        w = self._w[:, 0]
        # Integral = prod_i sqrt(pi)/(2*c_i) * (erf(c_i*w_i) + erf(c_i*(1-w_i)))
        # Using scipy.special.erf via numpy conversion
        c_np = bkd.to_numpy(c)
        w_np = bkd.to_numpy(w)
        import scipy.special

        erf_terms = scipy.special.erf(c_np * w_np) + scipy.special.erf(
            c_np * (1.0 - w_np)
        )
        integral = float(
            (math.sqrt(math.pi) / (2.0 * c_np) * erf_terms).prod()
        )
        return integral
