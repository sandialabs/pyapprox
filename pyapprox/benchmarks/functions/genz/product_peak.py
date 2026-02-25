"""Product Peak Genz function for quadrature benchmarks.

f(x) = prod_i (c_i^{-2} + (x_i - w_i)^2)^{-1}

Implements FunctionWithJacobianAndHVPProtocol.
"""

from typing import Generic, Sequence

from pyapprox.util.backends.protocols import Array, Backend


class ProductPeakFunction(Generic[Array]):
    """Product Peak Genz function.

    f(x) = prod_i (c_i^{-2} + (x_i - w_i)^2)^{-1}

    This function has a peak centered at w, with width controlled by c.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    c : Sequence[float]
        Coefficients controlling peak width in each dimension.
        Larger c means narrower peak. Shape: (nvars,).
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
        # g_i = c_i^{-2} + (x_i - w_i)^2
        g = 1.0 / self._c**2 + (samples - self._w) ** 2
        # f = prod(g_i^{-1}) = 1 / prod(g_i)
        return 1.0 / bkd.prod(g, axis=0, keepdims=True)

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
        g = 1.0 / self._c**2 + t**2  # (nvars, 1)
        f = 1.0 / bkd.prod(g)

        # df/dx_i = -2f * t_i / g_i
        jac = -2.0 * f * t / g
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
        g = 1.0 / self._c**2 + t**2  # (nvars, 1)
        f = 1.0 / bkd.prod(g)

        # h_i = -2*t_i/g_i, so df/dx_i = f * h_i
        h = -2.0 * t / g  # (nvars, 1)

        # dh_i/dx_i = -2/g_i * (1 - 2*t_i^2/g_i)
        dh_diag = -2.0 / g * (1.0 - 2.0 * t**2 / g)  # (nvars, 1)

        # H_ii = f * (dh_i/dx_i + h_i^2)
        # H_ij = f * h_i * h_j (i != j)
        # (H @ v)_i = f * (dh_i/dx_i * v_i + h_i * (h . v))
        h_dot_v = bkd.sum(h * vec)
        hvp = f * (dh_diag * vec + h * h_dot_v)
        return hvp

    def integrate(self) -> float:
        """Compute analytical integral over [0, 1]^nvars.

        Returns
        -------
        float
            The analytical integral value.
        """
        bkd = self._bkd
        # Integral = prod_i c_i * (arctan(c_i*(1-w_i)) + arctan(c_i*w_i))
        c = self._c[:, 0]
        w = self._w[:, 0]
        integral = bkd.prod(
            c * (bkd.arctan(c * (1.0 - w)) + bkd.arctan(c * w))
        )
        return float(integral)
