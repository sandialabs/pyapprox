"""Linear probability path for flow matching.

Implements the linear interpolation path x_t = (1-t)*x0 + t*x1,
which is the standard choice for conditional flow matching.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class LinearPath(Generic[Array]):
    """Linear probability path: x_t = (1-t)*x0 + t*x1.

    The signal and noise coefficients are:
        alpha(t) = t       (weight on target x1)
        sigma(t) = 1 - t   (weight on source x0)

    The conditional target vector field is:
        u_t(x | x0, x1) = x1 - x0

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def alpha(self, t: Array) -> Array:
        """Signal coefficient alpha(t) = t.

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``.

        Returns
        -------
        Array
            Shape ``(1, ns)``.
        """
        return t

    def sigma(self, t: Array) -> Array:
        """Noise coefficient sigma(t) = 1 - t.

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``.

        Returns
        -------
        Array
            Shape ``(1, ns)``.
        """
        return self._bkd.ones_like(t) - t

    def d_alpha(self, t: Array) -> Array:
        """Time derivative of alpha: d_alpha(t) = 1.

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``.

        Returns
        -------
        Array
            Shape ``(1, ns)``.
        """
        return self._bkd.ones_like(t)

    def d_sigma(self, t: Array) -> Array:
        """Time derivative of sigma: d_sigma(t) = -1.

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``.

        Returns
        -------
        Array
            Shape ``(1, ns)``.
        """
        return -self._bkd.ones_like(t)

    def interpolate(self, t: Array, x0: Array, x1: Array) -> Array:
        """Interpolate: x_t = sigma(t)*x0 + alpha(t)*x1 = (1-t)*x0 + t*x1.

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``.
        x0 : Array
            Source samples, shape ``(d, ns)``.
        x1 : Array
            Target samples, shape ``(d, ns)``.

        Returns
        -------
        Array
            Interpolated samples, shape ``(d, ns)``.
        """
        return self.sigma(t) * x0 + self.alpha(t) * x1

    def target_field(self, t: Array, x0: Array, x1: Array) -> Array:
        """Conditional target vector field: u_t = x1 - x0.

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``. Unused for linear path.
        x0 : Array
            Source samples, shape ``(d, ns)``.
        x1 : Array
            Target samples, shape ``(d, ns)``.

        Returns
        -------
        Array
            Target vector field, shape ``(d, ns)``.
        """
        return x1 - x0
