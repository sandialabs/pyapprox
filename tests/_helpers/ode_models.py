"""Small ODE models for tests, written against the backend.

``LotkaVolterra`` supplies its residual, its Jacobian matrix and its
Jacobian actions, so the matrix and matrix-free wrappers can be checked
against each other on one model.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class LotkaVolterra(Generic[Array]):
    r"""Predator-prey: :math:`x' = a x - b x p`, :math:`p' = c x p - d p`.

    The Jacobian is
    :math:`J = [[a - b p, -b x], [c p, c x - d]]`. States have shape (2,);
    the actions take and return blocks of shape (2, ncols).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        a: float = 1.0,
        b: float = 0.1,
        c: float = 0.075,
        d: float = 1.5,
    ) -> None:
        self._bkd = bkd
        self._a, self._b, self._c, self._d = a, b, c, d

    def coefficients(self) -> tuple[float, float, float, float]:
        """``(a, b, c, d)``."""
        return self._a, self._b, self._c, self._d

    def __call__(self, y: Array, t: float) -> Array:
        x, p = y[0], y[1]
        return self._bkd.stack(
            [self._a * x - self._b * x * p, self._c * x * p - self._d * p]
        )

    def jacobian(self, y: Array, t: float) -> Array:
        x, p = y[0], y[1]
        bkd = self._bkd
        return bkd.stack(
            [
                bkd.stack([self._a - self._b * p, -self._b * x]),
                bkd.stack([self._c * p, self._c * x - self._d]),
            ]
        )

    def apply_jacobian(self, y: Array, t: float, v: Array) -> Array:
        x, p = y[0], y[1]
        v_x, v_p = v[0:1], v[1:2]
        return self._bkd.vstack(
            [
                (self._a - self._b * p) * v_x - self._b * x * v_p,
                self._c * p * v_x + (self._c * x - self._d) * v_p,
            ]
        )

    def apply_jacobian_transpose(self, y: Array, t: float, w: Array) -> Array:
        x, p = y[0], y[1]
        w_x, w_p = w[0:1], w[1:2]
        return self._bkd.vstack(
            [
                (self._a - self._b * p) * w_x + self._c * p * w_p,
                -self._b * x * w_x + (self._c * x - self._d) * w_p,
            ]
        )
