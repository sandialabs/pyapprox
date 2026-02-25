"""Quadrature data container for flow matching.

Holds pre-assembled quadrature points and weights for computing the
conditional flow matching loss integral.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend


class FlowMatchingQuadData(Generic[Array]):
    """Container for flow matching quadrature data.

    Stores the quadrature points (t, x0, x1) and weights needed to
    approximate the CFM loss integral. Optionally includes conditioning
    variables c.

    Parameters
    ----------
    t : Array
        Time quadrature points, shape ``(1, n_quad)``.
    x0 : Array
        Source sample quadrature points, shape ``(d, n_quad)``.
    x1 : Array
        Target sample quadrature points, shape ``(d, n_quad)``.
    weights : Array
        Quadrature weights, shape ``(n_quad,)``.
    bkd : Backend[Array]
        Computational backend.
    c : Array, optional
        Conditioning variables, shape ``(m, n_quad)``.
    """

    def __init__(
        self,
        t: Array,
        x0: Array,
        x1: Array,
        weights: Array,
        bkd: Backend[Array],
        c: Optional[Array] = None,
    ) -> None:
        self._t = t
        self._x0 = x0
        self._x1 = x1
        self._weights = weights
        self._bkd = bkd
        self._c = c
        self._validate()

    def _validate(self) -> None:
        """Check shape consistency of all fields."""
        t_shape = self._t.shape
        x0_shape = self._x0.shape
        x1_shape = self._x1.shape
        w_shape = self._weights.shape

        if len(t_shape) != 2 or t_shape[0] != 1:
            raise ValueError(
                f"t must have shape (1, n_quad), got {t_shape}"
            )
        n_quad = t_shape[1]

        if len(x0_shape) != 2 or x0_shape[1] != n_quad:
            raise ValueError(
                f"x0 must have shape (d, {n_quad}), got {x0_shape}"
            )
        if len(x1_shape) != 2 or x1_shape[1] != n_quad:
            raise ValueError(
                f"x1 must have shape (d, {n_quad}), got {x1_shape}"
            )
        if x0_shape[0] != x1_shape[0]:
            raise ValueError(
                f"x0 and x1 must have same first dimension, "
                f"got {x0_shape[0]} and {x1_shape[0]}"
            )
        if len(w_shape) != 1 or w_shape[0] != n_quad:
            raise ValueError(
                f"weights must have shape ({n_quad},), got {w_shape}"
            )
        if self._c is not None:
            c_shape = self._c.shape
            if len(c_shape) != 2 or c_shape[1] != n_quad:
                raise ValueError(
                    f"c must have shape (m, {n_quad}), got {c_shape}"
                )

    def t(self) -> Array:
        """Time quadrature points, shape ``(1, n_quad)``."""
        return self._t

    def x0(self) -> Array:
        """Source quadrature points, shape ``(d, n_quad)``."""
        return self._x0

    def x1(self) -> Array:
        """Target quadrature points, shape ``(d, n_quad)``."""
        return self._x1

    def weights(self) -> Array:
        """Quadrature weights, shape ``(n_quad,)``."""
        return self._weights

    def c(self) -> Optional[Array]:
        """Conditioning variables, shape ``(m, n_quad)`` or None."""
        return self._c

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def n_quad(self) -> int:
        """Number of quadrature points."""
        return self._t.shape[1]

    def d(self) -> int:
        """Spatial dimension (number of state variables)."""
        return self._x0.shape[0]

    def m(self) -> int:
        """Conditioning dimension, or 0 if no conditioning."""
        if self._c is None:
            return 0
        return self._c.shape[0]
