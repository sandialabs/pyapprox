r"""Non-autonomous quadratic ODE residual for testing time-context bugs.

Implements: f(y, t; p) = g(t) * (A·y + p[0]·y² + p[1])

where g(t) = 1 + 0.5*t multiplies everything. Any method evaluating f
at the wrong time gets the wrong g(t) factor, exposing timing convention
bugs in the time-stepper infrastructure.

All derivatives carry the g(t) factor:
    df/dy = g(t) * (A + 2·p[0]·diag(y))
    df/dp = g(t) * [y², 1]
    d²f/dy² = g(t) * 2·p[0]·I  (diagonal tensor)
    d²f/(dy dp) = g(t) * [2·y, 0]
    d²f/dp² = 0
"""

from typing import Generic, Optional

from pyapprox.ode.mass_matrix import IdentityMassMatrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


def _require_1d(name: str, arr: Array) -> None:
    """Raise if a direction vector is not 1D (silent broadcast risk)."""
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1D, got shape {tuple(arr.shape)}"
        )


def _g(t: float) -> float:
    return 1.0 + 0.5 * t


class TimeModulatedQuadraticODE(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    r"""Non-autonomous quadratic ODE for testing time-context correctness.

    .. math::

        f(y, t; p) = g(t) \bigl(A y + p_0 y^2 + p_1\bigr),
        \quad g(t) = 1 + \tfrac{1}{2} t

    Parameters
    ----------
    Amat : Array
        Linear stability matrix. Shape: (nstates, nstates)
    bkd : Backend
        Backend for array operations.
    """

    def __init__(self, Amat: Array, bkd: Backend[Array]) -> None:
        validate_backend(bkd)
        self._Amat = Amat
        self._bkd = bkd
        self._time = 0.0
        self._param: Optional[Array] = None
        self._nstates = Amat.shape[0]
        self._nparams = 2
        self._mass = IdentityMassMatrix(self._nstates, bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nparams(self) -> int:
        return self._nparams

    def set_time(self, time: float) -> None:
        self._time = time

    def set_param(self, param: Array) -> None:
        """Set the parameters.

        Parameters
        ----------
        param : Array
            Parameters [quad_coeff, constant]. Shape: (2,)
        """
        if param.ndim != 1:
            raise ValueError(
                f"param must be 1D with shape (nparams,), got shape "
                f"{tuple(param.shape)}"
            )
        self._param = param

    def __call__(self, state: Array) -> Array:
        r"""Evaluate f(y, t) = g(t) * (A·y + p[0]·y² + p[1])."""
        if self._param is None:
            raise RuntimeError("Must call set_param() first")
        gt = _g(self._time)
        p0 = self._param[0]
        p1 = self._param[1]
        return gt * (self._Amat @ state + p0 * state**2 + p1)

    def jacobian(self, state: Array) -> Array:
        r"""Compute df/dy = g(t) * (A + 2·p[0]·diag(y))."""
        if self._param is None:
            raise RuntimeError("Must call set_param() first")
        gt = _g(self._time)
        return gt * (self._Amat + 2.0 * self._param[0] * self._bkd.diag(state))

    def mass_matrix(self) -> IdentityMassMatrix[Array]:
        return self._mass

    def param_jacobian(self, state: Array) -> Array:
        r"""Compute df/dp = g(t) * [y², 1]."""
        gt = _g(self._time)
        jac = self._bkd.zeros((self._nstates, self._nparams))
        jac = self._bkd.copy(jac)
        jac[:, 0] = gt * state**2
        jac[:, 1] = gt
        return jac

    def initial_param_jacobian(self) -> Array:
        return self._bkd.zeros((self._nstates, self._nparams))

    # =====================================================================
    # HVP methods — all carry g(t)
    # =====================================================================

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        r"""Compute λᵀ·(d²f/dy²)·w = g(t) * 2·p[0]·λ·w (elementwise)."""
        if self._param is None:
            raise RuntimeError("Must call set_param() first")
        gt = _g(self._time)
        _require_1d("wvec", wvec)
        return gt * 2.0 * self._param[0] * adj_state * wvec

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        r"""Compute λᵀ·(d²f/dy dp)·v = g(t) * 2·λ·y·v[0]."""
        gt = _g(self._time)
        _require_1d("vvec", vvec)
        return gt * 2.0 * adj_state * state * vvec[0]

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        r"""Compute λᵀ·(d²f/dp dy)·w. Returns shape (nparams,)."""
        gt = _g(self._time)
        result = self._bkd.zeros((self._nparams,))
        result = self._bkd.copy(result)
        _require_1d("wvec", wvec)
        result[0] = gt * 2.0 * self._bkd.sum(adj_state * state * wvec)
        return result

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        r"""Compute λᵀ·(d²f/dp²)·v = 0."""
        return self._bkd.zeros((self._nparams,))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )
