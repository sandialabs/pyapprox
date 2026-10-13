"""
Tikhonov augmentation of transient functionals.

Wraps any transient functional Q(y, p) as
:math:`Q(y, p) + (\\alpha/2)\\, p^\\top W_q\\, p` (:math:`W_q = I` by
default), adding a direct parameter dependence.
"""

from typing import Generic, Literal, Optional

from pyapprox.ode.functionals.derivatives import (
    TransientFunctionalDerivatives,
)
from pyapprox.ode.functionals.protocols import (
    TransientFunctionalProtocol,
)
from pyapprox.ode.time_quadrature import TrajectoryQuadratureProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class TikhonovAugmentedFunctional(Generic[Array]):
    """
    Augment a transient functional with a quadratic parameter cost.

    Computes :math:`Q(y, p) + (\\alpha/2)\\, p^\\top W_q\\, p` where Q is
    any transient functional. State derivatives delegate to the inner
    functional; the parameter derivatives gain :math:`\\alpha W_q p`
    (Jacobian) and :math:`\\alpha W_q v` (HVP). Its ``derivatives()``
    offers exactly what the inner functional's does.

    Parameters
    ----------
    inner : TransientFunctionalProtocol
        The functional to augment. Must have ``nqoi() == 1``.
    alpha : float
        Cost coefficient (exchange rate between Q and parameter effort).
    bkd : Backend
        Backend for array operations.
    weight : Array, optional
        Symmetric positive definite weight :math:`W_q`. Shape:
        ``(nparams, nparams)``. Identity when omitted.
    """

    def __init__(
        self,
        inner: TransientFunctionalProtocol[Array],
        alpha: float,
        bkd: Backend[Array],
        weight: Optional[Array] = None,
    ):
        validate_backend(bkd)
        if not isinstance(inner, TransientFunctionalProtocol):
            raise TypeError(
                "inner must satisfy "
                "TransientFunctionalProtocol, got "
                f"{type(inner).__name__}"
            )
        if inner.time_dependence() != "transient":
            raise TypeError(
                f"inner {type(inner).__name__} is a "
                f"{inner.time_dependence()} functional; this wrapper needs "
                "one that reads a trajectory"
            )
        if inner.nqoi() != 1:
            raise ValueError(
                f"inner must have nqoi() == 1, got {inner.nqoi()}: adding "
                "a scalar parameter cost to a multi-QoI functional is "
                "ambiguous"
            )
        if weight is not None:
            nparams = inner.nparams()
            if weight.ndim != 2 or weight.shape != (nparams, nparams):
                raise ValueError(
                    f"weight must have shape ({nparams}, {nparams}), got "
                    f"{weight.shape}"
                )
            asymmetry = bkd.max(bkd.abs(weight - weight.T))
            scale = bkd.max(bkd.abs(weight))
            if bkd.to_float(asymmetry) > 1e-12 * max(
                bkd.to_float(scale), 1.0
            ):
                raise ValueError("weight must be symmetric")
        self._inner = inner
        self._alpha = alpha
        self._weight = weight
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        return self._inner.nqoi()

    def nstates(self) -> int:
        """Return the number of state variables."""
        return self._inner.nstates()

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._inner.nparams()

    def nunique_params(self) -> int:
        """Return the number of parameters unique to the functional."""
        return self._inner.nunique_params()

    def time_dependence(self) -> Literal["transient"]:
        """Return the kind of state this functional reads.

        A trajectory of states, shape ``(nstates, ntimes)``.
        """
        return "transient"

    def inner(self) -> TransientFunctionalProtocol[Array]:
        """Return the wrapped functional."""
        return self._inner

    def alpha(self) -> float:
        """Return the cost coefficient."""
        return self._alpha

    def set_time_quadrature(
        self, quadrature: TrajectoryQuadratureProtocol[Array]
    ) -> None:
        """Forward the scheme's quadrature to the inner functional, which
        uses it or ignores it."""
        self._inner.set_time_quadrature(quadrature)

    def derivatives(self) -> TransientFunctionalDerivatives[Array]:
        """Return the inner functional's capabilities, adjusted.

        The Jacobians and HVPs are present exactly when the inner
        functional's are, the parameter ones adding the cost. The
        state-Jacobian action is the inner one, since the cost does not
        depend on the state. Per-QoI rows are not offered: the wrapper is
        scalar, and the inner rows would omit the cost.
        """
        inner = self._inner.derivatives()
        if inner.state_jacobian is None:
            return TransientFunctionalDerivatives.none()
        if inner.state_state_hvp is None:
            return TransientFunctionalDerivatives.first_order(
                self._state_jacobian,
                self._param_jacobian,
                apply_state_jacobian=inner.apply_state_jacobian,
            )
        return TransientFunctionalDerivatives.second_order(
            self._state_jacobian,
            self._param_jacobian,
            self._state_state_hvp,
            self._state_param_hvp,
            self._param_state_hvp,
            self._param_param_hvp,
            apply_state_jacobian=inner.apply_state_jacobian,
        )

    def _weighted(self, vec: Array) -> Array:
        """Return :math:`W_q v` (identity weight when W_q was omitted)."""
        if self._weight is None:
            return vec
        return self._bkd.dot(self._weight, vec)

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate :math:`Q(y, p) + (\\alpha/2) p^\\top W_q p`.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Augmented QoI. Shape: (1, 1)
        """
        cost = 0.5 * self._alpha * self._bkd.sum(
            param * self._weighted(param)
        )
        return self._inner(sol, param) + self._bkd.reshape(cost, (1, 1))

    # The derivative methods below read the inner functional's bundle when
    # called, since an inner functional's capability may depend on its
    # quadrature, injected after construction. They are reached only
    # through this wrapper's bundle, which offers each exactly when the
    # inner one has it; the None checks narrow the type.

    def _state_jacobian(self, sol: Array, param: Array) -> Array:
        """Compute dQ/dy; the parameter cost contributes nothing."""
        fn = self._inner.derivatives().state_jacobian
        if fn is None:
            raise RuntimeError("the inner functional has no Jacobians")
        return fn(sol, param)

    def _param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp, adding :math:`\\alpha (W_q p)^\\top`.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (1, nparams)
        """
        fn = self._inner.derivatives().param_jacobian
        if fn is None:
            raise RuntimeError("the inner functional has no Jacobians")
        return fn(sol, param) + self._alpha * self._weighted(param).T

    def _state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """Compute (d^2Q/dy^2)·w; the parameter cost contributes nothing."""
        fn = self._inner.derivatives().state_state_hvp
        if fn is None:
            raise RuntimeError("the inner functional has no HVPs")
        return fn(sol, param, time_idx, wvec)

    def _state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dy dp)·v; the parameter cost contributes nothing."""
        fn = self._inner.derivatives().state_param_hvp
        if fn is None:
            raise RuntimeError("the inner functional has no HVPs")
        return fn(sol, param, time_idx, vvec)

    def _param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """Compute (d^2Q/dp dy)·w; the parameter cost contributes nothing."""
        fn = self._inner.derivatives().param_state_hvp
        if fn is None:
            raise RuntimeError("the inner functional has no HVPs")
        return fn(sol, param, time_idx, wvec)

    def _param_param_hvp(
        self, sol: Array, param: Array, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dp^2)·v, adding :math:`\\alpha W_q v`."""
        fn = self._inner.derivatives().param_param_hvp
        if fn is None:
            raise RuntimeError("the inner functional has no HVPs")
        return fn(sol, param, vvec) + self._alpha * self._weighted(vvec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"inner={self._inner!r}, "
            f"alpha={self._alpha}, "
            f"weighted={self._weight is not None})"
        )
