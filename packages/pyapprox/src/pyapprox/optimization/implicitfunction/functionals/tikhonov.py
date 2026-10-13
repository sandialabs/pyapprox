"""
Tikhonov augmentation of steady functionals.

Wraps any steady functional Q(u, p) as
:math:`Q(u, p) + (\\alpha/2)\\, p^\\top W_q\\, p` (:math:`W_q = I` by
default), adding a direct parameter dependence.
"""

from typing import Generic, Literal, Optional

from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
    SteadyFunctionalHVPFn,
    SteadyFunctionalJacobianFn,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    SteadyFunctionalProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class TikhonovAugmentedSteadyFunctional(Generic[Array]):
    """
    Augment a steady functional with a quadratic parameter cost.

    Computes :math:`Q(u, p) + (\\alpha/2)\\, p^\\top W_q\\, p` where Q is
    any scalar steady functional. State derivatives are the inner
    functional's; the parameter derivatives gain
    :math:`\\alpha (W_q p)^\\top` (Jacobian) and :math:`\\alpha W_q v`
    (HVP). Its ``derivatives()`` offers exactly what the inner
    functional's does: none, the Jacobians, or the Jacobians and all four
    HVPs.

    Parameters
    ----------
    inner : SteadyFunctionalProtocol
        The functional to augment. Must have ``nqoi() == 1``.
    alpha : float
        Cost coefficient (exchange rate between Q and parameter effort).
    bkd : Backend
        Backend for array operations.
    weight : Array, optional
        Symmetric weight :math:`W_q`. Shape: ``(nparams, nparams)``.
        Identity when omitted.
    """

    def __init__(
        self,
        inner: SteadyFunctionalProtocol[Array],
        alpha: float,
        bkd: Backend[Array],
        weight: Optional[Array] = None,
    ) -> None:
        validate_backend(bkd)
        if not isinstance(inner, SteadyFunctionalProtocol):
            raise TypeError(
                "inner must satisfy SteadyFunctionalProtocol, got "
                f"{type(inner).__name__}"
            )
        # A transient functional has every member of the steady
        # protocol, so isinstance cannot reject it; the declared time
        # dependence does.
        if inner.time_dependence() != "steady":
            raise TypeError(
                f"inner {type(inner).__name__} is a "
                f"{inner.time_dependence()} functional; this wrapper needs "
                "one that reads a single state"
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
        # A steady functional's capability is fixed at construction, so
        # the inner callables are captured once; None when absent.
        inner_derivs = inner.derivatives()
        self._inner_param_jacobian: Optional[
            SteadyFunctionalJacobianFn[Array]
        ] = inner_derivs.param_jacobian
        self._inner_param_param_hvp: Optional[
            SteadyFunctionalHVPFn[Array]
        ] = inner_derivs.param_param_hvp
        self._derivs: SteadyFunctionalDerivatives[Array]
        inner_state_state = inner_derivs.state_state_hvp
        inner_state_param = inner_derivs.state_param_hvp
        inner_param_state = inner_derivs.param_state_hvp
        if inner_derivs.state_jacobian is None:
            self._derivs = SteadyFunctionalDerivatives.none()
        elif (
            inner_state_state is None
            or inner_state_param is None
            or inner_param_state is None
        ):
            self._derivs = SteadyFunctionalDerivatives.first_order(
                inner_derivs.state_jacobian, self._param_jacobian
            )
        else:
            # The cost does not depend on the state, so the state-facing
            # second derivatives are the inner functional's own.
            self._derivs = SteadyFunctionalDerivatives.second_order(
                inner_derivs.state_jacobian,
                self._param_jacobian,
                inner_state_state,
                inner_state_param,
                inner_param_state,
                self._param_param_hvp,
            )

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

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        return "steady"

    def inner(self) -> SteadyFunctionalProtocol[Array]:
        """Return the wrapped functional."""
        return self._inner

    def alpha(self) -> float:
        """Return the cost coefficient."""
        return self._alpha

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the inner functional's capabilities, the parameter
        fields adding the cost."""
        return self._derivs

    def _weighted(self, vec: Array) -> Array:
        """Return :math:`W_q v` (identity weight when W_q was omitted)."""
        if self._weight is None:
            return vec
        return self._bkd.dot(self._weight, vec)

    def __call__(self, state: Array, param: Array) -> Array:
        """
        Evaluate :math:`Q(u, p) + (\\alpha/2) p^\\top W_q p`.

        Parameters
        ----------
        state : Array
            State. Shape: (nstates, 1)
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
        return self._inner(state, param) + self._bkd.reshape(cost, (1, 1))

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute dQ/dp, adding :math:`\\alpha (W_q p)^\\top`.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (1, nparams)
        """
        fn = self._inner_param_jacobian
        if fn is None:
            raise RuntimeError(
                "param_jacobian is unavailable; check derivatives() before "
                "calling"
            )
        return fn(state, param) + self._alpha * self._weighted(param).T

    def _param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dp^2)·v, adding :math:`\\alpha W_q v`.

        Shape: (nparams, 1).
        """
        fn = self._inner_param_param_hvp
        if fn is None:
            raise RuntimeError(
                "param_param_hvp is unavailable; check derivatives() before "
                "calling"
            )
        return fn(state, param, vvec) + self._alpha * self._weighted(vvec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"inner={self._inner!r}, "
            f"alpha={self._alpha}, "
            f"weighted={self._weight is not None})"
        )
