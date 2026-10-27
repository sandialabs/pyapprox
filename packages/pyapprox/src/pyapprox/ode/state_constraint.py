r"""A state constraint's optional derivatives, and the identity constraint.

A constraint :math:`\Pi_t` maps a state to the state satisfying the
constraints at time :math:`t` (for example, one whose boundary entries are
the boundary data). The time integrator applies it at stored states (the
initial state and each step result); how it is computed, and what the
residual does between stored states, is the discretization's business.

Its derivatives are optional, so they travel in a frozen bundle returned
by ``derivatives()``. Every field is a callable returning an **operator**
between spaces of the stack; states and adjoints passed to the callables
are 1-D, as in :mod:`pyapprox.ode.residual_derivatives`.

- ``tangent(state, time)``: :math:`\Pi'`, state → state. Forward
  sensitivities apply it, adjoints its transpose.
- ``param(state, time)``: :math:`\partial \Pi/\partial p`, parameter →
  state, for constraint data that depend on the parameters.
- ``state_state(state, time, adj)``:
  :math:`\lambda^\top \partial^2 \Pi/\partial y^2`, state → state.
- ``state_param(state, time, adj)``:
  :math:`\lambda^\top \partial^2 \Pi/\partial y\,\partial p`, parameter →
  state.
- ``param_param(state, time, adj)``:
  :math:`\lambda^\top \partial^2 \Pi/\partial p^2`, parameter →
  parameter.

A term known to be identically zero is declared with
:data:`~pyapprox.ode.residual_derivatives.STATED_ZERO`, never by leaving it
out: absence means "not available", and a stated zero is a
:class:`~pyapprox.util.linalg.operators.ZeroOperator` a consumer may skip.
Every affine constraint (fixed values on some entries) has zero second
derivatives.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Union

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ParamFn,
    SecondDerivativeFn,
    StatedZero,
    second_derivative_or_zero,
)
from pyapprox.util.backends.protocols import Array, ArrayProtocol
from pyapprox.util.linalg.operators import (
    IdentityOperator,
    VectorSpaceProtocol,
    ZeroOperator,
)

A = TypeVar("A", bound=ArrayProtocol)

TangentFn = Callable[[Array, float], DerivativeOperator[Array]]

_FIELDS = ("tangent", "param", "state_state", "state_param", "param_param")
_SECOND_ORDER = ("state_state", "state_param", "param_param")


class _ZeroParam(Generic[Array]):
    """``param`` for constraint data independent of the parameters."""

    def __init__(self, op: ZeroOperator[Array]) -> None:
        self._op = op

    def __call__(self, state: Array, time: float) -> ZeroOperator[Array]:
        return self._op


@dataclass(frozen=True)
class StateConstraintDerivatives(Generic[Array]):
    """The optional derivatives of a state constraint.

    Build with the named constructors. Frozen: a constraint whose
    capability changes rebuilds its bundle, never mutates one.

    Invariants: ``param`` requires ``tangent`` and comes exactly with
    ``param_space``; the three second-order fields come together and
    require ``param``.
    """

    tangent: Optional[TangentFn[Array]] = None
    param: Optional[ParamFn[Array]] = None
    state_state: Optional[SecondDerivativeFn[Array]] = None
    state_param: Optional[SecondDerivativeFn[Array]] = None
    param_param: Optional[SecondDerivativeFn[Array]] = None
    param_space: Optional[VectorSpaceProtocol[Array]] = None

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"StateConstraintDerivatives field '{name}' must be "
                    f"callable or None; got {type(value).__name__}. Did you "
                    "pass an operator instead of the function returning it?"
                )
        if self.param is not None and self.tangent is None:
            raise ValueError("param requires tangent")
        if (self.param is not None) != (self.param_space is not None):
            raise ValueError("param_space comes exactly with param")
        present = [getattr(self, name) is not None for name in _SECOND_ORDER]
        if any(present) and not all(present):
            raise ValueError(
                "the three second-order fields come together: "
                + ", ".join(_SECOND_ORDER)
            )
        if all(present) and self.param is None:
            raise ValueError("second order requires param")

    @staticmethod
    def none() -> "StateConstraintDerivatives[A]":
        """No derivatives: the constraint can be applied only, so forward
        solves work and sensitivities do not."""
        return StateConstraintDerivatives()

    @staticmethod
    def state_only(tangent: TangentFn[A]) -> "StateConstraintDerivatives[A]":
        """The tangent and no parameters: forward sensitivities in the
        state, no parameter gradients."""
        return StateConstraintDerivatives(tangent=tangent)

    @staticmethod
    def first_order(
        state_space: VectorSpaceProtocol[A],
        param_space: VectorSpaceProtocol[A],
        tangent: TangentFn[A],
        param: Union[ParamFn[A], StatedZero],
    ) -> "StateConstraintDerivatives[A]":
        """The tangent and the parameter derivative: adjoint gradients.
        ``param=STATED_ZERO`` declares constraint data independent of
        :math:`p`."""
        return StateConstraintDerivatives(
            tangent=tangent,
            param=_param(param, state_space, param_space),
            param_space=param_space,
        )

    @staticmethod
    def second_order(
        state_space: VectorSpaceProtocol[A],
        param_space: VectorSpaceProtocol[A],
        tangent: TangentFn[A],
        param: Union[ParamFn[A], StatedZero],
        state_state: Union[SecondDerivativeFn[A], StatedZero],
        state_param: Union[SecondDerivativeFn[A], StatedZero],
        param_param: Union[SecondDerivativeFn[A], StatedZero],
    ) -> "StateConstraintDerivatives[A]":
        """First order plus the second-order terms: Hessian-vector
        products. Any term but the tangent may be ``STATED_ZERO``."""
        return StateConstraintDerivatives(
            tangent=tangent,
            param=_param(param, state_space, param_space),
            state_state=second_derivative_or_zero(
                state_state, state_space, state_space
            ),
            state_param=second_derivative_or_zero(
                state_param, param_space, state_space
            ),
            param_param=second_derivative_or_zero(
                param_param, param_space, param_space
            ),
            param_space=param_space,
        )


def _param(
    given: Union[ParamFn[A], StatedZero],
    state_space: VectorSpaceProtocol[A],
    param_space: VectorSpaceProtocol[A],
) -> ParamFn[A]:
    if isinstance(given, StatedZero):
        return _ZeroParam(ZeroOperator(param_space, state_space))
    return given


class _IdentityTangent(Generic[Array]):
    """The identity constraint's tangent."""

    def __init__(self, op: IdentityOperator[Array]) -> None:
        self._op = op

    def __call__(self, state: Array, time: float) -> IdentityOperator[Array]:
        return self._op


class IdentityConstraint(Generic[Array]):
    r""":math:`\Pi_t = I`: no constraint.

    ``constrain`` returns its input unchanged, so a time integrator given
    it produces exactly what it produces unconstrained. The tangent is an
    :class:`~pyapprox.util.linalg.operators.IdentityOperator`, declaring
    ``identity`` so a consumer may skip it; with a parameter space every
    other derivative is a stated zero, without one the bundle is state
    only.

    Parameters
    ----------
    state_space : VectorSpaceProtocol
        The state space.
    param_space : VectorSpaceProtocol, optional
        The parameter space, when parameter derivatives are needed.
    """

    def __init__(
        self,
        state_space: VectorSpaceProtocol[Array],
        param_space: Optional[VectorSpaceProtocol[Array]] = None,
    ) -> None:
        tangent = _IdentityTangent(IdentityOperator(state_space))
        self._derivs: StateConstraintDerivatives[Array] = (
            StateConstraintDerivatives.state_only(tangent)
            if param_space is None
            else StateConstraintDerivatives.second_order(
                state_space,
                param_space,
                tangent,
                param=STATED_ZERO,
                state_state=STATED_ZERO,
                state_param=STATED_ZERO,
                param_param=STATED_ZERO,
            )
        )

    def constrain(self, state: Array, time: float) -> Array:
        """``state`` itself."""
        return state

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
