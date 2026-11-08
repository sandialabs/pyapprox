r"""Frozen bundle of a residual's optional derivatives: :math:`R(u; p)`.

The residual of an implicit function :math:`R(u; p) = 0` always evaluates
:math:`R`; its derivatives are optional, so they travel in a bundle
returned by ``derivatives()``. Absence of a capability is ``None``, never
a missing attribute; a term known to be identically zero is declared with
:data:`STATED_ZERO`, never by leaving it out, and becomes a
:class:`~pyapprox.util.linalg.operators.ZeroOperator` between the right
spaces.

Every field is a callable returning an **operator** between spaces of the
stack, never a matrix; states, adjoints and parameters are all vectors of
one type ``V``, and the states and adjoints passed to the callables are in
the residual's state form.

- ``jacobian(state, point)``: :math:`\partial R/\partial u`, labeled with
  ``point`` (a ``JacobianSnapshotProtocol``), at one of two levels
  (:class:`ResidualJacobian`): depending on :math:`u`, or constant (an
  affine :math:`R`), which a solver may factor once.
- ``param(state)``: :math:`\partial R/\partial p`, parameter → state.
- ``state_state(state, adj)``: :math:`\lambda^\top \partial^2 R/\partial
  u^2`, state → state (symmetric).
- ``state_param(state, adj)``: :math:`\lambda^\top \partial^2 R/\partial
  u\, \partial p`, parameter → state; its transpose is the
  parameter-state term.
- ``param_param(state, adj)``: :math:`\lambda^\top \partial^2 R/\partial
  p^2`, parameter → parameter.

Fields read the current parameters when called, so a bundle is fixed for
its residual's life; which fields are present never changes.

A time-indexed right-hand side :math:`f(u, t; p)` is a family of such
residuals, one per time: :meth:`~pyapprox.ode.residual_derivatives.
ODEResidualDerivatives.at_time` gives the bundle of one member.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Union

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    VectorSpaceProtocol,
    ZeroOperator,
)

V = TypeVar("V")
W = TypeVar("W")
F_co = TypeVar("F_co", covariant=True)
G = TypeVar("G")

DerivativeOperator = ActionOperatorProtocol[V, FormatProtocol]
"""A derivative that is applied but never solved with: any format."""
GeneralJacobianFn = Callable[[V, LinearizationPoint], JacobianSnapshotProtocol[V, G]]
"""A Jacobian that may depend on the state."""
ConstantJacobianFn = Callable[[LinearizationPoint], JacobianSnapshotProtocol[V, G]]
"""A Jacobian independent of the state: :math:`R` affine."""
ResidualParamFn = Callable[[V], DerivativeOperator[V]]
ResidualSecondDerivativeFn = Callable[[V, V], DerivativeOperator[V]]


class StatedZero:
    """The type of :data:`STATED_ZERO`."""

    def __repr__(self) -> str:
        return "STATED_ZERO"


STATED_ZERO = StatedZero()
"""Declares a derivative term identically zero (not merely unavailable)."""


class _ZeroSecondDerivative(Generic[V]):
    """An adjoint-weighted second derivative that is identically zero."""

    def __init__(self, op: ZeroOperator[V]) -> None:
        self._op = op

    def __call__(self, state: V, adj: V) -> ZeroOperator[V]:
        return self._op


def _second_derivative(
    given: Union[ResidualSecondDerivativeFn[W], StatedZero],
    domain: VectorSpaceProtocol[W],
    range: VectorSpaceProtocol[W],
) -> ResidualSecondDerivativeFn[W]:
    if isinstance(given, StatedZero):
        return _ZeroSecondDerivative(ZeroOperator(domain, range))
    return given


def zero_second_derivative(op: ZeroOperator[W]) -> ResidualSecondDerivativeFn[W]:
    """A second derivative declared zero, returning ``op``: what
    :data:`STATED_ZERO` becomes in a bundle."""
    return _ZeroSecondDerivative(op)


def is_stated_zero(field: object) -> bool:
    """Whether a bundle field is a second derivative declared zero."""
    return isinstance(field, _ZeroSecondDerivative)


class ResidualJacobian(Generic[V, F_co]):
    r""":math:`\partial R/\partial u` and whether it depends on :math:`u`.

    Always called as ``jacobian(state, point)``. Build it with
    :meth:`general` (a function of ``(state, point)``) or :meth:`constant`
    (a function of ``point`` alone: :math:`R` is affine in :math:`u`, and a
    function that is never given the state cannot depend on it).
    """

    def __init__(
        self,
        general: Optional[GeneralJacobianFn[V, F_co]] = None,
        constant: Optional[ConstantJacobianFn[V, F_co]] = None,
    ) -> None:
        if (general is None) == (constant is None):
            raise ValueError("a Jacobian holds exactly one function")
        given = general if general is not None else constant
        if not callable(given):
            raise TypeError(f"a Jacobian holds a function; got {type(given).__name__}")
        self._general, self._constant = general, constant

    @staticmethod
    def general(fn: GeneralJacobianFn[W, G]) -> "ResidualJacobian[W, G]":
        """A Jacobian that may depend on the state."""
        return ResidualJacobian(general=fn)

    @staticmethod
    def constant(fn: ConstantJacobianFn[W, G]) -> "ResidualJacobian[W, G]":
        """A Jacobian independent of the state: :math:`R` affine."""
        return ResidualJacobian(constant=fn)

    def __call__(
        self, state: V, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, F_co]:
        """:math:`J` at ``state``, labeled ``point``; a constant Jacobian is
        not given the state."""
        if self._constant is not None:
            return self._constant(point)
        if self._general is None:
            raise AssertionError("a Jacobian holds exactly one function")
        return self._general(state, point)

    def affine(self) -> bool:
        """Whether :math:`R` is affine in the state, its Jacobian constant."""
        return self._constant is not None

    def __repr__(self) -> str:
        level = "constant" if self.affine() else "general"
        return f"{type(self).__name__}.{level}"


@dataclass(frozen=True)
class ResidualDerivatives(Generic[V, F_co]):
    """The optional derivatives of a residual :math:`R(u; p)`.

    Build with the named constructors. Frozen: a residual whose capability
    changes rebuilds its bundle, never mutates one.

    Invariants: ``param`` requires ``jacobian`` and comes with
    ``param_space``; the three second-order fields come together and
    require ``param``; with a constant Jacobian (:math:`R` affine)
    ``state_state`` is the stated zero.
    """

    jacobian: Optional[ResidualJacobian[V, F_co]] = None
    param: Optional[ResidualParamFn[V]] = None
    state_state: Optional[ResidualSecondDerivativeFn[V]] = None
    state_param: Optional[ResidualSecondDerivativeFn[V]] = None
    param_param: Optional[ResidualSecondDerivativeFn[V]] = None
    param_space: Optional[VectorSpaceProtocol[V]] = None

    def __post_init__(self) -> None:
        for name in ("param", "state_state", "state_param", "param_param"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"ResidualDerivatives field '{name}' must be callable or "
                    f"None; got {type(value).__name__}"
                )
        if self.jacobian is not None and not isinstance(
            self.jacobian, ResidualJacobian
        ):
            raise TypeError(
                "jacobian must be a ResidualJacobian; got "
                f"{type(self.jacobian).__name__}"
            )
        if self.param is not None and self.jacobian is None:
            raise ValueError("param requires jacobian")
        if (self.param is None) != (self.param_space is None):
            raise ValueError("param_space comes exactly with param")
        second = [
            self.state_state is not None,
            self.state_param is not None,
            self.param_param is not None,
        ]
        if any(second) and not all(second):
            raise ValueError(
                "the three second-order fields come together: state_state, "
                "state_param, param_param"
            )
        if all(second) and self.param is None:
            raise ValueError("second order requires param")
        if (
            self.jacobian is not None
            and self.jacobian.affine()
            and self.state_state is not None
            and not is_stated_zero(self.state_state)
        ):
            raise ValueError(
                "a constant Jacobian means R is affine, with zero state "
                "curvature: state_state must be STATED_ZERO"
            )

    @staticmethod
    def none() -> "ResidualDerivatives[W, G]":
        """No derivatives."""
        return ResidualDerivatives()

    @staticmethod
    def state_only(jacobian: ResidualJacobian[W, G]) -> "ResidualDerivatives[W, G]":
        """The state Jacobian alone: Newton solves, no parameter
        derivatives."""
        return ResidualDerivatives(jacobian=jacobian)

    @staticmethod
    def first_order(
        param_space: VectorSpaceProtocol[W],
        jacobian: ResidualJacobian[W, G],
        param: ResidualParamFn[W],
    ) -> "ResidualDerivatives[W, G]":
        """The state and parameter Jacobians: gradients."""
        return ResidualDerivatives(
            jacobian=jacobian, param=param, param_space=param_space
        )

    @staticmethod
    def second_order(
        state_space: VectorSpaceProtocol[W],
        param_space: VectorSpaceProtocol[W],
        jacobian: ResidualJacobian[W, G],
        param: ResidualParamFn[W],
        state_state: Union[ResidualSecondDerivativeFn[W], StatedZero],
        state_param: Union[ResidualSecondDerivativeFn[W], StatedZero],
        param_param: Union[ResidualSecondDerivativeFn[W], StatedZero],
    ) -> "ResidualDerivatives[W, G]":
        """First order plus the adjoint-weighted second derivatives:
        Hessian-vector products. Any of these may be ``STATED_ZERO``; with
        a constant Jacobian ``state_state`` must be."""
        return ResidualDerivatives(
            jacobian=jacobian,
            param=param,
            state_state=_second_derivative(state_state, state_space, state_space),
            state_param=_second_derivative(state_param, param_space, state_space),
            param_param=_second_derivative(param_param, param_space, param_space),
            param_space=param_space,
        )
