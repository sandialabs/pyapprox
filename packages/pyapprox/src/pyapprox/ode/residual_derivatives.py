r"""Frozen bundle of an ODE residual's optional derivatives.

A residual :math:`f(y, t; p)` of :math:`M \dot y = f` always evaluates
:math:`f`; its derivatives are optional, so they travel in a bundle
returned by ``derivatives()``. Absence of a capability is ``None``, never
a missing attribute, and consumers decide what they can do by
None-checking the bundle once rather than by ``isinstance`` on the
residual.

Every field is an operator or an action, never a matrix, so a
distributed or matrix-free residual fills it as easily as a dense one:

- ``jacobian``: a :class:`~pyapprox.ode.protocols.JacobianOperatorProtocol`
  producer of :math:`\partial f/\partial y` as labeled operators. Declare
  one with :func:`dense_jacobian` (a private method returning the dense
  matrix) or :func:`matrix_free_jacobian` (its two actions).
- ``param``: :class:`ParamActions`, ``jvp(state, time, v) -> (nstates,)``
  for :math:`(\partial f/\partial p) v` with ``v`` of shape ``(nparams,)``
  and ``vjp(state, time, w) -> (nparams,)`` for
  :math:`(\partial f/\partial p)^\top w`. Declare from a small matrix with
  :func:`param_actions`.
- ``initial_state``: :class:`InitialStateDerivatives` of :math:`y_0(p)`:
  ``jvp(v) -> (nstates,)``, ``vjp(w) -> (nparams,)`` and, for second
  order, ``hvp(weight, v) -> (nparams,)`` for
  :math:`\sum_i w_i (\partial^2 y_{0,i}/\partial p^2) v`.
- second-order contractions with an adjoint :math:`\lambda`, all four
  together: ``state_state(state, time, adj, w) -> (nstates,)`` for
  :math:`\lambda^\top (\partial^2 f/\partial y^2) w`,
  ``state_param(state, time, adj, v) -> (nstates,)``,
  ``param_state(state, time, adj, w) -> (nparams,)`` and
  ``param_param(state, time, adj, v) -> (nparams,)``.

Vectors are 1-D backend arrays. A term known to be identically zero is
declared with :data:`STATED_ZERO`, never by leaving it out: the
constructor puts an exact zero in its place and records its name in
``zeros``, so a consumer may skip it, and absence keeps meaning "not
available". The bundle is generic over the Jacobian's format and
covariant in it, so a system builder keeps the static pairing of format
and solver and a consumer that ignores the format accepts any bundle.
"""

from dataclasses import dataclass, fields
from typing import Callable, Generic, Optional, TypeVar, Union

from pyapprox.ode.dense_jacobian_operator import DenseJacobianOperator
from pyapprox.ode.matrix_free_jacobian_operator import MatrixFreeJacobianOperator
from pyapprox.ode.protocols import JacobianOperatorProtocol
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import SpaceDescriptor

A = TypeVar("A", bound=ArrayProtocol)
F_co = TypeVar("F_co", covariant=True)
G = TypeVar("G")

StateActionFn = Callable[[Array, float, Array], Array]
InitialActionFn = Callable[[Array], Array]
InitialHVPFn = Callable[[Array, Array], Array]
ContractionFn = Callable[[Array, float, Array, Array], Array]
JacobianMatrixFn = Callable[[Array, float], Array]
JacobianBlockFn = Callable[[Array, float, Array], Array]

_SECOND_ORDER = ("state_state", "state_param", "param_state", "param_param")


class _StatedZero:
    """The type of :data:`STATED_ZERO`."""

    def __repr__(self) -> str:
        return "STATED_ZERO"


STATED_ZERO = _StatedZero()
"""Declares a derivative term identically zero (not merely unavailable)."""


def _require_callable(owner: str, name: str, value: object) -> None:
    if value is not None and not callable(value):
        raise TypeError(
            f"{owner} field '{name}' must be callable or None; got "
            f"{type(value).__name__}. Did you pass a computed value instead "
            "of the function itself?"
        )


@dataclass(frozen=True)
class ParamActions(Generic[Array]):
    r""":math:`\partial f/\partial p` by its two actions, which come together:
    the tangent sweep needs ``jvp``, the adjoint gradient ``vjp``."""

    jvp: StateActionFn[Array]
    vjp: StateActionFn[Array]

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_callable("ParamActions", field.name, getattr(self, field.name))


@dataclass(frozen=True)
class InitialStateDerivatives(Generic[Array]):
    r"""Derivatives of the initial state :math:`y_0(p)`; ``hvp`` only for
    second order."""

    jvp: InitialActionFn[Array]
    vjp: InitialActionFn[Array]
    hvp: Optional[InitialHVPFn[Array]] = None

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_callable(
                "InitialStateDerivatives", field.name, getattr(self, field.name)
            )


class _ZeroInitialAction(Generic[Array]):
    """An exact zero vector of length ``size`` from one vector argument."""

    def __init__(self, bkd: Backend[Array], size: int) -> None:
        self._bkd, self._size = bkd, size

    def __call__(self, vec: Array) -> Array:
        return self._bkd.zeros((self._size,))


class _ZeroInitialHVP(Generic[Array]):
    """An exact zero initial-state curvature, of length ``nparams``."""

    def __init__(self, bkd: Backend[Array], nparams: int) -> None:
        self._bkd, self._nparams = bkd, nparams

    def __call__(self, weight: Array, vec: Array) -> Array:
        return self._bkd.zeros((self._nparams,))


class _ZeroContraction(Generic[Array]):
    """An exact zero second-order contraction of length ``size``."""

    def __init__(self, bkd: Backend[Array], size: int) -> None:
        self._bkd, self._size = bkd, size

    def __call__(self, state: Array, time: float, adj: Array, vec: Array) -> Array:
        return self._bkd.zeros((self._size,))


@dataclass(frozen=True)
class ODEResidualDerivatives(Generic[Array, F_co]):
    """The optional derivatives of an ODE residual.

    Build with the named constructors. Frozen: a residual whose
    capability changes rebuilds its bundle, never mutates one.

    Invariants: ``param`` and ``initial_state`` come together and
    require ``jacobian``; the four second-order contractions come
    together and require ``param`` and an ``initial_state`` with
    ``hvp``; ``nparams`` is positive exactly when ``param`` is present.
    """

    jacobian: Optional[JacobianOperatorProtocol[Array, Array, F_co]] = None
    param: Optional[ParamActions[Array]] = None
    initial_state: Optional[InitialStateDerivatives[Array]] = None
    state_state: Optional[ContractionFn[Array]] = None
    state_param: Optional[ContractionFn[Array]] = None
    param_state: Optional[ContractionFn[Array]] = None
    param_param: Optional[ContractionFn[Array]] = None
    nparams: int = 0
    zeros: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in _SECOND_ORDER:
            _require_callable("ODEResidualDerivatives", name, getattr(self, name))
        if self.jacobian is not None and not isinstance(
            self.jacobian, JacobianOperatorProtocol
        ):
            raise TypeError(
                "jacobian must satisfy JacobianOperatorProtocol, got "
                f"{type(self.jacobian).__name__}; declare a matrix with "
                "dense_jacobian or actions with matrix_free_jacobian"
            )
        first = (self.param is not None, self.initial_state is not None)
        if first[0] != first[1]:
            raise ValueError("param and initial_state come together")
        if first[0] and self.jacobian is None:
            raise ValueError("param and initial_state require jacobian")
        if first[0] != (self.nparams > 0):
            raise ValueError(
                "nparams must be positive exactly when param is present, got "
                f"{self.nparams}"
            )
        present = [getattr(self, name) is not None for name in _SECOND_ORDER]
        if any(present) and not all(present):
            raise ValueError(
                "the four second-order contractions come together: "
                + ", ".join(_SECOND_ORDER)
            )
        if all(present):
            if self.initial_state is None or self.initial_state.hvp is None:
                raise ValueError(
                    "second order requires param and an initial_state with hvp"
                )
        elif self.initial_state is not None and self.initial_state.hvp is not None:
            raise ValueError("initial_state.hvp is for second order only")

    @staticmethod
    def none() -> "ODEResidualDerivatives[A, G]":
        """No derivatives: the residual can be stepped explicitly only."""
        return ODEResidualDerivatives()

    @staticmethod
    def state_only(
        jacobian: JacobianOperatorProtocol[A, A, G],
    ) -> "ODEResidualDerivatives[A, G]":
        """A state Jacobian and no parameters: implicit stepping, no
        parameter gradients."""
        return ODEResidualDerivatives(jacobian=jacobian)

    @staticmethod
    def first_order(
        bkd: Backend[A],
        nstates: int,
        nparams: int,
        jacobian: JacobianOperatorProtocol[A, A, G],
        param: ParamActions[A],
        initial_state: Union[InitialStateDerivatives[A], _StatedZero],
    ) -> "ODEResidualDerivatives[A, G]":
        """The state Jacobian and the parameter derivatives: adjoint
        gradients. ``initial_state=STATED_ZERO`` declares :math:`y_0`
        independent of :math:`p`."""
        zeros: set[str] = set()
        initial = _initial_state(bkd, nstates, nparams, initial_state, False, zeros)
        return ODEResidualDerivatives(
            jacobian=jacobian,
            param=param,
            initial_state=initial,
            nparams=nparams,
            zeros=frozenset(zeros),
        )

    @staticmethod
    def second_order(
        bkd: Backend[A],
        nstates: int,
        nparams: int,
        jacobian: JacobianOperatorProtocol[A, A, G],
        param: ParamActions[A],
        initial_state: Union[InitialStateDerivatives[A], _StatedZero],
        state_state: Union[ContractionFn[A], _StatedZero],
        state_param: Union[ContractionFn[A], _StatedZero],
        param_state: Union[ContractionFn[A], _StatedZero],
        param_param: Union[ContractionFn[A], _StatedZero],
    ) -> "ODEResidualDerivatives[A, G]":
        """First order plus the four second-order contractions: Hessian-
        vector products. Any term may be ``STATED_ZERO``; an
        ``initial_state`` given explicitly must include ``hvp``."""
        zeros: set[str] = set()
        initial = _initial_state(bkd, nstates, nparams, initial_state, True, zeros)
        sizes = {
            "state_state": nstates,
            "state_param": nstates,
            "param_state": nparams,
            "param_param": nparams,
        }
        given = {
            "state_state": state_state,
            "state_param": state_param,
            "param_state": param_state,
            "param_param": param_param,
        }
        resolved: dict[str, ContractionFn[A]] = {}
        for name, value in given.items():
            if isinstance(value, _StatedZero):
                resolved[name] = _ZeroContraction(bkd, sizes[name])
                zeros.add(name)
            else:
                resolved[name] = value
        return ODEResidualDerivatives(
            jacobian=jacobian,
            param=param,
            initial_state=initial,
            state_state=resolved["state_state"],
            state_param=resolved["state_param"],
            param_state=resolved["param_state"],
            param_param=resolved["param_param"],
            nparams=nparams,
            zeros=frozenset(zeros),
        )


def _initial_state(
    bkd: Backend[A],
    nstates: int,
    nparams: int,
    given: Union[InitialStateDerivatives[A], _StatedZero],
    second_order: bool,
    zeros: set[str],
) -> InitialStateDerivatives[A]:
    """The initial-state group, with exact zeros for ``STATED_ZERO``."""
    if isinstance(given, _StatedZero):
        zeros.add("initial_state")
        return InitialStateDerivatives(
            jvp=_ZeroInitialAction(bkd, nstates),
            vjp=_ZeroInitialAction(bkd, nparams),
            hvp=_ZeroInitialHVP(bkd, nparams) if second_order else None,
        )
    return given


class _JacobianMatrixFunction(Generic[Array]):
    """A function returning the Jacobian matrix, as the model
    ``DenseJacobianOperator`` reads."""

    def __init__(self, matrix: JacobianMatrixFn[Array]) -> None:
        self._matrix = matrix

    def jacobian(self, state: Array, time: float) -> Array:
        return self._matrix(state, time)


def dense_jacobian(
    matrix: JacobianMatrixFn[Array], nstates: int, bkd: Backend[Array]
) -> DenseJacobianOperator[Array]:
    """Declare a Jacobian given as a dense matrix.

    Parameters
    ----------
    matrix : callable
        ``(state, time) -> (nstates, nstates)`` backend array; typically
        a private bound method.
    nstates : int
        The number of states.
    bkd : Backend
        The backend of the states.
    """
    descriptor = SpaceDescriptor("state", nstates, backend_dtype_name(bkd))
    return DenseJacobianOperator(_JacobianMatrixFunction(matrix), descriptor, bkd)


class _JacobianActionFunctions(Generic[Array]):
    """Two action functions, as the model ``MatrixFreeJacobianOperator``
    reads."""

    def __init__(
        self, apply: JacobianBlockFn[Array], apply_transpose: JacobianBlockFn[Array]
    ) -> None:
        self._apply, self._apply_transpose = apply, apply_transpose

    def apply_jacobian(self, state: Array, time: float, v: Array) -> Array:
        return self._apply(state, time, v)

    def apply_jacobian_transpose(self, state: Array, time: float, w: Array) -> Array:
        return self._apply_transpose(state, time, w)


def matrix_free_jacobian(
    apply: JacobianBlockFn[Array],
    apply_transpose: JacobianBlockFn[Array],
    nstates: int,
    bkd: Backend[Array],
) -> MatrixFreeJacobianOperator[Array]:
    r"""Declare a Jacobian given by its actions.

    Parameters
    ----------
    apply, apply_transpose : callable
        ``(state, time, block) -> block`` for :math:`J v` and
        :math:`J^\top w`, blocks of shape ``(nstates, ncols)``.
    nstates : int
        The number of states.
    bkd : Backend
        The backend of the states.
    """
    descriptor = SpaceDescriptor("state", nstates, backend_dtype_name(bkd))
    return MatrixFreeJacobianOperator(
        _JacobianActionFunctions(apply, apply_transpose), descriptor
    )


class _ParamMatrixActions(Generic[Array]):
    r"""The actions of :math:`\partial f/\partial p` given as a matrix."""

    def __init__(self, matrix: JacobianMatrixFn[Array], bkd: Backend[Array]) -> None:
        self._matrix, self._bkd = matrix, bkd

    def jvp(self, state: Array, time: float, vec: Array) -> Array:
        return self._bkd.dot(self._matrix(state, time), vec)

    def vjp(self, state: Array, time: float, vec: Array) -> Array:
        return self._bkd.dot(self._matrix(state, time).T, vec)


def param_actions(
    matrix: JacobianMatrixFn[Array], bkd: Backend[Array]
) -> ParamActions[Array]:
    r"""Declare :math:`\partial f/\partial p` given as a small dense matrix.

    Parameters
    ----------
    matrix : callable
        ``(state, time) -> (nstates, nparams)`` backend array; typically
        a private bound method. Each action forms it once.
    bkd : Backend
        The backend of the states.
    """
    actions = _ParamMatrixActions(matrix, bkd)
    return ParamActions(jvp=actions.jvp, vjp=actions.vjp)
