r"""Frozen bundle of an ODE residual's optional derivatives.

A residual :math:`f(y, t; p)` of :math:`M \dot y = f` always evaluates
:math:`f`; its derivatives are optional, so they travel in a bundle
returned by ``derivatives()``. Absence of a capability is ``None``, never
a missing attribute, and consumers decide what they can do by
None-checking the bundle once rather than by ``isinstance`` on the
residual.

Every field is a callable returning an **operator** between spaces of
the stack, never a matrix. The parameter space is a space of the stack
like the state space (a field-valued parameter is as large as the state
and distributed like it), so the parameter derivatives are rectangular
operators, and a large or distributed residual fills the bundle as
easily as a small dense one. A few scalar parameters are a small space of
the same stack, so states, adjoints and parameters are all vectors of
one type ``V``: the bundle is ``ODEResidualDerivatives[V, F]``, and on
the backend stack ``V`` is the backend array. Operators act on blocks of
shape ``(size, ncols)``; states and adjoints passed to the callables are
1-D.

- ``jacobian(state, time, point)``: :math:`\partial f/\partial y`,
  state → state, labeled with ``point`` (a ``JacobianSnapshotProtocol``)
  because solvers pair with it and may reuse its setup.
- ``param(state, time)``: :math:`\partial f/\partial p`, parameter →
  state; its transpose gives the adjoint gradient's contribution.
- ``initial_state()``: :math:`\partial y_0/\partial p`, parameter →
  state.
- ``initial_state_curvature(weight)``:
  :math:`\sum_i w_i\,\partial^2 y_{0,i}/\partial p^2`, parameter →
  parameter.
- ``state_state(state, time, adj)``:
  :math:`\lambda^\top \partial^2 f/\partial y^2`, state → state
  (symmetric).
- ``state_param(state, time, adj)``:
  :math:`\lambda^\top \partial^2 f/\partial y\,\partial p`, parameter →
  state; its transpose is the parameter-state term, so the two can never
  disagree.
- ``param_param(state, time, adj)``:
  :math:`\lambda^\top \partial^2 f/\partial p^2`, parameter → parameter.

A term known to be identically zero is declared with :data:`STATED_ZERO`,
never by leaving it out: the constructor puts a
:class:`~pyapprox.util.linalg.operators.ZeroOperator` between the right
spaces in its place, whose properties declare ``zero``, so a consumer may
skip it, and absence keeps meaning "not available". The bundle is
covariant in the Jacobian's format, so a system builder keeps its static
pairing of format and solver and a consumer that ignores the format
accepts any bundle. A bundle describes one discretization and is rebuilt
when it changes (a remesh).

A bundle is valid for the parameters set when it was fetched: after
``set_param`` a consumer fetches ``derivatives()`` again, so a residual
may build fields that hold data computed for one parameter value (an
assembled operator, say). Which fields are present must not depend on
the parameter values.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, ParamSpec, TypeVar, Union

from pyapprox.ode.dense_jacobian_operator import DenseJacobianOperator
from pyapprox.ode.matrix_free_jacobian_operator import MatrixFreeJacobianOperator
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendVectorSpace,
    DenseFormat,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearizedOperator,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
    VectorSpaceProtocol,
    ZeroOperator,
)

V = TypeVar("V")
"""The vector type of states, adjoints and parameters: one stack's."""
W = TypeVar("W")
F_co = TypeVar("F_co", covariant=True)
G = TypeVar("G")
P = ParamSpec("P")

DerivativeOperator = ActionOperatorProtocol[V, FormatProtocol]
"""A derivative that is applied but never solved with: any format."""
JacobianFn = Callable[[V, float, LinearizationPoint], JacobianSnapshotProtocol[V, G]]
StateFreeJacobianFn = Callable[
    [float, LinearizationPoint], JacobianSnapshotProtocol[V, G]
]
"""A Jacobian that cannot see the state: :math:`J(t)`, :math:`f` affine."""
ConstantJacobianFn = Callable[[LinearizationPoint], JacobianSnapshotProtocol[V, G]]
"""A Jacobian that sees neither state nor time: :math:`f` affine with a
time-invariant :math:`J`."""
ParamFn = Callable[[V, float], DerivativeOperator[V]]
InitialStateFn = Callable[[], DerivativeOperator[V]]
CurvatureFn = Callable[[V], DerivativeOperator[V]]
SecondDerivativeFn = Callable[[V, float, V], DerivativeOperator[V]]

_FIELDS = (
    "jacobian",
    "param",
    "initial_state",
    "initial_state_curvature",
    "state_state",
    "state_param",
    "param_param",
)
_SECOND_ORDER = (
    "initial_state_curvature",
    "state_state",
    "state_param",
    "param_param",
)


class StatedZero:
    """The type of :data:`STATED_ZERO`."""

    def __repr__(self) -> str:
        return "STATED_ZERO"


STATED_ZERO = StatedZero()
"""Declares a derivative term identically zero (not merely unavailable)."""


class _ZeroInitialState(Generic[V]):
    """``initial_state`` for a :math:`y_0` independent of :math:`p`."""

    def __init__(self, op: ZeroOperator[V]) -> None:
        self._op = op

    def __call__(self) -> ZeroOperator[V]:
        return self._op


class _ZeroCurvature(Generic[V]):
    """``initial_state_curvature`` for a :math:`y_0` affine in :math:`p`."""

    def __init__(self, op: ZeroOperator[V]) -> None:
        self._op = op

    def __call__(self, weight: V) -> ZeroOperator[V]:
        return self._op


class _ZeroSecondDerivative(Generic[V]):
    """An adjoint-weighted second derivative that is identically zero."""

    def __init__(self, op: ZeroOperator[V]) -> None:
        self._op = op

    def __call__(self, state: V, time: float, adj: V) -> ZeroOperator[V]:
        return self._op


class Jacobian(Generic[V, F_co]):
    r""":math:`\partial f/\partial y` and what it may depend on.

    Always called as ``jacobian(state, time, point)``, so every consumer
    uses one signature. What it depends on is fixed by the named
    constructor the residual used, that is by the function it could
    supply: :meth:`general` takes ``(state, time, point)``,
    :meth:`state_free` takes ``(time, point)`` (a Jacobian that cannot see
    the state, so :math:`f` is affine in it), and :meth:`constant` takes
    ``(point)`` (affine, and the Jacobian time-invariant). A function that
    is never given the state cannot depend on it, so the declaration
    cannot disagree with the Jacobian; time-dependent forcing and boundary
    data live in :math:`f`, not in :math:`J`, and keep a Jacobian
    constant.

    Holds exactly one of the three functions; build it with the named
    constructors.
    """

    def __init__(
        self,
        general: Optional[JacobianFn[V, F_co]] = None,
        state_free: Optional[StateFreeJacobianFn[V, F_co]] = None,
        constant: Optional[ConstantJacobianFn[V, F_co]] = None,
    ) -> None:
        given = [fn for fn in (general, state_free, constant) if fn is not None]
        if len(given) != 1:
            raise ValueError("a Jacobian holds exactly one function")
        if not callable(given[0]):
            raise TypeError(
                f"a Jacobian holds a function; got {type(given[0]).__name__}"
            )
        self._general, self._state_free, self._constant = general, state_free, constant

    @staticmethod
    def general(fn: JacobianFn[W, G]) -> "Jacobian[W, G]":
        """A Jacobian that may depend on the state and the time."""
        return Jacobian(general=fn)

    @staticmethod
    def state_free(fn: StateFreeJacobianFn[W, G]) -> "Jacobian[W, G]":
        """A Jacobian of the time only: :math:`f` is affine in the state."""
        return Jacobian(state_free=fn)

    @staticmethod
    def constant(fn: ConstantJacobianFn[W, G]) -> "Jacobian[W, G]":
        """A Jacobian of neither state nor time: :math:`f` is affine in
        the state and its Jacobian time-invariant."""
        return Jacobian(constant=fn)

    def __call__(
        self, state: V, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, F_co]:
        """:math:`J` at ``(state, time)``, labeled ``point``; a state-free
        or constant Jacobian is not given what it cannot depend on."""
        if self._constant is not None:
            return self._constant(point)
        if self._state_free is not None:
            return self._state_free(time, point)
        if self._general is None:
            raise AssertionError("a Jacobian holds exactly one function")
        return self._general(state, time, point)

    def affine(self) -> bool:
        """Whether :math:`f` is affine in the state: the Jacobian cannot
        see it."""
        return self._general is None

    def time_invariant(self) -> bool:
        """Whether the Jacobian is time-invariant (and so :math:`f`
        affine): it sees neither state nor time."""
        return self._constant is not None

    def __repr__(self) -> str:
        level = (
            "constant"
            if self.time_invariant()
            else "state_free" if self.affine() else "general"
        )
        return f"{type(self).__name__}.{level}"


SnapshotMap = Callable[[JacobianSnapshotProtocol[V, G]], JacobianSnapshotProtocol[V, G]]
"""A map of a Jacobian snapshot to another of the same format, the same at
every state and time."""


class _MappedGeneral(Generic[V, G]):
    def __init__(self, fn: JacobianFn[V, G], transform: SnapshotMap[V, G]) -> None:
        self._fn, self._transform = fn, transform

    def __call__(
        self, state: V, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, G]:
        return self._transform(self._fn(state, time, point))


class _MappedStateFree(Generic[V, G]):
    def __init__(
        self, fn: StateFreeJacobianFn[V, G], transform: SnapshotMap[V, G]
    ) -> None:
        self._fn, self._transform = fn, transform

    def __call__(
        self, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, G]:
        return self._transform(self._fn(time, point))


class _MappedConstant(Generic[V, G]):
    def __init__(
        self, fn: ConstantJacobianFn[V, G], transform: SnapshotMap[V, G]
    ) -> None:
        self._fn, self._transform = fn, transform

    def __call__(self, point: LinearizationPoint) -> JacobianSnapshotProtocol[V, G]:
        return self._transform(self._fn(point))


def map_jacobian(
    jacobian: Jacobian[W, G], transform: SnapshotMap[W, G]
) -> Jacobian[W, G]:
    """``jacobian`` with ``transform`` applied to every snapshot, at the
    same level: a map that is the same at every state and time (zeroing a
    fixed set of rows, say) sees neither, so a state-free or constant
    Jacobian stays one."""
    if jacobian._constant is not None:
        return Jacobian.constant(_MappedConstant(jacobian._constant, transform))
    if jacobian._state_free is not None:
        return Jacobian.state_free(_MappedStateFree(jacobian._state_free, transform))
    if jacobian._general is None:
        raise AssertionError("a Jacobian holds exactly one function")
    return Jacobian.general(_MappedGeneral(jacobian._general, transform))


def as_jacobian(
    jacobian: Union[JacobianFn[W, G], Jacobian[W, G]],
) -> Jacobian[W, G]:
    """``jacobian`` as a :class:`Jacobian`: a plain function of the state
    and time is a general one."""
    if isinstance(jacobian, Jacobian):
        return jacobian
    return Jacobian.general(jacobian)


@dataclass(frozen=True)
class ODEResidualDerivatives(Generic[V, F_co]):
    """The optional derivatives of an ODE residual.

    Build with the named constructors. Frozen: a residual whose
    capability changes rebuilds its bundle, never mutates one.

    Invariants: ``param`` and ``initial_state`` come together, require
    ``jacobian`` and come with ``param_space``; the four second-order
    fields come together and require them; a Jacobian that cannot see the
    state (:math:`f` affine, see :class:`Jacobian`) has zero state
    curvature, so with second order present ``state_state`` must be the
    stated zero.

    The constructors take the Jacobian as a :class:`Jacobian`, or as a
    plain function of state and time (a general one).
    """

    jacobian: Optional[Jacobian[V, F_co]] = None
    param: Optional[ParamFn[V]] = None
    initial_state: Optional[InitialStateFn[V]] = None
    initial_state_curvature: Optional[CurvatureFn[V]] = None
    state_state: Optional[SecondDerivativeFn[V]] = None
    state_param: Optional[SecondDerivativeFn[V]] = None
    param_param: Optional[SecondDerivativeFn[V]] = None
    param_space: Optional[VectorSpaceProtocol[V]] = None

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"ODEResidualDerivatives field '{name}' must be callable "
                    f"or None; got {type(value).__name__}. Did you pass an "
                    "operator or a matrix instead of the function returning it?"
                )
        first = (self.param is not None, self.initial_state is not None)
        if first[0] != first[1]:
            raise ValueError("param and initial_state come together")
        if first[0] and self.jacobian is None:
            raise ValueError("param and initial_state require jacobian")
        if first[0] != (self.param_space is not None):
            raise ValueError("param_space comes exactly with param")
        present = [getattr(self, name) is not None for name in _SECOND_ORDER]
        if any(present) and not all(present):
            raise ValueError(
                "the four second-order fields come together: "
                + ", ".join(_SECOND_ORDER)
            )
        if all(present) and not first[0]:
            raise ValueError("second order requires param and initial_state")
        if self.jacobian is not None and not isinstance(self.jacobian, Jacobian):
            raise TypeError(
                "jacobian must be a Jacobian (the named constructors wrap a "
                f"plain function); got {type(self.jacobian).__name__}"
            )
        affine = self.jacobian is not None and self.jacobian.affine()
        if (
            affine
            and self.state_state is not None
            and not isinstance(self.state_state, _ZeroSecondDerivative)
        ):
            raise ValueError(
                "a Jacobian that cannot see the state means f is affine, with "
                "zero state curvature: state_state must be STATED_ZERO"
            )

    @staticmethod
    def none() -> "ODEResidualDerivatives[W, G]":
        """No derivatives: the residual can be stepped explicitly only."""
        return ODEResidualDerivatives()

    @staticmethod
    def state_only(
        jacobian: Union[JacobianFn[W, G], Jacobian[W, G]],
    ) -> "ODEResidualDerivatives[W, G]":
        """A state Jacobian and no parameters: implicit stepping, no
        parameter gradients."""
        return ODEResidualDerivatives(jacobian=as_jacobian(jacobian))

    @staticmethod
    def first_order(
        state_space: VectorSpaceProtocol[W],
        param_space: VectorSpaceProtocol[W],
        jacobian: Union[JacobianFn[W, G], Jacobian[W, G]],
        param: ParamFn[W],
        initial_state: Union[InitialStateFn[W], StatedZero],
    ) -> "ODEResidualDerivatives[W, G]":
        """The state Jacobian and the parameter derivatives: adjoint
        gradients. ``initial_state=STATED_ZERO`` declares :math:`y_0`
        independent of :math:`p`."""
        return ODEResidualDerivatives(
            jacobian=as_jacobian(jacobian),
            param=param,
            initial_state=_initial_state(initial_state, state_space, param_space),
            param_space=param_space,
        )

    @staticmethod
    def second_order(
        state_space: VectorSpaceProtocol[W],
        param_space: VectorSpaceProtocol[W],
        jacobian: Union[JacobianFn[W, G], Jacobian[W, G]],
        param: ParamFn[W],
        initial_state: Union[InitialStateFn[W], StatedZero],
        initial_state_curvature: Union[CurvatureFn[W], StatedZero],
        state_state: Union[SecondDerivativeFn[W], StatedZero],
        state_param: Union[SecondDerivativeFn[W], StatedZero],
        param_param: Union[SecondDerivativeFn[W], StatedZero],
    ) -> "ODEResidualDerivatives[W, G]":
        """First order plus the second-order terms: Hessian-vector
        products. Any term may be ``STATED_ZERO``; with a Jacobian that
        cannot see the state, ``state_state`` must be."""
        curvature: CurvatureFn[W] = (
            _ZeroCurvature(ZeroOperator(param_space))
            if isinstance(initial_state_curvature, StatedZero)
            else initial_state_curvature
        )
        return ODEResidualDerivatives(
            jacobian=as_jacobian(jacobian),
            param=param,
            initial_state=_initial_state(initial_state, state_space, param_space),
            initial_state_curvature=curvature,
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


def _initial_state(
    given: Union[InitialStateFn[W], StatedZero],
    state_space: VectorSpaceProtocol[W],
    param_space: VectorSpaceProtocol[W],
) -> InitialStateFn[W]:
    if isinstance(given, StatedZero):
        return _ZeroInitialState(ZeroOperator(param_space, state_space))
    return given


def second_derivative_or_zero(
    given: Union[SecondDerivativeFn[W], StatedZero],
    domain: VectorSpaceProtocol[W],
    range: VectorSpaceProtocol[W],
) -> SecondDerivativeFn[W]:
    """``given``, or for :data:`STATED_ZERO` a second derivative returning
    a :class:`~pyapprox.util.linalg.operators.ZeroOperator` from
    ``domain`` to ``range``, for any bundle built from such fields."""
    if isinstance(given, StatedZero):
        return _ZeroSecondDerivative(ZeroOperator(domain, range))
    return given


def backend_state_space(bkd: Backend[Array], nstates: int) -> BackendVectorSpace[Array]:
    """The state space of a model on the backend stack, in the backend's
    configured dtype."""
    descriptor = SpaceDescriptor("state", nstates, backend_dtype_name(bkd))
    return BackendVectorSpace(descriptor, bkd)


def backend_spaces(
    bkd: Backend[Array], nstates: int, nparams: int
) -> tuple[BackendVectorSpace[Array], BackendVectorSpace[Array]]:
    """The state and parameter spaces of a model on the backend stack, in
    the backend's configured dtype."""
    descriptor = SpaceDescriptor("param", nparams, backend_dtype_name(bkd))
    return backend_state_space(bkd, nstates), BackendVectorSpace(descriptor, bkd)


class _JacobianMatrixFunction(Generic[Array]):
    """A function returning the Jacobian matrix, as the model
    ``DenseJacobianOperator`` reads."""

    def __init__(self, matrix: Callable[[Array, float], Array]) -> None:
        self._matrix = matrix

    def jacobian(self, state: Array, time: float) -> Array:
        return self._matrix(state, time)


def dense_jacobian(
    matrix: Callable[[Array, float], Array],
    state_space: VectorSpaceProtocol[Array],
    bkd: Backend[Array],
) -> Jacobian[Array, DenseFormat[Array]]:
    """Declare a Jacobian given as a dense matrix of the state and time.

    Parameters
    ----------
    matrix : callable
        ``(state, time) -> (nstates, nstates)`` backend array; typically
        a private bound method.
    state_space : VectorSpaceProtocol
        The state space, as given to the bundle's constructor.
    bkd : Backend
        The backend of the states.
    """
    producer = DenseJacobianOperator(
        _JacobianMatrixFunction(matrix), state_space.descriptor(), bkd
    )
    return Jacobian.general(producer.jacobian_operator)


class _LabeledConstant(Generic[V, G]):
    """``point ->`` a fixed operator labeled ``point``."""

    def __init__(self, op: ActionOperatorProtocol[V, G]) -> None:
        self._op = op

    def __call__(self, point: LinearizationPoint) -> LinearizedOperator[V, G]:
        return LinearizedOperator(self._op, point)


class _LabeledOfTime(Generic[V, G]):
    """``(time, point) ->`` the operator of the time, labeled ``point``."""

    def __init__(self, op: Callable[[float], ActionOperatorProtocol[V, G]]) -> None:
        self._op = op

    def __call__(
        self, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[V, G]:
        return LinearizedOperator(self._op(time), point)


def constant_jacobian(op: ActionOperatorProtocol[W, G]) -> Jacobian[W, G]:
    """Declare the Jacobian of an :math:`f` affine in the state whose
    Jacobian is time-invariant: one operator, of any stack and format.

    Parameters
    ----------
    op : ActionOperatorProtocol
        The Jacobian at every state and time, state to state; its format
        is the Jacobian's, so it pairs with the system's solver.
    """
    return Jacobian.constant(_LabeledConstant(op))


def state_free_jacobian(
    op: Callable[[float], ActionOperatorProtocol[W, G]],
) -> Jacobian[W, G]:
    """Declare the Jacobian of an :math:`f` affine in the state: an
    operator of the time only, of any stack and format.

    Parameters
    ----------
    op : callable
        ``time ->`` the Jacobian at that time, state to state. It is not
        given the state, so it cannot depend on it.
    """
    return Jacobian.state_free(_LabeledOfTime(op))


class _JacobianActionFunctions(Generic[Array]):
    """Two action functions, as the model ``MatrixFreeJacobianOperator``
    reads."""

    def __init__(
        self,
        apply: Callable[[Array, float, Array], Array],
        apply_transpose: Callable[[Array, float, Array], Array],
    ) -> None:
        self._apply, self._apply_transpose = apply, apply_transpose

    def apply_jacobian(self, state: Array, time: float, v: Array) -> Array:
        return self._apply(state, time, v)

    def apply_jacobian_transpose(self, state: Array, time: float, w: Array) -> Array:
        return self._apply_transpose(state, time, w)


def matrix_free_jacobian(
    apply: Callable[[Array, float, Array], Array],
    apply_transpose: Callable[[Array, float, Array], Array],
    state_space: VectorSpaceProtocol[Array],
) -> Jacobian[Array, MatrixFree]:
    r"""Declare a Jacobian given by its actions of the state and time.

    Parameters
    ----------
    apply, apply_transpose : callable
        ``(state, time, block) -> block`` for :math:`J v` and
        :math:`J^\top w`, blocks of shape ``(nstates, ncols)``.
    state_space : VectorSpaceProtocol
        The state space, as given to the bundle's constructor.
    """
    producer = MatrixFreeJacobianOperator(
        _JacobianActionFunctions(apply, apply_transpose), state_space.descriptor()
    )
    return Jacobian.general(producer.jacobian_operator)


SecondDerivativeActionFn = Callable[[Array, float, Array, Array], Array]
"""``(state, time, adj, vec) -> vec``: one column of a second derivative."""


class _SecondDerivativeOperator(Generic[Array]):
    """A second derivative at one ``(state, time, adj)``, applied a column
    at a time through the model's vector functions; ``MatrixFree``.
    ``out`` is ignored."""

    def __init__(
        self,
        actions: tuple[
            SecondDerivativeActionFn[Array], SecondDerivativeActionFn[Array]
        ],
        point: tuple[Array, float, Array],
        spaces: tuple[SpaceDescriptor, SpaceDescriptor],
        properties: OperatorProperties,
        bkd: Backend[Array],
    ) -> None:
        self._apply, self._apply_transpose = actions
        self._state, self._time, self._adj = point
        self._domain, self._range = spaces
        self._properties = properties
        self._bkd = bkd

    def domain(self) -> SpaceDescriptor:
        return self._domain

    def range(self) -> SpaceDescriptor:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> MatrixFree:
        return MatrixFree()

    def _columns(
        self, action: SecondDerivativeActionFn[Array], block: Array, size: int
    ) -> Array:
        if block.ndim != 2 or block.shape[0] != size:
            raise ValueError(
                f"input must have shape ({size}, ncols), got {tuple(block.shape)}"
            )
        return self._bkd.stack(
            [
                action(self._state, self._time, self._adj, block[:, k])
                for k in range(block.shape[1])
            ],
            axis=1,
        )

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        return self._columns(self._apply, x, self._domain.size)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self._columns(self._apply_transpose, y, self._range.size)


class _MatrixFreeSecondDerivative(Generic[Array]):
    """A second derivative from its two vector functions."""

    def __init__(
        self,
        apply: SecondDerivativeActionFn[Array],
        apply_transpose: SecondDerivativeActionFn[Array],
        domain: VectorSpaceProtocol[Array],
        range: VectorSpaceProtocol[Array],
        bkd: Backend[Array],
        properties: OperatorProperties,
    ) -> None:
        self._actions = (apply, apply_transpose)
        self._spaces = (domain.descriptor(), range.descriptor())
        self._bkd, self._properties = bkd, properties

    def __call__(
        self, state: Array, time: float, adj: Array
    ) -> _SecondDerivativeOperator[Array]:
        return _SecondDerivativeOperator(
            self._actions, (state, time, adj), self._spaces, self._properties, self._bkd
        )


def matrix_free_second_derivative(
    apply: SecondDerivativeActionFn[Array],
    apply_transpose: SecondDerivativeActionFn[Array],
    domain: VectorSpaceProtocol[Array],
    range: VectorSpaceProtocol[Array],
    bkd: Backend[Array],
    symmetric: bool = False,
) -> SecondDerivativeFn[Array]:
    r"""Declare an adjoint-weighted second derivative given by its actions.

    For a model whose second derivatives are vector functions
    ``(state, time, adj, vec) -> vec`` rather than matrices. The result
    applies them one column at a time; it refers to the state and
    adjoint it was made at rather than copying them.

    Parameters
    ----------
    apply, apply_transpose : callable
        The operator and its transpose on one vector: for
        ``state_param``, :math:`\lambda^\top f_{yp} v` and
        :math:`\lambda^\top f_{py} w`; for a symmetric ``state_state`` or
        ``param_param``, the same function twice.
    domain, range : VectorSpaceProtocol
        The operator's spaces, as for :func:`dense_operator`.
    bkd : Backend
        The backend of the states.
    symmetric : bool, default False
        Declare the operator symmetric; not verified.
    """
    return _MatrixFreeSecondDerivative(
        apply,
        apply_transpose,
        domain,
        range,
        bkd,
        OperatorProperties(symmetric=symmetric),
    )


CurvatureActionFn = Callable[[Array, Array], Array]
"""``(weight, vec) -> vec``: one column of the initial-state curvature."""


class _InitialStateCurvatureOperator(Generic[Array]):
    r""":math:`\sum_i w_i\,\partial^2 y_{0,i}/\partial p^2` at one weight,
    applied a column at a time; symmetric, ``MatrixFree``. ``out`` is
    ignored."""

    def __init__(
        self,
        hvp: CurvatureActionFn[Array],
        weight: Array,
        param_space: SpaceDescriptor,
        bkd: Backend[Array],
    ) -> None:
        self._hvp, self._weight = hvp, weight
        self._space, self._bkd = param_space, bkd

    def domain(self) -> SpaceDescriptor:
        return self._space

    def range(self) -> SpaceDescriptor:
        return self._space

    def properties(self) -> OperatorProperties:
        return OperatorProperties(symmetric=True)

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        size = self._space.size
        if x.ndim != 2 or x.shape[0] != size:
            raise ValueError(
                f"input must have shape ({size}, ncols), got {tuple(x.shape)}"
            )
        return self._bkd.stack(
            [self._hvp(self._weight, x[:, k]) for k in range(x.shape[1])], axis=1
        )

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self.apply(y)


class _MatrixFreeInitialStateCurvature(Generic[Array]):
    """The initial-state curvature from its vector function."""

    def __init__(
        self,
        hvp: CurvatureActionFn[Array],
        param_space: VectorSpaceProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        self._hvp, self._space, self._bkd = hvp, param_space.descriptor(), bkd

    def __call__(self, weight: Array) -> _InitialStateCurvatureOperator[Array]:
        return _InitialStateCurvatureOperator(self._hvp, weight, self._space, self._bkd)


def matrix_free_initial_state_curvature(
    hvp: CurvatureActionFn[Array],
    param_space: VectorSpaceProtocol[Array],
    bkd: Backend[Array],
) -> CurvatureFn[Array]:
    r"""Declare the initial-state curvature given by its action.

    Parameters
    ----------
    hvp : callable
        ``(weight, vec) -> vec`` for
        :math:`\sum_i w_i\,(\partial^2 y_{0,i}/\partial p^2)\,v`; the
        operator is symmetric, so the same function is its transpose.
    param_space : VectorSpaceProtocol
        The parameter space, as given to the bundle's constructor.
    bkd : Backend
        The backend of the states.
    """
    return _MatrixFreeInitialStateCurvature(hvp, param_space, bkd)


class _DenseOperatorFunction(Generic[P, Array]):
    """A function returning a dense matrix, as one returning the dense
    operator between two spaces; it takes the same arguments."""

    def __init__(
        self,
        matrix: Callable[P, Array],
        domain: VectorSpaceProtocol[Array],
        range: VectorSpaceProtocol[Array],
        bkd: Backend[Array],
        properties: OperatorProperties,
    ) -> None:
        self._matrix, self._bkd = matrix, bkd
        self._domain, self._range = domain.descriptor(), range.descriptor()
        self._properties = properties

    def __call__(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(
            self._matrix(*args, **kwargs),
            self._domain,
            self._bkd,
            range=self._range,
            properties=self._properties,
        )


def dense_operator(
    matrix: Callable[P, Array],
    domain: VectorSpaceProtocol[Array],
    range: VectorSpaceProtocol[Array],
    bkd: Backend[Array],
    symmetric: bool = False,
) -> _DenseOperatorFunction[P, Array]:
    r"""Declare any derivative but the Jacobian from a function returning
    its dense matrix.

    The result takes the same arguments as ``matrix``, so it fits the
    field it is given to: ``param(state, time)``, ``initial_state()``,
    ``initial_state_curvature(weight)``, or a second derivative
    ``(state, time, adj)``. Only for small spaces: a field-valued
    parameter supplies its operator directly.

    Parameters
    ----------
    matrix : callable
        Returns a ``(range.size, domain.size)`` backend array; typically
        a private bound method.
    domain, range : VectorSpaceProtocol
        The operator's spaces, as given to the bundle's constructor:
        parameter → state for ``param``, ``initial_state`` and
        ``state_param``; state → state for ``state_state``; parameter →
        parameter for ``initial_state_curvature`` and ``param_param``.
    bkd : Backend
        The backend of the states.
    symmetric : bool, default False
        Declare the matrix symmetric (``state_state``, ``param_param``,
        ``initial_state_curvature``); the dense operator verifies it.
    """
    return _DenseOperatorFunction(
        matrix, domain, range, bkd, OperatorProperties(symmetric=symmetric)
    )
