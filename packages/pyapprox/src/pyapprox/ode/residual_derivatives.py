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


@dataclass(frozen=True)
class ODEResidualDerivatives(Generic[V, F_co]):
    """The optional derivatives of an ODE residual.

    Build with the named constructors. Frozen: a residual whose
    capability changes rebuilds its bundle, never mutates one.

    Invariants: ``param`` and ``initial_state`` come together, require
    ``jacobian`` and come with ``param_space``; the four second-order
    fields come together and require them.
    """

    jacobian: Optional[JacobianFn[V, F_co]] = None
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

    @staticmethod
    def none() -> "ODEResidualDerivatives[W, G]":
        """No derivatives: the residual can be stepped explicitly only."""
        return ODEResidualDerivatives()

    @staticmethod
    def state_only(jacobian: JacobianFn[W, G]) -> "ODEResidualDerivatives[W, G]":
        """A state Jacobian and no parameters: implicit stepping, no
        parameter gradients."""
        return ODEResidualDerivatives(jacobian=jacobian)

    @staticmethod
    def first_order(
        state_space: VectorSpaceProtocol[W],
        param_space: VectorSpaceProtocol[W],
        jacobian: JacobianFn[W, G],
        param: ParamFn[W],
        initial_state: Union[InitialStateFn[W], StatedZero],
    ) -> "ODEResidualDerivatives[W, G]":
        """The state Jacobian and the parameter derivatives: adjoint
        gradients. ``initial_state=STATED_ZERO`` declares :math:`y_0`
        independent of :math:`p`."""
        return ODEResidualDerivatives(
            jacobian=jacobian,
            param=param,
            initial_state=_initial_state(initial_state, state_space, param_space),
            param_space=param_space,
        )

    @staticmethod
    def second_order(
        state_space: VectorSpaceProtocol[W],
        param_space: VectorSpaceProtocol[W],
        jacobian: JacobianFn[W, G],
        param: ParamFn[W],
        initial_state: Union[InitialStateFn[W], StatedZero],
        initial_state_curvature: Union[CurvatureFn[W], StatedZero],
        state_state: Union[SecondDerivativeFn[W], StatedZero],
        state_param: Union[SecondDerivativeFn[W], StatedZero],
        param_param: Union[SecondDerivativeFn[W], StatedZero],
    ) -> "ODEResidualDerivatives[W, G]":
        """First order plus the second-order terms: Hessian-vector
        products. Any term may be ``STATED_ZERO``."""
        curvature: CurvatureFn[W] = (
            _ZeroCurvature(ZeroOperator(param_space))
            if isinstance(initial_state_curvature, StatedZero)
            else initial_state_curvature
        )
        return ODEResidualDerivatives(
            jacobian=jacobian,
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
) -> JacobianFn[Array, DenseFormat[Array]]:
    """Declare a Jacobian given as a dense matrix.

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
    return producer.jacobian_operator


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
) -> JacobianFn[Array, MatrixFree]:
    r"""Declare a Jacobian given by its actions.

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
    return producer.jacobian_operator


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
