r"""The implicit system of a semi-explicit model :math:`M \dot y = f(y, t)`.

``SemiExplicitSystem`` satisfies
:class:`~pyapprox.ode.protocols.ImplicitSystemProtocol` with
:math:`F = M \dot y - f(y, t)`, so the shifted Jacobian is
:math:`\sigma M - J`. The mass is an operator in the same stack and
format as the Jacobian, so the sum is stored when they are and pairs
with a solver of that format; a singular mass (a differential-algebraic
model such as Stokes) is fine, because implicit steps never invert it.

``dense_system`` and ``matrix_free_system`` build one from a model in a
single line, for a model whose Jacobian is a dense backend array or is
given by its actions.
"""

from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.dense_jacobian_operator import DenseJacobianOperator
from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.ode.matrix_free_jacobian_operator import MatrixFreeJacobianOperator
from pyapprox.ode.protocols import JacobianOperatorProtocol, ODEResidualProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendVectorSpace,
    DenseFormat,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import (
    GMRES,
    ActionOperatorProtocol,
    FormatProtocol,
    LinearizationPoint,
    LinearizedOperator,
    LinearSolverProtocol,
    MatrixFree,
    OperatorProperties,
    ScaledSum,
    SpaceDescriptor,
    VectorSpaceProtocol,
)

F = TypeVar("F", bound=FormatProtocol)


@runtime_checkable
class RightHandSideProtocol(Protocol[Array]):
    r""":math:`f(y, t)` of :math:`M \dot y = f(y, t)`."""

    def bkd(self) -> Backend[Array]:
        """The computational backend."""
        ...

    def __call__(self, state: Array, time: float) -> Array:
        """:math:`f(y, t)`. Shape: (nstates,)"""
        ...


class SemiExplicitSystem(Generic[Array, F]):
    r""":math:`F(t, y, \dot y) = M \dot y - f(y, t)` and its shifted Jacobian.

    Satisfies :class:`~pyapprox.ode.protocols.ImplicitSystemProtocol`
    with the stack's vectors being backend arrays of shape
    ``(nstates, ncols)``; states are 1-D backend arrays.

    Parameters
    ----------
    rhs : RightHandSideProtocol
        :math:`f(y, t)`.
    mass : ActionOperatorProtocol
        :math:`M`, in the Jacobian's format ``F``.
    jacobian : JacobianOperatorProtocol
        :math:`J = \partial f/\partial y`, in format ``F``.
    space : VectorSpaceProtocol
        The state space's vector operations, for the sum.
    solver : LinearSolverProtocol
        Solves with the shifted Jacobian; pairs with ``F``.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    SpaceMismatchError
        If the mass is not an operator on the space.
    """

    def __init__(
        self,
        rhs: RightHandSideProtocol[Array],
        mass: ActionOperatorProtocol[Array, F],
        jacobian: JacobianOperatorProtocol[Array, Array, F],
        space: VectorSpaceProtocol[Array],
        solver: LinearSolverProtocol[Array, F],
    ) -> None:
        for name, value, protocol in (
            ("rhs", rhs, RightHandSideProtocol),
            ("mass", mass, ActionOperatorProtocol),
            ("jacobian", jacobian, JacobianOperatorProtocol),
            ("space", space, VectorSpaceProtocol),
            ("solver", solver, LinearSolverProtocol),
        ):
            if not isinstance(value, protocol):
                raise TypeError(
                    f"{name} must satisfy {protocol.__name__}, "
                    f"got {type(value).__name__}"
                )
        descriptor = space.descriptor()
        descriptor.require_same(mass.domain(), "domain of the mass")
        descriptor.require_same(mass.range(), "range of the mass")
        self._rhs = rhs
        self._mass = mass
        self._jacobian = jacobian
        self._space = space
        self._solver = solver
        self._bkd = rhs.bkd()

    def mass(self) -> ActionOperatorProtocol[Array, F]:
        """:math:`M`."""
        return self._mass

    def space(self) -> VectorSpaceProtocol[Array]:
        """The state space's vector operations."""
        return self._space

    def residual(self, state: Array, state_rate: Array, time: float) -> Array:
        r""":math:`M \dot y - f(y, t)`. Shape: (nstates,)"""
        bkd = self._bkd
        size = self._space.descriptor().size
        mass_rate = self._mass.apply(bkd.reshape(state_rate, (size, 1)))
        return bkd.reshape(mass_rate, (size,)) - self._rhs(state, time)

    def shifted_jacobian(
        self,
        state: Array,
        state_rate: Array,
        time: float,
        shift: float,
        point: LinearizationPoint,
    ) -> LinearizedOperator[Array, F]:
        r""":math:`\sigma M - J(y, t)`, labeled with ``point``; it does not
        depend on :math:`\dot y`."""
        jacobian = self._jacobian.jacobian_operator(state, time, point)
        return LinearizedOperator(
            ScaledSum([(shift, self._mass), (-1.0, jacobian)], self._space), point
        )

    def solver(self) -> LinearSolverProtocol[Array, F]:
        return self._solver

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rhs={self._rhs!r}, solver={self._solver!r})"


def _state_space(
    mass: MassMatrixProtocol[Array], bkd: Backend[Array]
) -> tuple[Array, SpaceDescriptor]:
    """The model's mass as a dense array, and the state space it acts on,
    in the backend's configured dtype."""
    matrix = mass.as_matrix()
    descriptor = SpaceDescriptor("state", matrix.shape[0], backend_dtype_name(bkd))
    return matrix, descriptor


def _mass_properties(mass: MassMatrixProtocol[Array]) -> OperatorProperties:
    """What the model's mass declares: identity, or diagonal, or nothing."""
    if mass.is_identity():
        return OperatorProperties(
            symmetric=True, spd=True, diagonal=True, identity=True
        )
    if mass.is_diagonal():
        return OperatorProperties(symmetric=True, diagonal=True)
    return OperatorProperties()


def dense_system(
    model: ODEResidualProtocol[Array],
    solver: Optional[LinearSolverProtocol[Array, DenseFormat[Array]]] = None,
) -> SemiExplicitSystem[Array, DenseFormat[Array]]:
    """The implicit system of a model whose Jacobian is a dense array.

    The mass, the Jacobian and the shifted Jacobian are dense operators
    of the backend stack; the solver defaults to the backend's dense LU.
    Its mass is read from the model's ``mass_matrix()``.

    Parameters
    ----------
    model : ODEResidualProtocol
        Supplies :math:`f`, its dense ``jacobian`` and its mass.
    solver : LinearSolverProtocol, optional
        A solver for dense operators. Defaults to ``BackendDenseLU``.
    """
    bkd = model.bkd()
    mass = model.mass_matrix()
    matrix, descriptor = _state_space(mass, bkd)
    return SemiExplicitSystem(
        model,
        BackendDenseOperator(
            matrix, descriptor, bkd, properties=_mass_properties(mass)
        ),
        DenseJacobianOperator(model, descriptor, bkd),
        BackendVectorSpace(descriptor, bkd),
        BackendDenseLU(bkd) if solver is None else solver,
    )


@runtime_checkable
class MatrixFreeModelProtocol(Protocol[Array]):
    """A model given by :math:`f`, its mass and its Jacobian's actions."""

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array, time: float) -> Array: ...

    def mass_matrix(self) -> MassMatrixProtocol[Array]: ...

    def apply_jacobian(self, state: Array, time: float, v: Array) -> Array: ...

    def apply_jacobian_transpose(
        self, state: Array, time: float, w: Array
    ) -> Array: ...


class _MassAction(Generic[Array]):
    """A model's mass applied through its own ``apply``: no storage is
    formed, so the format is ``MatrixFree``."""

    def __init__(
        self,
        mass: MassMatrixProtocol[Array],
        descriptor: SpaceDescriptor,
        bkd: Backend[Array],
    ) -> None:
        self._mass = mass
        self._descriptor = descriptor
        self._bkd = bkd
        self._properties = _mass_properties(mass)

    def domain(self) -> SpaceDescriptor:
        return self._descriptor

    def range(self) -> SpaceDescriptor:
        return self._descriptor

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> MatrixFree:
        return MatrixFree()

    def _columns(self, apply: Callable[[Array], Array], x: Array) -> Array:
        size = self._descriptor.size
        if x.ndim != 2 or x.shape[0] != size:
            raise ValueError(
                f"input must have shape ({size}, ncols), got {tuple(x.shape)}"
            )
        return self._bkd.stack([apply(x[:, k]) for k in range(x.shape[1])], axis=1)

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        return self._columns(self._mass.apply, x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self._columns(self._mass.apply_transpose, y)


def matrix_free_system(
    model: MatrixFreeModelProtocol[Array],
    nstates: int,
    solver: Optional[LinearSolverProtocol[Array, MatrixFree]] = None,
) -> SemiExplicitSystem[Array, MatrixFree]:
    r"""The implicit system of a model whose Jacobian is given by its
    actions.

    Neither the Jacobian nor the mass is formed: both are applied through
    the model, so the shifted Jacobian :math:`\sigma M - J` is matrix-free
    and the solver defaults to GMRES. The model's actions see whole
    backend arrays, so this is single-process.

    Parameters
    ----------
    model : MatrixFreeModelProtocol
        Supplies :math:`f`, its mass and :math:`J v`, :math:`J^\top w`.
    nstates : int
        The number of states.
    solver : LinearSolverProtocol, optional
        A solver for matrix-free operators. Defaults to ``GMRES``.
    """
    bkd = model.bkd()
    descriptor = SpaceDescriptor("state", nstates, backend_dtype_name(bkd))
    space = BackendVectorSpace(descriptor, bkd)
    return SemiExplicitSystem(
        model,
        _MassAction(model.mass_matrix(), descriptor, bkd),
        MatrixFreeJacobianOperator(model, descriptor),
        space,
        GMRES(space, bkd) if solver is None else solver,
    )
