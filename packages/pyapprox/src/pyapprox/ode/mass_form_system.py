r"""What a stepper of :math:`M \dot y = f(y, t; p)` in mass form needs.

A :class:`MassFormSystem` holds, as one value, the pieces a mass-form
stepper reads: the residual (:math:`f` and its derivatives bundle), the
mass :math:`M`, the state space, a linear solver and the residual form.
The Jacobian is the residual's own, read from its bundle whenever it is
needed, so the system holds no second copy that could disagree with it.
The residual is typed with the solver's format ``F``, so a stage operator
:math:`M - \Delta t\, a_{ii} J` and its solver are paired statically.

A singular mass (a differential-algebraic model) is fine for implicit
stages, which solve with :math:`M - \Delta t\, a_{ii} J`; explicit
stages and a final block solve with :math:`M` itself.

:func:`dense_mass_form_system` builds one for a model on the backend
stack whose Jacobian is a dense array.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.ode.mass_matrix_system import mass_properties
from pyapprox.ode.residual_derivatives import (
    Jacobian,
    ODEResidualDerivatives,
    backend_state_space,
)
from pyapprox.ode.residual_form import BackendColumnForm, ResidualFormProtocol
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    InnerProductSpaceProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearSolverProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F")
F_co = TypeVar("F_co", covariant=True)


@runtime_checkable
class RightHandSideProtocol(Protocol[V, F_co]):
    r""":math:`f(y, t; p)` and its derivatives bundle, whose Jacobian is in
    format ``F_co``: all a mass-form stepper reads of the residual. States
    are in the residual's form."""

    def __call__(self, state: V, time: float) -> V: ...

    def derivatives(self) -> ODEResidualDerivatives[V, F_co]: ...


@runtime_checkable
class VersionedRightHandSideProtocol(RightHandSideProtocol[V, F_co], Protocol):
    """A right-hand side that reports its parameter version: the number of
    ``set_param`` calls so far (see
    :mod:`pyapprox.ode.protocols.differentiable_residual`)."""

    def parameters_version(self) -> int: ...


def _unversioned() -> int:
    """The version of a right-hand side without parameters: one for its
    life."""
    return 0


def parameters_version_reader(rhs: RightHandSideProtocol[V, F_co]) -> Callable[[], int]:
    """How to read ``rhs``'s parameter version: its own
    ``parameters_version``, or a constant 0 for a right-hand side without
    parameters. Whether it reports one is fixed for its life, so a consumer
    decides once, at construction."""
    if isinstance(rhs, VersionedRightHandSideProtocol):
        return rhs.parameters_version
    return _unversioned


@dataclass(frozen=True)
class MassFormSystem(Generic[V, A, F]):
    r""":math:`M \dot y = f(y, t; p)` as a mass-form stepper reads it.

    Attributes
    ----------
    rhs : RightHandSideProtocol
        :math:`f` and its derivatives bundle, taking states in the
        residual's form; the bundle's Jacobian, in format ``F``, is the
        system's (:meth:`jacobian`).
    mass : ActionOperatorProtocol
        :math:`M`, in format ``F``.
    space : InnerProductSpaceProtocol
        The state space; states are its single-column vectors.
    solver : LinearSolverProtocol
        Solves with :math:`M` and with :math:`M - \Delta t\, a_{ii} J`;
        pairs with ``F``.
    form : ResidualFormProtocol
        Converts the space's vectors to the residual's state form and
        back.

    A bundle without a Jacobian is accepted: an explicit method steps
    forward solving with :math:`M` alone. The consumers that need
    :math:`J` (an implicit method, the derivative sweeps) refuse such a
    system themselves; the Jacobian-reading methods here raise
    ``TypeError`` if called on one.
    """

    rhs: RightHandSideProtocol[V, F]
    mass: ActionOperatorProtocol[V, F]
    space: InnerProductSpaceProtocol[V, A]
    solver: LinearSolverProtocol[V, F]
    form: ResidualFormProtocol[V, V]

    def __post_init__(self) -> None:
        descriptor = self.space.descriptor()
        descriptor.require_same(self.mass.domain(), "domain of the mass")
        descriptor.require_same(self.mass.range(), "range of the mass")

    def jacobian(
        self, state: V, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, F]:
        r""":math:`J = \partial f/\partial y` in format ``F`` at ``state``
        (in the residual's form) and ``time``, labeled ``point``: the
        residual bundle's own Jacobian, read from it on every call."""
        return self._jacobian()(state, time, point)

    def is_affine(self) -> bool:
        """Whether :math:`f` is affine in the state: the residual's
        Jacobian cannot see it (see
        :class:`~pyapprox.ode.residual_derivatives.Jacobian`), so a stage
        equation is one linear solve."""
        return self._jacobian().affine()

    def jacobian_time_invariant(self) -> bool:
        """Whether the residual's Jacobian sees neither state nor time, so
        a stage operator is the same at every step of a given size."""
        return self._jacobian().time_invariant()

    def _jacobian(self) -> Jacobian[V, F]:
        jacobian = self.rhs.derivatives().jacobian
        if jacobian is None:
            raise TypeError(
                f"{type(self.rhs).__name__}'s derivatives bundle has no "
                "jacobian; only an explicit forward solve works without one"
            )
        return jacobian


@runtime_checkable
class DenseModelProtocol(Protocol[Array]):
    """A backend model with a mass and a bundle whose Jacobian is dense."""

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array, time: float) -> Array: ...

    def mass_matrix(self) -> MassMatrixProtocol[Array]: ...

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]: ...


def dense_mass_form_system(
    model: DenseModelProtocol[Array],
    solver: Optional[LinearSolverProtocol[Array, DenseFormat[Array]]] = None,
) -> MassFormSystem[Array, Array, DenseFormat[Array]]:
    """The mass-form system of a backend model with a dense Jacobian.

    Parameters
    ----------
    model : DenseModelProtocol
        Supplies :math:`f`, its mass and a bundle with a dense Jacobian.
    solver : LinearSolverProtocol, optional
        Defaults to the backend's dense LU.
    """
    bkd = model.bkd()
    mass = model.mass_matrix()
    matrix = mass.as_matrix()
    space: BackendVectorSpace[Array] = backend_state_space(bkd, matrix.shape[0])
    return MassFormSystem(
        rhs=model,
        mass=BackendDenseOperator(
            matrix, space.descriptor(), bkd, properties=mass_properties(mass)
        ),
        space=space,
        solver=BackendDenseLU(bkd) if solver is None else solver,
        form=BackendColumnForm(bkd),
    )
