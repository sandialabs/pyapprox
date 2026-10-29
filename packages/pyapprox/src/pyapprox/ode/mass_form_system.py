r"""What a stepper of :math:`M \dot y = f(y, t; p)` in mass form needs.

A :class:`MassFormSystem` holds, as one value, the pieces a mass-form
Runge-Kutta stepper reads: the residual (:math:`f` and its derivatives
bundle), the mass :math:`M`, the Jacobian producer, the state space, a
linear solver and the residual form. The Jacobian producer is held
beside the bundle, typed with the solver's format ``F``, so a stage
operator :math:`M - \Delta t\, a_{ii} J` and its solver are paired
statically; the bundle's other fields are only applied, so their format
does not matter.

A singular mass (a differential-algebraic model) is fine for implicit
stages, which solve with :math:`M - \Delta t\, a_{ii} J`; explicit
stages and a final block solve with :math:`M` itself.

:func:`dense_mass_form_system` builds one for a model on the backend
stack whose Jacobian is a dense array.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.ode.mass_matrix_system import mass_properties
from pyapprox.ode.residual_derivatives import (
    JacobianFn,
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
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearSolverProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F")


@runtime_checkable
class RightHandSideProtocol(Protocol[V]):
    r""":math:`f(y, t; p)` and its derivatives bundle: all a mass-form
    stepper reads of the residual. States are in the residual's form."""

    def __call__(self, state: V, time: float) -> V: ...

    def derivatives(self) -> ODEResidualDerivatives[V, FormatProtocol]: ...


@dataclass(frozen=True)
class MassFormSystem(Generic[V, A, F]):
    r""":math:`M \dot y = f(y, t; p)` as a mass-form stepper reads it.

    Attributes
    ----------
    rhs : RightHandSideProtocol
        :math:`f` and its derivatives bundle, taking states in the
        residual's form.
    mass : ActionOperatorProtocol
        :math:`M`, in format ``F``.
    jacobian : JacobianFn
        ``(state, time, point) ->`` :math:`J = \partial f/\partial y` in
        format ``F``, at a state in the residual's form.
    space : InnerProductSpaceProtocol
        The state space; states are its single-column vectors.
    solver : LinearSolverProtocol
        Solves with :math:`M` and with :math:`M - \Delta t\, a_{ii} J`;
        pairs with ``F``.
    form : ResidualFormProtocol
        Converts the space's vectors to the residual's state form and
        back.
    """

    rhs: RightHandSideProtocol[V]
    mass: ActionOperatorProtocol[V, F]
    jacobian: JacobianFn[V, F]
    space: InnerProductSpaceProtocol[V, A]
    solver: LinearSolverProtocol[V, F]
    form: ResidualFormProtocol[V, V]

    def __post_init__(self) -> None:
        descriptor = self.space.descriptor()
        descriptor.require_same(self.mass.domain(), "domain of the mass")
        descriptor.require_same(self.mass.range(), "range of the mass")


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

    Raises
    ------
    TypeError
        If the model's bundle has no Jacobian.
    """
    bkd = model.bkd()
    mass = model.mass_matrix()
    matrix = mass.as_matrix()
    space: BackendVectorSpace[Array] = backend_state_space(bkd, matrix.shape[0])
    jacobian = model.derivatives().jacobian
    if jacobian is None:
        raise TypeError(
            f"{type(model).__name__}'s derivatives bundle has no jacobian; a "
            "mass-form stepper with implicit stages needs one"
        )
    return MassFormSystem(
        rhs=model,
        mass=BackendDenseOperator(
            matrix, space.descriptor(), bkd, properties=mass_properties(mass)
        ),
        jacobian=jacobian,
        space=space,
        solver=BackendDenseLU(bkd) if solver is None else solver,
        form=BackendColumnForm(bkd),
    )
