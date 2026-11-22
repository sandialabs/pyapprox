r"""A collocation system as :math:`M \dot u = f(u, t; p)` with no row
treated.

:func:`collocation_unconstrained_system` gives the mass-form system of a
collocation physics: its interior operator :math:`f` collocated at every
node, boundary nodes included, and the mass :math:`M = \mathrm{diag}(m)`
of the physics' nodal weights. Collocation's mass is diagonal because each
row is the strong form enforced at one node and the nodal basis is
cardinal there: no integral couples neighboring nodes or carries their
spacing. A weight is one for a state that evolves, zero for an algebraic
row (a continuity constraint), or a coefficient multiplying the time
derivative.

The essential rows are then treated by the shared row strategies
(``differentiated_rows`` with ``CoupledRows`` or ``SymmetricRows``,
``RowPinning``, ``PinnedRowsConstraint``), which read the system's
essential declaration directly; collocation's mass is one on those rows.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.right_hand_side import CollocationRightHandSide
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.time_order import refuse_second_order_in_time
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    LinearSolverProtocol,
    OperatorProperties,
    SpaceDescriptor,
)


@runtime_checkable
class TransientInteriorOperatorProtocol(
    CollocationInteriorOperatorProtocol[Array], Protocol, Generic[Array]
):
    r"""An interior operator that declares the nodal weights of its time
    derivative, :math:`\mathrm{diag}(m)\, \dot u = f(u, t)`."""

    def interior_mass_weights(self) -> Array:
        """The nodal weights; zero on an algebraic row. Shape: (nstates,)"""
        ...


def _mass_operator(
    weights: Array, descriptor: SpaceDescriptor, bkd: Backend[Array]
) -> BackendDenseOperator[Array]:
    r""":math:`\mathrm{diag}(m)`, declared diagonal, SPD when every weight
    is positive and the identity when every weight is one."""
    # Read once at construction to declare the operator's properties
    if bkd.to_bool(bkd.any(weights < 0.0)):
        raise ValueError(
            "mass weights must be nonnegative: a negative weight reverses "
            "time on its row, an ill-posed evolution"
        )
    positive = bkd.to_bool(bkd.all(weights > 0.0))
    identity = bkd.to_float(bkd.max(bkd.abs(weights - 1.0))) == 0.0
    properties = OperatorProperties(
        symmetric=True, spd=positive, diagonal=True, identity=identity
    )
    return BackendDenseOperator(
        bkd.diag(weights), descriptor, bkd, properties=properties
    )


def collocation_unconstrained_system(
    system: CollocationSystem[Array],
    parameters: Optional[ParameterState[Array]] = None,
    solver: Optional[LinearSolverProtocol[Array, DenseFormat[Array]]] = None,
) -> MassFormSystem[Array, Array, DenseFormat[Array]]:
    r"""The mass-form system :math:`M \dot u = f(u, t; p)` of a collocation
    system's physics, no row treated.

    The right-hand side is the raw interior operator: the system's
    essential and closure declarations are not applied here. The essential
    rows go through the shared row strategies; the closure rows need a
    step-end closure. The mass is fixed for the system's life.

    Parameters
    ----------
    system : CollocationSystem
        Supplies the physics; its declarations are left to the row
        treatments.
    parameters : ParameterState, optional
        The parameters, shared with every system built on them.
    solver : LinearSolverProtocol, optional
        Solves with :math:`M` and the stage operators; defaults to the
        backend's dense LU.

    Raises
    ------
    TypeError
        If the physics is second order in time (a first-order path would
        integrate a relaxation, not dynamics), or does not declare its
        mass weights.
    ValueError
        If the weights do not have one entry per state or a weight is
        negative.
    """
    physics = system.physics()
    refuse_second_order_in_time(physics)
    if not isinstance(physics, TransientInteriorOperatorProtocol):
        raise TypeError(
            "the physics must declare its mass weights "
            "(TransientInteriorOperatorProtocol), got "
            f"{type(physics).__name__}"
        )
    bkd, nstates = system.bkd(), system.nstates()
    weights = physics.interior_mass_weights()
    if tuple(weights.shape) != (nstates,):
        raise ValueError(
            f"mass weights must have shape ({nstates},), got {tuple(weights.shape)}"
        )
    space = backend_state_space(bkd, nstates)
    return MassFormSystem(
        rhs=CollocationRightHandSide(physics, parameters),
        mass=_mass_operator(weights, space.descriptor(), bkd),
        space=space,
        solver=BackendDenseLU(bkd) if solver is None else solver,
        form=BackendColumnForm(bkd),
    )
