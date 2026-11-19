r"""A transient Galerkin system built for the tableau and multistep steppers.

:func:`galerkin_transient_system` builds, from a discretization, everything
a transient solve and its sweeps read: the solved system, whose essential
rows carry the rate of their data (:func:`~pyapprox.ode.differentiated_rows.
differentiated_rows` with :class:`~pyapprox.ode.differentiated_rows.
CoupledRows`) and are pinned at each step's end; the unconstrained system,
on which a quantity of the essential rows (the boundary flux) is read; and
the shared parameter state. :func:`solve_galerkin_transient` steps it with a
method the caller chooses, in the shape of
:func:`~pyapprox.pde.models.galerkin.steady_residual.solve_galerkin_steady`:
every choice is injected and nothing is kept between calls.
:func:`galerkin_transient_derivatives` gives the tangent and adjoint sweeps
of the stepper a solve used.
"""

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from pyapprox.ode.differentiated_rows import CoupledRows, differentiated_rows
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.pinned_rows import RowPinning
from pyapprox.ode.protocols.differentiable_stepper import StepperProtocol
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.step_plan import StepPlanProtocol
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_store import TrajectorySinkProtocol
from pyapprox.pde.boundary.protocols import ConstraintSetProtocol
from pyapprox.pde.galerkin.time_integration.essential_pinned_rows import (
    EssentialPinnedRows,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.pde.models.galerkin.unconstrained_system import (
    galerkin_unconstrained_system,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import FormatProtocol, LinearSolverProtocol

F = TypeVar("F", bound=FormatProtocol)


@dataclass(frozen=True)
class GalerkinTransientSystem(Generic[Array, F]):
    """What a transient Galerkin solve and its sweeps read.

    Built by :func:`galerkin_transient_system`; the fields share one
    discretization and one parameter state.

    Attributes
    ----------
    solved : MassFormSystem
        The system stepped: the essential rows in differentiated form,
        the coupling kept in the mass.
    unconstrained : MassFormSystem
        The system with no row treated, for quantities of the essential
        rows; not solved.
    essential_rows : EssentialPinnedRows
        The essential rows and their data, as the pin and quantities read
        on those rows (the boundary flux) declare them.
    row_pinning : RowPinning
        Pins the essential rows to their data at each step's end.
    constraints : ConstraintSetProtocol
        The essential constraints, which set the initial state's
        essential entries.
    parameters : ParameterState or None
        The shared parameters; none for a system without them.
    """

    solved: MassFormSystem[Array, Array, F]
    unconstrained: MassFormSystem[Array, Array, F]
    essential_rows: EssentialPinnedRows[Array]
    row_pinning: RowPinning[Array]
    constraints: ConstraintSetProtocol[Array]
    parameters: Optional[ParameterState[Array]]


def galerkin_transient_system(
    discretization: GalerkinStackDiscretization[Array, Array, F],
    solver: LinearSolverProtocol[Array, F],
    parameters: Optional[ParameterState[Array]] = None,
) -> GalerkinTransientSystem[Array, F]:
    r"""The transient system of ``discretization``.

    The essential rows are pinned at each step's end, with the coupling
    columns kept in the mass, so a one-step method whose step result is
    solved conserves :math:`1^\top M u` exactly. A stored-state constraint
    in place of the pin (a projection, the symmetric mass) is composed by
    hand from the same parts.

    Parameters
    ----------
    discretization : GalerkinStackDiscretization
        The system on a stack.
    solver : LinearSolverProtocol
        Solves with the mass and the stage operators; pairs with the
        stack's format.
    parameters : ParameterState, optional
        The parameters; none for a system without them.

    Raises
    ------
    TypeError
        If ``parameters`` is not a ``ParameterState``.
    ValueError
        If the parameterization writes objects the system does not hold,
        or the essential data supplies no rate.
    """
    unconstrained = galerkin_unconstrained_system(discretization, solver, parameters)
    rows = EssentialPinnedRows(discretization)
    return GalerkinTransientSystem(
        solved=differentiated_rows(unconstrained, rows, CoupledRows()),
        unconstrained=unconstrained,
        essential_rows=rows,
        row_pinning=RowPinning(rows),
        constraints=discretization.system().constraint_set(),
        parameters=parameters,
    )


def solve_galerkin_transient(
    system: GalerkinTransientSystem[Array, F],
    method: SteppingMethodProtocol[Array, Array],
    setups: MassShiftStrategy[Array, Array, F],
    steps: StepPlanProtocol,
    init_state: Array,
    sink: TrajectorySinkProtocol[Array],
) -> StepperProtocol[Array, Array]:
    """Step ``system`` through ``steps`` from ``init_state``.

    The essential entries of ``init_state`` are set to their data at the
    plan's origin first. The parameters are those currently set on
    ``system.parameters``.

    Parameters
    ----------
    system : GalerkinTransientSystem
        From :func:`galerkin_transient_system`.
    method : SteppingMethodProtocol
        The time-stepping method, with its nonlinear solver when implicit.
    setups : MassShiftStrategy
        Sets up the stepper's linear solves; its solver pairs with the
        system's. The caller owns it: clear it when the parameters change.
    steps : StepPlanProtocol
        The steps, from the time of the initial state.
    init_state : Array
        Shape ``(nstates, 1)``.
    sink : TrajectorySinkProtocol
        Receives the stored states: a store, a functional's sink, or
        several through a multi-sink.

    Returns
    -------
    StepperProtocol
        The stepper the solve used, for
        :func:`galerkin_transient_derivatives`.

    Raises
    ------
    TypeError
        If ``method`` does not satisfy ``SteppingMethodProtocol``.
    ValueError
        If ``init_state`` is not of shape ``(nstates, 1)``.
    """
    if not isinstance(method, SteppingMethodProtocol):
        raise TypeError(
            f"method must satisfy SteppingMethodProtocol, got {type(method).__name__}"
        )
    nstates = system.solved.space.descriptor().size
    if tuple(init_state.shape) != (nstates, 1):
        raise ValueError(
            f"init_state must have shape ({nstates}, 1), got {tuple(init_state.shape)}"
        )
    bkd = system.constraints.bkd()
    start = bkd.reshape(
        system.constraints.inject(init_state[:, 0], steps.origin()), (-1, 1)
    )
    stepper = method.stepper(system.solved, setups, system.row_pinning)
    Integrator(stepper).solve(start, steps, sink)
    return stepper


def galerkin_transient_derivatives(
    system: GalerkinTransientSystem[Array, F],
    stepper: StepperProtocol[Array, Array],
) -> TrajectoryDerivatives[Array, Array]:
    """The tangent and adjoint sweeps of ``stepper``, the one a
    :func:`solve_galerkin_transient` of ``system`` returned.

    The initial state's parameter derivatives are the parameterization's;
    its essential entries are data, so they do not depend on the
    parameters.
    """
    return TrajectoryDerivatives(
        stepper, None, ResidualInitialState(system.solved.rhs, system.solved.form)
    )
