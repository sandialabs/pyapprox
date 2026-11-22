r"""An skfem Galerkin subdomain as a Dirichlet-to-Neumann response, or
as a port-loaded one (Neumann-to-Dirichlet).

The subdomain's rows are tested equations, so the residual left on a
port row pinned to the port data is the boundary flux tested against
that row's basis function: the constraint reaction, which
:class:`~pyapprox.pde.coupling.pinned_response.PinnedReactionFlux` returns
as the port's output. For the same reason a tested flux given as port
data enters as a natural load on the port rows
(:func:`skfem_loaded_response`).

Bound to skfem and the SciPy stack (:func:`~pyapprox.pde.galerkin.
time_integration.skfem_discretization.skfem_discretization`): states are
NumPy columns and the Jacobian is CSR, so nothing built here is
differentiable through.
"""

from typing import Any, Optional, Sequence

from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import (
    RecomputeAlways,
    SetupStrategyProtocol,
    SteadyRequest,
)
from pyapprox.ode.differentiated_rows import CoupledRows, differentiated_rows
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.pinned_rows import (
    AtEveryBlock,
    AtStepEnd,
    PinPlacementProtocol,
    SplitRowPinning,
)
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.stepping_methods import RungeKutta, SteppingMethodProtocol
from pyapprox.ode.trajectory_store import TrajectorySinkProtocol
from pyapprox.pde.coupling.loaded_response import (
    PortRowsOutput,
    SteadyLoadedResponse,
)
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    PinnedReactionFlux,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.ports import BackendPortRows, PortMapProtocol
from pyapprox.pde.coupling.responses import Flux, PortQuantity, Trace
from pyapprox.pde.coupling.transient import (
    StageQuadratureFlux,
    StageReactions,
    StartContinuousLagrange,
    StepResidualReaction,
    TemporalTestFunctionsProtocol,
    TemporalTrialSpaceProtocol,
    WindowedResponse,
    WindowPortRows,
    runge_kutta_block_fractions,
)
from pyapprox.pde.galerkin.protocols.system import GalerkinTransientSystemProtocol
from pyapprox.pde.galerkin.time_integration.essential_pinned_rows import (
    EssentialPinnedRows,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.models.galerkin.unconstrained_system import (
    galerkin_unconstrained_system,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import CSRFormat, NumpyRowSet, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

# NumPy backend arrays: the dtype varies.
_Np = NDArray[Any]


def skfem_pinned_response(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    setups: SetupStrategyProtocol[SteadyRequest[_Np, CSRFormat], _Np, CSRFormat],
    nonlinear: NonlinearSolverProtocol[_Np, _Np],
    time: Optional[float] = None,
) -> SteadyPinnedResponse[_Np, _Np, CSRFormat]:
    """An skfem subdomain whose field DOFs ``ports`` are pinned to the
    port data, in that order, returning the reaction there.

    Parameters
    ----------
    system
        The composed skfem system. A port DOF must not be an essential
        row: wrap a condition covering one in
        :class:`~pyapprox.pde.boundary.except_dofs.ExceptDofs`.
    ports
        The port DOFs, in port order; for ``k`` components, node-major.
    setups
        Sets up the CSR Jacobian of the port-pinned residual, e.g.
        ``RecomputeAlways(ScipySparseLU())``.
    nonlinear
        Solves the subdomain problem.
    time
        The time the data are frozen at; omitted, the system must be
        declared time-invariant (:func:`~pyapprox.pde.models.galerkin.
        steady_residual.galerkin_steady_residual`).

    Raises
    ------
    ValueError
        If there are no port DOFs.
    """
    if len(ports) == 0:
        raise ValueError("an skfem response needs at least one port DOF")
    bkd = NumpyBkd()
    disc = skfem_discretization(system)
    states, theta = disc.space(), disc.parameter_space(len(ports))
    data: PortPinnedData[_Np, _Np] = PortPinnedData(
        NumpyRowSet(states.descriptor(), list(ports)),
        BackendPortRows(states.descriptor(), list(ports), bkd),
        BackendPortRows(theta.descriptor(), list(range(len(ports))), bkd),
        states,
        theta,
    )
    residual = galerkin_steady_residual(disc, time=time)
    return SteadyPinnedResponse(
        residual,
        data,
        PinnedReactionFlux(residual, data),
        states,
        setups,
        nonlinear,
        bkd,
    )


def skfem_windowed_response(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    method: SteppingMethodProtocol[_Np, _Np],
    setups: MassShiftStrategy[_Np, _Np, CSRFormat],
    init_state: _Np,
    sink: TrajectorySinkProtocol[_Np],
    origin: float = 0.0,
    nsubsteps: int = 1,
    port_placement: Optional[PinPlacementProtocol] = None,
    essential_placement: Optional[PinPlacementProtocol] = None,
) -> WindowedResponse[_Np, _Np]:
    """An skfem subdomain stepped through time windows, its field DOFs
    ``ports`` pinned to the port trace beside its essential rows, returning
    the time-discrete reaction there (see :mod:`~pyapprox.pde.coupling.
    transient`).

    The pinned rows are treated as :func:`~pyapprox.pde.models.galerkin.
    transient_system.galerkin_transient_system` treats essential rows:
    differentiated, with the coupling kept in the mass, pinned in the
    blocks each placement selects (:class:`~pyapprox.ode.pinned_rows.
    SplitRowPinning`), at each step's end by default.

    Parameters
    ----------
    system
        The composed skfem system. A port DOF must not be an essential
        row; the essential data must supply their rate.
    ports
        The port DOFs, in port order; for ``k`` components, node-major.
    method
        The time-stepping method, with its nonlinear solver when implicit.
    setups
        Sets up the stepper's linear solves, e.g.
        ``RecomputeAlways(ScipySparseLU())``.
    init_state
        The state at ``origin``, shape (nstates, 1); its essential and port
        entries are taken as given.
    sink
        Empty; receives the committed trajectory.
    origin
        The initial time.
    nsubsteps
        Equal steps per window.
    port_placement, essential_placement
        Which blocks of a step pin the port rows and the essential rows;
        :class:`~pyapprox.ode.pinned_rows.AtStepEnd` when omitted.

    Raises
    ------
    ValueError
        If there are no port DOFs, or as :class:`~pyapprox.pde.coupling.
        transient.WindowPortRows` and :class:`~pyapprox.pde.coupling.
        transient.WindowedResponse`.
    """
    unconstrained, rows, port_rows, pinning = _window_port_rows(
        system,
        ports,
        StartContinuousLagrange(),
        AtStepEnd() if port_placement is None else port_placement,
        AtStepEnd() if essential_placement is None else essential_placement,
    )
    solved = differentiated_rows(unconstrained, rows, CoupledRows())
    stepper = method.stepper(solved, setups, pinning)
    # Read for its residual and its Jacobian's action only: never solved.
    reaction = method.stepper(unconstrained, RecomputeAlways(ScipySparseLU()))
    return WindowedResponse(
        stepper,
        rows,
        StepResidualReaction(reaction, port_rows),
        init_state,
        origin,
        sink,
        NumpyBkd(),
        nsubsteps,
    )


def skfem_stagewise_response(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    method: RungeKutta[_Np, _Np],
    setups: MassShiftStrategy[_Np, _Np, CSRFormat],
    init_state: _Np,
    sink: TrajectorySinkProtocol[_Np],
    origin: float = 0.0,
    essential_placement: Optional[PinPlacementProtocol] = None,
) -> WindowedResponse[_Np, _Np]:
    """An skfem subdomain coupled stage by stage: one step per window,
    the port data one trace per block of ``method``'s step
    (:func:`~pyapprox.pde.coupling.transient.runge_kutta_block_fractions`),
    every block's port rows pinned to its own, the output each block's
    reaction (see :class:`~pyapprox.pde.coupling.transient.StageReactions`).
    Subdomains sharing the method then step as the single domain does, at
    the method's order.

    The essential rows are pinned at the step's end by default, as the
    single-domain step pins them, and solved with their data's rate in the
    other blocks. Parameters are as :func:`skfem_windowed_response`.

    Parameters
    ----------
    essential_placement
        Which blocks pin the essential rows;
        :class:`~pyapprox.ode.pinned_rows.AtStepEnd` when omitted.
        :class:`~pyapprox.ode.pinned_rows.AtEveryBlock` is another scheme,
        equal to the single-domain step only for time-invariant essential
        data.

    Raises
    ------
    ValueError
        If there are no port DOFs, or as :func:`~pyapprox.pde.coupling.
        transient.runge_kutta_block_fractions`.
    """
    fractions = runge_kutta_block_fractions(method.tableau())
    unconstrained, rows, port_rows, pinning = _window_port_rows(
        system,
        ports,
        StartContinuousLagrange(fractions),
        AtEveryBlock(),
        AtStepEnd() if essential_placement is None else essential_placement,
    )
    solved = differentiated_rows(unconstrained, rows, CoupledRows())
    stepper = method.stepper(solved, setups, pinning)
    # Read for its residual and its Jacobian's action only: never solved.
    reaction = method.stepper(unconstrained, RecomputeAlways(ScipySparseLU()))
    bkd = NumpyBkd()
    return WindowedResponse(
        stepper,
        rows,
        StageReactions(reaction, port_rows, bkd),
        init_state,
        origin,
        sink,
        bkd,
    )


def skfem_spacetime_response(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    method: RungeKutta[_Np, _Np],
    setups: MassShiftStrategy[_Np, _Np, CSRFormat],
    init_state: _Np,
    sink: TrajectorySinkProtocol[_Np],
    trial: TemporalTrialSpaceProtocol,
    test: TemporalTestFunctionsProtocol,
    origin: float = 0.0,
    nsubsteps: int = 1,
    port_placement: Optional[PinPlacementProtocol] = None,
    essential_placement: Optional[PinPlacementProtocol] = None,
) -> WindowedResponse[_Np, _Np]:
    """An skfem subdomain coupled through a space in time per window: the
    port rows pinned to the port data's function of time in ``trial``,
    ``method`` stepping ``nsubsteps`` equal steps inside the window, the
    output the port flux tested against ``test`` by the method's stage
    quadrature (:class:`~pyapprox.pde.coupling.transient.
    StageQuadratureFlux`). Subdomains may differ in method and step.

    Parameters are as :func:`skfem_windowed_response`, and:

    Parameters
    ----------
    trial
        The port data's space in time.
    test
        The test functions; as many as ``trial`` has unknowns.
    port_placement
        Which blocks pin the port rows;
        :class:`~pyapprox.ode.pinned_rows.AtEveryBlock` when omitted, so
        every stage reads the port data at its own time.
    essential_placement
        Which blocks pin the essential rows;
        :class:`~pyapprox.ode.pinned_rows.AtStepEnd` when omitted.

    Raises
    ------
    TypeError
        As :class:`~pyapprox.pde.coupling.transient.WindowPortRows` and
        :class:`~pyapprox.pde.coupling.transient.StageQuadratureFlux`.
    ValueError
        If there are no port DOFs, ``test`` and ``trial`` differ in count,
        or as :class:`~pyapprox.pde.coupling.transient.
        StageQuadratureFlux` and :class:`~pyapprox.pde.coupling.transient.
        WindowedResponse`.
    """
    unconstrained, rows, port_rows, pinning = _window_port_rows(
        system,
        ports,
        trial,
        AtEveryBlock() if port_placement is None else port_placement,
        AtStepEnd() if essential_placement is None else essential_placement,
    )
    # Read for its residual and its Jacobian's action only: never solved.
    reaction = method.stepper(unconstrained, RecomputeAlways(ScipySparseLU()))
    bkd = NumpyBkd()
    flux = StageQuadratureFlux(reaction, method.tableau(), port_rows, test, bkd)
    if flux.nfunctions() != rows.trial().nunknowns():
        raise ValueError(
            f"{flux.nfunctions()} test functions for {rows.trial().nunknowns()} "
            "unknown traces per port row: the skeleton residual would not be "
            "square"
        )
    solved = differentiated_rows(unconstrained, rows, CoupledRows())
    stepper = method.stepper(solved, setups, pinning)
    return WindowedResponse(
        stepper,
        rows,
        flux,
        init_state,
        origin,
        sink,
        bkd,
        nsubsteps,
    )


def _window_port_rows(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    trial: TemporalTrialSpaceProtocol,
    port_placement: PinPlacementProtocol,
    essential_placement: PinPlacementProtocol,
) -> tuple[
    MassFormSystem[_Np, _Np, CSRFormat],
    WindowPortRows[_Np, _Np],
    BackendPortRows[_Np],
    SplitRowPinning[_Np],
]:
    """The unconstrained system of ``system``, its essential and port rows
    with port data in ``trial`` over a window, the port rows, and the
    pin rule placing the port and the essential rows.

    Raises
    ------
    ValueError
        If there are no port DOFs.
    """
    if len(ports) == 0:
        raise ValueError("an skfem response needs at least one port DOF")
    bkd = NumpyBkd()
    disc = skfem_discretization(system)
    unconstrained = galerkin_unconstrained_system(disc, ScipySparseLU())
    states = unconstrained.space
    descriptor = states.descriptor()
    dofs = [bkd.to_int(d) for d in system.constraint_set().dofs()]
    port_rows = BackendPortRows(descriptor, list(ports), bkd)
    essential = EssentialPinnedRows(disc)
    rows = WindowPortRows(
        essential,
        NumpyRowSet(descriptor, dofs + list(ports)),
        port_rows,
        states,
        trial,
    )
    pinning = SplitRowPinning(
        rows,
        NumpyRowSet(descriptor, list(ports)),
        essential.rows(),
        port_placement,
        essential_placement,
    )
    return unconstrained, rows, port_rows, pinning


def skfem_loaded_response(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    setups: SetupStrategyProtocol[SteadyRequest[_Np, CSRFormat], _Np, CSRFormat],
    nonlinear: NonlinearSolverProtocol[_Np, _Np],
    load: Optional[PortMapProtocol[_Np]] = None,
    output: Optional[PortMapProtocol[_Np]] = None,
    receives: PortQuantity = Flux(),
    returns: PortQuantity = Trace(),
    time: Optional[float] = None,
) -> SteadyLoadedResponse[_Np, _Np, CSRFormat]:
    """An skfem subdomain whose port data enter as a natural load on its
    field DOFs ``ports``, in that order, returning the state there (see
    :mod:`~pyapprox.pde.coupling.loaded_response`).

    Parameters
    ----------
    system
        The composed skfem system. A port DOF must not be an essential
        row.
    ports
        The port DOFs, in port order; for ``k`` components, node-major.
    setups
        Sets up the CSR Jacobian, e.g. ``RecomputeAlways(ScipySparseLU())``.
    nonlinear
        Solves the subdomain problem.
    load
        Maps the port data to the load on the port rows; omitted, the
        identity: the data are a flux tested against the port DOFs' basis
        functions, in Neumann form (outward).
    output
        Maps the port DOFs' values to the output; omitted, the identity:
        the trace.
    receives, returns
        The kinds of the port data and the output; a flux in and a trace
        out by default.
    time
        As :func:`skfem_pinned_response`.

    Raises
    ------
    ValueError
        If there are no port DOFs, or as :class:`~pyapprox.pde.coupling.
        loaded_response.PortLoadResidual`.
    """
    if len(ports) == 0:
        raise ValueError("an skfem response needs at least one port DOF")
    bkd = NumpyBkd()
    disc = skfem_discretization(system)
    states, theta = disc.space(), disc.parameter_space(len(ports))
    data: PortPinnedData[_Np, _Np] = PortPinnedData(
        NumpyRowSet(states.descriptor(), list(ports)),
        BackendPortRows(states.descriptor(), list(ports), bkd),
        BackendPortRows(theta.descriptor(), list(range(len(ports))), bkd),
        states,
        theta,
    )
    residual = galerkin_steady_residual(disc, time=time)
    return SteadyLoadedResponse(
        residual,
        data,
        PortRowsOutput(data, output),
        states,
        setups,
        nonlinear,
        bkd,
        load,
        receives,
        returns,
    )
