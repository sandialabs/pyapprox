r"""An skfem subdomain stepped through time windows (backward Euler).

The problem: :math:`u_t = \Delta u - u + 1 + x + 2y` on the unit square,
u = 1 on the left, u = 2 on the right, natural conditions above and
below, :math:`u_0 = 1 + x + \sin(\pi x)\sin(\pi y)`; split at x = 1/2
into two Q1 subdomains meshed as the single domain is, Δt = 0.01.

Each window takes its end trace from the single-domain run, so each
subdomain must reproduce that run's states, and the two reactions must
cancel on the interface (each is the subdomain's share of the
single-domain step residual there). With a steady state as initial state
and its trace held, the reaction over a window is Δt times the steady
reaction: the sign and scale of the steady flux.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.boundary.except_dofs import ExceptDofs
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeIndependent,
)
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.stacks.galerkin import (
    skfem_stagewise_response,
    skfem_windowed_response,
)
from pyapprox.pde.coupling.transient import (
    WindowedPoint,
    WindowedResponse,
    WindowedResponseProtocol,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.transient_system import (
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_skfem import (
    STEADY,
    CoupledProblem,
    Np,
    TensorQuadMesh,
    Value,
    box_problem,
    node_at,
)

_BKD = NumpyBkd()
_DT, _NSTEPS = 0.01, 3
_NONE = _BKD.zeros((0, 1))


def _forcing(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


def _sink(x: Np, u: Np) -> Np:
    return -u


def _sink_derivative(x: Np, u: Np) -> Np:
    return -np.ones_like(u)


def _initial(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))


def _build(
    mesh: TensorQuadMesh, sides: Sequence[tuple[str, Value]], exempt: set[int]
) -> Any:
    bkd = mesh.bkd()
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=ConstantDiffusion(1.0),
        bkd=bkd,
        reaction=CallableReaction(_sink, _sink_derivative),
        forcing=TimeIndependent(_forcing),
    )
    conditions = []
    for side, v in sides:
        bc = DirichletBC(
            basis, side, v if isinstance(v, float) else TimeIndependent(v), bkd
        )
        on_side = exempt & {int(d) for d in bc.constrained_dofs()}
        conditions.append(ExceptDofs(bc, sorted(on_side)))
    return compose_galerkin_system(physics, conditions)


def _newton() -> Newton[Np, Np]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=20)


def _method(
    tableau: Callable[[], ButcherTableau] = backward_euler,
) -> RungeKutta[Np, Np]:
    made = tableau()
    return RungeKutta(made, None if made.is_explicit() else _newton())


class _Strip:
    """The split problem, the single-domain run and a windowed response
    per half."""

    def __init__(self, nsteps: int = _NSTEPS, dt: float = _DT) -> None:
        self.problem: CoupledProblem = box_problem(
            (2, 1), _build, [("left", 1.0), ("right", 2.0)]
        )
        whole, mesh = self.problem.single_domain_system()
        self.nodes = mesh.nodes()
        system = galerkin_transient_system(skfem_discretization(whole), ScipySparseLU())
        self.reference: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
        solve_galerkin_transient(
            system,
            _method(),
            RecomputeAlways(ScipySparseLU()),
            FixedStep(dt, (0.0, nsteps * dt)),
            np.reshape(_initial(mesh.nodes()), (-1, 1)),
            self.reference,
        )

    def half(
        self,
        subdomain: int,
        init: Any = None,
        nsubsteps: int = 1,
        tableau: Callable[[], ButcherTableau] = backward_euler,
    ) -> "_Half":
        """The windowed response of ``subdomain`` stepped by ``tableau``."""
        system, nodes, ports, dofs = self.problem.port_system(subdomain)
        rows = [node_at(self.nodes, nodes[:, j]) for j in range(nodes.shape[1])]
        state = self.reference.state(0)[rows] if init is None else init[rows]
        sink: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
        response = skfem_windowed_response(
            system,
            ports,
            _method(tableau),
            RecomputeAlways(ScipySparseLU()),
            state,
            sink,
            nsubsteps=nsubsteps,
        )
        return _Half(response, rows, ports, dofs, sink)

    def trace(self, index: int, half: "_Half") -> Np:
        """The reference trace on ``half``'s ports at state ``index``."""
        state: Np = self.reference.state(index)[half.rows]
        return state[half.ports]


@dataclass
class _Half:
    """A windowed response, the reference rows of its nodes, its port
    DOFs, its skeleton DOFs and its sink."""

    response: WindowedResponse[Np, Np]
    rows: list[int]
    ports: list[int]
    dofs: tuple[int, ...]
    sink: InMemoryTrajectoryStore[Np]


def _step_all(strip: _Strip, halves: list[_Half], end: int, window: TimeWindow) -> Np:
    """Linearize every half at the reference end trace, accept, and return
    the reactions summed on the skeleton."""
    balance = np.zeros((strip.problem.space.descriptor().size, 1))
    for half in halves:
        point = half.response.linearize(strip.trace(end, half), _NONE, window)
        np.add.at(balance, list(half.dofs), point.output())
        half.response.accept(point)
        _BKD.assert_allclose(
            half.response.state(), strip.reference.state(end)[half.rows], atol=1e-12
        )
        assert half.response.time() == window.end
    return balance


def test_windows_reproduce_the_single_domain_run() -> None:
    strip = _Strip()
    halves = [strip.half(s) for s in range(2)]
    for half in halves:
        assert isinstance(half.response, WindowedResponseProtocol)
        half.response.start()
    for n in range(_NSTEPS):
        balance = _step_all(strip, halves, n + 1, TimeWindow(n * _DT, (n + 1) * _DT))
        # Each reaction alone is of order 1e-3.
        _BKD.assert_allclose(balance, np.zeros_like(balance), atol=1e-12)
    for half in halves:
        sink = half.sink
        assert sink.ntimes() == _NSTEPS + 1 and sink.nsteps() == _NSTEPS
        _BKD.assert_allclose(
            np.array(sink.times()), np.arange(_NSTEPS + 1) * _DT, atol=1e-15
        )
        # Backward Euler's unknown is the new state.
        _BKD.assert_allclose(sink.window(-1).unknown.blocks[-1], sink.state(-1))
        _BKD.assert_allclose(
            sink.state(-1), strip.reference.state(-1)[half.rows], atol=1e-12
        )


def test_substeps_read_the_interpolated_trace() -> None:
    """With two substeps per window the middle state's port rows are the
    mean of the window's end traces; a held steady trace sums to the
    steady reaction over the window; the sink gets both steps."""
    strip = _Strip()
    half = strip.half(0, nsubsteps=2)
    response = half.response
    response.start()
    start, end = strip.trace(0, half), strip.trace(1, half)
    point = response.linearize(end, _NONE, TimeWindow(0.0, _DT))
    middle: Np = point.trial().state(1)
    _BKD.assert_allclose(middle[half.ports], 0.5 * (start + end), atol=1e-14)
    response.accept(point)
    assert half.sink.nsteps() == 2 and half.sink.time(1) == 0.5 * _DT
    steady, _ = strip.problem.single_domain()
    held = strip.half(0, init=steady, nsubsteps=2)
    trace = steady[held.rows][held.ports]
    output = held.response.linearize(trace, _NONE, TimeWindow(0.0, _DT)).output()
    piece, _, _ = strip.problem.piece(0)
    steady_output = piece.linearize(trace, _NONE, STEADY).output()
    _BKD.assert_allclose(output, _DT * steady_output, rtol=1e-9, atol=1e-13)


def test_held_steady_trace_gives_the_steady_reaction() -> None:
    strip = _Strip()
    steady, _ = strip.problem.single_domain()
    half = strip.half(0, init=steady)
    trace = steady[half.rows][half.ports]
    half.response.start()
    point = half.response.linearize(trace, _NONE, TimeWindow(0.0, _DT))
    _BKD.assert_allclose(point.state(), steady[half.rows], atol=1e-11)
    piece, _, _ = strip.problem.piece(0)
    steady_output = piece.linearize(trace, _NONE, STEADY).output()
    _BKD.assert_allclose(point.output(), _DT * steady_output, rtol=1e-9, atol=1e-13)


def test_an_unaccepted_window_changes_nothing() -> None:
    strip = _Strip()
    half = strip.half(0)
    response, sink = half.response, half.sink
    response.start()
    window = TimeWindow(0.0, _DT)
    before = response.state()
    wrong = response.linearize(strip.trace(1, half) + 0.3, _NONE, window)
    assert response.time() == 0.0 and response.state() is before
    assert sink.ntimes() == 1
    wrong_output = wrong.output()
    right = response.linearize(strip.trace(1, half), _NONE, window)
    # A point holds its own trial: a later solve leaves its output alone.
    assert wrong.output() is wrong_output
    assert not np.allclose(wrong_output, right.output())
    response.accept(right)
    _BKD.assert_allclose(
        response.state(), strip.reference.state(1)[half.rows], atol=1e-12
    )
    assert sink.ntimes() == 2
    with pytest.raises(ValueError, match="committed time"):
        response.accept(wrong)
    with pytest.raises(ValueError, match="committed time"):
        response.linearize(strip.trace(1, half), _NONE, window)


@pytest.mark.parametrize("nsubsteps", [1, 3])
@pytest.mark.parametrize(
    "tableau", [backward_euler, heun, implicit_midpoint, alexander_sdirk3]
)
def test_jvp_trace_matches_finite_differences(
    tableau: Callable[[], ButcherTableau], nsubsteps: int
) -> None:
    """For any tableau (one block, explicit stages, a final block apart
    from the last stage, a stiffly accurate three-stage method), with one
    step per window or three, the later steps reading the changed states
    through the forward tangent sweep."""
    strip = _Strip()
    half = strip.half(1, tableau=tableau, nsubsteps=nsubsteps)
    response = half.response
    n = response.nport()
    window = TimeWindow(0.0, _DT)

    def fun(samples: Np) -> Np:
        return np.hstack(
            [
                response.linearize(samples[:, j : j + 1], _NONE, window).output()
                for j in range(samples.shape[1])
            ]
        )

    def jvp(sample: Np, vec: Np) -> Np:
        tangent = response.linearize(sample, _NONE, window).derivatives().jvp_trace
        assert tangent is not None
        return tangent(vec)

    sample = strip.trace(1, half)
    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=n, nvars=n, fun=fun, jvp=jvp, bkd=_BKD)
    ).check_derivatives(sample, relative=True, central=True)[0]
    assert _BKD.to_float(_BKD.min(errors)) <= 1e-6
    tangent = response.linearize(sample, _NONE, window).derivatives().jvp_trace
    assert tangent is not None
    block = np.random.default_rng(0).normal(size=(n, 3))
    columns = np.hstack([tangent(block[:, j : j + 1]) for j in range(3)])
    _BKD.assert_allclose(tangent(block), columns, rtol=1e-12)
    with pytest.raises(ValueError):
        tangent(np.zeros((n + 1, 1)))


def test_stagewise_jvp_trace_matches_finite_differences() -> None:
    """SDIRK3 coupled stage by stage: three traces per port, every block
    pinned; the trace tangent passes the derivative checker."""
    strip = _Strip()
    system, nodes, ports, _ = strip.problem.port_system(1)
    rows = [node_at(strip.nodes, nodes[:, j]) for j in range(nodes.shape[1])]
    response = skfem_stagewise_response(
        system,
        ports,
        _method(alexander_sdirk3),
        RecomputeAlways(ScipySparseLU()),
        strip.reference.state(0)[rows],
        InMemoryTrajectoryStore(),
    )
    n = response.nport()
    assert n == 3 * len(ports)
    window = TimeWindow(0.0, _DT)

    def fun(samples: Np) -> Np:
        return np.hstack(
            [
                response.linearize(samples[:, j : j + 1], _NONE, window).output()
                for j in range(samples.shape[1])
            ]
        )

    def jvp(sample: Np, vec: Np) -> Np:
        tangent = response.linearize(sample, _NONE, window).derivatives().jvp_trace
        assert tangent is not None
        return tangent(vec)

    end: Np = strip.reference.state(1)[rows][ports]
    sample = np.repeat(end, 3, axis=0)
    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=n, nvars=n, fun=fun, jvp=jvp, bkd=_BKD)
    ).check_derivatives(sample, relative=True, central=True)[0]
    assert _BKD.to_float(_BKD.min(errors)) <= 1e-6


def test_invalid_use() -> None:
    strip = _Strip()
    half, other = strip.half(0), strip.half(1)
    response = half.response
    trace = strip.trace(1, half)
    window = TimeWindow(0.0, _DT)
    point = response.linearize(trace, _NONE, window)
    assert isinstance(point, WindowedPoint)
    with pytest.raises(ValueError, match="start"):
        response.accept(point)
    response.start()
    with pytest.raises(ValueError, match="already started"):
        response.start()
    other.response.start()
    with pytest.raises(ValueError, match="not one of this response's"):
        other.response.accept(point)
    with pytest.raises(ValueError, match="nonempty window"):
        response.linearize(trace, _NONE, STEADY)
    with pytest.raises(ValueError, match="committed time"):
        response.linearize(trace, _NONE, TimeWindow(_DT, 2 * _DT))
    with pytest.raises(ValueError, match="port_data"):
        response.linearize(trace[:-1], _NONE, window)
    with pytest.raises(ValueError, match="params"):
        response.linearize(trace, np.zeros((1, 1)), window)
    system, _, ports, _ = strip.problem.port_system(0)
    full: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
    full.append(0.0, response.state())
    for bad_ports, sink, nsubsteps, match in (
        (ports, full, 1, "sink must be empty"),
        (ports, InMemoryTrajectoryStore(), 0, "nsubsteps"),
        ([], InMemoryTrajectoryStore(), 1, "at least one port"),
    ):
        with pytest.raises(ValueError, match=match):
            skfem_windowed_response(
                system,
                bad_ports,
                _method(),
                RecomputeAlways(ScipySparseLU()),
                response.state(),
                sink,
                nsubsteps=nsubsteps,
            )
