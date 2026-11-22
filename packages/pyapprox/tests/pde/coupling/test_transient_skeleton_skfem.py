r"""A transient skeleton solve per window against the single-domain run.

The problem: :math:`u_t = \Delta u - u + 1 + x + 2y` on the unit square,
u = 1 on the left and u = 2 on the right, natural above and below, from
:math:`u_0 = 1 + x + \sin(\pi x)\sin(\pi y)`. The square is split into
Q1 subdomains meshed as the single domain is (two or four vertical
slabs, or 2 x 2 with a cross point and natural outer sides throughout).
Each window is one step of the subdomains' method: backward Euler with
Δt = 0.01, or forward Euler with Δt = 1e-4 (below its stability limit
on these meshes). The coupled solve must reproduce the single-domain
run: every subdomain state, and the skeleton state equal to the
single-domain trace, at every window's end; the reactions must balance
on the interface in every window.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.accumulating_sink import MultiSink
from pyapprox.ode.npz_trajectory_store import NpzTrajectoryStore
from pyapprox.ode.pinned_rows import AtEveryBlock, PinPlacementProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectoryStoreProtocol,
)
from pyapprox.pde.boundary.except_dofs import ExceptDofs
from pyapprox.pde.boundary.signal import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import (
    PortQuantity,
    ResponsePointProtocol,
    TimeWindow,
)
from pyapprox.pde.coupling.skeleton import CoupledSubdomain, DenseSchur
from pyapprox.pde.coupling.spaces import ComponentSkeletonSpace
from pyapprox.pde.coupling.stacks.galerkin import (
    skfem_stagewise_response,
    skfem_windowed_response,
)
from pyapprox.pde.coupling.transient import (
    WindowedResponseProtocol,
    runge_kutta_block_fractions,
)
from pyapprox.pde.coupling.transient_skeleton import TransientSkeletonSolver
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
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_skfem import (
    CoupledProblem,
    Np,
    SystemBuilder,
    TensorQuadMesh,
    Value,
    box_problem,
    node_at,
)

_BKD = NumpyBkd()
_NWINDOWS = 3


def _forcing(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


def _sink(x: Np, u: Np) -> Np:
    return -u


def _sink_derivative(x: Np, u: Np) -> Np:
    return -np.ones_like(u)


def _initial(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))


def _moving(x: Np, time: float) -> Np:
    return np.full(x.shape[1], 1.0 + np.sin(20.0 * time))


def _moving_rate(x: Np, time: float) -> Np:
    return np.full(x.shape[1], 20.0 * np.cos(20.0 * time))


def _build_moving_left(
    mesh: TensorQuadMesh, sides: Sequence[tuple[str, Value]], exempt: set[int]
) -> Any:
    """As :func:`_build`, the left side's value :math:`1 + \\sin 20t`
    with its rate in place of the value given."""
    return _build(mesh, sides, exempt, moving_left=True)


def _build(
    mesh: TensorQuadMesh,
    sides: Sequence[tuple[str, Value]],
    exempt: set[int],
    moving_left: bool = False,
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
        value: Union[float, TimeIndependent, BoundarySignal] = (
            v if isinstance(v, float) else TimeIndependent(v)
        )
        if moving_left and side == "left":
            value = BoundarySignal(
                TimeDependent(_moving), [TimeDependent(_moving_rate)]
            )
        bc = DirichletBC(basis, side, value, bkd)
        on_side = exempt & {int(d) for d in bc.constrained_dofs()}
        conditions.append(ExceptDofs(bc, sorted(on_side)))
    return compose_galerkin_system(physics, conditions)


def _method(tableau: Callable[[], ButcherTableau]) -> RungeKutta[Np, Np]:
    made = tableau()
    return RungeKutta(
        made, None if made.is_explicit() else Newton(atol=1e-13, rtol=0.0, maxiters=20)
    )


@dataclass
class _Accepted:
    """A windowed response that records each accepted point's output:
    the protocol satisfied by delegation."""

    inner: WindowedResponseProtocol[Np]
    outputs: list[Np]

    def bkd(self) -> NumpyBkd:
        return _BKD

    def nport(self) -> int:
        return self.inner.nport()

    def receives(self) -> PortQuantity:
        return self.inner.receives()

    def returns(self) -> PortQuantity:
        return self.inner.returns()

    def nparams(self) -> int:
        return 0

    def nqoi(self) -> int:
        return 0

    def linearize(
        self, port_data: Np, params: Np, window: TimeWindow
    ) -> ResponsePointProtocol[Np]:
        return self.inner.linearize(port_data, params, window)

    def time(self) -> float:
        return self.inner.time()

    def start(self) -> None:
        self.inner.start()

    def accept(self, point: ResponsePointProtocol[Np]) -> None:
        self.inner.accept(point)
        self.outputs.append(point.output())


class _Run:
    """The single-domain run and the coupled pieces of one partition."""

    def __init__(
        self,
        ncells: tuple[int, int],
        essentials: Sequence[tuple[str, Value]],
        tableau: Callable[[], ButcherTableau],
        dt: float,
        store: Optional[Callable[[int], TrajectoryStoreProtocol[Np]]] = None,
        nwindows: int = _NWINDOWS,
        build: SystemBuilder = _build,
    ) -> None:
        self.nwindows = nwindows
        self.dt = dt
        self.problem: CoupledProblem = box_problem(ncells, build, essentials)
        whole, mesh = self.problem.single_domain_system()
        self.nodes = mesh.nodes()
        system = galerkin_transient_system(skfem_discretization(whole), ScipySparseLU())
        self.reference: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
        solve_galerkin_transient(
            system,
            _method(tableau),
            RecomputeAlways(ScipySparseLU()),
            FixedStep(dt, (0.0, nwindows * dt)),
            np.reshape(_initial(mesh.nodes()), (-1, 1)),
            self.reference,
        )
        skel = self.problem.skeleton_coords()
        self.at_skeleton = self.problem.mesh_dofs(
            [node_at(self.nodes, skel[:, d]) for d in range(skel.shape[1])]
        )
        self.rows: list[list[int]] = []
        self.sinks: list[TrajectoryStoreProtocol[Np]] = []
        self.recorded: list[_Accepted] = []
        descriptor = self.problem.space.descriptor()
        self.subdomains: list[CoupledSubdomain[Np, Np]] = []
        for s in range(self.problem.topo.nsubdomains()):
            sub, nodes, ports, dofs = self.problem.port_system(s)
            rows = [node_at(self.nodes, nodes[:, j]) for j in range(nodes.shape[1])]
            sink: TrajectoryStoreProtocol[Np] = (
                InMemoryTrajectoryStore() if store is None else store(s)
            )
            response = skfem_windowed_response(
                sub,
                ports,
                _method(tableau),
                RecomputeAlways(ScipySparseLU()),
                self.reference.state(0)[rows],
                sink,
            )
            recorded = _Accepted(response, [])
            self.rows.append(rows)
            self.sinks.append(sink)
            self.recorded.append(recorded)
            self.subdomains.append(
                CoupledSubdomain(
                    recorded,
                    MatchingNodalTransfer(BackendPortRows(descriptor, dofs, _BKD)),
                )
            )

    def solver(self) -> TransientSkeletonSolver[Np, Np, Np, DenseFormat[Np]]:
        space = BackendVectorSpace(self.problem.space.descriptor(), _BKD)
        lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(_BKD))
        return TransientSkeletonSolver(
            space,
            self.subdomains,
            DenseSchur(_BKD),
            lu,
            Newton(atol=1e-11, rtol=1e-11, maxiters=20),
        )

    def windows(self) -> list[TimeWindow]:
        return [
            TimeWindow(n * self.dt, (n + 1) * self.dt) for n in range(self.nwindows)
        ]

    def lam0(self) -> Np:
        state: Np = self.reference.state(0)[self.at_skeleton]
        return state


_EDGES: list[tuple[str, Value]] = [("left", 1.0), ("right", 2.0)]
_CASES = [
    ((2, 1), _EDGES, backward_euler, 0.01),
    ((4, 1), _EDGES, backward_euler, 0.01),
    ((2, 2), [], backward_euler, 0.01),
    ((2, 1), _EDGES, forward_euler, 1e-4),
    # Its stage value on every row is the mean of the step's ends, which
    # the interpolated trace gives exactly.
    ((2, 1), _EDGES, implicit_midpoint, 0.01),
]


def _stagewise(
    run: _Run,
    tableau: Callable[[], ButcherTableau],
    essential_placement: Optional[PinPlacementProtocol] = None,
) -> tuple[list[list[int]], list[InMemoryTrajectoryStore[Np]], Np]:
    """``run``'s partition coupled stage by stage: each subdomain's
    reference rows, its sink, and the interface trajectory's states."""
    problem = run.problem
    nblocks = len(runge_kutta_block_fractions(tableau()))
    space = ComponentSkeletonSpace(problem.scalar, nblocks)
    descriptor = space.descriptor()
    subdomains: list[CoupledSubdomain[Np, Np]] = []
    sinks: list[InMemoryTrajectoryStore[Np]] = []
    rows_of: list[list[int]] = []
    for s in range(problem.topo.nsubdomains()):
        sub, nodes, ports, _ = problem.port_system(s)
        rows = [node_at(run.nodes, nodes[:, j]) for j in range(nodes.shape[1])]
        sink: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
        response = skfem_stagewise_response(
            sub,
            ports,
            _method(tableau),
            RecomputeAlways(ScipySparseLU()),
            run.reference.state(0)[rows],
            sink,
            essential_placement=essential_placement,
        )
        dofs = subdomain_port_dofs(problem.topo, space, s)
        transfer = MatchingNodalTransfer(BackendPortRows(descriptor, dofs, _BKD))
        subdomains.append(CoupledSubdomain(response, transfer))
        sinks.append(sink)
        rows_of.append(rows)
    lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(_BKD))
    solver = TransientSkeletonSolver(
        BackendVectorSpace(descriptor, _BKD),
        subdomains,
        DenseSchur(_BKD),
        lu,
        Newton(atol=1e-11, rtol=1e-11, maxiters=20),
    )
    # Every block's first guess is the start trace.
    lam0 = np.repeat(run.lam0(), nblocks, axis=0)
    interface: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
    solver.solve(lam0, run.windows(), interface)
    states = np.hstack([interface.state(n) for n in range(run.nwindows + 1)])
    return rows_of, sinks, states


@pytest.mark.parametrize(
    "tableau", [crank_nicolson, implicit_midpoint, alexander_sdirk3]
)
def test_stagewise_coupling_is_the_single_domain_step(
    tableau: Callable[[], ButcherTableau],
) -> None:
    """One trace per block of the step, every block pinned to its own and
    the reactions balanced block by block: the coupled run is the
    single-domain run (pinned at the step's end, constant essential data)
    to 1e-10 at every window, so SDIRK3 keeps its third order. The
    skeleton holds one component per block."""
    run = _Run((2, 1), _EDGES, tableau, 0.01)
    nblocks = len(runge_kutta_block_fractions(tableau()))
    rows_of, sinks, interface = _stagewise(run, tableau)
    for rows, sink in zip(rows_of, sinks):
        for n in range(_NWINDOWS + 1):
            _BKD.assert_allclose(
                sink.state(n), run.reference.state(n)[rows], atol=1e-10
            )
    # The last block is at the window's end: the single-domain trace.
    for n in range(1, _NWINDOWS + 1):
        _BKD.assert_allclose(
            interface[nblocks - 1 :: nblocks, n : n + 1],
            run.reference.state(n)[run.at_skeleton],
            atol=1e-10,
        )


def test_stagewise_coupling_with_moving_essential_data() -> None:
    """SDIRK3 stage by stage with u = 1 + sin 20t on the left, its rate
    supplied: the essential rows pinned at the step's end (the default)
    and solved with the data's rate in the inner stages, as the
    single-domain step treats them, so the coupled run equals it to
    1e-10. Pinned in every block instead (the control), the essential
    rows hold the data at each stage's time, another scheme: the run
    differs from the single-domain one by far more than the tolerance."""
    run = _Run((2, 1), _EDGES, alexander_sdirk3, 0.01, build=_build_moving_left)
    rows_of, sinks, _ = _stagewise(run, alexander_sdirk3)
    for rows, sink in zip(rows_of, sinks):
        for n in range(_NWINDOWS + 1):
            _BKD.assert_allclose(
                sink.state(n), run.reference.state(n)[rows], atol=1e-10
            )
    rows_of, sinks, _ = _stagewise(run, alexander_sdirk3, AtEveryBlock())
    gap = max(
        float(np.abs(sink.state(-1) - run.reference.state(-1)[rows]).max())
        for rows, sink in zip(rows_of, sinks)
    )
    assert gap > 1e-6, f"every-block pin differs by only {gap:.2e}"


def test_block_fractions() -> None:
    assert runge_kutta_block_fractions(backward_euler()) == (1.0,)
    assert runge_kutta_block_fractions(crank_nicolson()) == (1.0,)
    assert runge_kutta_block_fractions(implicit_midpoint()) == (0.5, 1.0)
    sdirk3 = alexander_sdirk3()
    assert runge_kutta_block_fractions(sdirk3) == tuple(sdirk3.c)
    # Heun's last stage and final block are both at the step's end.
    with pytest.raises(ValueError, match="distinct times"):
        runge_kutta_block_fractions(heun())


@pytest.mark.parametrize(
    "tableau, rate", [(implicit_midpoint, 2.0), (alexander_sdirk3, 2.0)]
)
def test_temporal_order(tableau: Callable[[], ButcherTableau], rate: float) -> None:
    """The coupled run against a fine single-domain SDIRK3 run at T =
    0.04, one step per window, Δt = 0.01 to 0.0025. The reference is on
    the same mesh, so both solve the same semi-discrete system and their
    difference is temporal error alone; its own step (6.25e-4) leaves an
    error near 1e-7, far below the measured ones. One trace per window,
    interpolated linearly, caps the order at 2: SDIRK3 alone converges at
    about 2.9 on these steps, coupled at 2.06 (finer port data in time
    lift the cap)."""
    final = 0.04
    fine = _Run((2, 1), _EDGES, alexander_sdirk3, final / 64, nwindows=64)
    errors = []
    for nwindows in (4, 8, 16):
        run = _Run((2, 1), _EDGES, tableau, final / nwindows, nwindows=nwindows)
        run.solver().solve(run.lam0(), run.windows(), InMemoryTrajectoryStore())
        errors.append(
            max(
                float(np.abs(sink.state(-1) - fine.reference.state(-1)[rows]).max())
                for rows, sink in zip(run.rows, run.sinks)
            )
        )
    observed = np.log2(errors[-2] / errors[-1])
    assert abs(observed - rate) < 0.15, f"rate {observed:.2f}, errors {errors}"


@pytest.mark.parametrize("ncells, essentials, tableau, dt", _CASES)
def test_windows_reproduce_the_single_domain_run(
    ncells: tuple[int, int],
    essentials: Sequence[tuple[str, Value]],
    tableau: Callable[[], ButcherTableau],
    dt: float,
) -> None:
    run = _Run(ncells, essentials, tableau, dt)
    interface: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
    last = run.solver().solve(run.lam0(), run.windows(), interface)
    assert interface.ntimes() == _NWINDOWS + 1
    for n in range(_NWINDOWS + 1):
        _BKD.assert_allclose(
            interface.state(n), run.reference.state(n)[run.at_skeleton], atol=1e-10
        )
    _BKD.assert_allclose(last, interface.state(-1), atol=0.0)
    _BKD.assert_allclose(
        np.array(interface.times()), np.arange(_NWINDOWS + 1) * dt, atol=1e-15
    )
    for rows, sink in zip(run.rows, run.sinks):
        assert sink.nsteps() == _NWINDOWS
        for n in range(_NWINDOWS + 1):
            _BKD.assert_allclose(
                sink.state(n), run.reference.state(n)[rows], atol=1e-10
            )
    # The reactions balance on the interface in every window; each alone
    # is not small.
    nskel = run.problem.space.descriptor().size
    for n in range(_NWINDOWS):
        balance = np.zeros((nskel, 1))
        largest = 0.0
        for sd, recorded in zip(run.subdomains, run.recorded):
            output = recorded.outputs[n]
            largest = max(largest, float(np.abs(output).max()))
            balance = sd.transfer.gather_add(output, balance)
        assert largest > 1e3 * 1e-10
        _BKD.assert_allclose(balance, np.zeros_like(balance), atol=1e-10)


def test_each_subdomain_writes_its_own_files(tmp_path: Path) -> None:
    """Each subdomain and the interface keep their trajectories in their
    own directories, as the ranks of a distributed run would; the files
    hold the in-memory run."""

    def files(s: int) -> NpzTrajectoryStore[Np]:
        return NpzTrajectoryStore(str(tmp_path / f"subdomain_{s}"), _BKD)

    on_disk = _Run((4, 1), _EDGES, backward_euler, 0.01, files)
    interface: NpzTrajectoryStore[Np] = NpzTrajectoryStore(
        str(tmp_path / "interface"), _BKD
    )
    on_disk.solver().solve(on_disk.lam0(), on_disk.windows(), interface)
    in_memory = _Run((4, 1), _EDGES, backward_euler, 0.01)
    kept: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
    in_memory.solver().solve(in_memory.lam0(), in_memory.windows(), kept)
    for n in range(_NWINDOWS + 1):
        _BKD.assert_allclose(interface.state(n), kept.state(n), atol=0.0)
    for disk, memory in zip(on_disk.sinks, in_memory.sinks):
        assert isinstance(disk, NpzTrajectoryStore)
        assert len(os.listdir(disk.directory())) == _NWINDOWS + 1
        for step in range(_NWINDOWS):
            got, want = disk.window(step), memory.window(step)
            assert got.span == want.span
            _BKD.assert_allclose(got.unknown.blocks[0], want.unknown.blocks[0])
            _BKD.assert_allclose(got.past_states[0], want.past_states[0])
    assert sorted(os.listdir(tmp_path)) == [
        "interface",
        "subdomain_0",
        "subdomain_1",
        "subdomain_2",
        "subdomain_3",
    ]


def test_the_interface_trajectory_fans_out() -> None:
    run = _Run((2, 1), _EDGES, backward_euler, 0.01)
    kept: list[InMemoryTrajectoryStore[Np]] = [
        InMemoryTrajectoryStore(),
        InMemoryTrajectoryStore(),
    ]
    run.solver().solve(run.lam0(), run.windows(), MultiSink(kept))
    for store in kept:
        assert store.nsteps() == _NWINDOWS
        window = store.window(-1)
        # A window's unknown is its solution; it read the one before.
        _BKD.assert_allclose(window.unknown.blocks[0], store.state(-1), atol=0.0)
        _BKD.assert_allclose(window.past_states[0], store.state(-2), atol=0.0)


def test_invalid_use() -> None:
    run = _Run((2, 1), _EDGES, backward_euler, 0.01)
    solver = run.solver()
    windows = run.windows()
    lam0 = run.lam0()

    def fresh() -> InMemoryTrajectoryStore[Np]:
        return InMemoryTrajectoryStore()

    with pytest.raises(ValueError, match="at least one window"):
        solver.solve(lam0, [], fresh())
    with pytest.raises(ValueError, match="contiguous"):
        solver.solve(lam0, [windows[0], windows[2]], fresh())
    with pytest.raises(ValueError, match="responses at"):
        solver.solve(lam0, windows[1:], fresh())
    with pytest.raises(ValueError, match="nonempty"):
        solver.solve(lam0, [TimeWindow(0.0, 0.0)], fresh())
    full = fresh()
    full.append(0.0, lam0)
    with pytest.raises(ValueError, match="interface sink must be empty"):
        solver.solve(lam0, windows, full)
    steady, _, _ = run.problem.piece(0)
    space = BackendVectorSpace(run.problem.space.descriptor(), _BKD)
    lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(_BKD))
    with pytest.raises(TypeError, match="WindowedResponseProtocol"):
        TransientSkeletonSolver(
            space,
            [CoupledSubdomain(steady, run.subdomains[0].transfer)],
            DenseSchur(_BKD),
            lu,
            Newton(atol=1e-11, rtol=1e-11, maxiters=20),
        )
