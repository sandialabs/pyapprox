r"""The new transient path against the old one, with the old scheme.

The old scheme is the raw interior system with the essential and closure
rows pinned at the end of each step, no differentiated rows: on the new
path, ``collocation_unconstrained_system`` stepped by ``RungeKuttaStepper``
with a ``ClosurePinning`` at the step end; on the old path,
``CollocationModel.solve_transient`` with the same conditions set on the
physics. Three problems, each stepped by backward Euler, forward Euler,
Crank-Nicolson and Heun:

- 1D ADR (12 nodes), Dirichlet left, gradient Robin right;
- 1D periodic ADR (12 nodes);
- ADR on a 6 x 6 polar mesh, time-dependent Dirichlet data on all four
  sides.

Every stored state agrees to 1e-10. This checks the new machinery
against the old numbers while both paths exist; it is no claim that the
step-end scheme is the right one, which the temporal-order cases judge.
"""

import math
from dataclasses import dataclass
from typing import Callable, Generic, List

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.integration import Integrator
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import constant_dirichlet_bc
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic import PeriodicBC
from pyapprox.pde.collocation.boundary.periodic_rows import (
    PeriodicPart,
    PeriodicRows,
)
from pyapprox.pde.collocation.boundary.robin import RobinBC
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.closure_pinning import ClosurePinning
from pyapprox.pde.models.collocation.unconstrained_system import (
    collocation_unconstrained_system,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton

_N1D = 12
_DELTAT = 5e-4
_NSTEPS = 8

_METHODS: List[str] = ["backward_euler", "forward_euler", "crank_nicolson", "heun"]
_TABLEAUX: dict[str, Callable[[], ButcherTableau]] = {
    "backward_euler": backward_euler,
    "forward_euler": forward_euler,
    "crank_nicolson": crank_nicolson,
    "heun": heun,
}


@dataclass
class _Problem(Generic[Array]):
    """One problem on both paths: the new system, the old physics with
    its conditions set, and the initial state."""

    system: CollocationSystem[Array]
    old: AdvectionDiffusionReaction[Array]
    init: Array


def _adr_1d(
    bkd: Backend[Array], mesh: TransformedMesh1D[Array]
) -> AdvectionDiffusionReaction[Array]:
    return AdvectionDiffusionReaction(
        ChebyshevBasis1D(mesh, bkd),
        bkd,
        diffusion=0.2,
        velocity=[bkd.full((_N1D,), 0.5)],
        reaction=-0.4,
    )


def _dirichlet_robin(bkd: Backend[Array]) -> _Problem[Array]:
    mesh = TransformedMesh1D(_N1D, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    normal = GradientNormalOperator(
        bkd, right, mesh.boundary_normals(1), [basis.derivative_matrix(1, 0)]
    )
    alpha, beta, data = 2.0, 0.5, 0.3
    layout = StateLayout.uniform(_N1D, 1)
    robin: List[ClosureRowsProtocol[Array]] = [
        RobinRows(bkd, layout, [RobinPart(right, normal, alpha, beta, data)])
    ]
    system: CollocationSystem[Array] = CollocationSystem(
        _adr_1d(bkd, mesh),
        DirichletRows(bkd, layout, [DirichletPart(left, 1.0)]),
        robin,
    )
    old = _adr_1d(bkd, mesh)
    old.set_boundary_conditions(
        [
            constant_dirichlet_bc(bkd, left, 1.0),
            RobinBC(bkd, right, normal, alpha, beta, bkd.full((1,), data)),
        ]
    )
    nodes = mesh.points()[0]
    return _Problem(system, old, 1.0 + 0.5 * (nodes + 1.0) * bkd.cos(nodes))


def _periodic(bkd: Backend[Array]) -> _Problem[Array]:
    mesh = TransformedMesh1D(_N1D, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    primary, partner = mesh.boundary_indices(0), mesh.boundary_indices(1)
    derivative = basis.derivative_matrix(1, 0)
    rows: List[ClosureRowsProtocol[Array]] = [
        PeriodicRows(
            bkd,
            StateLayout.uniform(_N1D, 1),
            mesh.reference_points(),
            [PeriodicPart(primary, partner, derivative)],
        )
    ]
    system: CollocationSystem[Array] = CollocationSystem(_adr_1d(bkd, mesh), None, rows)
    old = _adr_1d(bkd, mesh)
    old.set_boundary_conditions([PeriodicBC(bkd, primary, partner, derivative)])
    nodes = mesh.points()[0]
    return _Problem(system, old, bkd.cos(math.pi * nodes) + 0.2)


def _boundary_value(coords: Array, time: float) -> Array:
    return (1.0 + time) * (coords[0] + 0.5 * coords[1])


def _polar_dirichlet(bkd: Backend[Array]) -> _Problem[Array]:
    transform = PolarTransform((1.0, 2.0), (-math.pi / 2, math.pi / 2), bkd)
    mesh = TransformedMesh2D(6, 6, bkd, transform)
    points = mesh.points()

    def physics() -> AdvectionDiffusionReaction[Array]:
        return AdvectionDiffusionReaction(ChebyshevBasis2D(mesh, bkd), bkd, 0.5)

    sides = [mesh.boundary_indices(side) for side in range(4)]
    parts = [
        DirichletPart(
            indices,
            NodalSupplier(TimeDependent(_boundary_value), points[:, indices]),
        )
        for indices in sides
    ]
    npts = points.shape[1]
    system: CollocationSystem[Array] = CollocationSystem(
        physics(), DirichletRows(bkd, StateLayout.uniform(npts, 1), parts)
    )
    old = physics()

    def values(indices: Array) -> Callable[[float], Array]:
        return lambda time: _boundary_value(points[:, indices], time)

    old.set_boundary_conditions(
        [DirichletBC(bkd, indices, values(indices)) for indices in sides]
    )
    radius = bkd.sqrt(points[0] ** 2 + points[1] ** 2)
    bump = bkd.sin(math.pi * (radius - 1.0)) * bkd.cos(
        bkd.arctan2(points[1], points[0])
    )
    return _Problem(system, old, _boundary_value(points, 0.0) + bump)


_PROBLEMS: dict[str, Callable[[Backend[Array]], _Problem[Array]]] = {
    "dirichlet_robin": _dirichlet_robin,
    "periodic": _periodic,
    "polar_dirichlet": _polar_dirichlet,
}


def _new_trajectory(
    bkd: Backend[Array], problem: _Problem[Array], method: str, times: List[float]
) -> List[Array]:
    """The stored states on the new path, every row pinned at the step
    end."""
    system = collocation_unconstrained_system(problem.system)
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
    stepper = RungeKuttaStepper(
        _TABLEAUX[method](),
        system,
        RecomputeAlways(system.solver),
        newton,
        ClosurePinning(problem.system),
    )
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    Integrator(stepper).solve(
        bkd.reshape(problem.init, (-1, 1)), GivenSizes.from_times(times), store
    )
    return [store.state(n)[:, 0] for n in range(store.ntimes())]


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("name", list(_PROBLEMS))
def test_trajectories_equal_the_old_path(
    bkd: Backend[Array], name: str, method: str
) -> None:
    problem = _PROBLEMS[name](bkd)
    times = [_DELTAT * n for n in range(_NSTEPS + 1)]
    new = _new_trajectory(bkd, problem, method, times)
    config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
        method=method,
        init_time=0.0,
        final_time=_DELTAT * _NSTEPS,
        deltat=_DELTAT,
        newton_tol=1e-13,
        newton_maxiter=20,
        verbosity=0,
    )
    old, _ = CollocationModel(problem.old, bkd).solve_transient(problem.init, config)
    assert old.shape[1] == len(new)
    for n, state in enumerate(new):
        bkd.assert_allclose(state, old[:, n], rtol=1e-10, atol=1e-10)
