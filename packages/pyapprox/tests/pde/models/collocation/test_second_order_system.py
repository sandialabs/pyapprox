r"""Tests for the doubled first-order system of collocation elastodynamics.

A linear-elastic square :math:`[-1, 1]^2` (5 x 5 Chebyshev nodes,
:math:`\lambda = 1`, :math:`\mu = 0.5`, :math:`\rho = 2`), driven on its
left edge by :math:`u_x = g(t)(1 - y^2)`, :math:`u_y = 0`, with
:math:`g(t) = 0.3 \sin 2t + 0.1 t`. The bar case has zero traction on
the other three edges (closure rows, imposed directly on the velocity
rows); the clamped case has zero displacement there instead. Until the
collocation Dirichlet data take their time derivatives from a boundary
signal, the driven edge's data are a local declaration supplying them.
"""

import math
from typing import Callable, Generic, List, Optional

import numpy as np
import pytest

from pyapprox.implicitfunction.implicit_map import ArrayParameters, ImplicitMapFunction
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.integration import Integrator
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import AtEveryBlock, PinnedRowsParamDerivatives
from pyapprox.ode.second_order.assembled_blocks import backend_dense_state_blocks
from pyapprox.ode.second_order.closure_forms import DirectClosureForm
from pyapprox.ode.second_order.condensed_blocks import (
    backend_dense_condensed_state_blocks,
)
from pyapprox.ode.second_order.reduction import InitialVelocityDerivatives
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import backward_euler, crank_nicolson
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import TractionNormalOperator
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh2D
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.closure_pinning import ClosureRelation
from pyapprox.pde.models.collocation.second_order_system import (
    CollocationSecondOrderSystem,
    SecondOrderStorageProtocol,
    collocation_second_order_system,
)
from pyapprox.pde.models.collocation.unconstrained_system import (
    collocation_unconstrained_system,
)
from pyapprox.pde.parameterizations.collocation_elasticity import (
    CollocationElasticityParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.second_order_cases import G, prescribed_g

_N1D = 5
_NPTS = _N1D * _N1D
_NSTATES = 2 * _NPTS
_RHO = 2.0
_FREE_SIDES = (BoundarySide.RIGHT, BoundarySide.BOTTOM, BoundarySide.TOP)


class _DrivenEdge(Generic[Array]):
    r"""Dirichlet rows with :math:`u_x = g(t) (1 - y^2)` on the left edge
    and zero on every other part, declaring the data's time derivatives
    by order."""

    def __init__(
        self,
        bkd: Backend[Array],
        mesh: TransformedMesh2D[Array],
        parts: List[DirichletPart[Array]],
    ) -> None:
        self._bkd = bkd
        self._rows = DirichletRows(bkd, StateLayout.uniform(_NPTS, 2), parts).rows()
        left = mesh.boundary_indices(BoundarySide.LEFT)
        y = mesh.points()[1]
        profile = bkd.zeros((_NSTATES,))
        self._profile = bkd.index_update(profile, left, 1.0 - y[left] ** 2)

    def rows(self) -> BackendRowSet[Array]:
        return self._rows

    def values(self, time: float) -> Array:
        return self._column(time, 0)

    def _column(self, time: float, order: int) -> Array:
        return self._bkd.reshape(self._profile * prescribed_g(time, order), (-1, 1))

    def time_derivative(self, order: int) -> Optional[Callable[[float], Array]]:
        return _EdgeDerivative(self, order)

    def rates(self) -> Optional[Callable[[float], Array]]:
        return self.time_derivative(1)

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        return PinnedRowsParamDerivatives.independent()

    def is_time_invariant(self) -> bool:
        return False


class _EdgeDerivative(Generic[Array]):
    def __init__(self, edge: _DrivenEdge[Array], order: int) -> None:
        self._edge, self._order = edge, order

    def __call__(self, time: float) -> Array:
        return self._edge._column(time, self._order)


def _mesh(bkd: Backend[Array]) -> TransformedMesh2D[Array]:
    return TransformedMesh2D(_N1D, _N1D, bkd)


def _physics(
    bkd: Backend[Array], mesh: TransformedMesh2D[Array], density: Optional[float]
) -> LinearElasticityPhysics[Array]:
    return LinearElasticityPhysics(
        ChebyshevBasis2D(mesh, bkd), bkd, lamda=1.0, mu=0.5, density=density
    )


def _left_parts(mesh: TransformedMesh2D[Array]) -> List[DirichletPart[Array]]:
    left = mesh.boundary_indices(BoundarySide.LEFT)
    return [DirichletPart(left, 0.0, 0), DirichletPart(left, 0.0, 1)]


def _bar(bkd: Backend[Array]) -> CollocationSystem[Array]:
    """The driven left edge; zero traction on the other three."""
    mesh = _mesh(bkd)
    physics = _physics(bkd, mesh, _RHO)
    closures: List[ClosureRowsProtocol[Array]] = []
    for side in _FREE_SIDES:
        indices, normals = mesh.boundary_indices(side), mesh.boundary_normals(side)
        for component in (0, 1):
            operator = TractionNormalOperator(bkd, indices, normals, physics, component)
            closures.append(
                RobinRows(
                    bkd,
                    StateLayout.uniform(_NPTS, 2),
                    [RobinPart(indices, operator, 0.0, 1.0, 0.0, component)],
                )
            )
    return CollocationSystem(
        physics, _DrivenEdge(bkd, mesh, _left_parts(mesh)), closures
    )


def _clamped(
    bkd: Backend[Array], physics: LinearElasticityPhysics[Array]
) -> CollocationSystem[Array]:
    """The driven left edge; zero displacement on the other three."""
    mesh = _mesh(bkd)
    parts = _left_parts(mesh)
    for side in _FREE_SIDES:
        indices = mesh.boundary_indices(side)
        parts += [DirichletPart(indices, 0.0, 0), DirichletPart(indices, 0.0, 1)]
    return CollocationSystem(physics, _DrivenEdge(bkd, mesh, parts))


def _doubled(
    system: CollocationSystem[Array],
    blocks: SecondOrderStorageProtocol[Array, G],
    parameters: Optional[ParameterState[Array]] = None,
) -> CollocationSecondOrderSystem[Array, G]:
    return collocation_second_order_system(
        system,
        blocks,
        AtEveryBlock(),
        None,
        DirectClosureForm(),
        InitialVelocityDerivatives.independent(),
        parameters=parameters,
    )


def _trajectory(
    bkd: Backend[Array],
    doubled: CollocationSecondOrderSystem[Array, G],
    method: str,
) -> InMemoryTrajectoryStore[Array]:
    system = doubled.system()
    tableau = backward_euler() if method == "backward_euler" else crank_nicolson()
    newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=10)
    stepper = RungeKutta(tableau, newton).stepper(
        system, RecomputeAlways(system.solver), doubled.pinning()
    )
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    start = doubled.initial_state(bkd.zeros((2 * _NSTATES, 1)), 0.0)
    Integrator(stepper).solve(start, FixedStep(0.02, (0.0, 0.2)), store)
    return store


@pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
def test_bar_keeps_its_boundary_rows(bkd: Backend[Array], method: str) -> None:
    """Every stored state after the first carries g, g' on the driven
    edge and zero traction on the free edges to 1e-9, in both storages,
    which agree to 1e-9."""
    system = _bar(bkd)
    relation = ClosureRelation(system)
    closure_rows = list(relation.rows().indices())
    edge = _DrivenEdge(bkd, _mesh(bkd), _left_parts(_mesh(bkd)))
    driven = list(edge.rows().indices())
    stores = [
        _trajectory(
            bkd, _doubled(system, backend_dense_state_blocks(bkd, _NSTATES)), method
        ),
        _trajectory(
            bkd,
            _doubled(system, backend_dense_condensed_state_blocks(bkd, _NSTATES)),
            method,
        ),
    ]
    assert stores[0].ntimes() == stores[1].ntimes() == 11
    for k in range(1, stores[0].ntimes()):
        time = stores[0].time(k)
        for store in stores:
            state = store.state(k)
            u, v = state[:_NSTATES], state[_NSTATES:]
            bkd.assert_allclose(
                u[driven], edge.values(time)[driven], rtol=1e-12, atol=1e-14
            )
            bkd.assert_allclose(
                v[driven],
                edge._column(time, 1)[driven],
                rtol=1e-12,
                atol=1e-14,
            )
            traction = relation.residual(u, time)[closure_rows]
            assert bkd.to_float(bkd.max(bkd.abs(traction))) <= 1e-9
        bkd.assert_allclose(
            stores[1].state(k), stores[0].state(k), rtol=1e-9, atol=1e-11
        )
    final = stores[0].state(stores[0].ntimes() - 1)
    # The drive reached the interior: the solution is not trivially zero.
    assert bkd.to_float(bkd.max(bkd.abs(final[:_NSTATES]))) > 1e-3


def test_mass_is_the_density(bkd: Backend[Array]) -> None:
    """The assembled mass is diag(I, rho I)."""
    doubled = _doubled(_bar(bkd), backend_dense_state_blocks(bkd, _NSTATES))
    mass = doubled.system().mass.apply(bkd.eye(2 * _NSTATES))
    expected = np.diag(np.concatenate([np.ones(_NSTATES), np.full(_NSTATES, _RHO)]))
    bkd.assert_allclose(mass, bkd.asarray(expected), rtol=1e-14)


class TestRefusals:
    def test_first_order_path_names_this_entry_point(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="collocation_second_order_system"):
            collocation_unconstrained_system(_bar(bkd))

    def test_no_density(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        system = _clamped(bkd, _physics(bkd, mesh, None))
        with pytest.raises(ValueError, match="declares no density"):
            _doubled(system, backend_dense_state_blocks(bkd, _NSTATES))

    def test_essential_rows_without_time_derivatives(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        essential = DirichletRows(bkd, StateLayout.uniform(_NPTS, 2), _left_parts(mesh))
        system = CollocationSystem(_physics(bkd, mesh, _RHO), essential)
        with pytest.raises(TypeError, match="TimeDerivativeRowsProtocol"):
            _doubled(system, backend_dense_state_blocks(bkd, _NSTATES))

    def test_storage_without_row_lifting(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="SecondOrderStorageProtocol"):
            _doubled(_bar(bkd), None)  # type: ignore[arg-type]


def _endpoint(
    bkd: Backend[Array], blocks: SecondOrderStorageProtocol[Array, G]
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    """u_x(T) at the center node of the clamped square, its Young's
    modulus E = 10 + sum_k theta_k phi_k in three modes; the initial
    state, zero, does not depend on theta."""
    mesh = _mesh(bkd)
    physics = _physics(bkd, mesh, _RHO)
    points = mesh.points()
    coords = points[0] + points[1]
    modes = bkd.stack(
        [bkd.sin((k + 1) * math.pi * coords / 2.0) / (k + 1) for k in range(3)],
        axis=1,
    )
    facade = CollocationElasticityParameterization(
        physics,
        youngs_modulus_map=MeshKLEFieldMap(bkd, bkd.full((_NPTS,), 10.0), modes),
        poisson_ratio=0.3,
        bkd=bkd,
    )
    parameters = ParameterState(facade)
    doubled = _doubled(_clamped(bkd, physics), blocks, parameters)
    system = doubled.system()
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=10)
    center = _NPTS // 2
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd,
            1,
            state_terms=(EndpointTerm(StateEntries(bkd, 2 * _NSTATES, [center])),),
        )
    )
    implicit = TransientImplicitMap(
        doubled,
        doubled,
        RungeKutta(crank_nicolson(), newton),
        RecomputeAlways(system.solver),
        FixedStep(0.02, (0.0, 0.2)),
        bkd.zeros((2 * _NSTATES, 1)),
        functional,
    )
    return ImplicitMapFunction(implicit, ArrayParameters(bkd))


def test_gradient_in_the_modulus(numpy_bkd: Backend[Array]) -> None:
    """The adjoint gradient of the center displacement in E's three modes
    matches central differences to 1e-6, and the condensed storage's to
    1e-9."""
    bkd = numpy_bkd
    sample = bkd.asarray([[0.4], [-0.3], [0.2]])
    assembled = _endpoint(bkd, backend_dense_state_blocks(bkd, _NSTATES))
    condensed = _endpoint(bkd, backend_dense_condensed_state_blocks(bkd, _NSTATES))
    jacobian = assembled.derivatives().jacobian
    condensed_jacobian = condensed.derivatives().jacobian
    assert jacobian is not None and condensed_jacobian is not None
    gradient = jacobian(sample)
    assert bkd.to_float(bkd.max(bkd.abs(gradient))) > 1e-8
    bkd.assert_allclose(condensed_jacobian(sample), gradient, rtol=1e-9)
    step = 1e-5
    for k in range(3):
        shift = bkd.asarray(np.eye(3)[:, [k]] * step)
        difference = (assembled(sample + shift) - assembled(sample - shift)) / (
            2 * step
        )
        bkd.assert_allclose(gradient[0, k], difference[0, 0], rtol=1e-6)
