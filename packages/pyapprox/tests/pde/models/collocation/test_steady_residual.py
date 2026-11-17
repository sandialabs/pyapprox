"""Tests for the collocation steady residual.

Each case builds one physics, sets the old boundary conditions on it for
``CollocationParameterizedStateEquation``, and declares the same rows
for ``collocation_steady_residual``. The old path applies its conditions
in list order, the last winning a shared node; the cases list the
Dirichlet conditions last, which matches the new rule that essential
rows win. Value, R_u and R_p are compared at a random state.
"""

import math
from typing import Callable, Generic, List, Protocol, Tuple, Union

import numpy as np
import pytest

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import (
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
    TractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic import PeriodicBC
from pyapprox.pde.collocation.boundary.periodic_rows import PeriodicPart, PeriodicRows
from pyapprox.pde.collocation.boundary.robin import (
    flux_neumann_bc,
    gradient_robin_bc,
    traction_robin_bc,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.collocation.protocols.boundary import BoundaryConditionProtocol
from pyapprox.pde.collocation.protocols.physics import PhysicsProtocol
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation import create_collocation_model
from pyapprox.pde.models.collocation.closed_system import (
    CollocationSystem,
)
from pyapprox.pde.models.collocation.steady import (
    CollocationParameterizedStateEquation,
)
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.collocation_elasticity import (
    CollocationElasticityParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import LinearizationPoint
from pyapprox.util.rootfinding.nonlinear import Newton

_POINT = LinearizationPoint(0, ("test",))
_NMODES = 2


class _Applies(Protocol[Array]):
    def apply(self, x: Array) -> Array: ...


class _Case(Generic[Array]):
    """A physics with both paths built on it: ``physics`` already holds the
    old boundary conditions, ``system`` declares the same rows."""

    def __init__(
        self,
        bkd: Backend[Array],
        physics: PhysicsProtocol[Array],
        system: CollocationSystem[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        self.bkd = bkd
        self.nstates = system.nstates()
        model = create_collocation_model(
            physics, bkd, parameterization=parameterization
        )
        self.old = CollocationParameterizedStateEquation(
            model, bkd, parameterization=parameterization
        )
        self.parameters = ParameterState(parameterization)
        self.system = system
        self.residual: PinnedResidual[Array, DenseFormat[Array]] = (
            collocation_steady_residual(system, self.parameters)
        )
        self.param = bkd.asarray(
            np.random.default_rng(3).normal(0.0, 0.2, parameterization.nparams())
        )
        self.parameters.set_param(self.param)

    def state(self, scale: float = 0.5) -> Array:
        rng = np.random.default_rng(4)
        return self.bkd.asarray(rng.normal(0.0, scale, self.nstates))

    def compare(self, scale: float = 0.5) -> None:
        bkd, state, param = self.bkd, self.state(scale), self.param
        old_value = self.old(state[:, None], param[:, None])[:, 0]
        _close(bkd, self.residual(state[:, None])[:, 0], old_value)
        bundle = self.residual.derivatives()
        assert bundle.jacobian is not None and bundle.param is not None
        _close(
            bkd,
            _dense(bkd, bundle.jacobian(state[:, None], _POINT), self.nstates),
            self.old.state_jacobian(state[:, None], param[:, None]),
        )
        _close(
            bkd,
            _dense(bkd, bundle.param(state[:, None]), param.shape[0]),
            self.old.param_jacobian(state[:, None], param[:, None]),
        )


def _dense(bkd: Backend[Array], operator: _Applies[Array], ncols: int) -> Array:
    eye = bkd.eye(ncols)
    return bkd.hstack([operator.apply(eye[:, j : j + 1]) for j in range(ncols)])


def _close(bkd: Backend[Array], actual: Array, expected: Array) -> None:
    """Equal to 1e-12 relative to the larger entry."""
    scale = max(bkd.to_float(bkd.max(bkd.abs(expected))), 1.0)
    bkd.assert_allclose(actual, expected, rtol=0.0, atol=1e-12 * scale)


def _kle(bkd: Backend[Array], npts: int, mean: float) -> MeshKLEFieldMap[Array]:
    rng = np.random.default_rng(5)
    return MeshKLEFieldMap(
        bkd,
        bkd.full((npts,), mean),
        bkd.asarray(rng.uniform(-0.3, 0.3, (npts, _NMODES))),
    )


def _adr_1d(
    bkd: Backend[Array],
) -> Tuple[AdvectionDiffusionReaction[Array], TransformedMesh1D[Array]]:
    """1D ADR, 16 points: KLE diffusion about 2, velocity 0.3, reaction
    -0.4, forcing sin(pi x)."""
    mesh = TransformedMesh1D(16, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=2.0,
        velocity=[bkd.full((16,), 0.3)],
        reaction=-0.4,
        forcing=NodalField(bkd.sin(math.pi * nodes)),
    )
    return physics, mesh


def _adr_parameterization(
    bkd: Backend[Array], physics: AdvectionDiffusionReaction[Array]
) -> CollocationAdvectionDiffusionParameterization[Array]:
    return CollocationAdvectionDiffusionParameterization(
        physics, diffusion_map=_kle(bkd, physics.npts(), 2.0), bkd=bkd
    )


def _dirichlet_1d(bkd: Backend[Array]) -> _Case[Array]:
    """u(-1) = 0.5, u(1) = -0.3."""
    physics, mesh = _adr_1d(bkd)
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    physics.set_boundary_conditions(
        [DirichletBC(bkd, left, 0.5), DirichletBC(bkd, right, -0.3)]
    )
    essential = DirichletRows(
        bkd, 16, 1, [DirichletPart(left, 0.5), DirichletPart(right, -0.3)]
    )
    system = CollocationSystem(physics, essential)
    return _Case(bkd, physics, system, _adr_parameterization(bkd, physics))


def _flux_neumann_1d(bkd: Backend[Array]) -> _Case[Array]:
    """u(-1) = 0.5 and the total flux 0.2 at x = 1, the diffusion
    parameterized."""
    physics, mesh = _adr_1d(bkd)
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    normals = bkd.asarray(np.array([[1.0]]))
    physics.set_boundary_conditions(
        [
            flux_neumann_bc(bkd, right, normals, physics, values=0.2),
            DirichletBC(bkd, left, 0.5),
        ]
    )
    essential = DirichletRows(bkd, 16, 1, [DirichletPart(left, 0.5)])
    flux = RobinRows(
        bkd,
        16,
        1,
        [
            RobinPart(
                right, FluxNormalOperator(bkd, right, normals, physics), 0.0, 1.0, 0.2
            )
        ],
    )
    system = CollocationSystem(physics, essential, [flux])
    return _Case(bkd, physics, system, _adr_parameterization(bkd, physics))


def _adr_2d(
    bkd: Backend[Array], forcing: Union[NodalField[Array], NodalSupplier[Array]]
) -> Tuple[
    AdvectionDiffusionReaction[Array], TransformedMesh2D[Array], ChebyshevBasis2D[Array]
]:
    """2D ADR on a 6 x 6 mesh: diffusion 2, velocity (0.5, -0.2), reaction
    -0.4."""
    mesh = TransformedMesh2D(6, 6, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    npts = basis.npts()
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=2.0,
        velocity=[bkd.full((npts,), 0.5), bkd.full((npts,), -0.2)],
        reaction=-0.4,
        forcing=forcing,
    )
    return physics, mesh, basis


def _steady_forcing(bkd: Backend[Array]) -> NodalField[Array]:
    points = TransformedMesh2D(6, 6, bkd).points()
    return NodalField(bkd.sin(math.pi * points[0]) * points[1])


def _robin_2d(bkd: Backend[Array]) -> _Case[Array]:
    """Dirichlet left 0.5 and right -0.3; gradient Robin bottom and top
    (alpha 2, beta 0.5, g 0.3 and 0.4)."""
    physics, mesh, basis = _adr_2d(bkd, _steady_forcing(bkd))
    npts = basis.npts()
    derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
    old: List[BoundaryConditionProtocol[Array]] = [
        gradient_robin_bc(
            bkd,
            mesh.boundary_indices(side),
            mesh.boundary_normals(side),
            derivatives,
            2.0,
            0.5,
            0.1 * (side + 1),
        )
        for side in (2, 3)
    ]
    old += [
        DirichletBC(bkd, mesh.boundary_indices(0), 0.5),
        DirichletBC(bkd, mesh.boundary_indices(1), -0.3),
    ]
    physics.set_boundary_conditions(old)
    robin = RobinRows(
        bkd,
        npts,
        1,
        [
            RobinPart(
                mesh.boundary_indices(side),
                GradientNormalOperator(
                    bkd,
                    mesh.boundary_indices(side),
                    mesh.boundary_normals(side),
                    derivatives,
                ),
                2.0,
                0.5,
                0.1 * (side + 1),
            )
            for side in (2, 3)
        ],
    )
    essential = DirichletRows(
        bkd,
        npts,
        1,
        [
            DirichletPart(mesh.boundary_indices(0), 0.5),
            DirichletPart(mesh.boundary_indices(1), -0.3),
        ],
    )
    system = CollocationSystem(physics, essential, [robin])
    return _Case(bkd, physics, system, _adr_parameterization(bkd, physics))


def _periodic_2d(bkd: Backend[Array]) -> _Case[Array]:
    """Periodic in x; Dirichlet bottom and top with data x^2."""
    physics, mesh, basis = _adr_2d(bkd, _steady_forcing(bkd))
    npts = basis.npts()
    x = mesh.points()[0]
    dx = basis.derivative_matrix(1, 0)
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    bottom, top = mesh.boundary_indices(2), mesh.boundary_indices(3)
    physics.set_boundary_conditions(
        [
            PeriodicBC(bkd, left, right, dx),
            DirichletBC(bkd, bottom, x[bottom] ** 2),
            DirichletBC(bkd, top, x[top] ** 2),
        ]
    )
    periodic = PeriodicRows(
        bkd, mesh.reference_points(), 1, [PeriodicPart(left, right, dx)]
    )
    essential = DirichletRows(
        bkd,
        npts,
        1,
        [
            DirichletPart(bottom, NodalField(x[bottom] ** 2)),
            DirichletPart(top, NodalField(x[top] ** 2)),
        ],
    )
    system = CollocationSystem(physics, essential, [periodic])
    return _Case(bkd, physics, system, _adr_parameterization(bkd, physics))


def _traction_2d(bkd: Backend[Array]) -> _Case[Array]:
    """Linear elasticity on a 6 x 6 mesh, lambda 1, mu 1, Young's modulus a
    KLE field about 2.5, Poisson ratio 0.25: clamped left; traction 0.1
    (component 0) and -0.05 (component 1) on the right, traction-free
    bottom and top."""
    mesh = TransformedMesh2D(6, 6, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    npts = basis.npts()
    physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)
    traction = {(1, 0): 0.1, (1, 1): -0.05}
    sides = [(side, component) for side in (1, 2, 3) for component in (0, 1)]
    old: List[BoundaryConditionProtocol[Array]] = [
        traction_robin_bc(
            bkd,
            mesh.boundary_indices(side),
            mesh.boundary_normals(side),
            physics,
            npts,
            component,
            0.0,
            1.0,
            traction.get((side, component), 0.0),
        )
        for side, component in sides
    ]
    left = mesh.boundary_indices(0)
    old += [DirichletBC(bkd, left, 0.0), DirichletBC(bkd, left + npts, 0.0)]
    physics.set_boundary_conditions(old)
    closure = RobinRows(
        bkd,
        npts,
        2,
        [
            RobinPart(
                mesh.boundary_indices(side),
                TractionNormalOperator(
                    bkd,
                    mesh.boundary_indices(side),
                    mesh.boundary_normals(side),
                    physics,
                    component,
                ),
                0.0,
                1.0,
                traction.get((side, component), 0.0),
                component,
            )
            for side, component in sides
        ],
    )
    essential = DirichletRows(
        bkd, npts, 2, [DirichletPart(left, 0.0, 0), DirichletPart(left, 0.0, 1)]
    )
    system = CollocationSystem(physics, essential, [closure])
    parameterization = CollocationElasticityParameterization(
        physics,
        youngs_modulus_map=_kle(bkd, npts, 2.5),
        poisson_ratio=0.25,
        bkd=bkd,
    )
    return _Case(bkd, physics, system, parameterization)


class TestEqualsTheOldPath:
    """Value, R_u and R_p equal CollocationParameterizedStateEquation's to
    1e-12 relative to the largest entry."""

    def test_dirichlet_1d(self, bkd: Backend[Array]) -> None:
        _dirichlet_1d(bkd).compare()

    def test_flux_neumann_1d(self, bkd: Backend[Array]) -> None:
        _flux_neumann_1d(bkd).compare()

    def test_dirichlet_and_robin_2d(self, bkd: Backend[Array]) -> None:
        _robin_2d(bkd).compare()

    def test_periodic_2d(self, bkd: Backend[Array]) -> None:
        _periodic_2d(bkd).compare()

    def test_traction_2d(self, bkd: Backend[Array]) -> None:
        _traction_2d(bkd).compare(scale=0.01)


def _newton_solve(case: _Case[Array]) -> Tuple[Array, int]:
    bkd = case.bkd
    system = SteadySystem(
        case.residual,
        backend_state_space(bkd, case.nstates),
        RecomputeAlways(BackendDenseLU(bkd)),
    )
    newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=30)
    point = system.solve(newton, bkd.zeros((case.nstates, 1)))
    assert point.converged()
    return point.state()[:, 0], point.info().iterations


class TestSolve:
    @pytest.mark.parametrize("build", [_dirichlet_1d, _robin_2d, _periodic_2d])
    def test_equals_solve_steady(
        self, bkd: Backend[Array], build: Callable[[Backend[Array]], _Case[Array]]
    ) -> None:
        """SteadySystem + Newton (atol 1e-12) equals the old solve to
        1e-10."""
        case = build(bkd)
        state, _ = _newton_solve(case)
        old = case.old.solve(bkd.zeros((case.nstates, 1)), case.param[:, None])[:, 0]
        bkd.assert_allclose(state, old, rtol=1e-10, atol=1e-10)

    def test_linear_problem_in_one_step(self, bkd: Backend[Array]) -> None:
        """Derivation-free: ADR with Dirichlet and linear Robin rows is
        linear in u, so one Newton step from zero solves it."""
        _, iterations = _newton_solve(_robin_2d(bkd))
        assert iterations == 1


def _growing(coords: Array, time: float) -> Array:
    """A positive datum varying in space and time."""
    return (1.0 + time) * (2.0 + coords[0] ** 2)


class _GrowingOnSide(Generic[Array]):
    """``_growing`` on a side's nodes, as the old DirichletBC reads a
    time-dependent datum."""

    def __init__(self, points: Array) -> None:
        self._points = points

    def __call__(self, time: float) -> Array:
        return _growing(self._points, time)


def _time_dependent_case(
    bkd: Backend[Array],
) -> Tuple[AdvectionDiffusionReaction[Array], CollocationSystem[Array]]:
    """2D ADR with forcing (1 + t)(2 + x^2), the old Dirichlet data the
    same on the left side, and gradient Robin rows on the bottom."""
    mesh = TransformedMesh2D(6, 6, bkd)
    points = mesh.points()
    supplier: TimeDependent[Array] = TimeDependent(_growing)
    physics, mesh, basis = _adr_2d(bkd, NodalSupplier(supplier, points))
    npts = basis.npts()
    left, bottom = mesh.boundary_indices(0), mesh.boundary_indices(2)
    derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
    physics.set_boundary_conditions(
        [
            gradient_robin_bc(
                bkd, bottom, mesh.boundary_normals(2), derivatives, 2.0, 0.5, 0.3
            ),
            DirichletBC(bkd, left, _GrowingOnSide(points[:, left])),
        ]
    )
    robin = RobinRows(
        bkd,
        npts,
        1,
        [
            RobinPart(
                bottom,
                GradientNormalOperator(
                    bkd, bottom, mesh.boundary_normals(2), derivatives
                ),
                2.0,
                0.5,
                0.3,
            )
        ],
    )
    essential = DirichletRows(
        bkd,
        npts,
        1,
        [DirichletPart(left, NodalSupplier(supplier, points[:, left]))],
    )
    return physics, CollocationSystem(physics, essential, [robin])


class TestTime:
    def test_time_dependent_coefficient_refused(self, bkd: Backend[Array]) -> None:
        _, system = _time_dependent_case(bkd)
        with pytest.raises(ValueError, match="AdvectionDiffusionReaction"):
            collocation_steady_residual(system)

    def test_time_dependent_dirichlet_data_refused(self, bkd: Backend[Array]) -> None:
        """Steady forcing, so only the Dirichlet data are time-dependent."""
        mesh = TransformedMesh2D(6, 6, bkd)
        points = mesh.points()
        physics, mesh, _ = _adr_2d(bkd, _steady_forcing(bkd))
        left = mesh.boundary_indices(0)
        supplier: TimeDependent[Array] = TimeDependent(_growing)
        essential = DirichletRows(
            bkd,
            physics.npts(),
            1,
            [DirichletPart(left, NodalSupplier(supplier, points[:, left]))],
        )
        with pytest.raises(ValueError, match="essential data are time-dependent"):
            collocation_steady_residual(CollocationSystem(physics, essential))

    def test_given_time_equals_the_old_path_there(self, bkd: Backend[Array]) -> None:
        """At t = 0.7 the value and R_u equal the old adapter's with the old
        conditions applied at 0.7."""
        time = 0.7
        physics, system = _time_dependent_case(bkd)
        residual = collocation_steady_residual(system, time=time)
        state = bkd.asarray(np.random.default_rng(4).normal(0.0, 0.5, physics.npts()))
        adapter = create_collocation_model(physics, bkd).adapter()
        old_value, old_jacobian = physics.apply_boundary_conditions(
            adapter(state, time), adapter.jacobian(state, time), state, time
        )
        _close(bkd, residual(state[:, None])[:, 0], old_value)
        jacobian = residual.derivatives().jacobian
        assert jacobian is not None
        _close(
            bkd,
            _dense(bkd, jacobian(state[:, None], _POINT), physics.npts()),
            old_jacobian,
        )

    def test_flux_rows_reading_a_time_dependent_diffusion_refused(
        self, bkd: Backend[Array]
    ) -> None:
        """The flux declares its time dependence from the diffusion it
        reads: steady forcing, so only the closure is time-dependent."""
        mesh = TransformedMesh2D(6, 6, bkd)
        supplier: TimeDependent[Array] = TimeDependent(_growing)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=NodalSupplier(supplier, mesh.points()),
            forcing=_steady_forcing(bkd),
        )
        assert physics.flux_is_time_dependent()
        bottom = mesh.boundary_indices(2)
        flux = RobinRows(
            bkd,
            npts,
            1,
            [
                RobinPart(
                    bottom,
                    FluxNormalOperator(bkd, bottom, mesh.boundary_normals(2), physics),
                    0.0,
                    1.0,
                    0.0,
                )
            ],
        )
        system = CollocationSystem(physics, None, [flux])
        # The interior is time-dependent too (it reads the diffusion), so
        # check the closure's own declaration and the piece named first
        assert not flux.is_time_invariant()
        with pytest.raises(ValueError, match="AdvectionDiffusionReaction"):
            collocation_steady_residual(system)

    def test_time_dependent_closure_data_refused(self, bkd: Backend[Array]) -> None:
        """Steady physics; gradient Robin data (1 + t)(2 + x^2) on the
        bottom: the closure is the piece named."""
        physics, mesh, basis = _adr_2d(bkd, _steady_forcing(bkd))
        bottom = mesh.boundary_indices(2)
        supplier: TimeDependent[Array] = TimeDependent(_growing)
        derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
        robin = RobinRows(
            bkd,
            basis.npts(),
            1,
            [
                RobinPart(
                    bottom,
                    GradientNormalOperator(
                        bkd, bottom, mesh.boundary_normals(2), derivatives
                    ),
                    1.0,
                    1.0,
                    NodalSupplier(supplier, mesh.points()[:, bottom]),
                )
            ],
        )
        with pytest.raises(ValueError, match=r"closure 0 \(RobinRows\)"):
            collocation_steady_residual(CollocationSystem(physics, None, [robin]))
