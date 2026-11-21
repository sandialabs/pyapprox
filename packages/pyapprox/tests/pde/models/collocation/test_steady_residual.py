"""Tests for the collocation steady residual: a linear problem solves in
one Newton step, and the builder refuses time-dependent pieces when no
time is given."""

import math
from typing import Tuple, Union

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import (
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.models.collocation.closed_system import (
    CollocationSystem,
)
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton


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


def _robin_2d(bkd: Backend[Array]) -> CollocationSystem[Array]:
    """Dirichlet left 0.5 and right -0.3; gradient Robin bottom and top
    (alpha 2, beta 0.5, g 0.3 and 0.4)."""
    physics, mesh, basis = _adr_2d(bkd, _steady_forcing(bkd))
    npts = basis.npts()
    derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
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
    return CollocationSystem(physics, essential, [robin])


class TestSolve:
    def test_linear_problem_in_one_step(self, bkd: Backend[Array]) -> None:
        """Derivation-free: ADR with Dirichlet and linear Robin rows is
        linear in u, so one Newton step from zero solves it."""
        system = _robin_2d(bkd)
        nstates = system.nstates()
        steady = SteadySystem(
            collocation_steady_residual(system),
            backend_state_space(bkd, nstates),
            RecomputeAlways(BackendDenseLU(bkd)),
        )
        newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=30)
        point = steady.solve(newton, bkd.zeros((nstates, 1)))
        assert point.converged()
        assert point.info().iterations == 1


def _growing(coords: Array, time: float) -> Array:
    """A positive datum varying in space and time."""
    return (1.0 + time) * (2.0 + coords[0] ** 2)


def _time_dependent_case(
    bkd: Backend[Array],
) -> Tuple[AdvectionDiffusionReaction[Array], CollocationSystem[Array]]:
    """2D ADR with forcing (1 + t)(2 + x^2), the Dirichlet data the same
    on the left side, and gradient Robin rows on the bottom."""
    mesh = TransformedMesh2D(6, 6, bkd)
    points = mesh.points()
    supplier: TimeDependent[Array] = TimeDependent(_growing)
    physics, mesh, basis = _adr_2d(bkd, NodalSupplier(supplier, points))
    npts = basis.npts()
    left, bottom = mesh.boundary_indices(0), mesh.boundary_indices(2)
    derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
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
