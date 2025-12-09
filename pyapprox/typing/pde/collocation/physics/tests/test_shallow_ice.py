"""Tests for Shallow Ice physics implementation."""

import unittest
import math
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import create_uniform_mesh_1d
from pyapprox.typing.pde.collocation.boundary import (
    constant_dirichlet_bc,
)
from pyapprox.typing.pde.collocation.physics.shallow_ice import (
    ShallowIcePhysics,
    create_shallow_ice,
)
from pyapprox.typing.pde.collocation.physics.tests.test_utils import (
    PhysicsTestBase,
    PhysicsNewtonResidual,
)
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
from pyapprox.typing.pde.collocation.manufactured_solutions.shallow_ice import (
    ManufacturedShallowIce,
)
from pyapprox.typing.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)


class TestShallowIcePhysics(PhysicsTestBase):
    """Tests for ShallowIcePhysics."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()

    def test_jacobian_derivative_checker(self):
        """Test Jacobian matches finite differences using DerivativeChecker."""
        bkd = self.bkd()
        npts = 12
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Flat bed
        bed = bkd.zeros((npts,))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness (physically meaningful)
        state = 100.0 + 50.0 * (1.0 - nodes ** 2)

        self.check_jacobian(physics, state, time=0.0)

    def test_jacobian_sloped_bed(self):
        """Test Jacobian with sloped bed topography."""
        bkd = self.bkd()
        npts = 12
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Sloped bed
        bed = 100.0 * nodes

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness
        state = 150.0 + 30.0 * (1.0 - nodes ** 2)

        self.check_jacobian(physics, state, time=0.0)

    def test_residual_at_manufactured_solution(self):
        """Verify residual is near zero at manufactured solution.

        Uses normalized parameters (A=1, rho=1) to avoid numerical issues
        with the highly nonlinear shallow ice equation. With glaciological
        parameters (A~1e-16, rho~917), the forcing terms are O(1e10) which
        causes numerical issues with spectral methods.
        """
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Normalized parameters for numerical stability
        A = 1.0
        rho = 1.0

        # Manufactured solution with smooth positive ice thickness
        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",  # small sloped bed
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        # Get forcing from manufactured solution
        # Manufactured: dH/dt - div(D*grad(s)) = f → f = -div(D*grad(s))
        # Physics:      residual = div(D*grad(s)) + f
        # For residual = 0: f = -div(D*grad(s)) which is what manufactured computes
        forcing = man_sol.functions["forcing"](nodes[None, :])
        if forcing.ndim == 2:
            forcing = forcing[:, 0]
        # NO negation needed - sign conventions match

        # Get bed elevation
        bed = man_sol.functions["bed"](nodes[None, :])
        if bed.ndim == 2:
            bed = bed[:, 0]

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        # Get exact solution
        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        # Check residual is near zero at manufactured solution
        self.check_residual_zero(physics, exact_solution, atol=1e-6)

    def test_solve_steady_from_small_perturbation(self):
        """Verify Newton converges from small perturbation.

        Uses normalized parameters for numerical stability.
        Tests that NewtonSolver converges when starting close to solution.
        """
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Normalized parameters for numerical stability
        A = 1.0
        rho = 1.0

        # Manufactured solution
        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        # Get forcing (no negation needed - sign conventions match)
        forcing = man_sol.functions["forcing"](nodes[None, :])
        if forcing.ndim == 2:
            forcing = forcing[:, 0]

        # Get bed elevation
        bed = man_sol.functions["bed"](nodes[None, :])
        if bed.ndim == 2:
            bed = bed[:, 0]

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        # Set boundary conditions
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)

        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        bc_left = constant_dirichlet_bc(bkd, left_idx, float(exact_solution[left_idx]))
        bc_right = constant_dirichlet_bc(bkd, right_idx, float(exact_solution[right_idx]))
        physics.set_boundary_conditions([bc_left, bc_right])

        # Small perturbation from exact solution
        np.random.seed(42)
        perturbation = 0.01 * exact_solution * np.random.randn(npts)
        initial_guess = exact_solution + bkd.array(perturbation)

        # Solve with NewtonSolver
        residual_wrapper = PhysicsNewtonResidual(physics, time=0.0)
        solver = NewtonSolver(residual_wrapper)
        solver.set_options(maxiters=50, atol=1e-10, rtol=1e-10)
        solution = solver.solve(initial_guess)

        bkd.assert_allclose(solution, exact_solution, atol=1e-6)

    def test_factory_function(self):
        """Test create_shallow_ice factory function."""
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        bed = bkd.zeros((npts,))

        physics = create_shallow_ice(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        self.assertEqual(physics.ncomponents(), 1)
        self.assertEqual(physics.nstates(), npts)

    def test_transient_thickness_evolution(self):
        """Test transient evolution of ice thickness.

        Shallow ice: dH/dt = div(D*grad(s)) + f
        For a simple test, start with uniform thickness and observe evolution.
        """
        bkd = self.bkd()
        npts = 15
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Flat bed
        bed = bkd.zeros((npts,))

        # Accumulation (positive forcing = ice growth)
        accumulation = 0.1 * bkd.ones((npts,))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0,
            forcing=lambda t: accumulation
        )

        model = CollocationModel(physics, bkd)

        # Initial thickness: parabolic profile
        H0 = 100.0 + 50.0 * (1.0 - nodes ** 2)

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=1.0,
            deltat=0.1,
        )

        solutions, times = model.solve_transient(H0, config)

        # Check solution is finite and physical (H > 0)
        H_final = solutions[:, -1]
        self.assertTrue(bkd.isfinite(bkd.norm(H_final)))
        self.assertGreater(float(bkd.min(H_final)), 0.0)

        # With positive accumulation, ice should grow (mean thickness increases)
        mean_initial = float(np.mean(np.asarray(H0)))
        mean_final = float(np.mean(np.asarray(H_final)))
        self.assertGreater(mean_final, mean_initial)


if __name__ == "__main__":
    unittest.main()
