"""Tests for Helmholtz physics implementation."""

import unittest
import math
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import create_uniform_mesh_1d
from pyapprox.typing.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.typing.pde.collocation.physics.helmholtz import (
    HelmholtzPhysics,
    create_helmholtz,
)
from pyapprox.typing.pde.collocation.physics.tests.test_utils import (
    PhysicsTestBase,
    PhysicsNewtonResidual,
)
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
from pyapprox.typing.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)


class TestHelmholtzPhysics(PhysicsTestBase):
    """Tests for HelmholtzPhysics."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()

    def test_jacobian_derivative_checker(self):
        """Test Jacobian matches finite differences using DerivativeChecker."""
        bkd = self.bkd()
        npts = 15
        basis = ChebyshevBasis1D(npts, bkd)

        physics = HelmholtzPhysics(basis, bkd, wave_number_sq=2.0)

        # Random state
        state = bkd.array(np.random.randn(physics.nstates()))

        self.check_jacobian(physics, state, time=0.0)

    def test_jacobian_with_forcing(self):
        """Test Jacobian with non-zero forcing."""
        bkd = self.bkd()
        npts = 12
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Add forcing term
        forcing = bkd.cos(nodes)

        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=2.0, forcing=lambda t: forcing
        )

        state = bkd.array(np.random.randn(physics.nstates()))

        self.check_jacobian(physics, state, time=0.0)

    def test_residual_at_manufactured_solution(self):
        """BEFORE: Verify residual = 0 at manufactured solution (no BCs)."""
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)

        # Manufactured solution for 1D Helmholtz
        # Helmholtz PDE: -u'' + k^2*u = f
        # Use u = sin(pi*x), k^2 = 1
        # u'' = -pi^2 * sin(pi*x)
        # -u'' = pi^2 * sin(pi*x)
        # k^2 * u = sin(pi*x)
        # f = pi^2 * sin(pi*x) + sin(pi*x) = (pi^2 + 1) * sin(pi*x)
        k_sq = 1.0
        nodes = basis.nodes()

        exact_solution = bkd.sin(math.pi * nodes)

        # Compute forcing for -Laplacian(u) + k^2*u = f
        # f = (pi^2 + k^2) * sin(pi*x)
        forcing = (math.pi ** 2 + k_sq) * bkd.sin(math.pi * nodes)

        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=k_sq, forcing=lambda t: forcing
        )

        # Check residual is near zero at manufactured solution
        self.check_residual_zero(physics, exact_solution, atol=1e-8)

    def test_solve_from_wrong_initial_guess(self):
        """AFTER: Verify convergence to exact solution from wrong guess."""
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Manufactured solution: u = (1 - x^2), satisfies u(-1) = u(1) = 0
        # -u'' + k^2*u = f where u = 1 - x^2
        # u' = -2x, u'' = -2
        # -(-2) + k^2*(1-x^2) = f
        # f = 2 + k^2*(1-x^2)
        k_sq = 1.0
        nodes = basis.nodes()
        exact_solution = 1.0 - nodes ** 2
        forcing = 2.0 + k_sq * (1.0 - nodes ** 2)

        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=k_sq, forcing=lambda t: forcing
        )

        # Set boundary conditions: u(-1) = 0, u(1) = 0
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        # Newton solve from wrong initial guess using NewtonSolver
        initial_guess = bkd.zeros((npts,))
        residual_wrapper = PhysicsNewtonResidual(physics, time=0.0)
        solver = NewtonSolver(residual_wrapper)
        solver.set_options(maxiters=20, atol=1e-12, rtol=1e-12)
        solution = solver.solve(initial_guess)

        bkd.assert_allclose(solution, exact_solution, atol=1e-10)

    def test_factory_function(self):
        """Test create_helmholtz factory function."""
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)

        physics = create_helmholtz(basis, bkd, wave_number_sq=2.0)

        self.assertEqual(physics.ncomponents(), 1)
        self.assertEqual(physics.nstates(), npts)
        self.assertAlmostEqual(physics.wave_number_sq(), 2.0)

    def test_transient_decay(self):
        """Test transient Helmholtz as reaction-diffusion decay.

        Helmholtz: du/dt = u'' - k²*u
        This is diffusion with decay. For u(x,0) = sin(pi*x) with
        homogeneous Dirichlet BCs, the exact solution is:
        u(x,t) = exp(-(pi² + k²)*t) * sin(pi*x)
        """
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        k_sq = 1.0
        physics = HelmholtzPhysics(basis, bkd, wave_number_sq=k_sq)

        # Set homogeneous Dirichlet BCs
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial condition: sin(pi*x)
        u0 = bkd.sin(math.pi * nodes)

        final_time = 0.1
        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=final_time,
            deltat=0.005,
        )

        solutions, times = model.solve_transient(u0, config)

        # Exact solution: exp(-(pi² + k²)*t) * sin(pi*x)
        decay_rate = math.pi ** 2 + k_sq
        u_exact_final = math.exp(-decay_rate * final_time) * bkd.sin(math.pi * nodes)

        # Use higher rtol due to time discretization error
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.05, atol=1e-10)

if __name__ == "__main__":
    unittest.main()
