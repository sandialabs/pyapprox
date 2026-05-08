"""Tests for Burgers physics implementation."""

import math

import numpy as np

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.manufactured_solutions.burgers import (
    ManufacturedBurgers1D,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.burgers import (
    BurgersPhysics1D,
    create_burgers_1d,
)
from tests._helpers.physics_test_utils import (
    PhysicsNewtonResidual,
    PhysicsTestBase,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.util.rootfinding.newton import NewtonSolver


class TestBurgersPhysics(PhysicsTestBase):
    """Tests for BurgersPhysics1D."""

    def test_jacobian_derivative_checker(self, bkd):
        """Test Jacobian matches finite differences using DerivativeChecker."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = BurgersPhysics1D(basis, bkd, viscosity=0.1)

        # Random state (positive to avoid any issues)
        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_non_conservative(self, bkd):
        """Test Jacobian for non-conservative form."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = BurgersPhysics1D(basis, bkd, viscosity=0.05, conservative=False)

        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_with_forcing(self, bkd):
        """Test Jacobian with non-zero forcing."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        forcing = 0.5 * bkd.sin(math.pi * nodes)

        physics = BurgersPhysics1D(basis, bkd, viscosity=0.1, forcing=lambda t: forcing)

        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_residual_at_manufactured_solution(self, bkd):
        """BEFORE: Verify residual = 0 at manufactured solution (no BCs)."""
        npts = 25
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # Manufactured solution for 1D Burgers
        # Use smooth solution that vanishes at boundaries
        nu = 0.1
        man_sol = ManufacturedBurgers1D(
            sol_str="sin(pi*x)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()

        # Get forcing from manufactured solution
        forcing = man_sol.functions["forcing"](nodes[None, :])
        if forcing.ndim == 2:
            forcing = forcing[:, 0]

        physics = BurgersPhysics1D(basis, bkd, viscosity=nu, forcing=lambda t: forcing)

        # Get exact solution
        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        # Check residual is near zero at manufactured solution
        self.check_residual_zero(bkd, physics, exact_solution, atol=1e-8)

    def test_residual_polynomial(self, bkd):
        """Test residual with polynomial solution.

        For u(x) = x^2 on [-1, 1]:
        du/dx = 2x
        d²u/dx² = 2
        u * du/dx = 2x^3
        Residual = nu*2 - 2*x^3 = 2*nu - 2*x^3
        """
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        nu = 0.5
        physics = BurgersPhysics1D(basis, bkd, viscosity=nu)

        # u = x^2
        nodes = basis.nodes()
        u = nodes**2

        residual = physics.residual(u, time=0.0)

        # Expected: nu * d²(x²)/dx² - x² * d(x²)/dx = nu*2 - x²*2x = 2*nu - 2*x³
        expected = 2.0 * nu - 2.0 * nodes**3
        bkd.assert_allclose(residual, expected, atol=1e-10)

    def test_solve_steady_from_wrong_initial_guess(self, bkd):
        """AFTER: Verify convergence to exact solution from wrong guess."""
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # For steady Burgers with nu * u'' - u * u' + f = 0
        # Choose simple manufactured solution: u = sin(pi*x)
        # with homogeneous Dirichlet BCs
        nu = 0.1
        man_sol = ManufacturedBurgers1D(
            sol_str="sin(pi*x)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()

        # Get forcing from manufactured solution
        forcing = man_sol.functions["forcing"](nodes[None, :])
        if forcing.ndim == 2:
            forcing = forcing[:, 0]

        physics = BurgersPhysics1D(basis, bkd, viscosity=nu, forcing=lambda t: forcing)

        # Set boundary conditions: u(-1) = 0, u(1) = 0
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        # Get exact solution
        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        # Newton solve from wrong initial guess using NewtonSolver
        initial_guess = bkd.zeros((npts,))
        residual_wrapper = PhysicsNewtonResidual(physics, time=0.0)
        solver = NewtonSolver(residual_wrapper)
        solver.set_options(maxiters=30, atol=1e-12, rtol=1e-12)
        solution = solver.solve(initial_guess)

        bkd.assert_allclose(solution, exact_solution, atol=1e-8)

    def test_factory_function(self, bkd):
        """Test create_burgers_1d factory function."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = create_burgers_1d(basis, bkd, viscosity=0.1)

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts
        assert abs(physics.viscosity() - 0.1) < 1e-7

    def test_transient_manufactured_solution(self, bkd):
        """Test transient evolution with time-dependent manufactured solution.

        For Burgers: du/dt = nu*d²u/dx² - u*du/dx + f
        Use manufactured solution u(x,t) = exp(-t)*sin(pi*x) which satisfies:
        du/dt = -exp(-t)*sin(pi*x)
        """
        npts = 25
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        nu = 0.1

        # Manufactured solution with time dependence: u = exp(-t)*sin(pi*x)
        man_sol = ManufacturedBurgers1D(
            sol_str="exp(-T)*sin(pi*x)",  # T is time symbol
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        # Forcing from manufactured solution
        def get_forcing(t):
            # Get time-dependent forcing
            forcing_2d = man_sol.functions["forcing"](nodes[None, :], t)
            return forcing_2d[:, 0] if forcing_2d.ndim == 2 else forcing_2d

        physics = BurgersPhysics1D(basis, bkd, viscosity=nu, forcing=get_forcing)

        # Set homogeneous Dirichlet BCs (sin(pi*x) = 0 at x = ±1)
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial condition at t=0
        u0 = bkd.sin(math.pi * nodes)

        # Time integration
        final_time = 0.5
        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(u0, config)

        # Compare to exact solution at final time
        u_exact_final = math.exp(-final_time) * bkd.sin(math.pi * nodes)
        # Use atol for boundary points near zero
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.05, atol=1e-10)

    def test_transient_manufactured_solution_crank_nicolson(self, bkd):
        """Test transient Burgers with Crank-Nicolson and manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev basis) and
        quadratic-in-time (exact for CN): u = (1-x^2)^2 * (1+T+T^2).
        With manufactured forcing, only spatial discretization error remains.
        """
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        nu = 0.1

        man_sol = ManufacturedBurgers1D(
            sol_str="(1 - x**2)**2*(1 + T + T**2)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        def get_forcing(t):
            forcing_2d = man_sol.functions["forcing"](nodes[None, :], t)
            return forcing_2d[:, 0] if forcing_2d.ndim == 2 else forcing_2d

        physics = BurgersPhysics1D(basis, bkd, viscosity=nu, forcing=get_forcing)

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        u0 = man_sol.functions["solution"](nodes[None, :], 0.0)

        final_time = 0.1
        config = TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(u0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact = man_sol.functions["solution"](nodes[None, :], t_final)

        bkd.assert_allclose(solutions[:, -1], u_exact, atol=1e-8)

    def test_transient_diffusion_dominated(self, bkd):
        """Test transient Burgers with high viscosity (diffusion dominated).

        When viscosity is large relative to convection, solution should
        behave like heat equation.
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # High viscosity makes it diffusion-dominated
        nu = 1.0
        physics = BurgersPhysics1D(basis, bkd, viscosity=nu)

        # Set homogeneous Dirichlet BCs
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial condition: sin(pi*x)
        u0 = bkd.sin(math.pi * nodes)

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.5,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(u0, config)

        # For diffusion-dominated, solution should decay like exp(-nu*pi²*t)
        # This is approximate since there's still some nonlinear convection
        decay_rate = nu * math.pi**2
        expected_decay = math.exp(-decay_rate * 0.5)

        # Check that solution has decayed
        initial_norm = float(bkd.norm(u0))
        final_norm = float(bkd.norm(solutions[:, -1]))
        assert final_norm / initial_norm < expected_decay * 1.5
