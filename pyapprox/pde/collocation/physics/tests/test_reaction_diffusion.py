"""Tests for reaction-diffusion physics implementations."""

import math

import numpy as np

from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedTwoSpeciesReactionDiffusion,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.fitzhugh_nagumo import (
    FitzHughNagumoPhysics,
    create_fitzhugh_nagumo,
)
from pyapprox.pde.collocation.physics.reaction_diffusion import (
    FitzHughNagumoReaction,
    LinearReaction,
    TwoSpeciesReactionDiffusionPhysics,
    create_two_species_reaction_diffusion,
)
from pyapprox.pde.collocation.physics.tests.test_utils import (
    PhysicsNewtonResidual,
    PhysicsTestBase,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)


class TestTwoSpeciesReactionDiffusion(PhysicsTestBase):
    """Tests for TwoSpeciesReactionDiffusionPhysics."""

    def test_jacobian_linear_reaction(self, bkd):
        """Test Jacobian with linear reaction."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # Linear reaction: R0 = -u0 + 0.5*u1, R1 = 0.5*u0 - u1
        reaction = LinearReaction(-1.0, 0.5, 0.5, -1.0, bkd)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd, diffusion0=0.1, diffusion1=0.05, reaction=reaction
        )

        # Random state
        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_fhn_reaction(self, bkd):
        """Test Jacobian with FitzHugh-Nagumo reaction."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        reaction = FitzHughNagumoReaction(
            alpha=0.1, eps=0.01, beta=0.5, gamma=1.0, bkd=bkd
        )

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd, diffusion0=1e-3, diffusion1=0.0, reaction=reaction
        )

        # State in reasonable range for FHN
        state = bkd.array(0.5 + 0.2 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_with_forcing(self, bkd):
        """Test Jacobian with forcing terms."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)
        forcing0 = bkd.sin(math.pi * nodes)
        forcing1 = bkd.cos(math.pi * nodes)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis,
            bkd,
            diffusion0=0.1,
            diffusion1=0.1,
            reaction=reaction,
            forcing0=lambda t: forcing0,
            forcing1=lambda t: forcing1,
        )

        state = bkd.array(np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_residual_at_manufactured_solution(self, bkd):
        """BEFORE: Verify residual = 0 at manufactured solution."""
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Manufactured solution: u0 = sin(pi*x), u1 = cos(pi*x)
        # For pure diffusion with no reaction:
        # D0 * u0'' = D0 * (-pi^2) * sin(pi*x) = f0
        # D1 * u1'' = D1 * (-pi^2) * cos(pi*x) = f1
        D0 = 0.1
        D1 = 0.05
        pi_sq = math.pi**2

        exact_u0 = bkd.sin(math.pi * nodes)
        exact_u1 = bkd.cos(math.pi * nodes)
        exact_solution = bkd.hstack([exact_u0, exact_u1])

        # For zero reaction, forcing must cancel the diffusion
        forcing0 = -D0 * (-pi_sq) * exact_u0  # = D0 * pi^2 * sin(pi*x)
        forcing1 = -D1 * (-pi_sq) * exact_u1  # = D1 * pi^2 * cos(pi*x)

        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis,
            bkd,
            diffusion0=D0,
            diffusion1=D1,
            reaction=reaction,
            forcing0=lambda t: forcing0,
            forcing1=lambda t: forcing1,
        )

        self.check_residual_zero(bkd, physics, exact_solution, atol=1e-8)

    def test_solve_steady_linear_system(self, bkd):
        """AFTER: Verify convergence for linear reaction-diffusion."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Simple test: u0 = u1 = 1 - x^2 (satisfies u(-1) = u(1) = 0)
        # Laplacian = -2
        # With no reaction, forcing = D * 2
        D0 = 0.1
        D1 = 0.05

        exact_u0 = 1.0 - nodes**2
        exact_u1 = 1.0 - nodes**2
        exact_solution = bkd.hstack([exact_u0, exact_u1])

        forcing0 = bkd.full((npts,), D0 * 2.0)
        forcing1 = bkd.full((npts,), D1 * 2.0)

        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis,
            bkd,
            diffusion0=D0,
            diffusion1=D1,
            reaction=reaction,
            forcing0=lambda t: forcing0,
            forcing1=lambda t: forcing1,
        )

        # Set BCs for both species at both boundaries
        # For vector physics with state [u0, u1], u0 uses indices 0:npts
        # and u1 uses indices npts:2*npts
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)

        # u0 boundaries at indices left_idx and right_idx
        bc_u0_left = zero_dirichlet_bc(bkd, bkd.array([left_idx]))
        bc_u0_right = zero_dirichlet_bc(bkd, bkd.array([right_idx]))
        # u1 boundaries at indices npts + left_idx and npts + right_idx
        bc_u1_left = zero_dirichlet_bc(bkd, bkd.array([npts + left_idx]))
        bc_u1_right = zero_dirichlet_bc(bkd, bkd.array([npts + right_idx]))

        physics.set_boundary_conditions(
            [bc_u0_left, bc_u0_right, bc_u1_left, bc_u1_right]
        )

        # Newton solve from wrong initial guess using NewtonSolver
        initial_guess = bkd.zeros((2 * npts,))
        residual_wrapper = PhysicsNewtonResidual(physics, time=0.0)
        solver = NewtonSolver(residual_wrapper)
        solver.set_options(maxiters=30, atol=1e-12, rtol=1e-12)
        solution = solver.solve(initial_guess)

        bkd.assert_allclose(solution, exact_solution, atol=1e-8)

    def test_factory_function(self, bkd):
        """Test create_two_species_reaction_diffusion factory."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics = create_two_species_reaction_diffusion(
            basis,
            bkd,
            diffusion0=0.1,
            diffusion1=0.05,
            reaction=reaction,
        )

        assert physics.ncomponents() == 2
        assert physics.nstates() == 2 * npts

    def test_transient_diffusion_only(self, bkd):
        """Test transient evolution with pure diffusion (no reaction).

        For u0, u1 = sin(pi*x) with homogeneous Dirichlet BCs and no reaction:
        du/dt = D * d²u/dx²
        Exact solution: u(x,t) = exp(-D*pi²*t) * sin(pi*x)
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        D0, D1 = 0.1, 0.05
        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd, diffusion0=D0, diffusion1=D1, reaction=reaction
        )

        # Set homogeneous Dirichlet BCs for both species
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_u0_left = zero_dirichlet_bc(bkd, bkd.array([left_idx]))
        bc_u0_right = zero_dirichlet_bc(bkd, bkd.array([right_idx]))
        bc_u1_left = zero_dirichlet_bc(bkd, bkd.array([npts + left_idx]))
        bc_u1_right = zero_dirichlet_bc(bkd, bkd.array([npts + right_idx]))
        physics.set_boundary_conditions(
            [bc_u0_left, bc_u0_right, bc_u1_left, bc_u1_right]
        )

        model = CollocationModel(physics, bkd)

        # Initial conditions
        u0_init = bkd.sin(math.pi * nodes)
        u1_init = bkd.sin(math.pi * nodes)
        state0 = bkd.hstack([u0_init, u1_init])

        final_time = 0.1
        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=final_time,
            deltat=0.005,
        )

        solutions, times = model.solve_transient(state0, config)

        # Exact solutions at final time
        u0_exact = math.exp(-D0 * math.pi**2 * final_time) * bkd.sin(math.pi * nodes)
        u1_exact = math.exp(-D1 * math.pi**2 * final_time) * bkd.sin(math.pi * nodes)
        exact_final = bkd.hstack([u0_exact, u1_exact])

        bkd.assert_allclose(solutions[:, -1], exact_final, rtol=0.02, atol=1e-10)

    def test_transient_diffusion_only_crank_nicolson(self, bkd):
        """Test transient reaction-diffusion with CN and manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev) and quadratic-in-time
        (exact for CN) with zero reaction. Tight tolerance since both spatial
        and temporal discretization errors are near machine precision.
        """
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        D0, D1 = 0.1, 0.05
        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=[
                "(1 - x**2)**2*(1 + T + T**2)",
                "(1 - x**2)**2*x*(1 + T + T**2)",
            ],
            nvars=1,
            diff_strs=[str(D0), str(D1)],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        def forcing0_fn(t):
            forcing = man_sol.functions["forcing"](nodes[None, :], t)
            return forcing[:, 0]

        def forcing1_fn(t):
            forcing = man_sol.functions["forcing"](nodes[None, :], t)
            return forcing[:, 1]

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis,
            bkd,
            diffusion0=D0,
            diffusion1=D1,
            reaction=reaction,
            forcing0=forcing0_fn,
            forcing1=forcing1_fn,
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_u0_left = zero_dirichlet_bc(bkd, bkd.array([left_idx]))
        bc_u0_right = zero_dirichlet_bc(bkd, bkd.array([right_idx]))
        bc_u1_left = zero_dirichlet_bc(bkd, bkd.array([npts + left_idx]))
        bc_u1_right = zero_dirichlet_bc(bkd, bkd.array([npts + right_idx]))
        physics.set_boundary_conditions(
            [bc_u0_left, bc_u0_right, bc_u1_left, bc_u1_right]
        )

        model = CollocationModel(physics, bkd)

        u_exact_0 = man_sol.functions["solution"](nodes[None, :], 0.0)
        state0 = bkd.concatenate([u_exact_0[:, 0], u_exact_0[:, 1]])

        final_time = 0.1
        config = TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact_final = man_sol.functions["solution"](nodes[None, :], t_final)
        exact_final = bkd.concatenate([u_exact_final[:, 0], u_exact_final[:, 1]])

        bkd.assert_allclose(solutions[:, -1], exact_final, atol=1e-8)


class TestFitzHughNagumoPhysics(PhysicsTestBase):
    """Tests for FitzHughNagumoPhysics."""

    def test_jacobian_derivative_checker(self, bkd):
        """Test Jacobian matches finite differences."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(
            basis, bkd, diffusion_v=1e-3, alpha=0.1, eps=0.01, beta=0.5, gamma=1.0
        )

        # State in reasonable FHN range
        np.random.seed(42)
        state = bkd.array(0.3 + 0.2 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_different_parameters(self, bkd):
        """Test Jacobian with different FHN parameters."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # Different parameter values
        physics = FitzHughNagumoPhysics(
            basis, bkd, diffusion_v=5e-4, alpha=0.2, eps=0.1, beta=1.0, gamma=0.5
        )

        np.random.seed(123)
        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_parameter_accessors(self, bkd):
        """Test parameter accessor methods."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(
            basis, bkd, alpha=0.15, eps=0.02, beta=0.6, gamma=0.9
        )

        assert abs(physics.alpha() - 0.15) < 1e-7
        assert abs(physics.eps() - 0.02) < 1e-7
        assert abs(physics.beta() - 0.6) < 1e-7
        assert abs(physics.gamma() - 0.9) < 1e-7

    def test_set_parameters(self, bkd):
        """Test updating FHN parameters."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(basis, bkd)

        # Update parameters
        physics.set_parameters(alpha=0.2, eps=0.05)

        assert abs(physics.alpha() - 0.2) < 1e-7
        assert abs(physics.eps() - 0.05) < 1e-7
        # Unchanged parameters
        assert abs(physics.beta() - 0.5) < 1e-7
        assert abs(physics.gamma() - 1.0) < 1e-7

    def test_factory_function(self, bkd):
        """Test create_fitzhugh_nagumo factory."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = create_fitzhugh_nagumo(
            basis, bkd, diffusion_v=1e-3, alpha=0.1, eps=0.01
        )

        assert physics.ncomponents() == 2
        assert physics.nstates() == 2 * npts
        assert abs(physics.alpha() - 0.1) < 1e-7

    def test_no_diffusion_on_recovery(self, bkd):
        """Verify recovery variable has zero diffusion."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(basis, bkd, diffusion_v=1e-3)

        d0, d1 = physics.diffusion()
        assert abs(d0 - 1e-3) < 1e-7
        assert abs(d1 - 0.0) < 1e-7

    def test_transient_evolution(self, bkd):
        """Test FitzHugh-Nagumo transient behavior.

        FHN is inherently transient with excitable dynamics. We test that
        the system evolves and doesn't blow up with physically reasonable
        initial conditions.
        """
        npts = 25
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        physics = FitzHughNagumoPhysics(
            basis, bkd, diffusion_v=1e-2, alpha=0.1, eps=0.01, beta=0.5, gamma=1.0
        )

        model = CollocationModel(physics, bkd)

        # Initial condition: small localized excitation
        v0 = 0.5 * bkd.exp(-10.0 * nodes**2)  # Excitatory variable
        w0 = bkd.zeros((npts,))  # Recovery variable starts at rest
        state0 = bkd.hstack([v0, w0])

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=1.0,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(state0, config)

        # Check solution is finite and bounded
        final_state = solutions[:, -1]
        assert bkd.isfinite(bkd.norm(final_state))
        assert float(bkd.norm(final_state)) < 100.0

        # Check that the system evolved (not stuck at initial)
        initial_norm = float(bkd.norm(state0))
        final_norm = float(bkd.norm(final_state))
        # FHN dynamics should change the state
        assert abs(initial_norm - final_norm) >= 10**(-2)


class TestFitzHughNagumoReaction:
    """Unit tests for FitzHughNagumoReaction class."""

    def test_reaction_values(self, bkd):
        """Test reaction term evaluation."""
        reaction = FitzHughNagumoReaction(0.1, 0.01, 0.5, 1.0, bkd)

        u0 = bkd.array([0.0, 0.5, 1.0])
        u1 = bkd.array([0.0, 0.1, 0.0])

        R0, R1 = reaction(u0, u1)

        # Manual calculation:
        # R0[0] = 0*(1-0)*(0-0.1) - 0 = 0
        # R0[1] = 0.5*(1-0.5)*(0.5-0.1) - 0.1 = 0.5*0.5*0.4 - 0.1 = 0.1 - 0.1 = 0
        # R0[2] = 1*(1-1)*(1-0.1) - 0 = 0
        expected_R0 = bkd.array([0.0, 0.0, 0.0])
        bkd.assert_allclose(R0, expected_R0, atol=1e-10)

        # R1[0] = 0.01*(0.5*0 - 1*0) = 0
        # R1[1] = 0.01*(0.5*0.5 - 1*0.1) = 0.01*(0.25 - 0.1) = 0.0015
        # R1[2] = 0.01*(0.5*1 - 1*0) = 0.005
        expected_R1 = bkd.array([0.0, 0.0015, 0.005])
        bkd.assert_allclose(R1, expected_R1, atol=1e-10)

    def test_reaction_jacobian(self, bkd):
        """Test reaction Jacobian computation."""
        reaction = FitzHughNagumoReaction(0.1, 0.01, 0.5, 1.0, bkd)

        u0 = bkd.array([0.5])
        u1 = bkd.array([0.1])

        dR0_du0, dR0_du1, dR1_du0, dR1_du1 = reaction.jacobian(u0, u1)

        # Check against finite differences
        eps = 1e-7
        R0_base, R1_base = reaction(u0, u1)

        R0_plus, _ = reaction(u0 + eps, u1)
        fd_dR0_du0 = (R0_plus - R0_base) / eps
        bkd.assert_allclose(dR0_du0, fd_dR0_du0, atol=1e-5)

        _, R1_plus = reaction(u0 + eps, u1)
        fd_dR1_du0 = (R1_plus - R1_base) / eps
        bkd.assert_allclose(dR1_du0, fd_dR1_du0, atol=1e-5)
