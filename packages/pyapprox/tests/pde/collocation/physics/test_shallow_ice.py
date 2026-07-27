"""Tests for Shallow Ice physics implementation."""


import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.shallow_ice import (
    ShallowIcePhysics,
    create_shallow_ice,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.manufactured.shallow_ice import (
    ManufacturedShallowIce,
)
from pyapprox.util.rootfinding.newton import NewtonSolver

from tests._helpers.physics_test_utils import (
    PhysicsDerivativeWrapper,
    PhysicsNewtonResidual,
    PhysicsTestBase,
)


class TestShallowIcePhysics(PhysicsTestBase):
    """Tests for ShallowIcePhysics."""

    def test_jacobian_derivative_checker(self, bkd):
        """Test Jacobian matches finite differences using DerivativeChecker."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Flat bed
        bed = bkd.zeros((npts,))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness (physically meaningful)
        state = 100.0 + 50.0 * (1.0 - nodes**2)

        # Exact Jacobian: limited only by the checker's FD noise
        # (measured sweep bottom ~1.8e-7)
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

    def test_jacobian_sloped_bed(self, bkd):
        """Test Jacobian with sloped bed topography."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Sloped bed
        bed = 100.0 * nodes

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness
        state = 150.0 + 30.0 * (1.0 - nodes**2)

        # Exact Jacobian: limited only by the checker's FD noise
        # (measured sweep bottom ~8e-8)
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

    def test_residual_at_manufactured_solution(self, bkd):
        """Verify residual is near zero at manufactured solution.

        Uses normalized parameters (A=1, rho=1) to avoid numerical issues
        with the highly nonlinear shallow ice equation. With glaciological
        parameters (A~1e-16, rho~917), the forcing terms are O(1e10) which
        causes numerical issues with spectral methods.
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
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
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho, forcing=lambda t: forcing
        )

        # Get exact solution
        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        # Spectral-truncation-limited (measured norm ~2.3e-7 at npts=20)
        self.check_residual_zero(bkd, physics, exact_solution, atol=5e-7)

    def test_solve_steady_from_small_perturbation(self, bkd):
        """Verify Newton converges from small perturbation.

        Uses normalized parameters for numerical stability.
        Tests that NewtonSolver converges when starting close to solution.
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
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
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho, forcing=lambda t: forcing
        )

        # Set boundary conditions
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)

        exact_solution = man_sol.functions["solution"](nodes[None, :])
        if exact_solution.ndim == 2:
            exact_solution = exact_solution[:, 0]

        li = bkd.to_int(left_idx)
        ri = bkd.to_int(right_idx)
        bc_left = constant_dirichlet_bc(
            bkd, left_idx, bkd.to_float(exact_solution[li])
        )
        bc_right = constant_dirichlet_bc(
            bkd, right_idx, bkd.to_float(exact_solution[ri])
        )
        physics.set_boundary_conditions([bc_left, bc_right])

        # Small perturbation from exact solution
        np.random.seed(42)
        pert_np = 0.01 * bkd.to_numpy(exact_solution) * np.random.randn(npts)
        initial_guess = exact_solution + bkd.array(pert_np)

        # Solve with NewtonSolver
        residual_wrapper = PhysicsNewtonResidual(physics, time=0.0)
        solver = NewtonSolver(residual_wrapper)
        solver.set_options(maxiters=50, atol=1e-10, rtol=1e-10)
        solution = solver.solve(initial_guess)

        # Newton-tolerance-limited (atol/rtol 1e-10; measured solution
        # error ~7.5e-13 with the exact Jacobian)
        bkd.assert_allclose(solution, exact_solution, atol=1e-11)

    def test_factory_function(self, bkd):
        """Test create_shallow_ice factory function."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        bed = bkd.zeros((npts,))

        physics = create_shallow_ice(
            basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts

    def _setup_transient_shallow_ice(self, bkd):
        """Set up transient shallow ice with manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev) and quadratic-in-time
        (exact for CN). Boundary values are constant: H(±1,t)=2.
        Normalized parameters (A=1, rho=1) for numerical stability.
        """
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        bc_mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)*(1 + T + T**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        def forcing_fn(t):
            return man_sol.functions["forcing"](nodes[None, :], t)

        bed = man_sol.functions["bed"](nodes[None, :])
        if bed.ndim == 2:
            bed = bed[:, 0]

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho, forcing=forcing_fn
        )

        exact_at_0 = man_sol.functions["solution"](nodes[None, :], 0.0)
        left_idx = bc_mesh.boundary_indices(0)
        right_idx = bc_mesh.boundary_indices(1)
        li = bkd.to_int(left_idx)
        ri = bkd.to_int(right_idx)
        bc_left = constant_dirichlet_bc(
            bkd, left_idx, bkd.to_float(exact_at_0[li])
        )
        bc_right = constant_dirichlet_bc(
            bkd, right_idx, bkd.to_float(exact_at_0[ri])
        )
        physics.set_boundary_conditions([bc_left, bc_right])

        return bkd, npts, nodes, man_sol, physics

    def _run_transient_shallow_ice(self, bkd, method, atol) :
        """Run transient shallow ice test with given method and tolerance."""
        bkd, npts, nodes, man_sol, physics = self._setup_transient_shallow_ice(bkd)
        model = CollocationModel(physics, bkd)

        state0 = man_sol.functions["solution"](nodes[None, :], 0.0)

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
            newton_tol=1e-10,
            newton_maxiter=20,
            lumped_mass=False,
            verbosity=0,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        exact_final = man_sol.functions["solution"](nodes[None, :], t_final)

        bkd.assert_allclose(solutions[:, -1], exact_final, atol=atol)

    @staticmethod
    def _make_shallow_ice(bkd, ndim, bed_type, friction, glen_exponent):
        """Build normalized (A=1, rho=1) shallow-ice physics on a
        1D or 2D Chebyshev basis with flat or varying bed."""
        if ndim == 1:
            mesh = TransformedMesh1D(10, bkd)
            basis = ChebyshevBasis1D(mesh, bkd)
            coords = [basis.nodes()]
        else:
            mesh = TransformedMesh2D(5, 5, bkd)
            basis = ChebyshevBasis2D(mesh, bkd)
            pts = mesh.points()
            coords = [pts[0], pts[1]]
        npts = basis.npts()
        # O(1) bed keeps residual magnitudes moderate so the
        # DerivativeChecker FD sweep bottoms out cleanly
        if bed_type == "flat":
            bed = bkd.zeros((npts,))
        else:
            bed = 0.5 * coords[0]
            if ndim == 2:
                bed = bed + 0.25 * coords[1] ** 2
        physics = ShallowIcePhysics(
            basis,
            bkd,
            bed=bed,
            friction=friction,
            A=1.0,
            rho=1.0,
            glen_exponent=glen_exponent,
        )
        return physics, npts

    @pytest.mark.parametrize("glen_exponent", [3.0, 4.0])
    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("bed_type", ["flat", "varying"])
    @pytest.mark.parametrize("friction", [float("inf"), 1.0])
    def test_jacobian_parametrized(
        self, bkd, glen_exponent, ndim, bed_type, friction
    ):
        """Exact Jacobian vs DerivativeChecker across n, dimension,
        bed topography, and sliding regimes.

        friction=inf makes friction_frac exactly zero (no sliding
        term); H > 0 is required for the fractional powers H^(n+1).
        """
        physics, npts = self._make_shallow_ice(
            bkd, ndim, bed_type, friction, glen_exponent
        )
        wrapper = PhysicsDerivativeWrapper(physics, time=0.0)
        checker = DerivativeChecker(wrapper)
        for _ in range(3):
            state = bkd.asarray(
                2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,))
            )
            errors = checker.check_derivatives(state[:, None], verbosity=0)
            # The FD sweep is V-shaped (truncation decay then roundoff
            # climb) bottoming near 9e-8; a wrong Jacobian plateaus
            # instead. Assert the measured bottom plus a calibrated
            # ratio (marginal cases reach ~1.3e-6).
            assert float(bkd.min(errors[0])) <= 1e-6
            assert checker.error_ratio(errors[0]) <= 5e-6

    def test_jacobian_forcing_independent(self, bkd):
        """dR/dH must be identical with and without forcing (additive,
        state-independent term)."""
        physics_plain, npts = self._make_shallow_ice(
            bkd, 2, "varying", 1.0, 3.0
        )
        forcing_vals = bkd.asarray(np.random.uniform(-1.0, 1.0, (npts,)))

        def forcing_fn(t):
            return forcing_vals

        physics_forced, _ = self._make_shallow_ice(bkd, 2, "varying", 1.0, 3.0)
        physics_forced._forcing_func = forcing_fn
        state = bkd.asarray(2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,)))
        jac_plain = physics_plain.jacobian(state, 0.0)
        jac_forced = physics_forced.jacobian(state, 0.0)
        bkd.assert_allclose(jac_forced, jac_plain, rtol=1e-12)

        wrapper = PhysicsDerivativeWrapper(physics_forced, time=0.0)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_diffusion_derivatives_n3_identity(self, bkd):
        """Exact n=3 collapse pinning the eps-regularization convention.

        For n=3: kappa_G = gamma*H^5 exactly (G^0 = 1, no eps
        dependence) and kappa_H = 5*gamma*H^4*(grad_s_sq+eps) +
        2*friction_frac*H.
        """
        friction = 2.0
        physics, npts = self._make_shallow_ice(bkd, 1, "varying", friction, 3.0)
        state = bkd.asarray(2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,)))
        _, grad_s_sq = physics._compute_surface_gradient(state)
        kappa_h, kappa_g = physics._compute_diffusion_derivatives(
            state, grad_s_sq
        )

        gravity = 9.81
        gamma = 2.0 * 1.0 * (1.0 * gravity) ** 3 / 5.0
        friction_frac = 1.0 * gravity / friction
        eps = 1e-12
        bkd.assert_allclose(kappa_g, gamma * state**5, rtol=1e-12)
        bkd.assert_allclose(
            kappa_h,
            5.0 * gamma * state**4 * (grad_s_sq + eps)
            + 2.0 * friction_frac * state,
            rtol=1e-12,
        )

    def test_jacobian_nonsymmetric_sloped_bed(self, bkd):
        """Guard against accidental symmetrization: the exact SIA
        Jacobian is nonsymmetric on a sloped bed."""
        physics, npts = self._make_shallow_ice(bkd, 1, "varying", 1.0, 3.0)
        state = bkd.asarray(2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,)))
        jac = physics.jacobian(state, 0.0)
        asym_norm = float(bkd.norm(jac - jac.T))
        assert asym_norm > 1e-8 * float(bkd.norm(jac))

    def test_transient_manufactured_backward_euler(self, bkd):
        """Test transient shallow ice with backward Euler.

        Time-discretization-limited (measured final error ~7.8e-8 at
        deltat=0.01 on both backends).
        """
        self._run_transient_shallow_ice(bkd, "backward_euler", atol=1e-6)

    def test_transient_manufactured_crank_nicolson(self, bkd):
        """Test transient shallow ice with Crank-Nicolson.

        Uses polynomial-in-space and quadratic-in-time manufactured solution.
        CN integrates quadratic-in-time exactly, so only roundoff remains
        (measured final error ~1.3e-13).
        """
        self._run_transient_shallow_ice(bkd, "crank_nicolson", atol=1e-11)
