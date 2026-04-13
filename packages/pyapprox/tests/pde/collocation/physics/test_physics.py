"""Tests for physics implementations."""

import math

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
    create_advection_diffusion,
    create_steady_diffusion,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)


class TestAdvectionDiffusionReaction:
    """Base test class for ADR physics."""

    def test_diffusion_residual_polynomial(self, bkd):
        """Test diffusion residual with polynomial solution.

        For u(x) = x^2 on [-1, 1]:
        laplacian(u) = 2
        With D = 1: residual = D * laplacian(u) = 2
        """
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # u = x^2
        nodes = basis.nodes()
        u = nodes**2

        residual = physics.residual(u, time=0.0)

        # Expected: laplacian(x^2) = 2 (constant)
        expected = bkd.full((npts,), 2.0)
        bkd.assert_allclose(residual, expected, atol=1e-10)

    def test_diffusion_jacobian(self, bkd):
        """Test diffusion Jacobian via finite differences."""
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Random state
        u = bkd.sin(math.pi * basis.nodes())

        # Compute analytical Jacobian
        jac = physics.jacobian(u, time=0.0)

        # Verify via finite differences
        eps = 1e-7
        jac_fd = bkd.zeros((npts, npts))
        for j in range(npts):
            u_plus = bkd.copy(u)
            u_plus[j] = u_plus[j] + eps
            u_minus = bkd.copy(u)
            u_minus[j] = u_minus[j] - eps
            res_plus = physics.residual(u_plus, 0.0)
            res_minus = physics.residual(u_minus, 0.0)
            for i in range(npts):
                jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        bkd.assert_allclose(jac, jac_fd, atol=1e-6)

    def test_advection_residual(self, bkd):
        """Test advection residual.

        For u(x) = sin(pi*x), v = 1:
        -v * du/dx = -cos(pi*x) * pi
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        # Velocity = 1
        velocity = [bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(basis, bkd, velocity=velocity)

        nodes = basis.nodes()
        u = bkd.sin(math.pi * nodes)

        residual = physics.residual(u, time=0.0)

        # Expected: -1 * d/dx[sin(pi*x)] = -pi * cos(pi*x)
        expected = -math.pi * bkd.cos(math.pi * nodes)
        bkd.assert_allclose(residual, expected, atol=1e-8)

    def test_advection_jacobian(self, bkd):
        """Test advection Jacobian via finite differences."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        velocity = [bkd.full((npts,), 2.0)]  # v = 2
        physics = AdvectionDiffusionReaction(basis, bkd, velocity=velocity)

        u = bkd.cos(math.pi * basis.nodes())

        jac = physics.jacobian(u, time=0.0)

        # Finite differences
        eps = 1e-7
        jac_fd = bkd.zeros((npts, npts))
        for j in range(npts):
            u_plus = bkd.copy(u)
            u_plus[j] = u_plus[j] + eps
            u_minus = bkd.copy(u)
            u_minus[j] = u_minus[j] - eps
            res_plus = physics.residual(u_plus, 0.0)
            res_minus = physics.residual(u_minus, 0.0)
            for i in range(npts):
                jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        bkd.assert_allclose(jac, jac_fd, atol=1e-6)

    def test_reaction_residual(self, bkd):
        """Test reaction residual.

        For u and r = 2: residual = 2 * u
        """
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, reaction=2.0)

        u = bkd.linspace(1.0, 5.0, npts)

        residual = physics.residual(u, time=0.0)

        expected = 2.0 * u
        bkd.assert_allclose(residual, expected, atol=1e-14)

    def test_forcing_residual(self, bkd):
        """Test forcing term."""
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        forcing = bkd.ones((npts,)) * 3.0
        physics = AdvectionDiffusionReaction(basis, bkd, forcing=lambda t: forcing)

        u = bkd.zeros((npts,))

        residual = physics.residual(u, time=0.0)

        # With no other terms, residual = forcing
        bkd.assert_allclose(residual, forcing, atol=1e-14)

    def test_combined_adr(self, bkd):
        """Test combined ADR physics."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        D = 0.1
        v = bkd.full((npts,), 1.0)
        r = -0.5

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=D, velocity=[v], reaction=r
        )

        u = bkd.sin(math.pi * basis.nodes())

        jac = physics.jacobian(u, time=0.0)

        # Verify Jacobian via finite differences
        eps = 1e-7
        jac_fd = bkd.zeros((npts, npts))
        for j in range(npts):
            u_plus = bkd.copy(u)
            u_plus[j] = u_plus[j] + eps
            u_minus = bkd.copy(u)
            u_minus[j] = u_minus[j] - eps
            res_plus = physics.residual(u_plus, 0.0)
            res_minus = physics.residual(u_minus, 0.0)
            for i in range(npts):
                jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        bkd.assert_allclose(jac, jac_fd, atol=1e-5)

    def test_boundary_condition_application(self, bkd):
        """Test applying boundary conditions."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Set Dirichlet BCs: u(-1) = 0, u(1) = 1
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = constant_dirichlet_bc(bkd, right_idx, 1.0)
        physics.set_boundary_conditions([bc_left, bc_right])

        # Some state
        u = bkd.linspace(0.0, 1.0, npts)

        residual = physics.residual(u, time=0.0)
        jacobian = physics.jacobian(u, time=0.0)

        # Apply BCs
        residual, jacobian = physics.apply_boundary_conditions(
            residual, jacobian, u, time=0.0
        )

        # Check boundary rows of Jacobian are identity
        bkd.assert_allclose(jacobian[0, :], bkd.eye(npts)[0, :], atol=1e-14)
        bkd.assert_allclose(jacobian[-1, :], bkd.eye(npts)[-1, :], atol=1e-14)


class TestDiffusionParameterization:
    """Test parameterized diffusion via DiffusionParameterization.

    Validates param_jacobian, nparams, and initial_param_jacobian using
    BasisExpansion + DiffusionParameterization (replaces the former
    TestAdvectionDiffusionReactionWithParam tests).
    """

    def test_param_jacobian(self, bkd):
        """Test parameter Jacobian via finite differences."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        # Parameterized diffusion: D = 1 + p0 * phi0 + p1 * phi1
        # where phi0 = 1 (constant), phi1 = x
        nodes = basis.nodes()
        phi0 = bkd.ones((npts,))
        phi1 = nodes

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])
        dp = create_diffusion_parameterization(bkd, basis, fm)

        # Set parameters
        param = bkd.array([0.5, 0.2])
        dp.apply(physics, param)

        # State
        u = bkd.sin(math.pi * nodes)

        # Analytical parameter Jacobian
        param_jac = dp.param_jacobian(physics, u, 0.0, param)

        # Verify via finite differences
        eps = 1e-7
        param_jac_fd = bkd.zeros((npts, 2))
        for j in range(2):
            param_plus = bkd.copy(param)
            param_plus[j] = param_plus[j] + eps
            param_minus = bkd.copy(param)
            param_minus[j] = param_minus[j] - eps

            dp.apply(physics, param_plus)
            res_plus = physics.residual(u, 0.0)
            dp.apply(physics, param_minus)
            res_minus = physics.residual(u, 0.0)

            for i in range(npts):
                param_jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        # Reset to original
        dp.apply(physics, param)

        bkd.assert_allclose(param_jac, param_jac_fd, atol=1e-5)

    def test_nparams(self, bkd):
        """Test nparams method."""
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        phi0 = bkd.ones((npts,))
        phi1 = basis.nodes()
        phi2 = basis.nodes() ** 2

        fm = BasisExpansion(bkd, 1.0, [phi0, phi1, phi2])
        dp = create_diffusion_parameterization(bkd, basis, fm)

        assert dp.nparams() == 3

    def test_initial_param_jacobian_zero(self, bkd):
        """Test initial param Jacobian is zero (IC does not depend on params)."""
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        phi0 = bkd.ones((npts,))

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(bkd, basis, fm)

        param = bkd.array([0.5])
        ic_jac = dp.initial_param_jacobian(physics, param)

        expected = bkd.zeros((npts, 1))
        bkd.assert_allclose(ic_jac, expected, atol=1e-14)


class TestFactoryFunctions:
    """Base test class for factory functions."""

    def test_create_steady_diffusion(self, bkd):
        """Test create_steady_diffusion factory."""
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = create_steady_diffusion(basis, bkd, diffusion=2.0)

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts

    def test_create_advection_diffusion(self, bkd):
        """Test create_advection_diffusion factory."""
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        velocity = [bkd.ones((npts,))]
        physics = create_advection_diffusion(
            basis, bkd, diffusion=1.0, velocity=velocity
        )

        assert physics.ncomponents() == 1


class TestADRTransient:
    """Test transient ADR with manufactured solutions.

    Uses quadratic-in-time manufactured solution so CN (2nd order) is
    exact while BE (1st order) has O(dt) temporal error.
    Solution: u = sin(pi*x)*(1 + T + T**2), homogeneous Dirichlet BCs.
    """

    def _setup_transient_adr(self, bkd) :
        """Create ADR physics with quadratic-in-time manufactured solution."""
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="sin(pi*x)*(1 + T + T**2)",
            nvars=1,
            diff_str="0.1",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        def forcing_fn(t):
            return man_sol.functions["forcing"](nodes[None, :], t)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, forcing=forcing_fn
        )

        # Homogeneous Dirichlet BCs (sin(pi*x) = 0 at x = +-1)
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        return physics, man_sol, nodes, bkd

    def _run_transient_adr(self, bkd, method, atol) :
        physics, man_sol, nodes, bkd = self._setup_transient_adr(bkd)
        model = CollocationModel(physics, bkd)

        u0 = man_sol.functions["solution"](nodes[None, :], 0.0)

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.005,
        )

        solutions, times = model.solve_transient(u0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact = man_sol.functions["solution"](nodes[None, :], t_final)

        bkd.assert_allclose(solutions[:, -1], u_exact, atol=atol)

    def test_transient_backward_euler(self, bkd):
        """Test transient ADR with backward Euler.

        BE is 1st order in time, so with dt=0.005 and quadratic-in-time
        solution there is O(dt) temporal error.
        """
        self._run_transient_adr(bkd, "backward_euler", atol=0.01)

    def test_transient_crank_nicolson(self, bkd):
        """Test transient ADR with Crank-Nicolson.

        CN is 2nd order in time. For quadratic-in-time solution,
        CN integrates the time derivative exactly.
        """
        self._run_transient_adr(bkd, "crank_nicolson", atol=1e-8)


# NumPy backend tests
