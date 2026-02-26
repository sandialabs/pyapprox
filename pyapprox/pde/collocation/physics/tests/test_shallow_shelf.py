"""Tests for Shallow Shelf Approximation physics implementations."""


import pytest
import numpy as np

from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import constant_dirichlet_bc
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.manufactured_solutions.shallow_shelf import (
    ManufacturedShallowShelfVelocityAndDepthEquations,
    ManufacturedShallowShelfVelocityEquations,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics.shallow_shelf import (
    ShallowShelfDepthPhysics,
    ShallowShelfDepthVelocityPhysics,
    ShallowShelfVelocityPhysics,
    create_shallow_shelf_depth,
    create_shallow_shelf_velocity,
)
from pyapprox.pde.collocation.physics.tests.test_utils import (
    PhysicsTestBase,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.util.test_utils import slow_test


class TestShallowShelfVelocityPhysics(PhysicsTestBase):
    """Tests for ShallowShelfVelocityPhysics."""

    def test_jacobian_derivative_checker(self, bkd):
        """Test Jacobian matches finite differences."""
        npts_1d = 6
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        # Uniform depth and flat bed
        depth = bkd.full((npts,), 1000.0)
        bed = bkd.zeros((npts,))

        physics = ShallowShelfVelocityPhysics(
            basis, bkd, depth=depth, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Small random velocities
        np.random.seed(42)
        state = bkd.array(0.1 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_sloped_bed(self, bkd):
        """Test Jacobian with sloped bed topography."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        # Create 2D bed slope (tile 1D nodes along y)
        nodes_x_1d = bkd.array(np.cos(np.pi * np.arange(npts_1d) / (npts_1d - 1)))
        # Tile to create 2D grid (npts_1d x npts_1d -> flattened npts)
        bed = 100.0 * bkd.array(np.tile(nodes_x_1d, npts_1d))
        depth = bkd.full((npts,), 800.0)

        physics = ShallowShelfVelocityPhysics(
            basis, bkd, depth=depth, bed=bed, friction=5e5, A=1e-16, rho=917.0
        )

        np.random.seed(123)
        state = bkd.array(0.05 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_ncomponents(self, bkd):
        """Test number of components."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        depth = bkd.full((npts,), 1000.0)
        bed = bkd.zeros((npts,))

        physics = ShallowShelfVelocityPhysics(
            basis, bkd, depth=depth, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # 2D: u and v velocity components
        assert physics.ncomponents() == 2
        assert physics.nstates() == 2 * npts

    def test_requires_2d_basis(self, bkd):
        """Test that 1D basis raises error."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        depth = bkd.full((npts,), 1000.0)
        bed = bkd.zeros((npts,))

        with pytest.raises(ValueError):
            ShallowShelfVelocityPhysics(
                basis, bkd, depth=depth, bed=bed, friction=1e6, A=1e-16, rho=917.0
            )

    def test_factory_function(self, bkd):
        """Test create_shallow_shelf_velocity factory."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        depth = bkd.full((npts,), 1000.0)
        bed = bkd.zeros((npts,))

        physics = create_shallow_shelf_velocity(
            basis, bkd, depth=depth, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        assert physics.ncomponents() == 2

    def test_set_depth(self, bkd):
        """Test updating depth."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        depth = bkd.full((npts,), 1000.0)
        bed = bkd.zeros((npts,))

        physics = ShallowShelfVelocityPhysics(
            basis, bkd, depth=depth, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )

        # Update depth
        new_depth = bkd.full((npts,), 800.0)
        physics.set_depth(new_depth)

        # Check that residual can still be computed
        state = bkd.zeros((physics.nstates(),))
        residual = physics.residual(state, time=0.0)
        assert residual.shape[0] == physics.nstates()

    @slow_test
    def test_residual_at_manufactured_solution(self, bkd):
        """Verify residual is near zero at manufactured solution.

        Uses a manufactured solution with smooth polynomial velocities.
        The forcing term is computed analytically so that the exact
        solution satisfies the SSA equations.

        Note: Uses default xy meshgrid indexing with C-order flattening,
        which matches the tensor product basis ordering.

        The SSA equations have highly nonlinear viscosity (Glen's flow law
        with n=3), requiring many points for spectral accuracy. With 50
        points per dimension, the residual norm reaches ~7.6e-9.
        """
        npts_1d = 50  # Need many points for nonlinear SSA
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        basis.npts()

        # Manufactured solution parameters
        depth_str = "1.0"
        bed_str = "0.0"
        friction_str = "1.0"
        A = 1.0
        rho = 1.0
        sol_strs = ["x**2 + y", "x + y**2"]

        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=sol_strs,
            nvars=2,
            bed_str=bed_str,
            depth_str=depth_str,
            friction_str=friction_str,
            A=A,
            rho=rho,
            bkd=bkd,
            oned=False,
        )

        # Get nodes using default xy indexing
        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        X, Y = np.meshgrid(nodes_x, nodes_y)  # default 'xy' indexing
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        nodes_2d = np.vstack([X_flat, Y_flat])

        # Get exact solution and forcing
        exact_sol_vals = man_sol.functions["solution"](nodes_2d)
        u_exact = exact_sol_vals[:, 0]
        v_exact = exact_sol_vals[:, 1]
        exact_state = bkd.hstack([u_exact, v_exact])

        forcing_vals = man_sol.functions["forcing"](nodes_2d)
        forcing_u = forcing_vals[:, 0]
        forcing_v = forcing_vals[:, 1]
        forcing = bkd.hstack([forcing_u, forcing_v])

        # Get fields
        depth_vals = man_sol.functions["depth"](nodes_2d).flatten()
        bed_vals = man_sol.functions["bed"](nodes_2d).flatten()
        friction_vals = man_sol.functions["friction"](nodes_2d).flatten()

        # Create physics with forcing
        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=depth_vals,
            bed=bed_vals,
            friction=friction_vals,
            A=A,
            rho=rho,
            forcing=lambda t: forcing,
        )

        # Compute residual at exact solution
        residual = physics.residual(exact_state, time=0.0)

        # Residual should be very small (spectral discretization error only)
        res_norm = float(bkd.norm(residual))
        assert res_norm < 1e-8, f"Residual norm {res_norm:.4e} too large"


class TestShallowShelfDepthPhysics(PhysicsTestBase):
    """Tests for ShallowShelfDepthPhysics."""

    def test_jacobian_constant_velocity(self, bkd):
        """Test Jacobian with constant velocity field."""
        npts_1d = 6
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = ShallowShelfDepthPhysics(basis, bkd)

        # Set constant velocity field (u, v)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 100.0),  # u
                bkd.full((npts,), 50.0),  # v
            ]
        )
        physics.set_velocities(velocity)

        # Positive depth
        np.random.seed(42)
        state = bkd.array(500.0 + 50.0 * np.random.randn(npts))
        state = bkd.array(np.maximum(np.asarray(state), 100.0))  # Ensure positive

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_2d(self, bkd):
        """Test Jacobian for 2D depth evolution."""
        npts_1d = 6
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = ShallowShelfDepthPhysics(basis, bkd)

        # Set velocity field (u, v)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 50.0),  # u
                bkd.full((npts,), 20.0),  # v
            ]
        )
        physics.set_velocities(velocity)

        # Positive depth
        np.random.seed(42)
        state = bkd.array(500.0 + 50.0 * np.random.randn(npts))
        state = bkd.array(np.maximum(np.asarray(state), 100.0))  # Ensure positive

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_residual_uniform_depth_and_velocity(self, bkd):
        """Test residual is zero for uniform depth and divergence-free velocity."""
        npts_1d = 8
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = ShallowShelfDepthPhysics(basis, bkd)

        # Uniform velocity (constant velocity is divergence-free)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 100.0),  # u
                bkd.full((npts,), 50.0),  # v
            ]
        )
        physics.set_velocities(velocity)

        # Uniform depth
        state = bkd.full((npts,), 1000.0)

        # For uniform H and (u,v): -div(H*vel) = -H*div(vel) - vel·grad(H) = 0
        residual = physics.residual(state, time=0.0)

        # Should be very small (not exactly zero due to spectral diff of constant)
        assert float(bkd.norm(residual)) < 1e-9

    def test_ncomponents(self, bkd):
        """Test number of components."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = ShallowShelfDepthPhysics(basis, bkd)

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts

    def test_requires_velocity_set(self, bkd):
        """Test that residual requires velocities to be set."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = ShallowShelfDepthPhysics(basis, bkd)
        state = bkd.full((npts,), 1000.0)

        with pytest.raises(RuntimeError):
            physics.residual(state, time=0.0)

    def test_requires_2d_basis(self, bkd):
        """Test that 1D basis raises error."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        with pytest.raises(ValueError):
            ShallowShelfDepthPhysics(basis, bkd)

    def test_factory_function(self, bkd):
        """Test create_shallow_shelf_depth factory."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = create_shallow_shelf_depth(basis, bkd)

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts

    def test_transient_depth_equilibrium(self, bkd):
        """Test transient equilibrium for depth evolution.

        Shallow shelf depth equation: dH/dt = -div(H*u)
        For uniform depth and divergence-free velocity, depth should remain constant.

        Note: Uses constant velocity which is divergence-free.
        """
        npts_1d = 8
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = ShallowShelfDepthPhysics(basis, bkd)

        # Set constant velocity field (divergence-free)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 100.0),  # u = constant
                bkd.full((npts,), 50.0),  # v = constant
            ]
        )
        physics.set_velocities(velocity)

        model = CollocationModel(physics, bkd)

        # Initial depth: uniform
        H0 = bkd.full((npts,), 800.0)

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.01,
            deltat=0.002,
        )

        solutions, times = model.solve_transient(H0, config)

        # Check solution is finite and physical (H > 0)
        H_final = solutions[:, -1]
        assert bkd.isfinite(bkd.norm(H_final))
        assert float(bkd.min(H_final)) > 0.0

        # With uniform depth and divergence-free velocity:
        # -div(H*u) = -H*div(u) - u·grad(H) = 0 (since div(u)=0 and grad(H)=0)
        # So depth should remain unchanged
        bkd.assert_allclose(H_final, H0, rtol=1e-6, atol=1e-10)

    def test_transient_depth_with_forcing(self, bkd):
        """Test transient depth evolution with source term.

        dH/dt = -div(H*u) + f
        With uniform depth, constant velocity (div-free), and constant f > 0,
        depth should increase linearly: H(t) = H0 + f*t
        """
        npts_1d = 8
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        # Constant accumulation rate
        f_val = 10.0  # depth increase per unit time
        forcing = bkd.full((npts,), f_val)

        physics = ShallowShelfDepthPhysics(basis, bkd, forcing=lambda t: forcing)

        # Set zero velocity (simplest case)
        velocity = bkd.zeros((2 * npts,))
        physics.set_velocities(velocity)

        model = CollocationModel(physics, bkd)

        # Initial depth: uniform
        H0 = bkd.full((npts,), 800.0)
        final_time = 0.1

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(H0, config)

        # Check solution is finite and physical (H > 0)
        H_final = solutions[:, -1]
        assert bkd.isfinite(bkd.norm(H_final))
        assert float(bkd.min(H_final)) > 0.0

        # Expected: H(t) = H0 + f*t
        H_expected = H0 + f_val * final_time
        bkd.assert_allclose(H_final, H_expected, rtol=0.01, atol=1e-6)

    def _setup_transient_depth_manufactured(self, bkd):
        """Set up transient depth with manufactured solution.

        Uses ManufacturedShallowShelfVelocityAndDepthEquations with:
        - Steady velocities u=x^2+y, v=x+y^2 (nonzero strain rate everywhere)
        - Time-dependent depth H = 2 + 0.3*(1-x^2)*(1-y^2)*(1+T+T^2)
          Boundaries constant: H(boundary,t) = 2
        - Quadratic-in-time for CN exactness

        Only depth forcing is used; velocity equations are not solved.
        """
        npts_1d = 10
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        nodes = mesh.points()  # (2, npts)

        man_sol = ManufacturedShallowShelfVelocityAndDepthEquations(
            vel_strs=["x**2 + y", "x + y**2"],
            nvars=2,
            bed_str="0.0",
            depth_str="2 + 0.3*(1 - x**2)*(1 - y**2)*(1 + T + T**2)",
            friction_str="1.0",
            A=1.0,
            rho=1.0,
            bkd=bkd,
            oned=False,
        )

        def forcing_fn(t):
            # depth_forcing includes dH/dt: f = dH/dt + div(H*vel)
            # Physics: R(H) = -div(H*vel) + f => dH/dt = R(H) = dH/dt ✓
            df = man_sol.functions["depth_forcing"](nodes, t)
            return df[:, 0] if df.ndim == 2 else df

        physics = ShallowShelfDepthPhysics(basis, bkd, forcing=forcing_fn)

        # Set prescribed velocity (steady)
        vel_vals = man_sol.functions["solution"](nodes, 0.0)
        u_vals = vel_vals[:, 1]  # component 1 = u
        v_vals = vel_vals[:, 2]  # component 2 = v
        velocity = bkd.hstack([u_vals, v_vals])
        physics.set_velocities(velocity)

        # Dirichlet BCs on all boundaries: H = 2 (constant)
        bc_list = []
        for bnd_id in range(mesh.nboundaries()):
            bnd_idx = mesh.boundary_indices(bnd_id)
            bc = constant_dirichlet_bc(bkd, bnd_idx, 2.0)
            bc_list.append(bc)
        physics.set_boundary_conditions(bc_list)

        return bkd, npts, nodes, man_sol, physics

    def _run_transient_depth_manufactured(self, bkd, method, atol) :
        """Run transient depth test with given method and tolerance."""
        bkd, npts, nodes, man_sol, physics = self._setup_transient_depth_manufactured(bkd)
        model = CollocationModel(physics, bkd)

        # Initial condition: H(x,y,0)
        sol0 = man_sol.functions["solution"](nodes, 0.0)
        state0 = sol0[:, 0]  # depth is component 0

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))

        exact_final = man_sol.functions["solution"](nodes, t_final)[:, 0]
        bkd.assert_allclose(solutions[:, -1], exact_final, atol=atol)

    def test_transient_depth_manufactured_backward_euler(self, bkd):
        """Test transient depth with backward Euler and manufactured solution."""
        self._run_transient_depth_manufactured(bkd, "backward_euler", atol=0.5)

    def test_transient_depth_manufactured_crank_nicolson(self, bkd):
        """Test transient depth with Crank-Nicolson and manufactured solution.

        Uses polynomial-in-space and quadratic-in-time manufactured solution.
        CN integrates quadratic-in-time exactly, so only spatial error remains.
        """
        self._run_transient_depth_manufactured(bkd, "crank_nicolson", atol=1e-8)


class TestShallowShelfDepthVelocityPhysics(PhysicsTestBase):
    """Tests for coupled ShallowShelfDepthVelocityPhysics."""

    def _create_coupled_physics(self, bkd, npts_1d=6) :
        """Create coupled depth-velocity physics for testing."""
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        bed = bkd.zeros((npts,))
        physics = ShallowShelfDepthVelocityPhysics(
            basis, bkd, bed=bed, friction=1.0, A=1.0, rho=1.0
        )
        return bkd, mesh, basis, npts, physics

    def test_ncomponents(self, bkd):
        """Test number of components is 3 (H, u, v)."""
        bkd, mesh, basis, npts, physics = self._create_coupled_physics(bkd)
        assert physics.ncomponents() == 3
        assert physics.nstates() == 3 * npts

    def test_mass_matrix_structure(self, bkd):
        """Test mass matrix is [[I, 0], [0, 0]]."""
        bkd, mesh, basis, npts, physics = self._create_coupled_physics(bkd)
        M = physics.mass_matrix()

        assert M.shape == (3 * npts, 3 * npts)

        # Top-left block should be identity
        bkd.assert_allclose(M[:npts, :npts], bkd.eye(npts), atol=1e-15)
        # Remaining blocks should be zero
        bkd.assert_allclose(M[:npts, npts:], bkd.zeros((npts, 2 * npts)), atol=1e-15)
        bkd.assert_allclose(M[npts:, :], bkd.zeros((2 * npts, 3 * npts)), atol=1e-15)

    def test_apply_mass_matrix(self, bkd):
        """Test apply_mass_matrix zeros velocity components."""
        bkd, mesh, basis, npts, physics = self._create_coupled_physics(bkd)

        vec = bkd.array(np.ones(3 * npts))
        result = physics.apply_mass_matrix(vec)

        # Depth part kept
        bkd.assert_allclose(result[:npts], bkd.ones((npts,)), atol=1e-15)
        # Velocity parts zeroed
        bkd.assert_allclose(result[npts:], bkd.zeros((2 * npts,)), atol=1e-15)

    def test_jacobian_derivative_checker(self, bkd):
        """Test Jacobian matches finite differences."""
        bkd, mesh, basis, npts, physics = self._create_coupled_physics(bkd)

        # State with positive depth and small velocities
        np.random.seed(42)
        H = 1.0 + 0.1 * np.random.randn(npts)
        u = 0.1 * np.random.randn(npts)
        v = 0.1 * np.random.randn(npts)
        state = bkd.array(np.concatenate([H, u, v]))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_requires_2d_basis(self, bkd):
        """Test that 1D basis raises error."""
        mesh = TransformedMesh1D(10, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        with pytest.raises(ValueError):
            ShallowShelfDepthVelocityPhysics(
                basis, bkd, bed=bkd.zeros((10,)), friction=1.0, A=1.0, rho=1.0
            )

    def _setup_transient_coupled_manufactured(self, bkd):
        """Set up transient coupled depth+velocity manufactured solution.

        Uses ManufacturedShallowShelfVelocityAndDepthEquations matching
        the legacy test pattern:
        - vel: u=(x+1)^2*(1+y)*(1+T), v=y^2 (nonzero strain rate)
        - depth: H = 0.1*(T+1) (time-dependent, uniform in space)
        - bed: s0 - alpha*x^2 - H0  (sloped)
        - friction=10, A=1, rho=1
        - Domain [0,1]^2 via AffineTransform2D
        """
        from pyapprox.pde.collocation.mesh import AffineTransform2D
        npts_1d = 8

        # Domain [0,1]^2
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd, transform=transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        nodes = mesh.points()  # (2, npts)

        s0, depth_val, alpha = 2, 0.1, 1e-1
        depth_str = f"{depth_val}*(T+1)"
        bed_str = f"{s0}-{alpha}*x**2-{depth_val}"
        vel_strs = ["(x+1)**2*(1+y)*(1+T)", "(y+1)**2"]
        friction_str = "10"
        A = 1
        rho = 1

        man_sol = ManufacturedShallowShelfVelocityAndDepthEquations(
            vel_strs=vel_strs,
            nvars=2,
            bed_str=bed_str,
            depth_str=depth_str,
            friction_str=friction_str,
            A=A,
            rho=rho,
            bkd=bkd,
            oned=False,
        )

        # Get bed at nodes
        bed_vals = man_sol.functions["bed"](nodes).flatten()
        friction_vals = man_sol.functions["friction"](nodes).flatten()

        def depth_forcing_fn(t):
            # depth_forcing includes dH/dt: f = dH/dt + div(H*vel)
            # Physics residual: R(H) = -div(H*vel) + f = dH/dt
            # Time integrator: M*(y-yn) - dt*R = M*(y-yn) - dt*dH/dt ≈ 0
            df = man_sol.functions["depth_forcing"](nodes, t)
            return df[:, 0] if df.ndim == 2 else df

        def velocity_forcing_fn(t):
            vf = man_sol.functions["velocity_forcing"](nodes, t)
            if vf.ndim == 2:
                return bkd.hstack([vf[:, 0], vf[:, 1]])
            return vf

        physics = ShallowShelfDepthVelocityPhysics(
            basis,
            bkd,
            bed=bed_vals,
            friction=friction_vals,
            A=A,
            rho=rho,
            g=9.81,
            depth_forcing=depth_forcing_fn,
            velocity_forcing=velocity_forcing_fn,
        )

        # Set Dirichlet BCs on all boundaries for all 3 components
        bc_list = []
        for bnd_id in range(mesh.nboundaries()):
            bnd_idx = mesh.boundary_indices(bnd_id)

            # Depth BC (component 0): H(boundary, t)
            def make_depth_bc_fn(idx, ms=man_sol, nd=nodes):
                def fn(t):
                    vals = ms.functions["solution"](nd, t)
                    return vals[idx, 0]  # depth is component 0

                return fn

            bc_H = DirichletBC(bkd, bnd_idx, make_depth_bc_fn(bnd_idx))
            bc_list.append(bc_H)

            # u-velocity BC (component 1): offset by npts
            u_idx = bnd_idx + npts

            def make_u_bc_fn(idx, ms=man_sol, nd=nodes):
                def fn(t):
                    vals = ms.functions["solution"](nd, t)
                    return vals[idx, 1]  # u is component 1

                return fn

            bc_u = DirichletBC(bkd, u_idx, make_u_bc_fn(bnd_idx))
            bc_list.append(bc_u)

            # v-velocity BC (component 2): offset by 2*npts
            v_idx = bnd_idx + 2 * npts

            def make_v_bc_fn(idx, ms=man_sol, nd=nodes):
                def fn(t):
                    vals = ms.functions["solution"](nd, t)
                    return vals[idx, 2]  # v is component 2

                return fn

            bc_v = DirichletBC(bkd, v_idx, make_v_bc_fn(bnd_idx))
            bc_list.append(bc_v)

        physics.set_boundary_conditions(bc_list)

        return bkd, npts, nodes, man_sol, physics

    def _run_transient_coupled_manufactured(self, bkd, method, rtol) :
        """Run transient coupled test with given method and tolerance."""
        bkd, npts, nodes, man_sol, physics = (
            self._setup_transient_coupled_manufactured(bkd)
        )
        model = CollocationModel(physics, bkd)

        # Initial condition: [H(0), u(0), v(0)]
        sol0 = man_sol.functions["solution"](nodes, 0.0)
        H0 = sol0[:, 0]
        u0 = sol0[:, 1]
        v0 = sol0[:, 2]
        state0 = bkd.hstack([H0, u0, v0])

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.1,
            newton_maxiter=50,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))

        exact_final = man_sol.functions["solution"](nodes, t_final)
        H_exact = exact_final[:, 0]
        u_exact = exact_final[:, 1]
        v_exact = exact_final[:, 2]
        exact_state = bkd.hstack([H_exact, u_exact, v_exact])

        bkd.assert_allclose(solutions[:, -1], exact_state, rtol=rtol)

    def test_transient_coupled_backward_euler(self, bkd):
        """Test transient coupled depth+velocity with backward Euler."""
        self._run_transient_coupled_manufactured(bkd, "backward_euler", rtol=1e-5)

    def test_transient_coupled_crank_nicolson(self, bkd):
        """Test transient coupled depth+velocity with Crank-Nicolson."""
        self._run_transient_coupled_manufactured(bkd, "crank_nicolson", rtol=1e-5)
