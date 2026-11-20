"""Tests for Shallow Shelf Approximation physics implementations."""

import pickle
from typing import Callable, Union

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import constant_dirichlet_bc
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics.coefficients import (
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.collocation.physics.shallow_shelf import (
    ShallowShelfDepthPhysics,
    ShallowShelfDepthVelocityPhysics,
    ShallowShelfVelocityPhysics,
    create_shallow_shelf_depth,
    create_shallow_shelf_velocity,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.manufactured.shallow_shelf import (
    ManufacturedShallowShelfVelocityAndDepthEquations,
)
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.physics_test_utils import (
    PhysicsTestBase,
)


class _VectorFunctionWrapper:
    """Adapts a 1D-array function and its assembly for DerivativeChecker."""

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        nqoi: int,
        fun: Callable[[Array], Array],
        jac: Callable[[Array], Array],
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._nqoi = nqoi
        self._fun = fun
        self._jac = jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, sample: Array) -> Array:
        return self._fun(sample[:, 0])[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._jac(sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


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
            basis,
            bkd,
            depth=NodalField(depth),
            bed=NodalField(bed),
            friction=1e6,
            A=1e-16,
            rho=917.0,
        )

        # Small random velocities
        np.random.seed(42)
        state = bkd.array(0.1 * np.random.randn(physics.nstates()))

        # Exact Jacobian: limited only by the checker's FD noise
        # (measured sweep bottom ~3e-8)
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

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
            basis,
            bkd,
            depth=NodalField(depth),
            bed=NodalField(bed),
            friction=5e5,
            A=1e-16,
            rho=917.0,
        )

        np.random.seed(123)
        state = bkd.array(0.05 * np.random.randn(physics.nstates()))

        # Exact Jacobian: limited only by the checker's FD noise
        # (measured sweep bottom ~1e-7)
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

    def test_ncomponents(self, bkd):
        """Test number of components."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        depth = bkd.full((npts,), 1000.0)
        bed = bkd.zeros((npts,))

        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=NodalField(depth),
            bed=NodalField(bed),
            friction=1e6,
            A=1e-16,
            rho=917.0,
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
                basis,
                bkd,
                depth=NodalField(depth),
                bed=NodalField(bed),
                friction=1e6,
                A=1e-16,
                rho=917.0,
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
            basis,
            bkd,
            depth=NodalField(depth),
            bed=NodalField(bed),
            friction=1e6,
            A=1e-16,
            rho=917.0,
        )

        assert physics.ncomponents() == 2

    def test_rewritten_depth_is_seen(self, bkd):
        """Rewriting the depth's values, as a parameterization or a
        coupling would, gives the residual of a physics built with the
        new depth: neither H nor the surface slope is cached."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        depth = NodalField(bkd.full((npts,), 1000.0))
        bed = NodalField(bkd.zeros((npts,)))
        physics = ShallowShelfVelocityPhysics(
            basis, bkd, depth=depth, bed=bed, friction=1e6, A=1e-16, rho=917.0
        )
        np.random.seed(5)
        state = bkd.array(0.1 * np.random.randn(physics.nstates()))
        before = physics.residual(state, time=0.0)

        new_depth = bkd.asarray(800.0 + 20.0 * np.random.uniform(-1, 1, npts))
        depth.set_values(new_depth)
        reference = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=NodalField(new_depth),
            bed=bed,
            friction=1e6,
            A=1e-16,
            rho=917.0,
        )
        after = physics.residual(state, time=0.0)
        bkd.assert_allclose(after, reference.residual(state, 0.0), rtol=1e-12)
        assert float(bkd.max(bkd.abs(after - before))) > 1e-3

    def test_jacobian_zero_friction(self, bkd):
        """Exact Jacobian with the sliding term switched off."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=800.0,
            bed=0.0,
            friction=0.0,
            A=1e-16,
            rho=917.0,
        )
        np.random.seed(7)
        state = bkd.array(0.1 * np.random.randn(physics.nstates()))
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

    def test_residual_depth_jacobian_derivative_checker(self, bkd):
        """d(R_vel)/dH matches finite differences (velocities fixed; the
        depth field is rewritten, and the surface gradient follows)."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        nodes_x_1d = bkd.array(np.cos(np.pi * np.arange(npts_1d) / (npts_1d - 1)))
        bed = 10.0 * bkd.array(np.tile(nodes_x_1d, npts_1d))
        depth_field = NodalField(bkd.full((npts,), 800.0))
        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=depth_field,
            bed=NodalField(bed),
            friction=1.0,
            A=1.0,
            rho=1.0,
        )
        np.random.seed(11)
        vel_state = bkd.array(0.1 * np.random.randn(2 * npts))
        depth0 = bkd.asarray(800.0 + 50.0 * np.random.uniform(-1.0, 1.0, (npts,)))

        def fun(depth):
            depth_field.set_values(depth)
            return physics.residual(vel_state, 0.0)

        def jac(depth):
            depth_field.set_values(depth)
            return physics.residual_depth_jacobian(vel_state, 0.0)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, npts, 2 * npts, fun, jac)
        )
        errors = checker.check_derivatives(depth0[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6


class TestShallowShelfDepthPhysics(PhysicsTestBase):
    """Tests for ShallowShelfDepthPhysics."""

    def test_jacobian_constant_velocity(self, bkd):
        """Test Jacobian with constant velocity field."""
        npts_1d = 6
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        # Constant velocity field (u, v)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 100.0),  # u
                bkd.full((npts,), 50.0),  # v
            ]
        )
        physics = ShallowShelfDepthPhysics(basis, bkd, NodalField(velocity))

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

        # Velocity field (u, v)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 50.0),  # u
                bkd.full((npts,), 20.0),  # v
            ]
        )
        physics = ShallowShelfDepthPhysics(basis, bkd, NodalField(velocity))

        # Positive depth
        np.random.seed(42)
        state = bkd.array(500.0 + 50.0 * np.random.randn(npts))
        state = bkd.array(np.maximum(np.asarray(state), 100.0))  # Ensure positive

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_residual_velocity_jacobian(self, bkd):
        """d(R_H)/d[vel] is exact: the residual is linear in the
        velocities, so the assembly reproduces residual differences
        exactly and passes the DerivativeChecker."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        velocity = NodalField(bkd.zeros((2 * npts,)))
        physics = ShallowShelfDepthPhysics(basis, bkd, velocity)
        np.random.seed(3)
        depth = bkd.asarray(500.0 + 50.0 * np.random.uniform(-1.0, 1.0, (npts,)))
        vel0 = bkd.array(0.1 * np.random.randn(2 * npts))

        def fun(vel):
            velocity.set_values(vel)
            return physics.residual(depth, 0.0)

        def jac(vel):
            velocity.set_values(vel)
            return physics.residual_velocity_jacobian(depth)

        # Exact linearity identity
        jmat = jac(vel0)
        bkd.assert_allclose(
            jmat @ vel0,
            fun(vel0) - fun(bkd.zeros((2 * npts,))),
            rtol=1e-12,
        )

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, 2 * npts, npts, fun, jac)
        )
        errors = checker.check_derivatives(vel0[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_residual_uniform_depth_and_velocity(self, bkd):
        """Test residual is zero for uniform depth and divergence-free velocity."""
        npts_1d = 8
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        # Uniform velocity (constant velocity is divergence-free)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 100.0),  # u
                bkd.full((npts,), 50.0),  # v
            ]
        )
        physics = ShallowShelfDepthPhysics(basis, bkd, NodalField(velocity))

        # Uniform depth
        state = bkd.full((npts,), 1000.0)

        # For uniform H and (u,v): -div(H*vel) = -H*div(vel) - vel·grad(H) = 0
        residual = physics.residual(state, time=0.0)

        # Should be very small (not exactly zero due to spectral diff of constant)
        assert float(bkd.norm(residual)) < 5e-9

    def test_ncomponents(self, bkd):
        """Test number of components."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = ShallowShelfDepthPhysics(basis, bkd, 0.0)

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts

    def test_requires_2d_basis(self, bkd):
        """Test that 1D basis raises error."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        with pytest.raises(ValueError):
            ShallowShelfDepthPhysics(basis, bkd, 0.0)

    def test_factory_function(self, bkd):
        """Test create_shallow_shelf_depth factory."""
        npts_1d = 5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        physics = create_shallow_shelf_depth(basis, bkd, 0.0)

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

        # Constant velocity field (divergence-free)
        velocity = bkd.hstack(
            [
                bkd.full((npts,), 100.0),  # u = constant
                bkd.full((npts,), 50.0),  # v = constant
            ]
        )
        physics = ShallowShelfDepthPhysics(basis, bkd, NodalField(velocity))

        model = CollocationModel(physics, bkd)

        # Initial depth: uniform
        H0 = bkd.full((npts,), 800.0)

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.01,
            deltat=0.002,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
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

        # Zero velocity (simplest case)
        physics = ShallowShelfDepthPhysics(
            basis, bkd, NodalField(bkd.zeros((2 * npts,))), forcing=NodalField(forcing)
        )

        model = CollocationModel(physics, bkd)

        # Initial depth: uniform
        H0 = bkd.full((npts,), 800.0)
        final_time = 0.1

        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
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

        # Prescribed velocity (steady)
        vel_vals = man_sol.functions["solution"](nodes, 0.0)
        u_vals = vel_vals[:, 1]  # component 1 = u
        v_vals = vel_vals[:, 2]  # component 2 = v
        velocity = bkd.hstack([u_vals, v_vals])
        physics = ShallowShelfDepthPhysics(
            basis, bkd, NodalField(velocity), forcing=forcing_fn
        )

        # Dirichlet BCs on all boundaries: H = 2 (constant)
        bc_list = []
        for bnd_id in range(mesh.nboundaries()):
            bnd_idx = mesh.boundary_indices(bnd_id)
            bc = constant_dirichlet_bc(bkd, bnd_idx, 2.0)
            bc_list.append(bc)
        physics.set_boundary_conditions(bc_list)

        return bkd, npts, nodes, man_sol, physics

    def _run_transient_depth_manufactured(self, bkd, method, atol):
        """Run transient depth test with given method and tolerance."""
        bkd, npts, nodes, man_sol, physics = self._setup_transient_depth_manufactured(
            bkd
        )
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
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
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

    def _create_coupled_physics(self, bkd, npts_1d=6):
        """Create coupled depth-velocity physics for testing."""
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        bed = bkd.zeros((npts,))
        physics = ShallowShelfDepthVelocityPhysics(
            basis, bkd, bed=NodalField(bed), friction=1.0, A=1.0, rho=1.0
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

        # Exact coupled Jacobian: limited only by the checker's FD
        # noise (measured sweep bottom ~8e-8)
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

    def test_jacobian_block_extraction(self, bkd):
        """The coupled residual and the Jacobian's sub-blocks equal those
        of standalone depth and velocity physics given the state's own
        depth and velocity."""
        bkd, mesh, basis, npts, physics = self._create_coupled_physics(bkd)
        np.random.seed(42)
        H = bkd.asarray(1.0 + 0.1 * np.random.randn(npts))
        u = bkd.asarray(0.1 * np.random.randn(npts))
        v = bkd.asarray(0.1 * np.random.randn(npts))
        state = bkd.hstack([H, u, v])

        jac = physics.jacobian(state, 0.0)
        vel_state = bkd.hstack([u, v])
        depth_physics = ShallowShelfDepthPhysics(basis, bkd, NodalField(vel_state))
        velocity_physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=NodalField(H),
            bed=physics.bed_function(),
            friction=1.0,
            A=1.0,
            rho=1.0,
        )
        bkd.assert_allclose(
            physics.residual(state, 0.0),
            bkd.hstack(
                [
                    depth_physics.residual(H, 0.0),
                    velocity_physics.residual(vel_state, 0.0),
                ]
            ),
            rtol=1e-12,
        )

        bkd.assert_allclose(
            jac[:npts, :npts],
            depth_physics.jacobian(H, 0.0),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            jac[:npts, npts:],
            depth_physics.residual_velocity_jacobian(H),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            jac[npts:, :npts],
            velocity_physics.residual_depth_jacobian(vel_state, 0.0),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            jac[npts:, npts:],
            velocity_physics.jacobian(vel_state, 0.0),
            rtol=1e-12,
        )

    def test_requires_2d_basis(self, bkd):
        """Test that 1D basis raises error."""
        mesh = TransformedMesh1D(10, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        with pytest.raises(ValueError):
            ShallowShelfDepthVelocityPhysics(
                basis, bkd, bed=0.0, friction=1.0, A=1.0, rho=1.0
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
            bed=NodalField(bed_vals),
            friction=NodalField(friction_vals),
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

    def _run_transient_coupled_manufactured(self, bkd, method, rtol):
        """Run transient coupled test with given method and tolerance."""
        bkd, npts, nodes, man_sol, physics = self._setup_transient_coupled_manufactured(
            bkd
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
            newton_tol=1e-10,
            verbosity=0,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))

        exact_final = man_sol.functions["solution"](nodes, t_final)
        H_exact = exact_final[:, 0]
        u_exact = exact_final[:, 1]
        v_exact = exact_final[:, 2]
        exact_state = bkd.hstack([H_exact, u_exact, v_exact])

        bkd.assert_allclose(solutions[:, -1], exact_state, rtol=rtol)

    @pytest.mark.slow_on("TorchBkd")
    def test_transient_coupled_backward_euler(self, bkd):
        """Test transient coupled depth+velocity with backward Euler."""
        self._run_transient_coupled_manufactured(bkd, "backward_euler", rtol=1e-5)

    @pytest.mark.slow_on("TorchBkd")
    def test_transient_coupled_crank_nicolson(self, bkd):
        """Test transient coupled depth+velocity with Crank-Nicolson."""
        self._run_transient_coupled_manufactured(bkd, "crank_nicolson", rtol=1e-5)


_NPTS_1D_SSA = 5


def _ssa_coords(bkd: Backend[Array]) -> Array:
    """Node coordinates. Shape: (2, npts)"""
    return TransformedMesh2D(_NPTS_1D_SSA, _NPTS_1D_SSA, bkd).points()


def _ssa_basis(bkd: Backend[Array]) -> ChebyshevBasis2D[Array]:
    mesh = TransformedMesh2D(_NPTS_1D_SSA, _NPTS_1D_SSA, bkd)
    return ChebyshevBasis2D(mesh, bkd)


def _thick(coords: Array) -> Array:
    return 1.0 + 0.1 * coords[0] * coords[1]


def _thickening(coords: Array, time: float) -> Array:
    return (1.0 + time) * (1.0 + 0.1 * coords[0] * coords[1])


def _ssa_declared(
    bkd: Backend[Array], dependent: bool, ncopies: int = 1
) -> NodalSupplier[Array]:
    """A declared positive supplier at the nodes, stacked ``ncopies``
    times."""
    coords = bkd.hstack([_ssa_coords(bkd)] * ncopies)
    if dependent:
        thickening: TimeDependent[Array] = TimeDependent(_thickening)
        return NodalSupplier(thickening, coords)
    thick: TimeIndependent[Array] = TimeIndependent(_thick)
    return NodalSupplier(thick, coords)


def _ssa_velocity(bkd: Backend[Array]) -> Array:
    npts = _NPTS_1D_SSA**2
    rng = np.random.default_rng(9)
    return bkd.asarray(0.1 * rng.standard_normal(2 * npts))


def _ssa_depth(bkd: Backend[Array]) -> Array:
    npts = _NPTS_1D_SSA**2
    rng = np.random.default_rng(10)
    return bkd.asarray(1.0 + 0.1 * rng.uniform(-1.0, 1.0, npts))


_ShelfPhysics = Union[
    ShallowShelfVelocityPhysics[Array],
    ShallowShelfDepthPhysics[Array],
    ShallowShelfDepthVelocityPhysics[Array],
]


def _ssa_physics(
    bkd: Backend[Array], kind: str, dependent: bool
) -> _ShelfPhysics[Array]:
    """One of the three physics with every coefficient a declared supplier,
    time-dependent or not."""
    basis = _ssa_basis(bkd)
    if kind == "velocity":
        return ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=_ssa_declared(bkd, dependent),
            bed=_ssa_declared(bkd, dependent),
            friction=_ssa_declared(bkd, dependent),
            A=1.0,
            rho=1.0,
            forcing=_ssa_declared(bkd, dependent, 2),
        )
    if kind == "depth":
        return ShallowShelfDepthPhysics(
            basis,
            bkd,
            _ssa_declared(bkd, dependent, 2),
            forcing=_ssa_declared(bkd, dependent),
        )
    return ShallowShelfDepthVelocityPhysics(
        basis,
        bkd,
        bed=_ssa_declared(bkd, dependent),
        friction=_ssa_declared(bkd, dependent),
        A=1.0,
        rho=1.0,
        depth_forcing=_ssa_declared(bkd, dependent),
        velocity_forcing=_ssa_declared(bkd, dependent, 2),
    )


def _ssa_state(bkd: Backend[Array], kind: str) -> Array:
    if kind == "velocity":
        return _ssa_velocity(bkd)
    if kind == "depth":
        return _ssa_depth(bkd)
    return bkd.hstack([_ssa_depth(bkd), _ssa_velocity(bkd)])


class TestShallowShelfInterior:
    @pytest.mark.parametrize("kind", ["velocity", "depth", "coupled"])
    def test_bundle_with_moving_coefficients(
        self, bkd: Backend[Array], kind: str
    ) -> None:
        """With every coefficient time-dependent, the bundle passes the
        checker and neither the operator nor its Jacobian is declared
        time-invariant."""
        physics = _ssa_physics(bkd, kind, dependent=True)
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert not physics.interior_jacobian_is_time_invariant()
        assert not physics.interior_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        report = ODEResidualBundleChecker(rhs).check(_ssa_state(bkd, kind), 0.3)
        assert report.failures() == []

    @pytest.mark.parametrize("kind", ["velocity", "depth", "coupled"])
    def test_declared_invariant(self, bkd: Backend[Array], kind: str) -> None:
        """Declared time-independent coefficients make the operator
        time-invariant, and the bundle passes the checker."""
        physics = _ssa_physics(bkd, kind, dependent=False)
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert physics.interior_is_time_invariant()
        assert physics.interior_jacobian_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        report = ODEResidualBundleChecker(rhs).check(_ssa_state(bkd, kind), 0.3)
        assert report.failures() == []

    def test_forcings_leave_jacobian_invariant(self, bkd: Backend[Array]) -> None:
        """Time-dependent forcings alone leave the coupled Jacobian
        time-invariant but not the operator."""
        physics = ShallowShelfDepthVelocityPhysics(
            _ssa_basis(bkd),
            bkd,
            bed=0.0,
            friction=1.0,
            A=1.0,
            rho=1.0,
            depth_forcing=_ssa_declared(bkd, True),
            velocity_forcing=_ssa_declared(bkd, True, 2),
        )
        assert physics.interior_jacobian_is_time_invariant()
        assert not physics.interior_is_time_invariant()

    def test_depth_read_at_evaluation_time(self, bkd: Backend[Array]) -> None:
        """A time-dependent depth gives, at time t, the velocity residual
        and Jacobian of the physics built from its values at t."""
        time = 0.7
        moving_depth = _ssa_declared(bkd, True)
        basis = _ssa_basis(bkd)

        def velocity_physics(
            depth: NodalSupplier[Array] | NodalField[Array],
        ) -> ShallowShelfVelocityPhysics[Array]:
            return ShallowShelfVelocityPhysics(
                basis, bkd, depth=depth, bed=0.0, friction=1.0, A=1.0, rho=1.0
            )

        moving = velocity_physics(moving_depth)
        reference = velocity_physics(NodalField(moving_depth.values(time)))
        state = _ssa_velocity(bkd)
        residual = moving.residual(state, time)
        bkd.assert_allclose(residual, reference.residual(state, 0.0), rtol=1e-12)
        bkd.assert_allclose(
            moving.jacobian(state, time), reference.jacobian(state, 0.0), rtol=1e-12
        )
        change = bkd.max(bkd.abs(moving.residual(state, 0.0) - residual))
        assert float(change) > 1e-3

    @pytest.mark.parametrize("kind", ["velocity", "depth", "coupled"])
    def test_pickle_round_trip(self, bkd: Backend[Array], kind: str) -> None:
        """Each physics pickles and evaluates the same after loading."""
        physics = _ssa_physics(bkd, kind, dependent=False)
        loaded = pickle.loads(pickle.dumps(physics))
        state = _ssa_state(bkd, kind)
        bkd.assert_allclose(
            loaded.residual(state, 0.3), physics.residual(state, 0.3), rtol=1e-14
        )
