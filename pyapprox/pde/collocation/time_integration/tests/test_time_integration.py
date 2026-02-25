"""Tests for time integration module."""

import math
import unittest
from typing import Generic

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    PhysicsToODEResidualAdapter,
    TimeIntegrationConfig,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestPhysicsToODEResidualAdapter(Generic[Array], unittest.TestCase):
    """Base test class for PhysicsToODEResidualAdapter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_adapter_basic_interface(self):
        """Test that adapter provides ODEResidual interface."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        # Check interface methods exist
        self.assertTrue(callable(adapter.bkd))
        self.assertTrue(callable(adapter.set_time))
        self.assertTrue(callable(adapter.jacobian))
        self.assertTrue(callable(adapter.mass_matrix))

        # __call__ should work
        state = bkd.zeros((npts,))
        adapter.set_time(0.0)
        result = adapter(state)
        self.assertEqual(result.shape, (npts,))

    def test_adapter_residual_consistency(self):
        """Test that adapter residual matches physics residual."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        nodes = basis.nodes()
        state = bkd.sin(math.pi * nodes)
        time = 0.5

        # Physics residual
        physics_res = physics.residual(state, time)

        # Adapter residual (without BCs)
        adapter.set_time(time)
        adapter_res = adapter(state)

        # Should match when no BCs
        bkd.assert_allclose(adapter_res, physics_res, atol=1e-14)

    def test_adapter_jacobian_consistency(self):
        """Test that adapter Jacobian matches physics Jacobian."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        nodes = basis.nodes()
        state = bkd.sin(math.pi * nodes)
        time = 0.5

        # Physics Jacobian
        physics_jac = physics.jacobian(state, time)

        # Adapter Jacobian (without BCs)
        adapter.set_time(time)
        adapter_jac = adapter.jacobian(state)

        bkd.assert_allclose(adapter_jac, physics_jac, atol=1e-14)

    def test_adapter_mass_matrix(self):
        """Test that adapter returns identity mass matrix."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        mass = adapter.mass_matrix(npts)
        expected = bkd.eye(npts)

        bkd.assert_allclose(mass, expected, atol=1e-14)

    def test_adapter_with_boundary_conditions(self):
        """Test that BCs are applied via physics.apply_boundary_conditions.

        The adapter returns the raw physics Jacobian. Boundary conditions
        are applied by CollocationModel._apply_boundary_conditions, which
        calls physics.apply_boundary_conditions on the Newton system.
        """
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Set BCs: u(-1) = 0, u(1) = 1
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = constant_dirichlet_bc(bkd, right_idx, 1.0)
        physics.set_boundary_conditions([bc_left, bc_right])

        # State that satisfies BCs
        nodes = basis.nodes()
        state = 0.5 * (nodes + 1.0)  # Linear from 0 to 1

        # Get raw Jacobian from physics, then apply BCs
        residual = physics.residual(state, 0.0)
        jacobian = physics.jacobian(state, 0.0)
        _, jacobian_with_bc = physics.apply_boundary_conditions(
            residual, jacobian, state, 0.0
        )

        # Boundary rows should be identity-like after applying BCs
        bkd.assert_allclose(jacobian_with_bc[0, :], bkd.eye(npts)[0, :], atol=1e-14)
        bkd.assert_allclose(jacobian_with_bc[-1, :], bkd.eye(npts)[-1, :], atol=1e-14)

    def test_adapter_param_jacobian_available(self):
        """Test that param_jacobian is available via parameterization."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        phi0 = bkd.ones((npts,))
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        fm = BasisExpansion(bkd, 1.0, [phi0])
        param = create_diffusion_parameterization(bkd, basis, fm)

        adapter = PhysicsToODEResidualAdapter(physics, bkd, parameterization=param)

        # Should have param_jacobian
        self.assertTrue(hasattr(adapter, "param_jacobian"))
        self.assertTrue(hasattr(adapter, "nparams"))
        self.assertTrue(hasattr(adapter, "set_param"))

        # Test that it works
        adapter.set_param(bkd.array([0.5]))
        nodes = basis.nodes()
        state = bkd.sin(math.pi * nodes)
        adapter.set_time(0.0)

        param_jac = adapter.param_jacobian(state)
        self.assertEqual(param_jac.shape, (npts, 1))

    def test_adapter_no_param_jacobian_for_basic_physics(self):
        """Test that basic physics does not have param_jacobian."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        adapter = PhysicsToODEResidualAdapter(physics, bkd)

        # Should NOT have param_jacobian
        self.assertFalse(hasattr(adapter, "param_jacobian"))
        self.assertFalse(hasattr(adapter, "nparams"))


class TestCollocationModel(Generic[Array], unittest.TestCase):
    """Base test class for CollocationModel."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_model_creation(self):
        """Test basic model creation."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        model = CollocationModel(physics, bkd)

        self.assertEqual(model.nstates(), npts)
        self.assertIs(model.physics(), physics)
        self.assertIs(model.bkd(), bkd)

    def test_solve_steady_poisson(self):
        """Test steady-state solve for Poisson equation.

        Solve: D * laplacian(u) + f = 0 with u(-1) = 0, u(1) = 0

        For u = sin(pi*x), laplacian(u) = -pi^2 * sin(pi*x)
        So D * laplacian(u) = -D * pi^2 * sin(pi*x)
        For residual = 0, need f = D * pi^2 * sin(pi*x)
        """
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        D = 1.0
        nodes = basis.nodes()

        # Forcing f such that D*laplacian(u) + f = 0 for u = sin(pi*x)
        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=D, forcing=forcing)

        # BCs: u(-1) = 0, u(1) = 0 (consistent with sin(pi*x))
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial guess - use exact solution for better convergence
        initial_guess = bkd.sin(math.pi * nodes) * 0.5

        # Solve
        u_steady = model.solve_steady(initial_guess, tol=1e-10, maxiter=50)

        # Compare to exact solution
        u_exact = bkd.sin(math.pi * nodes)
        bkd.assert_allclose(u_steady, u_exact, atol=1e-6)

    def test_solve_transient_decay(self):
        """Test transient solve for exponential decay.

        Solve: du/dt = -r * u with u(0) = 1
        Exact solution: u(t) = exp(-r * t)

        For backward Euler with small time step, should converge to exact.
        """
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 2.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)

        model = CollocationModel(physics, bkd)

        # Initial condition
        u0 = bkd.ones((npts,))

        # Time integration config with smaller time step for accuracy
        config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=1.0,
            deltat=0.01,  # Smaller dt for better accuracy
        )

        # Solve
        solutions, times = model.solve_transient(u0, config)

        # Compare to exact at final time
        u_exact_final = math.exp(-r * 1.0) * bkd.ones((npts,))
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.05)

    def test_transient_forward_euler(self):
        """Test Forward Euler time stepping."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 1.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)
        model = CollocationModel(physics, bkd)

        u0 = bkd.ones((npts,))
        config = TimeIntegrationConfig(
            method="forward_euler",
            final_time=0.5,
            deltat=0.01,  # Small dt for stability
        )

        solutions, times = model.solve_transient(u0, config)

        u_exact_final = math.exp(-r * 0.5) * bkd.ones((npts,))
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.02)

    def test_transient_heun(self):
        """Test Heun's method (RK2) time stepping."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 1.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)
        model = CollocationModel(physics, bkd)

        u0 = bkd.ones((npts,))
        config = TimeIntegrationConfig(
            method="heun",
            final_time=0.5,
            deltat=0.05,
        )

        solutions, times = model.solve_transient(u0, config)

        u_exact_final = math.exp(-r * 0.5) * bkd.ones((npts,))
        # Heun should be more accurate than Forward Euler
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.01)

    def test_transient_crank_nicolson(self):
        """Test Crank-Nicolson time stepping."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        r = 1.0
        physics = AdvectionDiffusionReaction(basis, bkd, reaction=-r)
        model = CollocationModel(physics, bkd)

        u0 = bkd.ones((npts,))
        config = TimeIntegrationConfig(
            method="crank_nicolson",
            final_time=0.5,
            deltat=0.1,
        )

        solutions, times = model.solve_transient(u0, config)

        u_exact_final = math.exp(-r * 0.5) * bkd.ones((npts,))
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.01)

    def test_transient_diffusion(self):
        """Test transient diffusion equation.

        Solve: du/dt = D * laplacian(u) with BCs and IC.
        """
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        D = 0.1
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=D)

        # BCs: u(-1) = 0, u(1) = 0
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial condition: sin(pi*x)
        nodes = basis.nodes()
        u0 = bkd.sin(math.pi * nodes)

        config = TimeIntegrationConfig(
            method="backward_euler",
            final_time=0.5,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(u0, config)

        # Exact solution: exp(-D * pi^2 * t) * sin(pi*x)
        u_exact_final = math.exp(-D * math.pi**2 * 0.5) * bkd.sin(math.pi * nodes)

        # Should be reasonably close
        # Use atol for boundary points which are near zero
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.05, atol=1e-10)

    def test_time_output_shape(self):
        """Test that output shapes are correct."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        model = CollocationModel(physics, bkd)

        u0 = bkd.zeros((npts,))
        config = TimeIntegrationConfig(
            method="backward_euler",
            final_time=0.5,
            deltat=0.1,
        )

        solutions, times = model.solve_transient(u0, config)

        # Should have 6 time points: 0, 0.1, 0.2, 0.3, 0.4, 0.5
        self.assertEqual(times.shape[0], 6)
        self.assertEqual(solutions.shape, (npts, 6))


# NumPy backend tests
class TestPhysicsToODEResidualAdapterNumpy(TestPhysicsToODEResidualAdapter):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestCollocationModelNumpy(TestCollocationModel):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
