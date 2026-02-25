"""Tests for physics on curvilinear (non-affine) coordinate transforms.

Phase 1: Polar transform + ADR physics
Phase 2: Elliptical transform + ADR physics
Phase 3: Transient polar ADR
Phase 4: Parameterized tests
Phase 5: Helmholtz on polar
Phase 6: Linear elasticity on polar

All tests use Dirichlet BCs only. Robin/Neumann BC tests are in Phase 7.

Configuration notes:
- Polar domain: r ∈ [1, 2], θ ∈ [-π/2, π/2] (annular, avoids r=0 singularity)
- Elliptical domain: u ∈ [0.5, 2], v ∈ [0.1, π-0.1] (avoids singularities)
- Resolution: 30×30 for polar (matches legacy), 25×25 for elliptical
- Tolerances: Looser than Cartesian due to curvilinear gradient factors
"""

import math
import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.boundary import (
    DirichletBC,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
    ManufacturedLinearElasticityEquations,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh2D,
)
from pyapprox.pde.collocation.mesh.transforms.elliptical import (
    EllipticalTransform,
)
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics import (
    AdvectionDiffusionReaction,
    HelmholtzPhysics,
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrapper to adapt physics interface for DerivativeChecker.

    DerivativeChecker expects:
    - __call__(sample) with sample shape (nvars, 1) returning (nqoi, 1)
    - jacobian(sample) with sample shape (nvars, 1) returning (nqoi, nvars)
    """

    def __init__(self, physics, bkd: Backend[Array], time: float = 0.0):
        self._physics = physics
        self._bkd = bkd
        self._time = time

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._physics.nstates()

    def nqoi(self) -> int:
        return self._physics.nstates()

    def __call__(self, sample: Array) -> Array:
        # sample shape: (nvars, 1)
        state = sample[:, 0]
        result = self._physics.residual(state, self._time)
        return result.reshape(-1, 1)

    def jacobian(self, sample: Array) -> Array:
        # sample shape: (nvars, 1)
        state = sample[:, 0]
        return self._physics.jacobian(state, self._time)


# =============================================================================
# Phase 1: Polar Transform + ADR Physics Tests
# =============================================================================


class TestPolarADR(Generic[Array], unittest.TestCase):
    """Test ADR physics on polar coordinate domains.

    Domain: r ∈ [1, 2], θ ∈ [-π/2, π/2] (quarter annulus)
    Manufactured solution: x**2*y**2 (Cartesian polynomial, smooth on polar domain)

    Tolerances are looser than Cartesian tests because:
    - Chebyshev interpolation must represent trigonometric gradient factors
    - More points needed for spectral accuracy on transformed gradients
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_polar_mesh_and_basis(self, npts_r: int = 30, npts_theta: int = 30):
        """Create polar mesh and basis.

        Note: Resolution 30x30 gives ~1e-10 residual for x^2*y^2 solution.
        The Cartesian polynomial x^2*y^2 = r^4*cos^2(θ)*sin^2(θ) involves
        trigonometric terms requiring sufficient resolution for spectral accuracy.
        """
        bkd = self.bkd()
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(-math.pi / 2, math.pi / 2),
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        return mesh, basis

    def _create_manufactured_solution(self, diff: float = 4.0, vel=None, react=0.0):
        """Create manufactured solution for ADR on polar domain."""
        bkd = self.bkd()
        vel_strs = ["0", "0"] if vel is None else [str(v) for v in vel]
        react_str = f"{react}*u" if react != 0.0 else "0"

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2*y**2",
            nvars=2,
            diff_str=str(diff),
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
            oned=True,
        )
        return man_sol

    def _apply_dirichlet_bcs(self, physics, mesh, man_sol, physical_pts):
        """Apply Dirichlet BCs using manufactured solution values."""
        bkd = self.bkd()
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_pts = physical_pts[:, bkd.to_numpy(boundary_idx).astype(int)]
            bc_values = man_sol.functions["solution"](bc_pts)
            bc = DirichletBC(bkd, boundary_idx, bc_values)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

    def _get_interior_indices(self, mesh, basis):
        """Get indices of interior (non-boundary) points."""
        bkd = self.bkd()
        boundary_set = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_set.add(int(bkd.to_numpy(idx)))
        return [i for i in range(basis.npts()) if i not in boundary_set]

    def test_polar_adr_residual_diffusion_only(self):
        """Test residual at exact solution for diffusion-only on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = self._create_manufactured_solution(diff=4.0)

        # Get physical coordinates and evaluate manufactured solution
        physical_pts = mesh.points()  # Shape: (2, npts)
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        # Create physics
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=4.0, forcing=lambda t: forcing
        )

        # Apply Dirichlet BCs
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        # Compute residual
        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Check interior residual
        interior_idx = self._get_interior_indices(mesh, basis)
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_idx])

        # Tight tolerance - spectral accuracy with 25x25 points
        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )

    def test_polar_adr_residual_with_advection(self):
        """Test residual with advection on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = self._create_manufactured_solution(diff=0.1, vel=[1.0, 2.0])

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        # Constant velocity field in Cartesian coordinates
        npts = basis.npts()
        velocity = [bkd.ones((npts,)), 2.0 * bkd.ones((npts,))]

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, velocity=velocity, forcing=lambda t: forcing
        )
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior_idx = self._get_interior_indices(mesh, basis)
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_idx])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )

    def test_polar_adr_residual_with_reaction(self):
        """Test residual with reaction on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = self._create_manufactured_solution(diff=1.0, react=2.0)

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, reaction=2.0, forcing=lambda t: forcing
        )
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior_idx = self._get_interior_indices(mesh, basis)
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_idx])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )

    def test_polar_adr_jacobian(self):
        """Test Jacobian correctness via finite differences on polar domain."""
        bkd = self.bkd()
        # Use smaller grid for Jacobian test (faster, Jacobian check doesn't need high
        # resolution)
        mesh, basis = self._create_polar_mesh_and_basis(npts_r=12, npts_theta=12)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        npts = basis.npts()
        sample = bkd.asarray([[0.1 * float(i) for i in range(npts)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)

    def test_polar_adr_solve(self):
        """Test numerical solve matches manufactured solution on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = self._create_manufactured_solution(diff=4.0)

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=4.0, forcing=lambda t: forcing
        )
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((basis.npts(),))
        u_numerical = model.solve_steady(initial_guess, tol=1e-8, maxiter=10)

        # Tight tolerance - Newton should converge in few iterations
        bkd.assert_allclose(u_numerical, u_exact, atol=1e-8)


class TestPolarADRNumpy(TestPolarADR[NDArray[Any]]):
    """NumPy backend tests for polar ADR."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolarADRTorch(TestPolarADR[torch.Tensor]):
    """PyTorch backend tests for polar ADR."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    @slow_test
    def test_polar_adr_solve(self):
        super().test_polar_adr_solve()


# =============================================================================
# Phase 2: Elliptical Transform + ADR Physics Tests
# =============================================================================


class TestEllipticalADR(Generic[Array], unittest.TestCase):
    """Test ADR physics on elliptical coordinate domains.

    Domain: u ∈ [0.5, 2], v ∈ [0.1, π-0.1] (avoids singularities at v=0,π)
    Focal distance: a = 1.0
    Manufactured solution: x**2*y**2
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_elliptical_mesh_and_basis(self, npts_u: int = 25, npts_v: int = 25):
        """Create elliptical mesh and basis.

        Note: 25x25 gives ~1e-10 residual for x^2*y^2 solution on elliptical domain.
        """
        bkd = self.bkd()
        transform = EllipticalTransform(
            u_bounds=(0.5, 2.0),
            v_bounds=(0.1, math.pi - 0.1),
            a=1.0,
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_u, npts_v, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        return mesh, basis

    def _create_manufactured_solution(self, diff: float = 1.0, vel=None, react=0.0):
        """Create manufactured solution for ADR on elliptical domain."""
        bkd = self.bkd()
        vel_strs = ["0", "0"] if vel is None else [str(v) for v in vel]
        react_str = f"{react}*u" if react != 0.0 else "0"

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2*y**2",
            nvars=2,
            diff_str=str(diff),
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
            oned=True,
        )
        return man_sol

    def _apply_dirichlet_bcs(self, physics, mesh, man_sol, physical_pts):
        """Apply Dirichlet BCs using manufactured solution values."""
        bkd = self.bkd()
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_pts = physical_pts[:, bkd.to_numpy(boundary_idx).astype(int)]
            bc_values = man_sol.functions["solution"](bc_pts)
            bc = DirichletBC(bkd, boundary_idx, bc_values)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

    def _get_interior_indices(self, mesh, basis):
        """Get indices of interior (non-boundary) points."""
        bkd = self.bkd()
        boundary_set = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_set.add(int(bkd.to_numpy(idx)))
        return [i for i in range(basis.npts()) if i not in boundary_set]

    def test_elliptical_adr_residual(self):
        """Test residual at exact solution on elliptical domain."""
        bkd = self.bkd()
        mesh, basis = self._create_elliptical_mesh_and_basis()

        man_sol = self._create_manufactured_solution(diff=1.0)

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior_idx = self._get_interior_indices(mesh, basis)
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_idx])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )

    def test_elliptical_adr_jacobian(self):
        """Test Jacobian correctness on elliptical domain."""
        bkd = self.bkd()
        mesh, basis = self._create_elliptical_mesh_and_basis(npts_u=12, npts_v=12)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        npts = basis.npts()
        sample = bkd.asarray([[0.1 * float(i) for i in range(npts)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    def test_elliptical_adr_solve(self):
        """Test numerical solve on elliptical domain."""
        bkd = self.bkd()
        mesh, basis = self._create_elliptical_mesh_and_basis()

        man_sol = self._create_manufactured_solution(diff=1.0)

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((basis.npts(),))
        u_numerical = model.solve_steady(initial_guess, tol=1e-8, maxiter=10)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-8)


class TestEllipticalADRNumpy(TestEllipticalADR[NDArray[Any]]):
    """NumPy backend tests for elliptical ADR."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEllipticalADRTorch(TestEllipticalADR[torch.Tensor]):
    """PyTorch backend tests for elliptical ADR."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)


# =============================================================================
# Phase 3: Time-Dependent Polar ADR Tests
# =============================================================================


class TestTransientPolarADR(Generic[Array], unittest.TestCase):
    """Test time-dependent ADR on polar domains.

    Manufactured solution: x**2*y**2*(1 + T) - linear in time
    For linear-in-time solutions, backward Euler and Crank-Nicolson are exact.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_polar_mesh_and_basis(self, npts_r: int = 25, npts_theta: int = 25):
        """Create polar mesh and basis."""
        bkd = self.bkd()
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(-math.pi / 2, math.pi / 2),
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        return mesh, basis

    def _create_transient_manufactured_solution(self, diff: float = 1.0):
        """Create time-dependent manufactured solution linear in time."""
        bkd = self.bkd()
        # Linear in time: backward Euler and Crank-Nicolson are exact
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2*y**2*(1 + T)",
            nvars=2,
            diff_str=str(diff),
            react_str="0",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )
        return man_sol

    def _apply_time_dependent_dirichlet_bcs(self, physics, mesh, man_sol, physical_pts):
        """Apply time-dependent Dirichlet BCs."""
        bkd = self.bkd()
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_pts = physical_pts[:, bkd.to_numpy(boundary_idx).astype(int)]

            # Time-dependent BC values
            def bc_values_fn(t, pts=bc_pts, ms=man_sol):
                return ms.functions["solution"](pts, t)

            bc = DirichletBC(bkd, boundary_idx, bc_values_fn)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

    @slow_test
    def test_transient_polar_backward_euler(self):
        """Test transient polar ADR with backward Euler.

        Linear-in-time solution => backward Euler is exact with any time step.
        """
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = self._create_transient_manufactured_solution(diff=1.0)

        physical_pts = mesh.points()

        # Time-dependent forcing
        def forcing_fn(t):
            return man_sol.functions["forcing"](physical_pts, t)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=forcing_fn
        )
        self._apply_time_dependent_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        model = CollocationModel(physics, bkd)

        # Initial condition at t=0
        u_initial = man_sol.functions["solution"](physical_pts, 0.0)

        # Large time step - backward Euler exact for linear-in-time
        config = TimeIntegrationConfig(
            final_time=1.0,
            deltat=0.25,
            method="backward_euler",
        )

        # Solve - returns (solutions, times) where solutions shape is (nstates, ntimes)
        solutions, times = model.solve_transient(u_initial, config)
        u_final = solutions[:, -1]
        t_final = float(bkd.to_numpy(times[-1]))

        # Compare to exact solution at final time
        u_exact_final = man_sol.functions["solution"](physical_pts, t_final)

        # Tight tolerance - only spatial discretization error
        bkd.assert_allclose(u_final, u_exact_final, atol=1e-8)

    def test_transient_polar_crank_nicolson(self):
        """Test transient polar ADR with Crank-Nicolson.

        Linear-in-time solution => Crank-Nicolson is exact with any time step.
        """
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = self._create_transient_manufactured_solution(diff=1.0)

        physical_pts = mesh.points()

        def forcing_fn(t):
            return man_sol.functions["forcing"](physical_pts, t)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=forcing_fn
        )
        self._apply_time_dependent_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        model = CollocationModel(physics, bkd)

        u_initial = man_sol.functions["solution"](physical_pts, 0.0)

        # Large time step - Crank-Nicolson exact for linear-in-time
        config = TimeIntegrationConfig(
            final_time=1.0,
            deltat=0.5,
            method="crank_nicolson",
        )

        # Solve - returns (solutions, times) where solutions shape is (nstates, ntimes)
        solutions, times = model.solve_transient(u_initial, config)
        u_final = solutions[:, -1]
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact_final = man_sol.functions["solution"](physical_pts, t_final)

        bkd.assert_allclose(u_final, u_exact_final, atol=1e-8)


class TestTransientPolarADRNumpy(TestTransientPolarADR[NDArray[Any]]):
    """NumPy backend tests for transient polar ADR."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTransientPolarADRTorch(TestTransientPolarADR[torch.Tensor]):
    """PyTorch backend tests for transient polar ADR."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    @slow_test
    def test_transient_polar_crank_nicolson(self):
        super().test_transient_polar_crank_nicolson()


# =============================================================================
# Phase 4: Parameterized Tests
# =============================================================================


# Note: Full parameterized tests require unittest_parametrize.
# For now, we add a few key configurations as explicit test methods.


class TestTransformADRConfigurations(Generic[Array], unittest.TestCase):
    """Test various ADR configurations on curvilinear domains."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_elliptical_with_advection(self):
        """Test elliptical domain with advection."""
        bkd = self.bkd()

        transform = EllipticalTransform(
            u_bounds=(0.5, 2.0),
            v_bounds=(0.1, math.pi - 0.1),
            a=1.0,
            bkd=bkd,
        )
        mesh = TransformedMesh2D(25, 25, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="x**2*y**2",
            nvars=2,
            diff_str="0.1",
            react_str="0",
            vel_strs=["1.0", "2.0"],
            bkd=bkd,
            oned=True,
        )

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        npts = basis.npts()
        velocity = [bkd.ones((npts,)), 2.0 * bkd.ones((npts,))]

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, velocity=velocity, forcing=lambda t: forcing
        )

        # Apply BCs
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_pts = physical_pts[:, bkd.to_numpy(boundary_idx).astype(int)]
            bc_values = man_sol.functions["solution"](bc_pts)
            bc = DirichletBC(bkd, boundary_idx, bc_values)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        # Check residual
        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Get interior indices
        boundary_set = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_set.add(int(bkd.to_numpy(idx)))
        interior_idx = [i for i in range(basis.npts()) if i not in boundary_set]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_idx])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )


class TestTransformADRConfigurationsNumpy(TestTransformADRConfigurations[NDArray[Any]]):
    """NumPy backend tests for ADR configurations."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTransformADRConfigurationsTorch(TestTransformADRConfigurations[torch.Tensor]):
    """PyTorch backend tests for ADR configurations."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)


# =============================================================================
# Phase 5: Helmholtz Physics on Polar Transform Tests
# =============================================================================


class TestPolarHelmholtz(Generic[Array], unittest.TestCase):
    """Test Helmholtz physics on polar coordinate domains.

    Domain: r in [1, 2], theta in [-pi/2, pi/2] (quarter annulus)
    Manufactured solution: x**2*y**2

    Sign convention:
    - ManufacturedHelmholtz produces forcing for: -Delta u - k^2*u = f
    - HelmholtzPhysics solves: -Delta u + k^2*u = f
    - So we negate k^2 when passing to HelmholtzPhysics.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_polar_mesh_and_basis(self, npts_r: int = 30, npts_theta: int = 30):
        bkd = self.bkd()
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(-math.pi / 2, math.pi / 2),
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        return mesh, basis

    def _apply_dirichlet_bcs(self, physics, mesh, man_sol, physical_pts):
        bkd = self.bkd()
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_pts = physical_pts[:, bkd.to_numpy(boundary_idx).astype(int)]
            bc_values = man_sol.functions["solution"](bc_pts)
            bc = DirichletBC(bkd, boundary_idx, bc_values)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

    def _get_interior_indices(self, mesh, basis):
        bkd = self.bkd()
        boundary_set = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_set.add(int(bkd.to_numpy(idx)))
        return [i for i in range(basis.npts()) if i not in boundary_set]

    def test_polar_helmholtz_residual(self):
        """Test Helmholtz residual at exact solution on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        k2 = 2.0
        man_sol = ManufacturedHelmholtz(
            sol_str="x**2*y**2",
            nvars=2,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        # Negate k2 due to sign convention mismatch
        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=-k2, forcing=lambda t: forcing
        )
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior_idx = self._get_interior_indices(mesh, basis)
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_idx])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
        )

    def test_polar_helmholtz_jacobian(self):
        """Test Helmholtz Jacobian via finite differences on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis(npts_r=12, npts_theta=12)

        k2 = 2.0
        physics = HelmholtzPhysics(basis, bkd, wave_number_sq=-k2)

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        npts = basis.npts()
        sample = bkd.asarray([[0.1 * float(i) for i in range(npts)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)

    def test_polar_helmholtz_solve(self):
        """Test numerical solve for Helmholtz on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        k2 = 2.0
        man_sol = ManufacturedHelmholtz(
            sol_str="x**2*y**2",
            nvars=2,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=-k2, forcing=lambda t: forcing
        )
        self._apply_dirichlet_bcs(physics, mesh, man_sol, physical_pts)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((basis.npts(),))
        u_numerical = model.solve_steady(initial_guess, tol=1e-8, maxiter=10)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-8)


class TestPolarHelmholtzNumpy(TestPolarHelmholtz[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolarHelmholtzTorch(TestPolarHelmholtz[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    @slow_test
    def test_polar_helmholtz_solve(self):
        super().test_polar_helmholtz_solve()


# =============================================================================
# Phase 6: Linear Elasticity on Polar Transform Tests
# =============================================================================


class TestPolarLinearElasticity(Generic[Array], unittest.TestCase):
    """Test linear elasticity physics on polar coordinate domains.

    Domain: r in [1, 2], theta in [-pi/2, pi/2] (quarter annulus)
    Manufactured solution: u = x**2*y**2, v = x**2*y**2*y
    Material: lambda = 1.0, mu = 1.0

    Linear elasticity state vector is [u at all pts, v at all pts].
    Boundary conditions applied to both displacement components.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_polar_mesh_and_basis(self, npts_r: int = 25, npts_theta: int = 25):
        bkd = self.bkd()
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(-math.pi / 2, math.pi / 2),
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        return mesh, basis

    def _get_interior_indices(self, mesh, npts):
        """Get interior indices for vector-valued problem (2*npts state)."""
        bkd = self.bkd()
        boundary_set = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                idx_int = int(bkd.to_numpy(idx))
                boundary_set.add(idx_int)
                boundary_set.add(idx_int + npts)
        return [i for i in range(2 * npts) if i not in boundary_set]

    def test_polar_elasticity_residual(self):
        """Test linear elasticity residual at exact solution on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["x**2*y**2", "x**2*y**2*y"],
            nvars=2,
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
            oned=True,
        )

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
        )

        # Apply Dirichlet BCs to both displacement components
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_pts = physical_pts[:, bkd.to_numpy(boundary_idx).astype(int)]

            # u-component BC
            bc_vals_u = man_sol.functions["solution"](bc_pts)[:, 0]
            bcs.append(DirichletBC(bkd, boundary_idx, bc_vals_u))

            # v-component BC (offset indices by npts)
            bc_vals_v = man_sol.functions["solution"](bc_pts)[:, 1]
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bcs.append(DirichletBC(bkd, boundary_idx_v, bc_vals_v))
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact_flat, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact_flat, 0.0), u_exact_flat, 0.0
        )

        interior_idx = self._get_interior_indices(mesh, npts)
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_idx])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-7
        )

    def test_polar_elasticity_jacobian(self):
        """Test linear elasticity Jacobian via finite differences on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis(npts_r=6, npts_theta=6)

        physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        nstates = physics.nstates()
        sample = bkd.asarray([[0.1 * float(i) for i in range(nstates)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    def test_polar_elasticity_solve(self):
        """Test numerical solve for linear elasticity on polar domain."""
        bkd = self.bkd()
        mesh, basis = self._create_polar_mesh_and_basis()

        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["x**2*y**2", "x**2*y**2*y"],
            nvars=2,
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
            oned=True,
        )

        physical_pts = mesh.points()
        u_exact = man_sol.functions["solution"](physical_pts)
        forcing = man_sol.functions["forcing"](physical_pts)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_pts = physical_pts[:, bkd.to_numpy(boundary_idx).astype(int)]

            bc_vals_u = man_sol.functions["solution"](bc_pts)[:, 0]
            bcs.append(DirichletBC(bkd, boundary_idx, bc_vals_u))

            bc_vals_v = man_sol.functions["solution"](bc_pts)[:, 1]
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bcs.append(DirichletBC(bkd, boundary_idx_v, bc_vals_v))
        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((2 * npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-8, maxiter=20)

        bkd.assert_allclose(u_numerical, u_exact_flat, atol=1e-7)


class TestPolarLinearElasticityNumpy(TestPolarLinearElasticity[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolarLinearElasticityTorch(TestPolarLinearElasticity[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    @slow_test
    def test_polar_elasticity_solve(self):
        super().test_polar_elasticity_solve()

    @slow_test
    def test_polar_elasticity_residual(self):
        super().test_polar_elasticity_residual()


if __name__ == "__main__":
    unittest.main()
