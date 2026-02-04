"""Tests for ADR manufactured solutions.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions for machine precision)
2. Jacobian correctness via DerivativeChecker
3. Numerical solution matches exact solution

Note: For residual tests, we use polynomial solutions that can be exactly
represented by Chebyshev interpolation. With n points, polynomials up to
degree n-1 are exactly represented. We use degree <= 4 to ensure machine
precision residuals even with moderate grid sizes.
"""

import unittest
import math
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
)
from pyapprox.typing.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    create_uniform_mesh_2d,
    create_uniform_mesh_3d,
)
from pyapprox.typing.pde.collocation.boundary import (
    DirichletBC,
    constant_dirichlet_bc,
    zero_dirichlet_bc,
)
from pyapprox.typing.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.typing.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.util.cartesian import cartesian_product_samples


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrapper to adapt physics interface for DerivativeChecker.

    DerivativeChecker expects:
    - bkd() -> Backend
    - nvars() -> int
    - nqoi() -> int
    - __call__(samples) -> Array
    - jacobian(sample) -> Array

    Physics provides:
    - residual(state, time)
    - jacobian(state, time)

    This wrapper fixes time=0 for derivative checking.
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
        """Evaluate residual at sample."""
        # sample is (nvars, 1) but physics expects (npts,)
        state = sample.flatten()
        result = self._physics.residual(state, self._time)
        return result.reshape(-1, 1)

    def jacobian(self, sample: Array) -> Array:
        """Evaluate Jacobian at sample."""
        state = sample.flatten()
        return self._physics.jacobian(state, self._time)


class TestManufacturedADR1D(Generic[Array], unittest.TestCase):
    """Test ADR manufactured solutions in 1D.

    For residual tests, we use polynomial solutions that can be exactly
    represented by Chebyshev interpolation:
    - u = (1-x**2)*x  (degree 3, zero at x=±1)
    - u = (1-x**2)*(1+x)  (degree 3, zero at x=±1)

    This gives machine precision residuals (< 1e-12).
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_steady_diffusion_residual(self):
        """Test residual = 0 at exact solution for steady diffusion.

        Uses polynomial solution u = (1-x**2)*x = x - x**3 (degree 3).
        With 10+ points, this should give machine precision.
        """
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial manufactured solution: u = (1-x**2)*x = x - x**3
        # This is degree 3 and zero at x=±1
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        # Get exact solution and forcing at nodes
        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Create physics with manufactured forcing
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        # Apply BCs: u(-1) = 0, u(1) = 0
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        # Compute residual at exact solution
        residual = physics.residual(u_exact, 0.0)

        # Apply BCs to residual
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Interior residual should be near machine precision
        bkd.assert_allclose(residual_with_bc[1:-1], bkd.zeros((npts - 2,)), atol=1e-12)

    def test_steady_diffusion_jacobian(self):
        """Test Jacobian correctness via DerivativeChecker."""
        bkd = self.bkd()
        npts = 15
        basis = ChebyshevBasis1D(npts, bkd)

        # Simple diffusion problem
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Wrap for DerivativeChecker
        wrapper = PhysicsDerivativeWrapper(physics, bkd)

        # Create random test point
        sample = bkd.asarray([[float(i) / npts for i in range(npts)]]).T

        # Check derivatives
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        # Should have small relative error at some epsilon
        min_error = float(bkd.min(errors[0]))
        self.assertLess(min_error, 1e-5)

    def test_steady_diffusion_solve(self):
        """Test numerical solution matches manufactured solution."""
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial manufactured solution
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Solve
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        # Should match exact solution to machine precision
        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)

    def test_advection_diffusion_residual(self):
        """Test residual for advection-diffusion problem.

        Uses polynomial solution u = (1-x**2)*(1+x) = 1 + x - x**2 - x**3 (degree 3).
        """
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial manufactured solution with advection
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 + x)",
            nvars=1,
            diff_str="0.1",
            react_str="0",
            vel_strs=["1.0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        velocity = [bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, velocity=velocity, forcing=lambda t: forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Interior residual should be near machine precision
        bkd.assert_allclose(residual_with_bc[1:-1], bkd.zeros((npts - 2,)), atol=1e-12)

    def test_advection_diffusion_jacobian(self):
        """Test Jacobian for advection-diffusion problem."""
        bkd = self.bkd()
        npts = 15
        basis = ChebyshevBasis1D(npts, bkd)

        velocity = [bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, velocity=velocity
        )

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        sample = bkd.asarray([[math.sin(float(i) / npts) for i in range(npts)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        min_error = float(bkd.min(errors[0]))
        self.assertLess(min_error, 1e-5)

    def test_reaction_diffusion_residual(self):
        """Test residual for reaction-diffusion problem.

        Uses polynomial solution u = (1-x**2)*x (degree 3).
        """
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial manufactured solution with reaction
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*x",
            nvars=1,
            diff_str="1.0",
            react_str="-u",  # Linear decay
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, reaction=-1.0, forcing=lambda t: forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        bkd.assert_allclose(residual_with_bc[1:-1], bkd.zeros((npts - 2,)), atol=1e-12)

    def test_reaction_diffusion_jacobian(self):
        """Test Jacobian for reaction-diffusion problem."""
        bkd = self.bkd()
        npts = 15
        basis = ChebyshevBasis1D(npts, bkd)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, reaction=-1.0
        )

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        sample = bkd.asarray([[math.sin(float(i) / npts) for i in range(npts)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        min_error = float(bkd.min(errors[0]))
        self.assertLess(min_error, 1e-5)

    def test_full_adr_residual(self):
        """Test residual for full ADR problem.

        Uses polynomial solution u = (1-x**2)*(1+2*x) (degree 3).
        """
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Full ADR with polynomial manufactured solution
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 + 2*x)",
            nvars=1,
            diff_str="0.1",
            react_str="-0.5*u",
            vel_strs=["1.0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        velocity = [bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(
            basis, bkd,
            diffusion=0.1,
            velocity=velocity,
            reaction=-0.5,
            forcing=lambda t: forcing,
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        bkd.assert_allclose(residual_with_bc[1:-1], bkd.zeros((npts - 2,)), atol=1e-12)

    def test_full_adr_solve(self):
        """Test numerical solution for full ADR problem."""
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 + 2*x)",
            nvars=1,
            diff_str="0.1",
            react_str="-0.5*u",
            vel_strs=["1.0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        velocity = [bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(
            basis, bkd,
            diffusion=0.1,
            velocity=velocity,
            reaction=-0.5,
            forcing=lambda t: forcing,
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)


class TestManufacturedADR2D(Generic[Array], unittest.TestCase):
    """Test ADR manufactured solutions in 2D.

    For residual tests, we use polynomial solutions that can be exactly
    represented by the tensor product Chebyshev basis:
    - u = (1-x**2)*(1-y**2) (degree 2 in each direction, zero on boundaries)
    - u = (1-x**2)*(1-y**2)*x*y (degree 3 in each direction)

    With 8x8 or more points, these give machine precision residuals.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_steady_diffusion_residual_2d(self):
        """Test residual = 0 for 2D steady diffusion.

        Uses polynomial solution u = (1-x**2)*(1-y**2) (degree 2 in each direction).
        """
        bkd = self.bkd()
        npts_x, npts_y = 8, 8
        basis = ChebyshevBasis2D(npts_x, npts_y, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        # Polynomial manufactured solution: u = (1-x**2)*(1-y**2)
        # Degree 2 in each direction, zero on all boundaries
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 - y**2)",
            nvars=2,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )

        # Construct mesh nodes from 1D nodes (tensor product ordering)
        # Use 'xy' indexing for compatibility with Kronecker product structure
        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)  # (2, npts)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        # Set BCs on all 4 boundaries
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Get interior indices
        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(idx)
        interior_indices = [
            i for i in range(basis.npts()) if i not in boundary_indices
        ]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        # Machine precision residual for polynomial solution
        bkd.assert_allclose(interior_residual, bkd.zeros(interior_residual.shape), atol=1e-12)

    def test_steady_diffusion_jacobian_2d(self):
        """Test Jacobian for 2D diffusion."""
        bkd = self.bkd()
        npts_x, npts_y = 8, 8
        basis = ChebyshevBasis2D(npts_x, npts_y, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        npts = basis.npts()
        sample = bkd.asarray([[0.1 * float(i) for i in range(npts)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        min_error = float(bkd.min(errors[0]))
        self.assertLess(min_error, 1e-5)

    def test_steady_diffusion_solve_2d(self):
        """Test numerical solution matches manufactured solution in 2D."""
        bkd = self.bkd()
        npts_x, npts_y = 10, 10
        basis = ChebyshevBasis2D(npts_x, npts_y, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 - y**2)",
            nvars=2,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0", "0"],
            bkd=bkd,
            oned=True,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((basis.npts(),))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-10)

    def test_advection_diffusion_2d(self):
        """Test residual for 2D advection-diffusion.

        Uses polynomial solution u = (1-x**2)*(1-y**2)*x (degree 3 in x, 2 in y).
        """
        bkd = self.bkd()
        npts_x, npts_y = 10, 10
        basis = ChebyshevBasis2D(npts_x, npts_y, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        # Polynomial with advection: degree 3 in x, degree 2 in y
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 - y**2)*x",
            nvars=2,
            diff_str="0.1",
            react_str="0",
            vel_strs=["1.0", "0.5"],
            bkd=bkd,
            oned=True,
        )

        # Construct mesh nodes from 1D nodes
        # Use 'xy' indexing for compatibility with Kronecker product structure
        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        velocity = [bkd.ones((npts,)), 0.5 * bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, velocity=velocity, forcing=lambda t: forcing
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Get interior
        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(idx)
        interior_indices = [
            i for i in range(basis.npts()) if i not in boundary_indices
        ]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        # Machine precision residual for polynomial solution
        bkd.assert_allclose(interior_residual, bkd.zeros(interior_residual.shape), atol=1e-12)


class TestManufacturedADRTransient(Generic[Array], unittest.TestCase):
    """Test transient ADR manufactured solutions.

    For transient tests, we use polynomial solutions in BOTH space and time:
    - u = (1 - 0.5*T)*(1-x**2)*x  (polynomial in space and time)

    This gives machine precision residuals.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_transient_diffusion_residual(self):
        """Test transient manufactured solution residual equals du/dt.

        For transient problems, residual = L(u) + f = du/dt at exact solution.
        Uses polynomial solution in both space and time.
        """
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Transient polynomial solution: u = (1 - 0.5*T)*(1-x**2)*x
        # du/dt = -0.5*(1-x**2)*x
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - 0.5*T)*(1 - x**2)*x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        time = 0.5

        u_exact = man_sol.functions["solution"](nodes, time)

        # The forcing includes both spatial and temporal terms
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: man_sol.functions["forcing"](nodes, t)
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        # The residual should equal du/dt
        residual = physics.residual(u_exact, time)

        # Expected: du/dt = -0.5*(1-x**2)*x
        # Compute expected at nodes
        x = nodes.flatten()
        expected_dudt = -0.5 * (1 - x**2) * x

        # Interior points only (BCs modify boundary residual)
        bkd.assert_allclose(residual[1:-1], expected_dudt[1:-1], atol=1e-12)

    def test_transient_diffusion_solve(self):
        """Test transient solve with manufactured solution.

        Uses polynomial solution for machine precision verification.
        """
        bkd = self.bkd()
        npts = 10
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial transient solution
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - 0.5*T)*(1 - x**2)*x",
            nvars=1,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes().reshape(1, -1)
        time_start = 0.0
        time_end = 0.1  # Short time to stay in valid range

        u0 = man_sol.functions["solution"](nodes, time_start)
        u_exact_end = man_sol.functions["solution"](nodes, time_end)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: man_sol.functions["forcing"](nodes, t)
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Solve with backward Euler (implicit, stable)
        config = TimeIntegrationConfig(
            method="backward_euler",
            deltat=0.01,
            init_time=time_start,
            final_time=time_end,
        )
        solutions, times = model.solve_transient(u0, config)

        # Final solution should match exact
        u_final = solutions[:, -1]
        bkd.assert_allclose(u_final, u_exact_end, atol=1e-3)  # Time discretization error


class TestManufacturedADR1DNumpy(TestManufacturedADR1D):
    """Numpy implementation of 1D ADR tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedADR2DNumpy(TestManufacturedADR2D):
    """Numpy implementation of 2D ADR tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedADRTransientNumpy(TestManufacturedADRTransient):
    """Numpy implementation of transient ADR tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


# =============================================================================
# Parameterized Tests
# =============================================================================
# These tests cover parameter combinations matching legacy tests

from unittest_parametrize import ParametrizedTestCase, parametrize

# ADR 1D test cases matching legacy itertools.product combinations
# NOTE: Non-standard domain cases (domain_01, domain_02) are commented out
# until domain transformation is implemented in the basis layer.
# See: plans/domain-transformation-implementation.md
ADR_1D_STEADY_CASES = [
    # (name, sol_str, diff, vel, react, npts, domain)
    ("diffusion_only", "(1 - x**2)*x", 1.0, [0.0], 0.0, 10, (-1.0, 1.0)),
    ("diff_4", "(1 - x**2)*x", 4.0, [0.0], 0.0, 10, (-1.0, 1.0)),
    ("with_advection", "(1 - x**2)*(1 + x)", 0.1, [1.0], 0.0, 15, (-1.0, 1.0)),
    ("with_reaction", "(1 - x**2)*x", 1.0, [0.0], -1.0, 10, (-1.0, 1.0)),
    ("full_adr", "(1 - x**2)*(1 + 2*x)", 0.1, [1.0], -0.5, 15, (-1.0, 1.0)),
    # TODO: Re-enable after implementing domain transformation
    # ("domain_01", "x*(1-x)", 1.0, [0.0], 0.0, 10, (0.0, 1.0)),
    # ("domain_02", "x*(2-x)", 1.0, [0.0], 0.0, 10, (0.0, 2.0)),
]


class TestADR1DParameterized(ParametrizedTestCase):
    """Parameterized tests for 1D ADR covering legacy test combinations."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,sol_str,diff,vel,react,npts,domain",
        ADR_1D_STEADY_CASES,
    )
    def test_steady_adr_1d_residual(self, name, sol_str, diff, vel, react, npts, domain):
        """Test residual = 0 at manufactured solution for various configurations."""
        bkd = self.bkd()
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, domain, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str=sol_str,
            nvars=1,
            diff_str=str(diff),
            react_str=f"{react}*u" if react != 0 else "0",
            vel_strs=[str(vel[0])],
            bkd=bkd,
            oned=True,
        )

        # Map nodes to domain
        nodes_ref = basis.nodes()  # in [-1, 1]
        lb, ub = domain
        nodes = lb + (nodes_ref + 1.0) / 2.0 * (ub - lb)
        nodes_2d = nodes.reshape(1, -1)

        u_exact = man_sol.functions["solution"](nodes_2d)
        forcing = man_sol.functions["forcing"](nodes_2d)

        # Create physics with velocity array
        velocity = [vel[0] * bkd.ones((npts,))] if vel[0] != 0 else None
        physics = AdvectionDiffusionReaction(
            basis, bkd,
            diffusion=diff,
            velocity=velocity,
            reaction=react,
            forcing=lambda t: forcing,
        )

        # Set BCs
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        # Compute residual at exact solution
        residual = physics.residual(u_exact, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Interior residual should be near machine precision
        bkd.assert_allclose(residual_bc[1:-1], bkd.zeros((npts - 2,)), atol=1e-10)

    @parametrize(
        "name,sol_str,diff,vel,react,npts,domain",
        ADR_1D_STEADY_CASES,
    )
    def test_steady_adr_1d_solve(self, name, sol_str, diff, vel, react, npts, domain):
        """Test numerical solution matches manufactured solution for various configs."""
        bkd = self.bkd()
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, domain, bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str=sol_str,
            nvars=1,
            diff_str=str(diff),
            react_str=f"{react}*u" if react != 0 else "0",
            vel_strs=[str(vel[0])],
            bkd=bkd,
            oned=True,
        )

        # Map nodes to domain
        nodes_ref = basis.nodes()
        lb, ub = domain
        nodes = lb + (nodes_ref + 1.0) / 2.0 * (ub - lb)
        nodes_2d = nodes.reshape(1, -1)

        u_exact = man_sol.functions["solution"](nodes_2d)
        forcing = man_sol.functions["forcing"](nodes_2d)

        velocity = [vel[0] * bkd.ones((npts,))] if vel[0] != 0 else None
        physics = AdvectionDiffusionReaction(
            basis, bkd,
            diffusion=diff,
            velocity=velocity,
            reaction=react,
            forcing=lambda t: forcing,
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-12, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact, atol=1e-8)


# ADR 2D test cases
ADR_2D_STEADY_CASES = [
    # (name, sol_str, diff, vel, npts)
    ("diffusion_only", "(1 - x**2)*(1 - y**2)", 1.0, [0.0, 0.0], (8, 8)),
    ("with_advection", "(1 - x**2)*(1 - y**2)*x", 0.1, [1.0, 0.5], (10, 10)),
]


class TestADR2DParameterized(ParametrizedTestCase):
    """Parameterized tests for 2D ADR."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,sol_str,diff,vel,npts",
        ADR_2D_STEADY_CASES,
    )
    def test_steady_adr_2d_residual(self, name, sol_str, diff, vel, npts):
        """Test 2D residual = 0 at manufactured solution."""
        bkd = self.bkd()
        npts_x, npts_y = npts
        basis = ChebyshevBasis2D(npts_x, npts_y, bkd)
        mesh = create_uniform_mesh_2d(npts, (-1.0, 1.0, -1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str=sol_str,
            nvars=2,
            diff_str=str(diff),
            react_str="0",
            vel_strs=[str(vel[0]), str(vel[1])],
            bkd=bkd,
            oned=True,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        total_npts = basis.npts()
        velocity = None
        if vel[0] != 0 or vel[1] != 0:
            velocity = [
                vel[0] * bkd.ones((total_npts,)),
                vel[1] * bkd.ones((total_npts,)),
            ]

        physics = AdvectionDiffusionReaction(
            basis, bkd,
            diffusion=diff,
            velocity=velocity,
            forcing=lambda t: forcing,
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Get interior indices
        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(idx)
        interior_indices = [
            i for i in range(basis.npts()) if i not in boundary_indices
        ]
        interior_residual = bkd.asarray([residual_bc[i] for i in interior_indices])

        bkd.assert_allclose(interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10)


# =============================================================================
# 3D ADR Tests
# =============================================================================

class TestManufacturedADR3D(Generic[Array], unittest.TestCase):
    """Test ADR manufactured solutions in 3D.

    For residual tests, we use polynomial solutions that can be exactly
    represented by the tensor product Chebyshev basis:
    - u = (1-x**2)*(1-y**2)*(1-z**2) (degree 2 in each direction, zero on boundaries)

    With 6x6x6 or more points, these give machine precision residuals.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_steady_diffusion_residual_3d(self):
        """Test residual = 0 for 3D steady diffusion.

        Uses polynomial solution u = (1-x**2)*(1-y**2)*(1-z**2) (degree 2 in each direction).
        """
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 6, 6, 6
        basis = ChebyshevBasis3D(npts_x, npts_y, npts_z, bkd)
        mesh = create_uniform_mesh_3d((npts_x, npts_y, npts_z), (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), bkd)

        # Polynomial manufactured solution: u = (1-x**2)*(1-y**2)*(1-z**2)
        # Degree 2 in each direction, zero on all boundaries
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 - y**2)*(1 - z**2)",
            nvars=3,
            diff_str="1.0",
            react_str="0",
            vel_strs=["0", "0", "0"],
            bkd=bkd,
            oned=True,
        )

        # Construct mesh nodes from 1D nodes (tensor product ordering)
        # Use first_dim_fastest=True to match Kronecker product structure
        nodes_1d = [basis.nodes_x(), basis.nodes_y(), basis.nodes_z()]
        nodes = cartesian_product_samples(nodes_1d, bkd, first_dim_fastest=True)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, forcing=lambda t: forcing
        )

        # Set BCs on all 6 boundaries
        bcs = []
        for side in range(6):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Get interior indices
        boundary_indices = set()
        for side in range(6):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(int(idx))
        interior_indices = [
            i for i in range(basis.npts()) if i not in boundary_indices
        ]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        # Machine precision residual for polynomial solution
        bkd.assert_allclose(interior_residual, bkd.zeros(interior_residual.shape), atol=1e-12)

    def test_steady_diffusion_jacobian_3d(self):
        """Test Jacobian for 3D diffusion."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 5, 5, 5
        basis = ChebyshevBasis3D(npts_x, npts_y, npts_z, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        npts = basis.npts()
        sample = bkd.asarray([[0.1 * float(i) / npts for i in range(npts)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        min_error = float(bkd.min(errors[0]))
        self.assertLess(min_error, 1e-5)

    def test_advection_diffusion_3d(self):
        """Test residual for 3D advection-diffusion.

        Uses polynomial solution u = (1-x**2)*(1-y**2)*(1-z**2)*x (degree 3 in x, 2 in y,z).
        """
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 8, 6, 6
        basis = ChebyshevBasis3D(npts_x, npts_y, npts_z, bkd)
        mesh = create_uniform_mesh_3d((npts_x, npts_y, npts_z), (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), bkd)

        # Polynomial with advection: degree 3 in x, degree 2 in y,z
        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="(1 - x**2)*(1 - y**2)*(1 - z**2)*x",
            nvars=3,
            diff_str="0.1",
            react_str="0",
            vel_strs=["1.0", "0.5", "0.2"],
            bkd=bkd,
            oned=True,
        )

        # Construct mesh nodes using first_dim_fastest for Kronecker product structure
        nodes_1d = [basis.nodes_x(), basis.nodes_y(), basis.nodes_z()]
        nodes = cartesian_product_samples(nodes_1d, bkd, first_dim_fastest=True)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        velocity = [bkd.ones((npts,)), 0.5 * bkd.ones((npts,)), 0.2 * bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, velocity=velocity, forcing=lambda t: forcing
        )

        bcs = []
        for side in range(6):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Get interior
        boundary_indices = set()
        for side in range(6):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(int(idx))
        interior_indices = [
            i for i in range(basis.npts()) if i not in boundary_indices
        ]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        # Machine precision residual for polynomial solution
        bkd.assert_allclose(interior_residual, bkd.zeros(interior_residual.shape), atol=1e-12)


class TestManufacturedADR3DNumpy(TestManufacturedADR3D):
    """Numpy implementation of 3D ADR tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


# ADR 3D test cases for parameterized tests
ADR_3D_STEADY_CASES = [
    # (name, sol_str, diff, vel, npts)
    ("diffusion_only", "(1 - x**2)*(1 - y**2)*(1 - z**2)", 1.0, [0.0, 0.0, 0.0], (5, 5, 5)),
    ("diff_4", "(1 - x**2)*(1 - y**2)*(1 - z**2)", 4.0, [0.0, 0.0, 0.0], (5, 5, 5)),
    ("with_advection", "(1 - x**2)*(1 - y**2)*(1 - z**2)*x", 0.1, [1.0, 0.5, 0.2], (6, 5, 5)),
]


class TestADR3DParameterized(ParametrizedTestCase):
    """Parameterized tests for 3D ADR."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,sol_str,diff,vel,npts",
        ADR_3D_STEADY_CASES,
    )
    def test_steady_adr_3d_residual(self, name, sol_str, diff, vel, npts):
        """Test 3D residual = 0 at manufactured solution."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = npts
        basis = ChebyshevBasis3D(npts_x, npts_y, npts_z, bkd)
        mesh = create_uniform_mesh_3d(npts, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), bkd)

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str=sol_str,
            nvars=3,
            diff_str=str(diff),
            react_str="0",
            vel_strs=[str(vel[0]), str(vel[1]), str(vel[2])],
            bkd=bkd,
            oned=True,
        )

        # Construct mesh nodes using first_dim_fastest for Kronecker product structure
        nodes_1d = [basis.nodes_x(), basis.nodes_y(), basis.nodes_z()]
        nodes = cartesian_product_samples(nodes_1d, bkd, first_dim_fastest=True)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        total_npts = basis.npts()
        velocity = None
        if vel[0] != 0 or vel[1] != 0 or vel[2] != 0:
            velocity = [
                vel[0] * bkd.ones((total_npts,)),
                vel[1] * bkd.ones((total_npts,)),
                vel[2] * bkd.ones((total_npts,)),
            ]

        physics = AdvectionDiffusionReaction(
            basis, bkd,
            diffusion=diff,
            velocity=velocity,
            forcing=lambda t: forcing,
        )

        bcs = []
        for side in range(6):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Get interior indices
        boundary_indices = set()
        for side in range(6):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(int(idx))
        interior_indices = [
            i for i in range(basis.npts()) if i not in boundary_indices
        ]
        interior_residual = bkd.asarray([residual_bc[i] for i in interior_indices])

        bkd.assert_allclose(interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
