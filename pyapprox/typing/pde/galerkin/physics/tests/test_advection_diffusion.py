"""Tests for LinearAdvectionDiffusionReaction physics.

This module contains:
1. Unit tests for matrix properties (TestLinearADRBase)
2. Parametrized convergence tests inspired by legacy test_finite_elements.py
   - 4 1D convergence rate tests with P2 elements
   - 4 2D convergence rate tests with P2 elements
   Uses manufactured solutions with natural (zero Neumann) boundary conditions.
"""

import unittest
from typing import Generic, Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver


class TestLinearADRBase(Generic[Array], unittest.TestCase):
    """Base test class for LinearAdvectionDiffusionReaction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_1d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_1d_mass_matrix_positive_definite(self) -> None:
        """Test mass matrix is positive definite."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        eigenvalues = np.linalg.eigvalsh(M_np)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_1d_stiffness_assembly(self) -> None:
        """Test stiffness matrix assembly in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=1.0, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        jac_np = self.bkd_inst.to_numpy(jac)

        # For pure diffusion (no reaction), -jacobian should be the
        # stiffness matrix, which should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_1d_residual_shape(self) -> None:
        """Test residual has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        res = physics.residual(u0, 0.0)

        self.assertEqual(res.shape, (physics.nstates(),))

    def test_1d_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        jac = physics.jacobian(u0, 0.0)

        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_2d_physics(self) -> None:
        """Test physics works in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=self.bkd_inst
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        # Initial condition
        u0 = physics.initial_condition(
            lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )

        # Check shapes
        self.assertEqual(u0.shape, (physics.nstates(),))

        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_with_forcing(self) -> None:
        """Test physics with forcing term."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def forcing(x):
            return np.ones(x.shape[1])

        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, forcing=forcing, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)

        # With forcing and u=0, residual should be non-zero
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_3d_physics(self) -> None:
        """Test physics works in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        # Initial condition
        u0 = physics.initial_condition(
            lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])
        )

        # Check shapes
        self.assertEqual(u0.shape, (physics.nstates(),))

        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

        jac = physics.jacobian(u0, 0.0)
        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_3d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 3D (pure diffusion)."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=1.0, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        jac_np = self.bkd_inst.to_numpy(jac)

        # For pure diffusion, -jacobian = stiffness matrix should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_3d_steady_state_solve(self) -> None:
        """Test steady-state solve in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=0.1,
            reaction=1.0,
            forcing=lambda x: np.ones(x.shape[1]),
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)
        result = solver.solve_linear()

        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1e-8)

    def test_manufactured_solution_1d(self) -> None:
        """Test convergence using manufactured solution in 1D.

        Use u_exact = cos(pi*x), which satisfies zero Neumann BCs at x=0,1.
        For diffusion-reaction: -D*u'' + r*u = f
        u'' = -pi^2 * cos(pi*x)
        f = D*pi^2*cos(pi*x) + r*cos(pi*x) = (D*pi^2 + r)*cos(pi*x)
        """
        D = 1.0
        r = 1.0

        def u_exact(x: NDArray[Any]) -> NDArray[Any]:
            return np.cos(np.pi * x[0])

        def forcing(x: NDArray[Any]) -> NDArray[Any]:
            return (D * np.pi**2 + r) * np.cos(np.pi * x[0])

        errors: List[float] = []
        mesh_sizes = [10, 20, 40]

        for nx in mesh_sizes:
            mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=self.bkd_inst)
            basis = LagrangeBasis(mesh, degree=1)

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                reaction=r,
                forcing=forcing,
                bkd=self.bkd_inst,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            u_num = self.bkd_inst.to_numpy(result.solution)
            dof_locs = self.bkd_inst.to_numpy(basis.dof_coordinates())
            u_ex = u_exact(dof_locs)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate (should be ~2 for P1 elements)
        errors_arr = np.array(errors)
        rates = np.log(errors_arr[:-1] / errors_arr[1:]) / np.log(2)

        # P1 elements should have convergence rate ~2
        self.assertTrue(np.all(rates > 1.5), f"Rates: {rates}")

    def test_manufactured_solution_2d(self) -> None:
        """Test convergence using manufactured solution in 2D.

        Use u_exact = cos(pi*x)*cos(pi*y), which satisfies zero Neumann BCs.
        Laplacian: -2*pi^2*cos(pi*x)*cos(pi*y)
        f = (2*D*pi^2 + r)*cos(pi*x)*cos(pi*y)
        """
        D = 1.0
        r = 1.0

        def u_exact(x: NDArray[Any]) -> NDArray[Any]:
            return np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])

        def forcing(x: NDArray[Any]) -> NDArray[Any]:
            return (2 * D * np.pi**2 + r) * np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])

        errors: List[float] = []
        mesh_sizes = [5, 10, 20]

        for n in mesh_sizes:
            mesh = StructuredMesh2D(
                nx=n, ny=n,
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                bkd=self.bkd_inst,
            )
            basis = LagrangeBasis(mesh, degree=1)

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                reaction=r,
                forcing=forcing,
                bkd=self.bkd_inst,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            u_num = self.bkd_inst.to_numpy(result.solution)
            dof_locs = self.bkd_inst.to_numpy(basis.dof_coordinates())
            u_ex = u_exact(dof_locs)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate
        errors_arr = np.array(errors)
        rates = np.log(errors_arr[:-1] / errors_arr[1:]) / np.log(2)

        # P1 elements should have convergence rate ~2
        self.assertTrue(np.all(rates > 1.5), f"Rates: {rates}")

    def test_manufactured_solution_3d(self) -> None:
        """Test convergence using manufactured solution in 3D.

        Use u_exact = cos(pi*x)*cos(pi*y)*cos(pi*z), which satisfies zero Neumann BCs.
        Laplacian: -3*pi^2*cos(pi*x)*cos(pi*y)*cos(pi*z)
        f = (3*D*pi^2 + r)*cos(pi*x)*cos(pi*y)*cos(pi*z)
        """
        D = 1.0
        r = 1.0

        def u_exact(x: NDArray[Any]) -> NDArray[Any]:
            return np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2])

        def forcing(x: NDArray[Any]) -> NDArray[Any]:
            return (
                (3 * D * np.pi**2 + r)
                * np.cos(np.pi * x[0])
                * np.cos(np.pi * x[1])
                * np.cos(np.pi * x[2])
            )

        errors: List[float] = []
        mesh_sizes = [3, 5, 8]

        for n in mesh_sizes:
            mesh = StructuredMesh3D(
                nx=n, ny=n, nz=n,
                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                bkd=self.bkd_inst,
            )
            basis = LagrangeBasis(mesh, degree=1)

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                reaction=r,
                forcing=forcing,
                bkd=self.bkd_inst,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            u_num = self.bkd_inst.to_numpy(result.solution)
            dof_locs = self.bkd_inst.to_numpy(basis.dof_coordinates())
            u_ex = u_exact(dof_locs)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate
        errors_arr = np.array(errors)
        rates = np.log(errors_arr[:-1] / errors_arr[1:]) / np.log(2)

        # P1 elements should have convergence rate ~2
        # Use lower threshold for 3D due to coarser meshes
        self.assertTrue(np.all(rates > 1.4), f"Rates: {rates}")


# =============================================================================
# Parametrized ADR Convergence Rate Tests with P2 Elements
# =============================================================================
#
# These tests verify convergence rates for the ADR solver with P2 elements.
# They are inspired by pyapprox/pde/galerkin/tests/test_finite_elements.py.
#
# Key design notes:
# - Use manufactured solutions with cos(k*pi*x) form that have zero Neumann
#   BCs (natural boundary conditions) to avoid explicit BC enforcement
# - Include reaction term (r > 0) to make the problem well-posed
# - Verify O(h^3) convergence rate for P2 elements with smooth solutions
#
# The 1D cases cover:
#   - Different wavenumbers: k=1, k=2
#   - With and without advection
# Total: 4 cases
#
# The 2D cases cover:
#   - Different wavenumbers
#   - With and without advection
# Total: 4 cases
# =============================================================================

# Test case format: (name, wavenumber_x, wavenumber_y, diffusivity, advection, reaction)
# For 1D: wavenumber_y is ignored (use 0)
# wavenumber: k in cos(k*pi*x) solution form

# 1D convergence rate test cases
ADR_1D_CONVERGENCE_CASES: List[Tuple[str, int, float, bool, float]] = [
    # (name, wavenumber, diffusivity, has_advection, reaction)
    ("1d_k1_diff_only", 1, 1.0, False, 1.0),
    ("1d_k2_diff_only", 2, 1.0, False, 1.0),
    ("1d_k1_with_adv", 1, 1.0, True, 1.0),
    ("1d_k2_with_adv", 2, 1.0, True, 1.0),
]

# 2D convergence rate test cases
ADR_2D_CONVERGENCE_CASES: List[Tuple[str, int, int, float, bool, float]] = [
    # (name, wavenumber_x, wavenumber_y, diffusivity, has_advection, reaction)
    ("2d_k11_diff_only", 1, 1, 1.0, False, 1.0),
    ("2d_k22_diff_only", 2, 2, 1.0, False, 1.0),
    ("2d_k11_with_adv", 1, 1, 1.0, True, 1.0),
    ("2d_k12_with_adv", 1, 2, 1.0, True, 1.0),
]


class TestParametrizedADR1DConvergence(ParametrizedTestCase):
    """Parametrized 1D ADR convergence tests with P2 elements.

    Uses P2 (quadratic) Lagrange elements with smooth manufactured solutions
    of the form u(x) = cos(k*pi*x) which have zero Neumann BCs naturally.
    Tests verify O(h^3) convergence rate for P2 elements.

    These tests are inspired by the 1D test cases from legacy
    test_finite_elements.py lines 636-664.
    """

    @parametrize(
        "name,wavenumber,diffusivity,has_advection,reaction",
        ADR_1D_CONVERGENCE_CASES,
    )
    def test_convergence_rate(
        self,
        name: str,
        wavenumber: int,
        diffusivity: float,
        has_advection: bool,
        reaction: float,
    ) -> None:
        """Test O(h^3) convergence for P2 elements."""
        bkd = NumpyBkd()
        mesh_sizes = [10, 20, 40]
        errors = []

        k = wavenumber
        D = diffusivity
        r = reaction

        # Exact solution: u(x) = cos(k*pi*x)
        # This satisfies zero Neumann BCs at x=0, x=1 naturally
        def u_exact(x: NDArray[Any]) -> NDArray[Any]:
            return np.cos(k * np.pi * x[0])

        # u'' = -k^2 * pi^2 * cos(k*pi*x)
        # Forcing for -D*u'' + r*u = f (no advection)
        # f = D*k^2*pi^2*cos(k*pi*x) + r*cos(k*pi*x)
        if has_advection:
            # With advection v*u' where v = 0.1
            # u' = -k*pi*sin(k*pi*x)
            v_coef = 0.1

            def forcing(x: NDArray[Any]) -> NDArray[Any]:
                return (
                    (D * (k * np.pi)**2 + r) * np.cos(k * np.pi * x[0])
                    - v_coef * k * np.pi * np.sin(k * np.pi * x[0])
                )
            velocity = bkd.asarray(np.array([v_coef]))
        else:

            def forcing(x: NDArray[Any]) -> NDArray[Any]:
                return (D * (k * np.pi)**2 + r) * np.cos(k * np.pi * x[0])
            velocity = None

        for nx in mesh_sizes:
            mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
            basis = LagrangeBasis(mesh, degree=2)  # P2 elements

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                velocity=velocity,
                reaction=r,
                forcing=forcing,
                bkd=bkd,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            dof_coords = bkd.to_numpy(basis.dof_coordinates())
            u_num = bkd.to_numpy(result.solution)
            u_ex = u_exact(dof_coords)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate (should be ~3 for P2 elements in L2 norm)
        errors_arr = np.array(errors)
        rates = np.log(errors_arr[:-1] / errors_arr[1:]) / np.log(2)

        # P2 elements should have convergence rate ~3
        # Use relaxed threshold for stability
        min_expected_rate = 2.5 if has_advection else 2.8
        self.assertTrue(
            np.all(rates > min_expected_rate),
            f"Test {name}: rates={rates} should be > {min_expected_rate}"
        )


class TestParametrizedADR2DConvergence(ParametrizedTestCase):
    """Parametrized 2D ADR convergence tests with P2 elements.

    Uses P2 (quadratic) Lagrange elements with smooth manufactured solutions
    of the form u(x,y) = cos(kx*pi*x)*cos(ky*pi*y) which have zero Neumann BCs.
    Tests verify O(h^3) convergence rate for P2 elements.

    These tests are inspired by the 2D test cases from legacy
    test_finite_elements.py lines 617-729.
    """

    @parametrize(
        "name,wavenumber_x,wavenumber_y,diffusivity,has_advection,reaction",
        ADR_2D_CONVERGENCE_CASES,
    )
    def test_convergence_rate(
        self,
        name: str,
        wavenumber_x: int,
        wavenumber_y: int,
        diffusivity: float,
        has_advection: bool,
        reaction: float,
    ) -> None:
        """Test O(h^3) convergence for P2 elements in 2D."""
        bkd = NumpyBkd()
        mesh_sizes = [5, 10, 20]
        errors = []

        kx, ky = wavenumber_x, wavenumber_y
        D = diffusivity
        r = reaction

        # Exact solution: u(x,y) = cos(kx*pi*x)*cos(ky*pi*y)
        # This satisfies zero Neumann BCs on [0,1]^2 naturally
        def u_exact(x: NDArray[Any]) -> NDArray[Any]:
            return np.cos(kx * np.pi * x[0]) * np.cos(ky * np.pi * x[1])

        # Laplacian = -(kx^2 + ky^2)*pi^2*cos(kx*pi*x)*cos(ky*pi*y)
        # Forcing for -D*Laplacian + r*u = f
        # f = D*(kx^2 + ky^2)*pi^2*u + r*u = (D*(kx^2+ky^2)*pi^2 + r)*u
        k_sq = kx**2 + ky**2
        if has_advection:
            # With advection v.grad(u) where v = [0.1, 0.1]
            # du/dx = -kx*pi*sin(kx*pi*x)*cos(ky*pi*y)
            # du/dy = -ky*pi*cos(kx*pi*x)*sin(ky*pi*y)
            vx, vy = 0.1, 0.1

            def forcing(x: NDArray[Any]) -> NDArray[Any]:
                u_val = np.cos(kx * np.pi * x[0]) * np.cos(ky * np.pi * x[1])
                dudx = -kx * np.pi * np.sin(kx * np.pi * x[0]) * np.cos(ky * np.pi * x[1])
                dudy = -ky * np.pi * np.cos(kx * np.pi * x[0]) * np.sin(ky * np.pi * x[1])
                return (
                    (D * k_sq * np.pi**2 + r) * u_val
                    + vx * dudx + vy * dudy
                )
            velocity = bkd.asarray(np.array([vx, vy]))
        else:

            def forcing(x: NDArray[Any]) -> NDArray[Any]:
                u_val = np.cos(kx * np.pi * x[0]) * np.cos(ky * np.pi * x[1])
                return (D * k_sq * np.pi**2 + r) * u_val
            velocity = None

        for n in mesh_sizes:
            mesh = StructuredMesh2D(
                nx=n, ny=n,
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                bkd=bkd,
            )
            basis = LagrangeBasis(mesh, degree=2)  # P2 elements

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                velocity=velocity,
                reaction=r,
                forcing=forcing,
                bkd=bkd,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            dof_coords = bkd.to_numpy(basis.dof_coordinates())
            u_num = bkd.to_numpy(result.solution)
            u_ex = u_exact(dof_coords)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate (should be ~3 for P2 elements)
        errors_arr = np.array(errors)
        rates = np.log(errors_arr[:-1] / errors_arr[1:]) / np.log(2)

        # P2 elements should have convergence rate ~3
        # Use relaxed threshold for stability
        min_expected_rate = 2.3 if has_advection else 2.5
        self.assertTrue(
            np.all(rates > min_expected_rate),
            f"Test {name}: rates={rates} should be > {min_expected_rate}"
        )


class TestLinearADRNumpy(TestLinearADRBase[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Try to import torch for dual-backend testing
try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestLinearADRTorch(TestLinearADRBase[torch.Tensor]):
        """PyTorch backend tests."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

except ImportError:
    pass


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()
