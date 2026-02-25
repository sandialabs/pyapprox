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
from scipy.sparse import issparse
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.solvers import SteadyStateSolver


class TestLinearADRBase(Generic[Array], unittest.TestCase):
    """Base test class for LinearAdvectionDiffusionReaction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def _to_dense(self, matrix):
        """Convert sparse or backend matrix to dense numpy array."""
        if issparse(matrix):
            return matrix.toarray()
        return self.bkd_inst.to_numpy(matrix)

    def test_1d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self._to_dense(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_1d_mass_matrix_positive_definite(self) -> None:
        """Test mass matrix is positive definite."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self._to_dense(M)

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
        jac_np = self._to_dense(jac)

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
        M_np = self._to_dense(M)

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
        jac_np = self._to_dense(jac)

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
        Uses ManufacturedAdvectionDiffusionReaction for consistent forcing.
        """
        D = 1.0
        r = 1.0
        bounds = [0.0, 1.0]

        # Create manufactured solution (always use NumpyBkd for skfem compatibility)
        numpy_bkd = NumpyBkd()
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="cos(pi*x)",
            diff_str=str(D),
            react_str=f"{r}*u",
            vel_strs=["0"],
            bkd=numpy_bkd,
        )
        u_exact = functions["solution"]
        forcing = functions["forcing"]

        errors: List[float] = []
        mesh_sizes = [10, 20, 40]

        for nx in mesh_sizes:
            mesh = StructuredMesh1D(nx=nx, bounds=(bounds[0], bounds[1]), bkd=self.bkd_inst)
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
        Uses ManufacturedAdvectionDiffusionReaction for consistent forcing.
        """
        D = 1.0
        r = 1.0
        bounds = [0.0, 1.0, 0.0, 1.0]

        # Create manufactured solution (always use NumpyBkd for skfem compatibility)
        numpy_bkd = NumpyBkd()
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="cos(pi*x)*cos(pi*y)",
            diff_str=str(D),
            react_str=f"{r}*u",
            vel_strs=["0", "0"],
            bkd=numpy_bkd,
        )
        u_exact = functions["solution"]
        forcing = functions["forcing"]

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
        Uses ManufacturedAdvectionDiffusionReaction for consistent forcing.
        """
        D = 1.0
        r = 1.0
        bounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

        # Create manufactured solution (always use NumpyBkd for skfem compatibility)
        numpy_bkd = NumpyBkd()
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="cos(pi*x)*cos(pi*y)*cos(pi*z)",
            diff_str=str(D),
            react_str=f"{r}*u",
            vel_strs=["0", "0", "0"],
            bkd=numpy_bkd,
        )
        u_exact = functions["solution"]
        forcing = functions["forcing"]

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
        bounds = [0.0, 1.0]

        # Build solution string: cos(k*pi*x)
        sol_str = f"cos({k}*pi*x)"

        # Build velocity string
        if has_advection:
            v_coef = 0.1
            vel_strs = [str(v_coef)]
            velocity = bkd.asarray(np.array([v_coef]))
        else:
            vel_strs = ["0"]
            velocity = None

        # Create manufactured solution using the infrastructure
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=str(D),
            react_str=f"{r}*u",
            vel_strs=vel_strs,
            bkd=bkd,
        )
        u_exact = functions["solution"]
        forcing = functions["forcing"]

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
        bounds = [0.0, 1.0, 0.0, 1.0]

        # Build solution string: cos(kx*pi*x)*cos(ky*pi*y)
        sol_str = f"cos({kx}*pi*x)*cos({ky}*pi*y)"

        # Build velocity strings
        if has_advection:
            vx, vy = 0.1, 0.1
            vel_strs = [str(vx), str(vy)]
            velocity = bkd.asarray(np.array([vx, vy]))
        else:
            vel_strs = ["0", "0"]
            velocity = None

        # Create manufactured solution using the infrastructure
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=str(D),
            react_str=f"{r}*u",
            vel_strs=vel_strs,
            bkd=bkd,
        )
        u_exact = functions["solution"]
        forcing = functions["forcing"]

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
    from pyapprox.util.backends.torch import TorchBkd

    class TestLinearADRTorch(TestLinearADRBase[torch.Tensor]):
        """PyTorch backend tests."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_manufactured_solution_1d(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_manufactured_solution_2d(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_manufactured_solution_3d(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_3d_steady_state_solve(self) -> None:
            pass

except ImportError:
    pass


# =============================================================================
# Parametrized ADR Exact Reproduction Tests with P2 Elements and BCs
# =============================================================================
#
# These tests verify exact reproduction for linear/quadratic solutions with
# P2 elements and various boundary conditions (Dirichlet, Neumann, Robin).
# They are inspired by pyapprox/pde/galerkin/tests/test_finite_elements.py
# test_advection_diffusion_reaction (lines 617-765).
#
# With P2 (quadratic) elements:
# - Linear solutions (e.g., "x", "x+2*y") should be reproduced exactly
# - Quadratic solutions (e.g., "x**2", "x**2*y**2") should be reproduced exactly
# - Expected relative error: < 1e-8
#
# 1D cases (24 total):
#   - bounds: [0, 1] or [0, 1.1]
#   - bndry_types: ["D", "N"], ["R", "D"], ["R", "R"]
#   - velocity: none or advective
#   - reaction: 0 or linear
#
# 2D cases (9 total):
#   - Various BC combinations
#   - Linear and quadratic solutions
# =============================================================================

# Import manufactured solution infrastructure
from pyapprox.pde.galerkin.manufactured import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
)

# 1D exact reproduction test cases with various BCs
# Format: (name, bounds, bndry_types, sol_str, diffusivity, vel_str, reaction)
# vel_str: velocity string for manufactured solution ("1e-16*x" = no advection)
# reaction: reaction coefficient (0.0 = no reaction, positive = source)
#
# 24 cases from itertools.product:
#   bounds: [0, 1], [0, 1.1]                          (2)
#   bndry_types: ["D", "N"], ["R", "D"], ["R", "R"]   (3)
#   vel_str: "1e-16*x" (none), "(1+x)/10" (advective) (2)
#   reaction: 0.0, 2.0                                 (2)
# Total: 2 × 3 × 2 × 2 = 24
ADR_1D_EXACT_CASES: List[Tuple[str, List[float], List[str], str, float, str, float]] = [
    # --- Domain [0, 1], no advection, no reaction ---
    ("1d_b01_DN_nov_nor", [0.0, 1.0], ["D", "N"], "x", 4.0, "1e-16*x", 0.0),
    ("1d_b01_RD_nov_nor", [0.0, 1.0], ["R", "D"], "x", 4.0, "1e-16*x", 0.0),
    ("1d_b01_RR_nov_nor", [0.0, 1.0], ["R", "R"], "x", 4.0, "1e-16*x", 0.0),
    # --- Domain [0, 1], no advection, reaction=2.0 ---
    ("1d_b01_DN_nov_r2", [0.0, 1.0], ["D", "N"], "x", 4.0, "1e-16*x", 2.0),
    ("1d_b01_RD_nov_r2", [0.0, 1.0], ["R", "D"], "x", 4.0, "1e-16*x", 2.0),
    ("1d_b01_RR_nov_r2", [0.0, 1.0], ["R", "R"], "x", 4.0, "1e-16*x", 2.0),
    # --- Domain [0, 1], advection=(1+x)/10, no reaction ---
    ("1d_b01_DN_vel_nor", [0.0, 1.0], ["D", "N"], "x", 4.0, "(1+x)/10", 0.0),
    ("1d_b01_RD_vel_nor", [0.0, 1.0], ["R", "D"], "x", 4.0, "(1+x)/10", 0.0),
    ("1d_b01_RR_vel_nor", [0.0, 1.0], ["R", "R"], "x", 4.0, "(1+x)/10", 0.0),
    # --- Domain [0, 1], advection=(1+x)/10, reaction=2.0 ---
    ("1d_b01_DN_vel_r2", [0.0, 1.0], ["D", "N"], "x", 4.0, "(1+x)/10", 2.0),
    ("1d_b01_RD_vel_r2", [0.0, 1.0], ["R", "D"], "x", 4.0, "(1+x)/10", 2.0),
    ("1d_b01_RR_vel_r2", [0.0, 1.0], ["R", "R"], "x", 4.0, "(1+x)/10", 2.0),
    # --- Domain [0, 1.1], no advection, no reaction ---
    ("1d_b011_DN_nov_nor", [0.0, 1.1], ["D", "N"], "x", 4.0, "1e-16*x", 0.0),
    ("1d_b011_RD_nov_nor", [0.0, 1.1], ["R", "D"], "x", 4.0, "1e-16*x", 0.0),
    ("1d_b011_RR_nov_nor", [0.0, 1.1], ["R", "R"], "x", 4.0, "1e-16*x", 0.0),
    # --- Domain [0, 1.1], no advection, reaction=2.0 ---
    ("1d_b011_DN_nov_r2", [0.0, 1.1], ["D", "N"], "x", 4.0, "1e-16*x", 2.0),
    ("1d_b011_RD_nov_r2", [0.0, 1.1], ["R", "D"], "x", 4.0, "1e-16*x", 2.0),
    ("1d_b011_RR_nov_r2", [0.0, 1.1], ["R", "R"], "x", 4.0, "1e-16*x", 2.0),
    # --- Domain [0, 1.1], advection=(1+x)/10, no reaction ---
    ("1d_b011_DN_vel_nor", [0.0, 1.1], ["D", "N"], "x", 4.0, "(1+x)/10", 0.0),
    ("1d_b011_RD_vel_nor", [0.0, 1.1], ["R", "D"], "x", 4.0, "(1+x)/10", 0.0),
    ("1d_b011_RR_vel_nor", [0.0, 1.1], ["R", "R"], "x", 4.0, "(1+x)/10", 0.0),
    # --- Domain [0, 1.1], advection=(1+x)/10, reaction=2.0 ---
    ("1d_b011_DN_vel_r2", [0.0, 1.1], ["D", "N"], "x", 4.0, "(1+x)/10", 2.0),
    ("1d_b011_RD_vel_r2", [0.0, 1.1], ["R", "D"], "x", 4.0, "(1+x)/10", 2.0),
    ("1d_b011_RR_vel_r2", [0.0, 1.1], ["R", "R"], "x", 4.0, "(1+x)/10", 2.0),
]

# 2D exact reproduction test cases with various BCs
# Format: (name, bounds, bndry_types, sol_str, diffusivity, vel_strs, reaction)
# vel_strs: velocity strings for manufactured solution (["1e-16*x", "1e-16*y"] = no advection)
# reaction: reaction coefficient (0.0 = no reaction, positive = source)
ADR_2D_EXACT_CASES: List[Tuple[str, List[float], List[str], str, float, List[str], float]] = [
    # Linear solution, all Dirichlet
    ("2d_lin_DDDD", [0.0, 1.0, 0.0, 1.0], ["D", "D", "D", "D"], "x+2*y", 4.0, ["1e-16*x", "1e-16*y"], 0.0),
    # Linear solution, mixed BCs
    ("2d_lin_DRDN", [0.0, 1.0, 0.0, 1.0], ["D", "R", "D", "N"], "x+2*y", 4.0, ["1e-16*x", "1e-16*y"], 0.0),
    ("2d_lin_NRDR", [0.0, 1.0, 0.0, 1.0], ["N", "R", "D", "R"], "x+2*y", 4.0, ["1e-16*x", "1e-16*y"], 0.0),
    # Quadratic solution x*y
    ("2d_xy_DDDD", [0.0, 1.0, 0.0, 1.0], ["D", "D", "D", "D"], "x*y", 4.0, ["1e-16*x", "1e-16*y"], 0.0),
    ("2d_xy_DRDD", [0.0, 1.0, 0.0, 1.0], ["D", "R", "D", "D"], "x*y", 4.0, ["1e-16*x", "1e-16*y"], 0.0),
    # Quadratic solution x^2 + y^2
    ("2d_quad_DDDD", [0.0, 1.0, 0.0, 1.0], ["D", "D", "D", "D"], "x**2+y**2", 4.0, ["1e-16*x", "1e-16*y"], 0.0),
    ("2d_quad_DRDN", [0.0, 1.0, 0.0, 1.0], ["D", "R", "D", "N"], "x**2+y**2", 4.0, ["1e-16*x", "1e-16*y"], 0.0),
    # Higher-order quadratic x^2*y^2
    ("2d_x2y2_DDDD", [0.0, 1.0, 0.0, 1.0], ["D", "D", "D", "D"], "x**2*y**2", 1.0, ["1e-16*x", "1e-16*y"], 0.0),
    ("2d_x2y2_DRDD", [0.0, 1.0, 0.0, 1.0], ["D", "R", "D", "D"], "x**2*y**2", 1.0, ["1e-16*x", "1e-16*y"], 0.0),
]


class TestParametrizedADR1DExact(ParametrizedTestCase):
    """Parametrized 1D ADR exact reproduction tests with P2 elements.

    Uses P2 (quadratic) Lagrange elements with linear solutions and various
    boundary conditions. With linear solutions, P2 elements should reproduce
    the exact solution to near machine precision.

    These tests are inspired by the 1D test cases from legacy
    test_finite_elements.py lines 636-694.
    """

    @parametrize(
        "name,bounds,bndry_types,sol_str,diffusivity,vel_str,reaction",
        ADR_1D_EXACT_CASES,
    )
    def test_exact_reproduction(
        self,
        name: str,
        bounds: List[float],
        bndry_types: List[str],
        sol_str: str,
        diffusivity: float,
        vel_str: str,
        reaction: float,
    ) -> None:
        """Test exact reproduction for linear solution with P2 elements."""
        bkd = NumpyBkd()
        nx = 10  # Mesh size

        # Create mesh and basis
        mesh = StructuredMesh1D(
            nx=nx,
            bounds=(bounds[0], bounds[1]),
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=2)  # P2 elements

        # Create manufactured solution functions
        # Note: for linear reaction, use react_str = "r*u" format
        react_str = f"{reaction}*u" if reaction != 0 else "0*u"
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=f"{diffusivity}+1e-16*x",
            react_str=react_str,
            vel_strs=[vel_str],
            bkd=bkd,
        )

        # Create adapter and boundary conditions
        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)
        bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

        # Get exact solution and forcing
        exact_sol_func = adapter.solution_function()
        forcing_func = adapter.forcing_for_galerkin()

        # Get velocity function adapted for Galerkin (skfem) shape convention
        if "1e-16" in vel_str:
            velocity = None
        else:
            velocity = adapter.velocity_for_galerkin()

        # Create physics with boundary conditions
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=diffusivity,
            velocity=velocity,
            reaction=reaction,
            forcing=forcing_func,
            boundary_conditions=bc_set.all_conditions(),
            bkd=bkd,
        )

        # Solve
        solver = SteadyStateSolver(physics, tol=1e-12)
        result = solver.solve_linear()

        # Compute error
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        u_num = bkd.to_numpy(result.solution)
        u_exact = exact_sol_func(dof_coords)
        if u_exact.ndim > 1:
            u_exact = u_exact[:, 0] if u_exact.shape[1] == 1 else u_exact.flatten()

        # Compute relative error (handle near-zero solutions)
        u_norm = np.linalg.norm(u_exact)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact)

        # P2 elements should exactly reproduce linear solutions
        self.assertLess(
            rel_error, 1e-8,
            f"Test {name}: rel_error={rel_error:.2e} should be < 1e-8"
        )


# 1D conservative advection exact reproduction test cases
# Only cases with velocity are meaningful (conservative = non-conservative without advection)
# Format: (name, bounds, bndry_types, sol_str, diffusivity, vel_str, reaction)
ADR_1D_CONSERVATIVE_CASES: List[Tuple[str, List[float], List[str], str, float, str, float]] = [
    ("1d_cons_DN_vel_nor", [0.0, 1.0], ["D", "N"], "x", 4.0, "(1+x)/10", 0.0),
    ("1d_cons_RD_vel_r2", [0.0, 1.0], ["R", "D"], "x", 4.0, "(1+x)/10", 2.0),
    ("1d_cons_RR_vel_nor", [0.0, 1.0], ["R", "R"], "x", 4.0, "(1+x)/10", 0.0),
    ("1d_cons_RR_vel_r2", [0.0, 1.0], ["R", "R"], "x", 4.0, "(1+x)/10", 2.0),
]


class TestParametrizedADR1DConservative(ParametrizedTestCase):
    """Parametrized 1D ADR exact reproduction tests with conservative advection.

    Uses P2 elements with linear solutions and conservative advection form
    div(v*u). The manufactured solution is created with conservative=True to
    match the PDE. The adapter uses only the diffusive flux for BCs because
    the Galerkin weak form drops the advective boundary term from IBP.
    """

    @parametrize(
        "name,bounds,bndry_types,sol_str,diffusivity,vel_str,reaction",
        ADR_1D_CONSERVATIVE_CASES,
    )
    def test_exact_reproduction(
        self,
        name: str,
        bounds: List[float],
        bndry_types: List[str],
        sol_str: str,
        diffusivity: float,
        vel_str: str,
        reaction: float,
    ) -> None:
        """Test exact reproduction with conservative advection and P2 elements."""
        bkd = NumpyBkd()
        nx = 10

        mesh = StructuredMesh1D(
            nx=nx,
            bounds=(bounds[0], bounds[1]),
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=2)

        react_str = f"{reaction}*u" if reaction != 0 else "0*u"
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=f"{diffusivity}+1e-16*x",
            react_str=react_str,
            vel_strs=[vel_str],
            bkd=bkd,
            conservative=True,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, bkd, conservative=True,
        )
        bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

        exact_sol_func = adapter.solution_function()
        forcing_func = adapter.forcing_for_galerkin()
        velocity = adapter.velocity_for_galerkin()

        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=diffusivity,
            velocity=velocity,
            reaction=reaction,
            forcing=forcing_func,
            boundary_conditions=bc_set.all_conditions(),
            bkd=bkd,
            conservative=True,
        )

        solver = SteadyStateSolver(physics, tol=1e-12)
        result = solver.solve_linear()

        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        u_num = bkd.to_numpy(result.solution)
        u_exact = exact_sol_func(dof_coords)
        if u_exact.ndim > 1:
            u_exact = u_exact[:, 0] if u_exact.shape[1] == 1 else u_exact.flatten()

        u_norm = np.linalg.norm(u_exact)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact)

        self.assertLess(
            rel_error, 1e-8,
            f"Test {name}: rel_error={rel_error:.2e} should be < 1e-8"
        )


class TestParametrizedADR2DExact(ParametrizedTestCase):
    """Parametrized 2D ADR exact reproduction tests with P2 elements.

    Uses P2 (quadratic) Lagrange elements with linear/quadratic solutions
    and various boundary conditions. P2 elements should reproduce solutions
    up to degree 2 exactly.

    These tests are inspired by the 2D test cases from legacy
    test_finite_elements.py lines 617-729.
    """

    @parametrize(
        "name,bounds,bndry_types,sol_str,diffusivity,vel_strs,reaction",
        ADR_2D_EXACT_CASES,
    )
    def test_exact_reproduction(
        self,
        name: str,
        bounds: List[float],
        bndry_types: List[str],
        sol_str: str,
        diffusivity: float,
        vel_strs: List[str],
        reaction: float,
    ) -> None:
        """Test exact reproduction for linear/quadratic solution with P2 elements."""
        bkd = NumpyBkd()
        nx, ny = 5, 5  # Mesh size

        # Create mesh and basis
        mesh = StructuredMesh2D(
            nx=nx, ny=ny,
            bounds=[
                (bounds[0], bounds[1]),
                (bounds[2], bounds[3]),
            ],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=2)  # P2 elements

        # Create manufactured solution functions
        react_str = f"{reaction}*u" if reaction != 0 else "0*u"
        functions, _ = create_adr_manufactured_test(
            bounds=bounds,
            sol_str=sol_str,
            diff_str=f"{diffusivity}+1e-16*x+1e-16*y",
            react_str=react_str,
            vel_strs=vel_strs,
            bkd=bkd,
        )

        # Create adapter and boundary conditions
        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)
        bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

        # Get exact solution and forcing
        exact_sol_func = adapter.solution_function()
        forcing_func = adapter.forcing_for_galerkin()

        # Parse velocity (check if effectively zero)
        has_velocity = any("1e-16" not in v for v in vel_strs)
        if has_velocity:
            velocity = bkd.asarray(np.array([0.1, 0.1]))
        else:
            velocity = None

        # Create physics with boundary conditions
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=diffusivity,
            velocity=velocity,
            reaction=reaction,
            forcing=forcing_func,
            boundary_conditions=bc_set.all_conditions(),
            bkd=bkd,
        )

        # Solve
        solver = SteadyStateSolver(physics, tol=1e-12)
        result = solver.solve_linear()

        # Compute error
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        u_num = bkd.to_numpy(result.solution)
        u_exact = exact_sol_func(dof_coords)
        if u_exact.ndim > 1:
            u_exact = u_exact[:, 0] if u_exact.shape[1] == 1 else u_exact.flatten()

        # Compute relative error
        u_norm = np.linalg.norm(u_exact)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact)

        # P2 elements should exactly reproduce quadratic solutions
        self.assertLess(
            rel_error, 1e-8,
            f"Test {name}: rel_error={rel_error:.2e} should be < 1e-8"
        )


from pyapprox.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()
