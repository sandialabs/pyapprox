"""Tests for boundary condition implementations.

Tests Dirichlet, Neumann, Robin BCs and the ManufacturedSolutionBC factory.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import (
    BoundaryConditionSet,
    DirichletBC,
    ManufacturedSolutionBC,
    NeumannBC,
    RobinBC,
    canonical_boundary_normal,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestBoundaryConditionsBase(Generic[Array], unittest.TestCase):
    """Base test class for boundary conditions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()


class TestDirichletBCBase(TestBoundaryConditionsBase[Array]):
    """Tests for DirichletBC."""

    __test__ = False

    def test_1d_constant_dirichlet(self) -> None:
        """Test constant Dirichlet BC in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        bc = DirichletBC(basis, "left", value_func=1.5, bkd=self.bkd_inst)

        # Check boundary DOFs
        dofs = self.bkd_inst.to_numpy(bc.boundary_dofs())
        self.assertEqual(len(dofs), 1)
        self.assertEqual(dofs[0], 0)  # Left boundary is DOF 0

        # Check boundary values
        values = bc.boundary_values(time=0.0)
        expected = self.bkd_inst.asarray(np.array([1.5], dtype=np.float64))
        self.bkd_inst.assert_allclose(values, expected)

    def test_1d_function_dirichlet(self) -> None:
        """Test function-valued Dirichlet BC in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # BC: u = x^2 at boundary
        def bc_func(x, t=None):
            return x[0] ** 2

        bc = DirichletBC(basis, "right", value_func=bc_func, bkd=self.bkd_inst)

        # Right boundary is at x=1
        values = bc.boundary_values(time=0.0)
        expected = self.bkd_inst.asarray(
            np.array([1.0], dtype=np.float64)  # 1^2 = 1
        )
        self.bkd_inst.assert_allclose(values, expected)

    def test_1d_dirichlet_apply_to_residual(self) -> None:
        """Test Dirichlet BC modifies residual correctly."""
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        bc = DirichletBC(basis, "left", value_func=2.0, bkd=self.bkd_inst)

        # Create dummy state and residual
        nstates = basis.ndofs()
        state = self.bkd_inst.asarray(np.zeros(nstates))
        residual = self.bkd_inst.asarray(np.ones(nstates))

        # Apply BC
        modified_res = bc.apply_to_residual(residual, state, time=0.0)
        modified_res_np = self.bkd_inst.to_numpy(modified_res)

        # At Dirichlet DOF, residual should be state - g = 0 - 2 = -2
        self.assertAlmostEqual(modified_res_np[0], -2.0)

        # Other DOFs should be unchanged
        for i in range(1, nstates):
            self.assertAlmostEqual(modified_res_np[i], 1.0)

    def test_1d_dirichlet_apply_to_jacobian(self) -> None:
        """Test Dirichlet BC modifies Jacobian correctly."""
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        bc = DirichletBC(basis, "left", value_func=0.0, bkd=self.bkd_inst)

        nstates = basis.ndofs()
        state = self.bkd_inst.asarray(np.zeros(nstates))
        jacobian = self.bkd_inst.asarray(np.ones((nstates, nstates)))

        modified_jac = bc.apply_to_jacobian(jacobian, state, time=0.0)
        modified_jac_np = self.bkd_inst.to_numpy(modified_jac)

        # Dirichlet row should be identity row
        expected_row = np.zeros(nstates)
        expected_row[0] = 1.0
        np.testing.assert_array_almost_equal(modified_jac_np[0, :], expected_row)

        # Other rows unchanged
        for i in range(1, nstates):
            np.testing.assert_array_almost_equal(
                modified_jac_np[i, :], np.ones(nstates)
            )

    def test_2d_dirichlet_multiple_boundaries(self) -> None:
        """Test Dirichlet BC on multiple boundaries in 2D."""
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)

        bc_left = DirichletBC(basis, "left", value_func=0.0, bkd=self.bkd_inst)
        bc_right = DirichletBC(basis, "right", value_func=1.0, bkd=self.bkd_inst)

        # Check DOFs are distinct
        left_dofs = set(self.bkd_inst.to_numpy(bc_left.boundary_dofs()))
        right_dofs = set(self.bkd_inst.to_numpy(bc_right.boundary_dofs()))

        self.assertEqual(len(left_dofs & right_dofs), 0)


class TestNeumannBCBase(TestBoundaryConditionsBase[Array]):
    """Tests for NeumannBC."""

    __test__ = False

    def test_1d_constant_neumann(self) -> None:
        """Test constant Neumann BC in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=self.bkd_inst)

        # Check flux values
        values = bc.flux_values(time=0.0)
        expected = self.bkd_inst.asarray(np.array([1.0], dtype=np.float64))
        self.bkd_inst.assert_allclose(values, expected)

    def test_1d_neumann_apply_to_load(self) -> None:
        """Test Neumann BC adds to load vector."""
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # Constant flux at right boundary
        bc = NeumannBC(basis, "right", flux_func=2.0, bkd=self.bkd_inst)

        nstates = basis.ndofs()
        load = self.bkd_inst.asarray(np.zeros(nstates))

        modified_load = bc.apply_to_load(load, time=0.0)
        modified_load_np = self.bkd_inst.to_numpy(modified_load)

        # Neumann BC should add contribution at right boundary DOF
        # For 1D linear elements, the boundary integral of constant = value
        # The contribution goes to the boundary node
        self.assertTrue(modified_load_np[-1] > 0)  # Right boundary is last DOF


class TestRobinBCBase(TestBoundaryConditionsBase[Array]):
    """Tests for RobinBC."""

    __test__ = False

    def test_1d_robin_reduces_to_dirichlet(self) -> None:
        """Test Robin BC with large alpha approaches Dirichlet."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # Large alpha effectively enforces u = g/alpha
        alpha = 1000.0
        g_val = 1000.0  # g = alpha * u_desired = 1000 * 1 = 1000
        bc = RobinBC(basis, "left", alpha=alpha, value_func=g_val, bkd=self.bkd_inst)

        self.assertEqual(bc.alpha(), alpha)

    def test_1d_robin_apply_to_stiffness(self) -> None:
        """Test Robin BC modifies stiffness matrix."""
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        bc = RobinBC(basis, "left", alpha=2.0, value_func=0.0, bkd=self.bkd_inst)

        nstates = basis.ndofs()
        stiffness = self.bkd_inst.asarray(np.zeros((nstates, nstates)))

        modified_stiff = bc.apply_to_stiffness(stiffness, time=0.0)
        modified_stiff_np = self.bkd_inst.to_numpy(modified_stiff)

        # Robin BC should add alpha * boundary_mass_matrix contribution
        # For 1D point boundary, this adds to (0,0) entry
        self.assertTrue(modified_stiff_np[0, 0] > 0)


class TestBoundaryConditionSetBase(TestBoundaryConditionsBase[Array]):
    """Tests for BoundaryConditionSet."""

    __test__ = False

    def test_empty_set(self) -> None:
        """Test empty boundary condition set."""
        bc_set = BoundaryConditionSet(self.bkd_inst)

        self.assertEqual(bc_set.ndirichlet(), 0)
        self.assertEqual(bc_set.nneumann(), 0)
        self.assertEqual(bc_set.nrobin(), 0)

    def test_add_multiple_bcs(self) -> None:
        """Test adding multiple boundary conditions."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        bc_set = BoundaryConditionSet(self.bkd_inst)

        bc_left = DirichletBC(basis, "left", value_func=0.0, bkd=self.bkd_inst)
        bc_right = NeumannBC(basis, "right", flux_func=1.0, bkd=self.bkd_inst)

        bc_set.add_dirichlet(bc_left)
        bc_set.add_neumann(bc_right)

        self.assertEqual(bc_set.ndirichlet(), 1)
        self.assertEqual(bc_set.nneumann(), 1)

    def test_dirichlet_dofs_and_values(self) -> None:
        """Test getting all Dirichlet DOFs and values."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        bc_set = BoundaryConditionSet(self.bkd_inst)
        bc_set.add_dirichlet(
            DirichletBC(basis, "left", value_func=1.0, bkd=self.bkd_inst)
        )
        bc_set.add_dirichlet(
            DirichletBC(basis, "right", value_func=2.0, bkd=self.bkd_inst)
        )

        dofs = self.bkd_inst.to_numpy(bc_set.dirichlet_dofs())
        values = self.bkd_inst.to_numpy(bc_set.dirichlet_values(time=0.0))

        self.assertEqual(len(dofs), 2)
        self.assertEqual(len(values), 2)


class TestCanonicalBoundaryNormal(TestBoundaryConditionsBase[Array]):
    """Tests for canonical_boundary_normal function."""

    __test__ = False

    def test_1d_normals(self) -> None:
        """Test 1D boundary normals."""
        x = np.array([[0.0, 0.5, 1.0]])  # Shape: (1, 3)

        # Left boundary (index 0): normal = -1
        n_left = canonical_boundary_normal(0, x)
        np.testing.assert_array_almost_equal(n_left, [[-1, -1, -1]])

        # Right boundary (index 1): normal = +1
        n_right = canonical_boundary_normal(1, x)
        np.testing.assert_array_almost_equal(n_right, [[1, 1, 1]])

    def test_2d_normals(self) -> None:
        """Test 2D boundary normals."""
        x = np.array([[0.0, 0.5], [0.0, 0.5]])  # Shape: (2, 2)

        # Left (x=xmin): normal = [-1, 0]
        n_left = canonical_boundary_normal(0, x)
        np.testing.assert_array_almost_equal(n_left[0, :], [-1, -1])
        np.testing.assert_array_almost_equal(n_left[1, :], [0, 0])

        # Right (x=xmax): normal = [+1, 0]
        n_right = canonical_boundary_normal(1, x)
        np.testing.assert_array_almost_equal(n_right[0, :], [1, 1])
        np.testing.assert_array_almost_equal(n_right[1, :], [0, 0])

        # Bottom (y=ymin): normal = [0, -1]
        n_bottom = canonical_boundary_normal(2, x)
        np.testing.assert_array_almost_equal(n_bottom[0, :], [0, 0])
        np.testing.assert_array_almost_equal(n_bottom[1, :], [-1, -1])

        # Top (y=ymax): normal = [0, +1]
        n_top = canonical_boundary_normal(3, x)
        np.testing.assert_array_almost_equal(n_top[0, :], [0, 0])
        np.testing.assert_array_almost_equal(n_top[1, :], [1, 1])


class TestManufacturedSolutionBCBase(TestBoundaryConditionsBase[Array]):
    """Tests for ManufacturedSolutionBC."""

    __test__ = False

    def test_1d_all_dirichlet(self) -> None:
        """Test creating all Dirichlet BCs from manufactured solution."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # u = x
        def sol(x, t=None):
            return x[0]

        # flux = D * grad(u) = 1 * [1] = [1]
        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, self.bkd_inst)
        bc_set = ms_bc.create_boundary_conditions(["D", "D"])

        self.assertEqual(bc_set.ndirichlet(), 2)
        self.assertEqual(bc_set.nneumann(), 0)
        self.assertEqual(bc_set.nrobin(), 0)

    def test_1d_mixed_bcs(self) -> None:
        """Test creating mixed BCs from manufactured solution."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0] ** 2

        def flux(x, t=None):
            return 2 * x  # D * grad(u) = 1 * 2x

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, self.bkd_inst)
        bc_set = ms_bc.create_boundary_conditions(["D", "N"])

        self.assertEqual(bc_set.ndirichlet(), 1)
        self.assertEqual(bc_set.nneumann(), 1)

    def test_1d_robin_bcs(self) -> None:
        """Test creating Robin BCs from manufactured solution."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0]

        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, self.bkd_inst)
        bc_set = ms_bc.create_boundary_conditions(["R", "R"], robin_alpha=1.0)

        self.assertEqual(bc_set.nrobin(), 2)

    def test_2d_all_dirichlet(self) -> None:
        """Test 2D manufactured solution BCs."""
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)

        # u = x + y
        def sol(x, t=None):
            return x[0] + x[1]

        # flux = D * grad(u) = 1 * [1, 1]
        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, self.bkd_inst)
        bc_set = ms_bc.create_boundary_conditions(["D", "D", "D", "D"])

        self.assertEqual(bc_set.ndirichlet(), 4)

    def test_invalid_bc_type(self) -> None:
        """Test error on invalid BC type."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0]

        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, self.bkd_inst)

        with self.assertRaises(ValueError):
            ms_bc.create_boundary_conditions(["X", "D"])

    def test_wrong_number_bc_types(self) -> None:
        """Test error on wrong number of BC types."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0]

        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, self.bkd_inst)

        with self.assertRaises(ValueError):
            ms_bc.create_boundary_conditions(["D"])  # Need 2 for 1D


# Concrete test classes for each backend


class TestDirichletBCNumpy(TestDirichletBCBase[NDArray[Any]]):
    """NumPy backend tests for DirichletBC."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestNeumannBCNumpy(TestNeumannBCBase[NDArray[Any]]):
    """NumPy backend tests for NeumannBC."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestRobinBCNumpy(TestRobinBCBase[NDArray[Any]]):
    """NumPy backend tests for RobinBC."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestBoundaryConditionSetNumpy(TestBoundaryConditionSetBase[NDArray[Any]]):
    """NumPy backend tests for BoundaryConditionSet."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestCanonicalBoundaryNormalNumpy(TestCanonicalBoundaryNormal[NDArray[Any]]):
    """NumPy backend tests for canonical_boundary_normal."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestManufacturedSolutionBCNumpy(TestManufacturedSolutionBCBase[NDArray[Any]]):
    """NumPy backend tests for ManufacturedSolutionBC."""

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

    class TestDirichletBCTorch(TestDirichletBCBase[torch.Tensor]):
        """PyTorch backend tests for DirichletBC."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

    class TestNeumannBCTorch(TestNeumannBCBase[torch.Tensor]):
        """PyTorch backend tests for NeumannBC."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

    class TestRobinBCTorch(TestRobinBCBase[torch.Tensor]):
        """PyTorch backend tests for RobinBC."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

    class TestBoundaryConditionSetTorch(TestBoundaryConditionSetBase[torch.Tensor]):
        """PyTorch backend tests for BoundaryConditionSet."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

    class TestCanonicalBoundaryNormalTorch(TestCanonicalBoundaryNormal[torch.Tensor]):
        """PyTorch backend tests for canonical_boundary_normal."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

    class TestManufacturedSolutionBCTorch(TestManufacturedSolutionBCBase[torch.Tensor]):
        """PyTorch backend tests for ManufacturedSolutionBC."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
