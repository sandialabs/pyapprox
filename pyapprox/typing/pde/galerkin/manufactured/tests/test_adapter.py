"""Tests for manufactured solution adapter for Galerkin tests.

Tests the integration of collocation manufactured solutions with
Galerkin finite element boundary conditions and physics.
"""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.manufactured import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
    create_helmholtz_manufactured_test,
)

from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestGalerkinManufacturedAdapterBase(Generic[Array], unittest.TestCase):
    """Base test class for GalerkinManufacturedSolutionAdapter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()


class TestADRManufacturedBase(TestGalerkinManufacturedAdapterBase[Array]):
    """Tests for ADR manufactured solution integration."""

    __test__ = False

    def test_create_adr_1d_linear(self) -> None:
        """Test creating 1D ADR manufactured solution."""
        bounds = [0.0, 1.0]
        functions, nvars = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        self.assertEqual(nvars, 1)
        self.assertIn("solution", functions)
        self.assertIn("forcing", functions)
        self.assertIn("flux", functions)

    def test_create_adr_2d_quadratic(self) -> None:
        """Test creating 2D ADR manufactured solution."""
        bounds = [0.0, 1.0, 0.0, 1.0]
        functions, nvars = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="x**2 + y**2",
            diff_str="1.0",
            react_str="u",
            vel_strs=["0", "0"],
            bkd=self.bkd_inst,
        )

        self.assertEqual(nvars, 2)
        self.assertIn("solution", functions)
        self.assertIn("forcing", functions)

    def test_adapter_creates_dirichlet_bc(self) -> None:
        """Test adapter creates Dirichlet boundary conditions."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        bc_set = adapter.create_boundary_conditions(["D", "D"])

        self.assertEqual(bc_set.ndirichlet(), 2)
        self.assertEqual(bc_set.nneumann(), 0)
        self.assertEqual(bc_set.nrobin(), 0)

    def test_adapter_creates_mixed_bcs(self) -> None:
        """Test adapter creates mixed boundary conditions."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        bc_set = adapter.create_boundary_conditions(["D", "N"])

        self.assertEqual(bc_set.ndirichlet(), 1)
        self.assertEqual(bc_set.nneumann(), 1)

    def test_adapter_creates_robin_bc(self) -> None:
        """Test adapter creates Robin boundary conditions."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        bc_set = adapter.create_boundary_conditions(["R", "R"], robin_alpha=1.0)

        self.assertEqual(bc_set.nrobin(), 2)

    def test_adapter_forcing_for_galerkin(self) -> None:
        """Test adapter provides correctly shaped forcing function."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x**2",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        forcing = adapter.forcing_for_galerkin()

        # Test that forcing returns 1D array
        x = np.array([[0.0, 0.5, 1.0]])
        f_vals = forcing(x)

        self.assertEqual(f_vals.ndim, 1)
        self.assertEqual(len(f_vals), 3)

    def test_adapter_solution_function(self) -> None:
        """Test adapter provides solution function."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        sol = adapter.solution_function()

        # Test solution at x=0.5
        x = np.array([[0.5]])
        sol_val = sol(x)

        # u = x, so u(0.5) = 0.5
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(sol_val.flatten(), expected)

    def test_adapter_2d_creates_4_bcs(self) -> None:
        """Test adapter creates 4 boundary conditions in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_str="x + y",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0", "0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        bc_set = adapter.create_boundary_conditions(["D", "D", "D", "D"])

        self.assertEqual(bc_set.ndirichlet(), 4)

    def test_adapter_invalid_bc_type_raises(self) -> None:
        """Test adapter raises on invalid BC type."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        with self.assertRaises(ValueError):
            adapter.create_boundary_conditions(["X", "D"])

    def test_adapter_wrong_bc_count_raises(self) -> None:
        """Test adapter raises on wrong number of BC types."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        with self.assertRaises(ValueError):
            adapter.create_boundary_conditions(["D"])  # Need 2 for 1D


class TestHelmholtzManufacturedBase(TestGalerkinManufacturedAdapterBase[Array]):
    """Tests for Helmholtz manufactured solution integration."""

    __test__ = False

    def test_create_helmholtz_1d(self) -> None:
        """Test creating 1D Helmholtz manufactured solution."""
        bounds = [0.0, 1.0]
        functions, nvars = create_helmholtz_manufactured_test(
            bounds=bounds,
            sol_str="x",
            sqwavenum_str="4+1e-16*x",
            bkd=self.bkd_inst,
        )

        self.assertEqual(nvars, 1)
        self.assertIn("solution", functions)
        self.assertIn("forcing", functions)
        self.assertIn("sqwavenum", functions)

    def test_create_helmholtz_2d(self) -> None:
        """Test creating 2D Helmholtz manufactured solution."""
        bounds = [0.0, 1.0, 0.0, 1.0]
        functions, nvars = create_helmholtz_manufactured_test(
            bounds=bounds,
            sol_str="x**2*y**2",
            sqwavenum_str="1+1e-16*x",
            bkd=self.bkd_inst,
        )

        self.assertEqual(nvars, 2)
        self.assertIn("solution", functions)
        self.assertIn("forcing", functions)

    def test_helmholtz_adapter_creates_bcs(self) -> None:
        """Test Helmholtz adapter creates boundary conditions."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_helmholtz_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            sqwavenum_str="4+1e-16*x",
            bkd=self.bkd_inst,
        )

        adapter = GalerkinManufacturedSolutionAdapter(
            basis, functions, self.bkd_inst
        )

        bc_set = adapter.create_boundary_conditions(["D", "D"])

        self.assertEqual(bc_set.ndirichlet(), 2)


# Concrete test classes for each backend

class TestADRManufacturedNumpy(TestADRManufacturedBase[NDArray[Any]]):
    """NumPy backend tests for ADR manufactured solutions."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestHelmholtzManufacturedNumpy(TestHelmholtzManufacturedBase[NDArray[Any]]):
    """NumPy backend tests for Helmholtz manufactured solutions."""

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

    class TestADRManufacturedTorch(TestADRManufacturedBase[torch.Tensor]):
        """PyTorch backend tests for ADR manufactured solutions."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

    class TestHelmholtzManufacturedTorch(
        TestHelmholtzManufacturedBase[torch.Tensor]
    ):
        """PyTorch backend tests for Helmholtz manufactured solutions."""

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
