"""Tests for manufactured solution adapter for Galerkin tests.

Tests the integration of collocation manufactured solutions with
Galerkin finite element boundary conditions and physics.
"""


import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.manufactured import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
    create_helmholtz_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
class TestADRManufacturedBase:
    """Tests for ADR manufactured solution integration."""

    def test_create_adr_1d_linear(self, numpy_bkd) -> None:
        """Test creating 1D ADR manufactured solution."""
        bkd = numpy_bkd
        bounds = [0.0, 1.0]
        functions, nvars = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        assert nvars == 1
        assert "solution" in functions
        assert "forcing" in functions
        assert "flux" in functions

    def test_create_adr_2d_quadratic(self, numpy_bkd) -> None:
        """Test creating 2D ADR manufactured solution."""
        bkd = numpy_bkd
        bounds = [0.0, 1.0, 0.0, 1.0]
        functions, nvars = create_adr_manufactured_test(
            bounds=bounds,
            sol_str="x**2 + y**2",
            diff_str="1.0",
            react_str="u",
            vel_strs=["0", "0"],
            bkd=bkd,
        )

        assert nvars == 2
        assert "solution" in functions
        assert "forcing" in functions

    def test_adapter_creates_dirichlet_bc(self, numpy_bkd) -> None:
        """Test adapter creates Dirichlet boundary conditions."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        bc_set = adapter.create_boundary_conditions(["D", "D"])

        assert bc_set.ndirichlet() == 2
        assert bc_set.nneumann() == 0
        assert bc_set.nrobin() == 0

    def test_adapter_creates_mixed_bcs(self, numpy_bkd) -> None:
        """Test adapter creates mixed boundary conditions."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        bc_set = adapter.create_boundary_conditions(["D", "N"])

        assert bc_set.ndirichlet() == 1
        assert bc_set.nneumann() == 1

    def test_adapter_creates_robin_bc(self, numpy_bkd) -> None:
        """Test adapter creates Robin boundary conditions."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        bc_set = adapter.create_boundary_conditions(["R", "R"], robin_alpha=1.0)

        assert bc_set.nrobin() == 2

    def test_adapter_forcing_for_galerkin(self, numpy_bkd) -> None:
        """Test adapter provides correctly shaped forcing function."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x**2",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        forcing = adapter.forcing_for_galerkin()

        # Test that forcing returns 1D array
        x = np.array([[0.0, 0.5, 1.0]])
        f_vals = forcing(x)

        assert f_vals.ndim == 1
        assert len(f_vals) == 3

    def test_adapter_solution_function(self, numpy_bkd) -> None:
        """Test adapter provides solution function."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        sol = adapter.solution_function()

        # Test solution at x=0.5
        x = np.array([[0.5]])
        sol_val = sol(x)

        # u = x, so u(0.5) = 0.5
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(sol_val.flatten(), expected)

    def test_adapter_2d_creates_4_bcs(self, numpy_bkd) -> None:
        """Test adapter creates 4 boundary conditions in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_str="x + y",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0", "0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        bc_set = adapter.create_boundary_conditions(["D", "D", "D", "D"])

        assert bc_set.ndirichlet() == 4

    def test_adapter_invalid_bc_type_raises(self, numpy_bkd) -> None:
        """Test adapter raises on invalid BC type."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        with pytest.raises(ValueError):
            adapter.create_boundary_conditions(["X", "D"])

    def test_adapter_wrong_bc_count_raises(self, numpy_bkd) -> None:
        """Test adapter raises on wrong number of BC types."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        with pytest.raises(ValueError):
            adapter.create_boundary_conditions(["D"])  # Need 2 for 1D


class TestHelmholtzManufacturedBase:
    """Tests for Helmholtz manufactured solution integration."""

    def test_create_helmholtz_1d(self, numpy_bkd) -> None:
        """Test creating 1D Helmholtz manufactured solution."""
        bkd = numpy_bkd
        bounds = [0.0, 1.0]
        functions, nvars = create_helmholtz_manufactured_test(
            bounds=bounds,
            sol_str="x",
            sqwavenum_str="4+1e-16*x",
            bkd=bkd,
        )

        assert nvars == 1
        assert "solution" in functions
        assert "forcing" in functions
        assert "sqwavenum" in functions

    def test_create_helmholtz_2d(self, numpy_bkd) -> None:
        """Test creating 2D Helmholtz manufactured solution."""
        bkd = numpy_bkd
        bounds = [0.0, 1.0, 0.0, 1.0]
        functions, nvars = create_helmholtz_manufactured_test(
            bounds=bounds,
            sol_str="x**2*y**2",
            sqwavenum_str="1+1e-16*x",
            bkd=bkd,
        )

        assert nvars == 2
        assert "solution" in functions
        assert "forcing" in functions

    def test_helmholtz_adapter_creates_bcs(self, numpy_bkd) -> None:
        """Test Helmholtz adapter creates boundary conditions."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        functions, _ = create_helmholtz_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            sqwavenum_str="4+1e-16*x",
            bkd=bkd,
        )

        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)

        bc_set = adapter.create_boundary_conditions(["D", "D"])

        assert bc_set.ndirichlet() == 2


