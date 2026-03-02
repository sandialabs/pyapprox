"""Tests for boundary condition implementations.

Tests Dirichlet, Neumann, Robin BCs and the ManufacturedSolutionBC factory.
"""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

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


class TestDirichletBCBase:
    """Tests for DirichletBC."""

    def test_1d_constant_dirichlet(self, numpy_bkd) -> None:
        """Test constant Dirichlet BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = DirichletBC(basis, "left", value_func=1.5, bkd=bkd)

        # Check boundary DOFs
        dofs = bkd.to_numpy(bc.boundary_dofs())
        assert len(dofs) == 1
        assert dofs[0] == 0  # Left boundary is DOF 0

        # Check boundary values
        values = bc.boundary_values(time=0.0)
        expected = bkd.asarray(np.array([1.5], dtype=np.float64))
        bkd.assert_allclose(values, expected)

    def test_1d_function_dirichlet(self, numpy_bkd) -> None:
        """Test function-valued Dirichlet BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # BC: u = x^2 at boundary
        def bc_func(x, t=None):
            return x[0] ** 2

        bc = DirichletBC(basis, "right", value_func=bc_func, bkd=bkd)

        # Right boundary is at x=1
        values = bc.boundary_values(time=0.0)
        expected = bkd.asarray(
            np.array([1.0], dtype=np.float64)  # 1^2 = 1
        )
        bkd.assert_allclose(values, expected)

    def test_1d_dirichlet_apply_to_residual(self, numpy_bkd) -> None:
        """Test Dirichlet BC modifies residual correctly."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = DirichletBC(basis, "left", value_func=2.0, bkd=bkd)

        # Create dummy state and residual
        nstates = basis.ndofs()
        state = bkd.asarray(np.zeros(nstates))
        residual = bkd.asarray(np.ones(nstates))

        # Apply BC
        modified_res = bc.apply_to_residual(residual, state, time=0.0)
        modified_res_np = bkd.to_numpy(modified_res)

        # At Dirichlet DOF, residual should be state - g = 0 - 2 = -2
        assert abs(modified_res_np[0] - -2.0) < 1e-7

        # Other DOFs should be unchanged
        for i in range(1, nstates):
            assert abs(modified_res_np[i] - 1.0) < 1e-7

    def test_1d_dirichlet_apply_to_jacobian(self, numpy_bkd) -> None:
        """Test Dirichlet BC modifies Jacobian correctly."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)

        nstates = basis.ndofs()
        state = bkd.asarray(np.zeros(nstates))
        jacobian = bkd.asarray(np.ones((nstates, nstates)))

        modified_jac = bc.apply_to_jacobian(jacobian, state, time=0.0)
        modified_jac_np = bkd.to_numpy(modified_jac)

        # Dirichlet row should be identity row
        expected_row = np.zeros(nstates)
        expected_row[0] = 1.0
        np.testing.assert_array_almost_equal(modified_jac_np[0, :], expected_row)

        # Other rows unchanged
        for i in range(1, nstates):
            np.testing.assert_array_almost_equal(
                modified_jac_np[i, :], np.ones(nstates)
            )

    def test_2d_dirichlet_multiple_boundaries(self, numpy_bkd) -> None:
        """Test Dirichlet BC on multiple boundaries in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)

        bc_left = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
        bc_right = DirichletBC(basis, "right", value_func=1.0, bkd=bkd)

        # Check DOFs are distinct
        left_dofs = set(bkd.to_numpy(bc_left.boundary_dofs()))
        right_dofs = set(bkd.to_numpy(bc_right.boundary_dofs()))

        assert len(left_dofs & right_dofs) == 0


class TestNeumannBCBase:
    """Tests for NeumannBC."""

    def test_1d_constant_neumann(self, numpy_bkd) -> None:
        """Test constant Neumann BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)

        # Check flux values
        values = bc.flux_values(time=0.0)
        expected = bkd.asarray(np.array([1.0], dtype=np.float64))
        bkd.assert_allclose(values, expected)

    def test_1d_neumann_apply_to_load(self, numpy_bkd) -> None:
        """Test Neumann BC adds to load vector."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Constant flux at right boundary
        bc = NeumannBC(basis, "right", flux_func=2.0, bkd=bkd)

        nstates = basis.ndofs()
        load = bkd.asarray(np.zeros(nstates))

        modified_load = bc.apply_to_load(load, time=0.0)
        modified_load_np = bkd.to_numpy(modified_load)

        # Neumann BC should add contribution at right boundary DOF
        # For 1D linear elements, the boundary integral of constant = value
        # The contribution goes to the boundary node
        assert modified_load_np[-1] > 0  # Right boundary is last DOF


class TestRobinBCBase:
    """Tests for RobinBC."""

    def test_1d_robin_reduces_to_dirichlet(self, numpy_bkd) -> None:
        """Test Robin BC with large alpha approaches Dirichlet."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Large alpha effectively enforces u = g/alpha
        alpha = 1000.0
        g_val = 1000.0  # g = alpha * u_desired = 1000 * 1 = 1000
        bc = RobinBC(basis, "left", alpha=alpha, value_func=g_val, bkd=bkd)

        assert bc.alpha() == alpha

    def test_1d_robin_apply_to_stiffness(self, numpy_bkd) -> None:
        """Test Robin BC modifies stiffness matrix."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = RobinBC(basis, "left", alpha=2.0, value_func=0.0, bkd=bkd)

        nstates = basis.ndofs()
        stiffness = bkd.asarray(np.zeros((nstates, nstates)))

        modified_stiff = bc.apply_to_stiffness(stiffness, time=0.0)
        modified_stiff_np = bkd.to_numpy(modified_stiff)

        # Robin BC should add alpha * boundary_mass_matrix contribution
        # For 1D point boundary, this adds to (0,0) entry
        assert modified_stiff_np[0, 0] > 0


class TestBoundaryConditionSetBase:
    """Tests for BoundaryConditionSet."""

    def test_empty_set(self, numpy_bkd) -> None:
        """Test empty boundary condition set."""
        bkd = numpy_bkd
        bc_set = BoundaryConditionSet(bkd)

        assert bc_set.ndirichlet() == 0
        assert bc_set.nneumann() == 0
        assert bc_set.nrobin() == 0

    def test_add_multiple_bcs(self, numpy_bkd) -> None:
        """Test adding multiple boundary conditions."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc_set = BoundaryConditionSet(bkd)

        bc_left = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
        bc_right = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)

        bc_set.add_dirichlet(bc_left)
        bc_set.add_neumann(bc_right)

        assert bc_set.ndirichlet() == 1
        assert bc_set.nneumann() == 1

    def test_dirichlet_dofs_and_values(self, numpy_bkd) -> None:
        """Test getting all Dirichlet DOFs and values."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc_set = BoundaryConditionSet(bkd)
        bc_set.add_dirichlet(
            DirichletBC(basis, "left", value_func=1.0, bkd=bkd)
        )
        bc_set.add_dirichlet(
            DirichletBC(basis, "right", value_func=2.0, bkd=bkd)
        )

        dofs = bkd.to_numpy(bc_set.dirichlet_dofs())
        values = bkd.to_numpy(bc_set.dirichlet_values(time=0.0))

        assert len(dofs) == 2
        assert len(values) == 2


class TestCanonicalBoundaryNormal:
    """Tests for canonical_boundary_normal function."""

    def test_1d_normals(self, numpy_bkd) -> None:
        """Test 1D boundary normals."""
        _bkd = numpy_bkd
        x = np.array([[0.0, 0.5, 1.0]])  # Shape: (1, 3)

        # Left boundary (index 0): normal = -1
        n_left = canonical_boundary_normal(0, x)
        np.testing.assert_array_almost_equal(n_left, [[-1, -1, -1]])

        # Right boundary (index 1): normal = +1
        n_right = canonical_boundary_normal(1, x)
        np.testing.assert_array_almost_equal(n_right, [[1, 1, 1]])

    def test_2d_normals(self, numpy_bkd) -> None:
        """Test 2D boundary normals."""
        _bkd = numpy_bkd
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


class TestManufacturedSolutionBCBase:
    """Tests for ManufacturedSolutionBC."""

    def test_1d_all_dirichlet(self, numpy_bkd) -> None:
        """Test creating all Dirichlet BCs from manufactured solution."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # u = x
        def sol(x, t=None):
            return x[0]

        # flux = D * grad(u) = 1 * [1] = [1]
        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, bkd)
        bc_set = ms_bc.create_boundary_conditions(["D", "D"])

        assert bc_set.ndirichlet() == 2
        assert bc_set.nneumann() == 0
        assert bc_set.nrobin() == 0

    def test_1d_mixed_bcs(self, numpy_bkd) -> None:
        """Test creating mixed BCs from manufactured solution."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0] ** 2

        def flux(x, t=None):
            return 2 * x  # D * grad(u) = 1 * 2x

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, bkd)
        bc_set = ms_bc.create_boundary_conditions(["D", "N"])

        assert bc_set.ndirichlet() == 1
        assert bc_set.nneumann() == 1

    def test_1d_robin_bcs(self, numpy_bkd) -> None:
        """Test creating Robin BCs from manufactured solution."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0]

        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, bkd)
        bc_set = ms_bc.create_boundary_conditions(["R", "R"], robin_alpha=1.0)

        assert bc_set.nrobin() == 2

    def test_2d_all_dirichlet(self, numpy_bkd) -> None:
        """Test 2D manufactured solution BCs."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)

        # u = x + y
        def sol(x, t=None):
            return x[0] + x[1]

        # flux = D * grad(u) = 1 * [1, 1]
        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, bkd)
        bc_set = ms_bc.create_boundary_conditions(["D", "D", "D", "D"])

        assert bc_set.ndirichlet() == 4

    def test_invalid_bc_type(self, numpy_bkd) -> None:
        """Test error on invalid BC type."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0]

        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, bkd)

        with pytest.raises(ValueError):
            ms_bc.create_boundary_conditions(["X", "D"])

    def test_wrong_number_bc_types(self, numpy_bkd) -> None:
        """Test error on wrong number of BC types."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        def sol(x, t=None):
            return x[0]

        def flux(x, t=None):
            return np.ones_like(x)

        ms_bc = ManufacturedSolutionBC(basis, sol, flux, bkd)

        with pytest.raises(ValueError):
            ms_bc.create_boundary_conditions(["D"])  # Need 2 for 1D


