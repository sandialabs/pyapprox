"""Tests for boundary condition implementations.

Tests Dirichlet, Neumann and Robin BCs and the BoundaryConditionSet.
"""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.boundary import (
    EssentialBCProtocol,
    WeakFormBCProtocol,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import (
    BoundaryConditionSet,
    CallableDirichletBC,
    DirectDirichletBC,
    DirichletBC,
    NeumannBC,
    NormalDependentDataProtocol,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.util.backends.numpy import NumpyBkd


class _NormalX:
    """Time-independent data equal to the normal's first component."""

    def evaluate(
        self,
        coords: NDArray[np.floating[Any]],
        normal: NDArray[np.floating[Any]],
        time: float,
    ) -> NDArray[np.floating[Any]]:
        return np.asarray(normal[0])

    def is_time_dependent(self) -> bool:
        return False


class TestDirichletBCBase:
    """Tests for DirichletBC."""

    def test_1d_constant_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
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

    def test_1d_function_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
        """Test function-valued Dirichlet BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # BC: u = x^2 at boundary
        def bc_func(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            return np.asarray(x[0] ** 2)

        bc = DirichletBC(basis, "right", value_func=bc_func, bkd=bkd)

        # Right boundary is at x=1
        values = bc.boundary_values(time=0.0)
        expected = bkd.asarray(
            np.array([1.0], dtype=np.float64)  # 1^2 = 1
        )
        bkd.assert_allclose(values, expected)

    def test_2d_dirichlet_multiple_boundaries(self, numpy_bkd: NumpyBkd) -> None:
        """Test Dirichlet BC on multiple boundaries in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
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

    def test_1d_constant_neumann(self, numpy_bkd: NumpyBkd) -> None:
        """Test constant Neumann BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)

        # A point boundary integrates g * phi to g at its node.
        load = bc.apply_to_load(bkd.zeros((basis.ndofs(),)), time=0.0)
        expected = np.zeros(basis.ndofs())
        expected[-1] = 1.0
        bkd.assert_allclose(load, bkd.asarray(expected))

    def test_normal_dependent_data_gets_outward_normal(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Data g(x, n, t) = n_x is -1 at the left end and +1 at the right."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        assert isinstance(_NormalX(), NormalDependentDataProtocol)
        zeros = bkd.zeros((basis.ndofs(),))
        left = NeumannBC(basis, "left", flux_func=_NormalX(), bkd=bkd)
        right = NeumannBC(basis, "right", flux_func=_NormalX(), bkd=bkd)
        bkd.assert_allclose(
            left.apply_to_load(zeros, 0.0)[0:1], bkd.asarray([-1.0])
        )
        bkd.assert_allclose(
            right.apply_to_load(zeros, 0.0)[-1:], bkd.asarray([1.0])
        )
        assert not left.is_time_dependent()

    def test_1d_neumann_apply_to_load(self, numpy_bkd: NumpyBkd) -> None:
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

    def test_1d_robin_reduces_to_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
        """Test Robin BC with large alpha approaches Dirichlet."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Large alpha effectively enforces u = g/alpha
        alpha = 1000.0
        g_val = 1000.0  # g = alpha * u_desired = 1000 * 1 = 1000
        bc = RobinBC(basis, "left", alpha=alpha, value_func=g_val, bkd=bkd)

        assert bc.alpha() == alpha

    def test_1d_robin_apply_to_stiffness(self, numpy_bkd: NumpyBkd) -> None:
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

    def test_empty_set(self, numpy_bkd: NumpyBkd) -> None:
        """Test empty boundary condition set."""
        bkd = numpy_bkd
        bc_set = BoundaryConditionSet(bkd)

        assert bc_set.ndirichlet() == 0
        assert bc_set.nneumann() == 0
        assert bc_set.nrobin() == 0

    def test_add_multiple_bcs(self, numpy_bkd: NumpyBkd) -> None:
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

    def test_dirichlet_dofs_and_values(self, numpy_bkd: NumpyBkd) -> None:
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


class TestRoleProtocols:
    """Concrete BC classes satisfy exactly one of the two disjoint roles."""

    def test_dirichlet_bcs_are_essential_only(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bcs = [
            DirichletBC(basis, "left", value_func=1.0, bkd=bkd),
            DirectDirichletBC([0], [1.0], bkd),
            CallableDirichletBC([0], lambda t: np.array([t]), bkd),
        ]
        for bc in bcs:
            assert isinstance(bc, EssentialBCProtocol), bc
            assert not isinstance(bc, WeakFormBCProtocol), bc

    def test_neumann_robin_are_weak_form_only(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bcs = [
            NeumannBC(basis, "right", flux_func=1.0, bkd=bkd),
            RobinBC(basis, "right", alpha=2.0, value_func=1.0, bkd=bkd),
        ]
        for bc in bcs:
            assert isinstance(bc, WeakFormBCProtocol), bc
            assert not isinstance(bc, EssentialBCProtocol), bc

    def test_constrained_accessors_match_legacy(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc = DirichletBC(basis, "left", value_func=3.0, bkd=bkd)
        bkd.assert_allclose(
            bkd.asarray(bc.constrained_dofs(), dtype=bkd.double_dtype()),
            bkd.asarray(bc.boundary_dofs(), dtype=bkd.double_dtype()),
        )
        bkd.assert_allclose(
            bc.constrained_values(0.0), bc.boundary_values(0.0)
        )

    def test_neumann_stiffness_and_jacobian_are_identity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)
        n = basis.ndofs()
        K = bkd.asarray(np.eye(n))
        state = bkd.asarray(np.zeros(n))
        assert bc.apply_to_stiffness(K, 0.0) is K
        assert bc.apply_to_jacobian(K, state, 0.0) is K

    def test_neumann_residual_adds_load(self, numpy_bkd: NumpyBkd) -> None:
        """The term enters F of M du/dt = F with a plus sign."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)
        n = basis.ndofs()
        state = bkd.asarray(np.zeros(n))
        res = bc.apply_to_residual(bkd.asarray(np.zeros(n)), state, 0.0)
        load = bc.apply_to_load(bkd.asarray(np.zeros(n)), 0.0)
        bkd.assert_allclose(res, load)

    def test_bc_set_role_accessors(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc_set = BoundaryConditionSet(bkd)
        dirichlet = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
        neumann = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)
        robin = RobinBC(basis, "right", alpha=1.0, value_func=0.0, bkd=bkd)
        bc_set.add_dirichlet(dirichlet)
        bc_set.add_neumann(neumann)
        bc_set.add_robin(robin)
        assert bc_set.essential_bcs() == [dirichlet]
        assert bc_set.weak_form_bcs() == [neumann, robin]
