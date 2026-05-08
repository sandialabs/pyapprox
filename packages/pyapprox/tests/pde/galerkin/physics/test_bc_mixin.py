"""Tests for GalerkinBCMixin."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Generic

import numpy as np
from scipy.sparse import csr_matrix

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.util.backends.protocols import Array


class _ConcreteMixinUser(GalerkinBCMixin[Array], Generic[Array]):
    """Minimal class using GalerkinBCMixin for testing."""

    def __init__(self, bkd, boundary_conditions=None):
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []


def _make_basis(bkd, nx=10):
    """Create a simple 1D Lagrange basis for testing."""
    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    return LagrangeBasis(mesh, degree=1)


class TestGalerkinBCMixin:
    def _make_user(self, bkd, boundary_conditions=None) :
        return _ConcreteMixinUser(bkd, boundary_conditions)

    # --- _apply_bc_to_stiffness ---

    def test_apply_bc_to_stiffness_no_bcs(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        user = self._make_user(bkd)
        n = 5
        K = csr_matrix(np.eye(n))
        result = user._apply_bc_to_stiffness(K, 0.0)
        # Should be unchanged
        bkd.assert_allclose(
            bkd.asarray(result.toarray()),
            bkd.asarray(np.eye(n)),
        )

    def test_apply_bc_to_stiffness_robin(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="right",
            alpha=2.0,
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin])
        n = basis.ndofs()
        K = csr_matrix(np.zeros((n, n)))
        result = user._apply_bc_to_stiffness(K, 0.0)
        # Robin should have added boundary mass terms
        result_dense = result.toarray() if hasattr(result, "toarray") else result
        assert not np.allclose(result_dense, 0.0), "Robin BC should modify stiffness"

    def test_apply_bc_to_stiffness_skips_dirichlet(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [dirichlet])
        n = basis.ndofs()
        K = csr_matrix(np.eye(n))
        result = user._apply_bc_to_stiffness(K, 0.0)
        # Dirichlet should NOT modify stiffness
        bkd.assert_allclose(
            bkd.asarray(result.toarray()),
            bkd.asarray(np.eye(n)),
        )

    # --- _apply_bc_to_load ---

    def test_apply_bc_to_load_neumann(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        neumann = NeumannBC(
            basis=basis,
            boundary_name="right",
            flux_func=1.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [neumann])
        n = basis.ndofs()
        load = np.zeros(n)
        result = user._apply_bc_to_load(load, 0.0)
        # Neumann should add flux contribution
        assert not np.allclose(result, 0.0), "Neumann BC should modify load"

    def test_apply_bc_to_load_robin(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="right",
            alpha=2.0,
            value_func=1.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin])
        n = basis.ndofs()
        load = np.zeros(n)
        result = user._apply_bc_to_load(load, 0.0)
        assert not np.allclose(result, 0.0), "Robin BC should modify load"

    # --- dirichlet_dof_info ---

    def test_dirichlet_dof_info_no_bcs(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        user = self._make_user(bkd)
        dofs, vals = user.dirichlet_dof_info(0.0)
        assert len(bkd.to_numpy(dofs)) == 0
        assert len(bkd.to_numpy(vals)) == 0

    def test_dirichlet_dof_info_with_dirichlet(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=5.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [dirichlet])
        dofs, vals = user.dirichlet_dof_info(0.0)
        dofs_np = bkd.to_numpy(dofs)
        vals_np = bkd.to_numpy(vals)
        assert len(dofs_np) > 0
        # Left boundary of 1D mesh is DOF 0
        assert 0 in dofs_np
        bkd.assert_allclose(
            bkd.asarray(vals_np),
            bkd.asarray(np.full_like(vals_np, 5.0)),
        )

    def test_dirichlet_dof_info_skips_robin(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=5.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin])
        dofs, vals = user.dirichlet_dof_info(0.0)
        # Robin should be skipped — no Dirichlet DOFs
        assert len(bkd.to_numpy(dofs)) == 0

    def test_dirichlet_dof_info_robin_then_dirichlet(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=0.0,
            bkd=bkd,
        )
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="right",
            value_func=1.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin, dirichlet])
        dofs, vals = user.dirichlet_dof_info(0.0)
        dofs_np = bkd.to_numpy(dofs)
        # Only right boundary DOF, not left (Robin)
        n = basis.ndofs()
        assert n - 1 in dofs_np
        assert 0 not in dofs_np

    # --- _apply_dirichlet_to_residual ---

    def test_apply_dirichlet_to_residual(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [dirichlet])
        n = basis.ndofs()
        residual = np.ones(n)
        state = np.zeros(n)
        result = user._apply_dirichlet_to_residual(residual, state, 0.0)
        result_np = bkd.to_numpy(bkd.asarray(result))
        # Dirichlet DOF (index 0) should have residual = state - g = 0 - 0 = 0
        bkd.assert_allclose(
            bkd.asarray([result_np[0]]),
            bkd.asarray([0.0]),
        )
        # Interior DOFs should remain 1.0
        bkd.assert_allclose(
            bkd.asarray(result_np[1:]),
            bkd.asarray(np.ones(n - 1)),
        )

    def test_apply_dirichlet_to_residual_skips_robin(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin])
        n = basis.ndofs()
        residual = np.ones(n)
        state = np.zeros(n)
        result = user._apply_dirichlet_to_residual(residual, state, 0.0)
        result_np = bkd.to_numpy(bkd.asarray(result))
        # Robin should be skipped — residual unchanged
        bkd.assert_allclose(
            bkd.asarray(result_np),
            bkd.asarray(np.ones(n)),
        )

    # --- _apply_dirichlet_to_jacobian ---

    def test_apply_dirichlet_to_jacobian(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [dirichlet])
        n = basis.ndofs()
        jacobian = csr_matrix(2.0 * np.eye(n))
        state = np.zeros(n)
        result = user._apply_dirichlet_to_jacobian(jacobian, state, 0.0)
        result_dense = result.toarray() if hasattr(result, "toarray") else result
        # Dirichlet DOF row should be identity (1 on diagonal, 0 elsewhere)
        bkd.assert_allclose(
            bkd.asarray([result_dense[0, 0]]),
            bkd.asarray([1.0]),
        )
        bkd.assert_allclose(
            bkd.asarray(result_dense[0, 1:]),
            bkd.asarray(np.zeros(n - 1)),
        )
        # Interior rows should be unchanged
        bkd.assert_allclose(
            bkd.asarray(np.diag(result_dense)[1:]),
            bkd.asarray(2.0 * np.ones(n - 1)),
        )

    # --- apply_boundary_conditions ---

    def test_apply_boundary_conditions_robin_then_dirichlet(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="right",
            alpha=1.0,
            value_func=0.0,
            bkd=bkd,
        )
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin, dirichlet])
        n = basis.ndofs()
        residual = np.ones(n)
        jacobian = csr_matrix(np.eye(n))
        state = np.zeros(n)
        res, jac = user.apply_boundary_conditions(residual, jacobian, state, 0.0)
        # Robin should have modified right boundary
        # Dirichlet should have replaced left boundary row
        res_np = bkd.to_numpy(bkd.asarray(res))
        bkd.assert_allclose(
            bkd.asarray([res_np[0]]),
            bkd.asarray([0.0]),
        )

    def test_apply_boundary_conditions_none_residual(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [dirichlet])
        n = basis.ndofs()
        jacobian = csr_matrix(2.0 * np.eye(n))
        state = np.zeros(n)
        res, jac = user.apply_boundary_conditions(None, jacobian, state, 0.0)
        assert res is None
        assert jac is not None

    def test_apply_boundary_conditions_none_jacobian(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [dirichlet])
        n = basis.ndofs()
        residual = np.ones(n)
        state = np.zeros(n)
        res, jac = user.apply_boundary_conditions(residual, None, state, 0.0)
        assert res is not None
        assert jac is None
