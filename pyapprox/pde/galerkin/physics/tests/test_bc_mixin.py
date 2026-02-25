"""Tests for GalerkinBCMixin."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class _ConcreteMixinUser(GalerkinBCMixin[Array], Generic[Array]):
    """Minimal class using GalerkinBCMixin for testing."""

    def __init__(self, bkd, boundary_conditions=None):
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []


def _make_basis(bkd, nx=10):
    """Create a simple 1D Lagrange basis for testing."""
    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    return LagrangeBasis(mesh, degree=1)


class TestGalerkinBCMixin(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_user(self, boundary_conditions=None):
        return _ConcreteMixinUser(self._bkd, boundary_conditions)

    # --- _apply_bc_to_stiffness ---

    def test_apply_bc_to_stiffness_no_bcs(self) -> None:
        user = self._make_user()
        n = 5
        K = csr_matrix(np.eye(n))
        result = user._apply_bc_to_stiffness(K, 0.0)
        # Should be unchanged
        self._bkd.assert_allclose(
            self._bkd.asarray(result.toarray()),
            self._bkd.asarray(np.eye(n)),
        )

    def test_apply_bc_to_stiffness_robin(self) -> None:
        basis = _make_basis(self._bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="right",
            alpha=2.0,
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([robin])
        n = basis.ndofs()
        K = csr_matrix(np.zeros((n, n)))
        result = user._apply_bc_to_stiffness(K, 0.0)
        # Robin should have added boundary mass terms
        result_dense = result.toarray() if hasattr(result, "toarray") else result
        self.assertFalse(
            np.allclose(result_dense, 0.0),
            "Robin BC should modify stiffness",
        )

    def test_apply_bc_to_stiffness_skips_dirichlet(self) -> None:
        basis = _make_basis(self._bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([dirichlet])
        n = basis.ndofs()
        K = csr_matrix(np.eye(n))
        result = user._apply_bc_to_stiffness(K, 0.0)
        # Dirichlet should NOT modify stiffness
        self._bkd.assert_allclose(
            self._bkd.asarray(result.toarray()),
            self._bkd.asarray(np.eye(n)),
        )

    # --- _apply_bc_to_load ---

    def test_apply_bc_to_load_neumann(self) -> None:
        basis = _make_basis(self._bkd)
        neumann = NeumannBC(
            basis=basis,
            boundary_name="right",
            flux_func=1.0,
            bkd=self._bkd,
        )
        user = self._make_user([neumann])
        n = basis.ndofs()
        load = np.zeros(n)
        result = user._apply_bc_to_load(load, 0.0)
        # Neumann should add flux contribution
        self.assertFalse(
            np.allclose(result, 0.0),
            "Neumann BC should modify load",
        )

    def test_apply_bc_to_load_robin(self) -> None:
        basis = _make_basis(self._bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="right",
            alpha=2.0,
            value_func=1.0,
            bkd=self._bkd,
        )
        user = self._make_user([robin])
        n = basis.ndofs()
        load = np.zeros(n)
        result = user._apply_bc_to_load(load, 0.0)
        self.assertFalse(
            np.allclose(result, 0.0),
            "Robin BC should modify load",
        )

    # --- dirichlet_dof_info ---

    def test_dirichlet_dof_info_no_bcs(self) -> None:
        user = self._make_user()
        dofs, vals = user.dirichlet_dof_info(0.0)
        self.assertEqual(len(self._bkd.to_numpy(dofs)), 0)
        self.assertEqual(len(self._bkd.to_numpy(vals)), 0)

    def test_dirichlet_dof_info_with_dirichlet(self) -> None:
        basis = _make_basis(self._bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=5.0,
            bkd=self._bkd,
        )
        user = self._make_user([dirichlet])
        dofs, vals = user.dirichlet_dof_info(0.0)
        dofs_np = self._bkd.to_numpy(dofs)
        vals_np = self._bkd.to_numpy(vals)
        self.assertGreater(len(dofs_np), 0)
        # Left boundary of 1D mesh is DOF 0
        self.assertIn(0, dofs_np)
        self._bkd.assert_allclose(
            self._bkd.asarray(vals_np),
            self._bkd.asarray(np.full_like(vals_np, 5.0)),
        )

    def test_dirichlet_dof_info_skips_robin(self) -> None:
        basis = _make_basis(self._bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=5.0,
            bkd=self._bkd,
        )
        user = self._make_user([robin])
        dofs, vals = user.dirichlet_dof_info(0.0)
        # Robin should be skipped — no Dirichlet DOFs
        self.assertEqual(len(self._bkd.to_numpy(dofs)), 0)

    def test_dirichlet_dof_info_robin_then_dirichlet(self) -> None:
        basis = _make_basis(self._bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=0.0,
            bkd=self._bkd,
        )
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="right",
            value_func=1.0,
            bkd=self._bkd,
        )
        user = self._make_user([robin, dirichlet])
        dofs, vals = user.dirichlet_dof_info(0.0)
        dofs_np = self._bkd.to_numpy(dofs)
        # Only right boundary DOF, not left (Robin)
        n = basis.ndofs()
        self.assertIn(n - 1, dofs_np)
        self.assertNotIn(0, dofs_np)

    # --- _apply_dirichlet_to_residual ---

    def test_apply_dirichlet_to_residual(self) -> None:
        basis = _make_basis(self._bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([dirichlet])
        n = basis.ndofs()
        residual = np.ones(n)
        state = np.zeros(n)
        result = user._apply_dirichlet_to_residual(residual, state, 0.0)
        result_np = self._bkd.to_numpy(self._bkd.asarray(result))
        # Dirichlet DOF (index 0) should have residual = state - g = 0 - 0 = 0
        self._bkd.assert_allclose(
            self._bkd.asarray([result_np[0]]),
            self._bkd.asarray([0.0]),
        )
        # Interior DOFs should remain 1.0
        self._bkd.assert_allclose(
            self._bkd.asarray(result_np[1:]),
            self._bkd.asarray(np.ones(n - 1)),
        )

    def test_apply_dirichlet_to_residual_skips_robin(self) -> None:
        basis = _make_basis(self._bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([robin])
        n = basis.ndofs()
        residual = np.ones(n)
        state = np.zeros(n)
        result = user._apply_dirichlet_to_residual(residual, state, 0.0)
        result_np = self._bkd.to_numpy(self._bkd.asarray(result))
        # Robin should be skipped — residual unchanged
        self._bkd.assert_allclose(
            self._bkd.asarray(result_np),
            self._bkd.asarray(np.ones(n)),
        )

    # --- _apply_dirichlet_to_jacobian ---

    def test_apply_dirichlet_to_jacobian(self) -> None:
        basis = _make_basis(self._bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([dirichlet])
        n = basis.ndofs()
        jacobian = csr_matrix(2.0 * np.eye(n))
        state = np.zeros(n)
        result = user._apply_dirichlet_to_jacobian(jacobian, state, 0.0)
        result_dense = result.toarray() if hasattr(result, "toarray") else result
        # Dirichlet DOF row should be identity (1 on diagonal, 0 elsewhere)
        self._bkd.assert_allclose(
            self._bkd.asarray([result_dense[0, 0]]),
            self._bkd.asarray([1.0]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray(result_dense[0, 1:]),
            self._bkd.asarray(np.zeros(n - 1)),
        )
        # Interior rows should be unchanged
        self._bkd.assert_allclose(
            self._bkd.asarray(np.diag(result_dense)[1:]),
            self._bkd.asarray(2.0 * np.ones(n - 1)),
        )

    # --- apply_boundary_conditions ---

    def test_apply_boundary_conditions_robin_then_dirichlet(self) -> None:
        basis = _make_basis(self._bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="right",
            alpha=1.0,
            value_func=0.0,
            bkd=self._bkd,
        )
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([robin, dirichlet])
        n = basis.ndofs()
        residual = np.ones(n)
        jacobian = csr_matrix(np.eye(n))
        state = np.zeros(n)
        res, jac = user.apply_boundary_conditions(residual, jacobian, state, 0.0)
        # Robin should have modified right boundary
        # Dirichlet should have replaced left boundary row
        res_np = self._bkd.to_numpy(self._bkd.asarray(res))
        self._bkd.assert_allclose(
            self._bkd.asarray([res_np[0]]),
            self._bkd.asarray([0.0]),
        )

    def test_apply_boundary_conditions_none_residual(self) -> None:
        basis = _make_basis(self._bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([dirichlet])
        n = basis.ndofs()
        jacobian = csr_matrix(2.0 * np.eye(n))
        state = np.zeros(n)
        res, jac = user.apply_boundary_conditions(None, jacobian, state, 0.0)
        self.assertIsNone(res)
        self.assertIsNotNone(jac)

    def test_apply_boundary_conditions_none_jacobian(self) -> None:
        basis = _make_basis(self._bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=0.0,
            bkd=self._bkd,
        )
        user = self._make_user([dirichlet])
        n = basis.ndofs()
        residual = np.ones(n)
        state = np.zeros(n)
        res, jac = user.apply_boundary_conditions(residual, None, state, 0.0)
        self.assertIsNotNone(res)
        self.assertIsNone(jac)


class TestGalerkinBCMixinNumpy(TestGalerkinBCMixin[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


from pyapprox.util.test_utils import load_tests  # noqa: F401

if __name__ == "__main__":
    unittest.main()
