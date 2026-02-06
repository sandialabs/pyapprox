"""Tests for boundary conditions."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
)
from pyapprox.typing.pde.collocation.boundary import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    PeriodicBC,
    constant_dirichlet_bc,
    zero_dirichlet_bc,
    zero_neumann_bc,
    homogeneous_robin_bc,
    gradient_robin_bc,
    gradient_neumann_bc,
)
from pyapprox.typing.pde.collocation.boundary.normal_operators import (
    _LegacyNormalOperator,
)


class TestDirichletBC(Generic[Array], unittest.TestCase):
    """Base test class for Dirichlet boundary conditions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_dirichlet_constant(self):
        """Test constant Dirichlet BC."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Left boundary: u = 2.0
        left_idx = mesh.boundary_indices(0)
        bc = constant_dirichlet_bc(bkd, left_idx, 2.0)

        # Test residual
        state = bkd.ones((npts,))
        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 0.0)

        # res[0] = u[0] - 2.0 = 1.0 - 2.0 = -1.0
        self.assertAlmostEqual(float(residual[0]), -1.0, places=12)

        # Test Jacobian
        jacobian = bkd.zeros((npts, npts))
        jacobian = bc.apply_to_jacobian(jacobian, state, 0.0)

        # Row 0 should be [1, 0, 0, 0, 0]
        bkd.assert_allclose(jacobian[0, :], bkd.eye(npts)[0, :], atol=1e-14)

    def test_dirichlet_zero(self):
        """Test zero Dirichlet BC."""
        bkd = self.bkd()
        npts = 5
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)
        bc = zero_dirichlet_bc(bkd, left_idx)

        state = bkd.ones((npts,)) * 3.0
        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 0.0)

        # res[0] = u[0] - 0 = 3.0
        self.assertAlmostEqual(float(residual[0]), 3.0, places=12)

    def test_dirichlet_time_dependent(self):
        """Test time-dependent Dirichlet BC."""
        bkd = self.bkd()
        npts = 5
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)

        # g(t) = t + 1
        def g(t):
            return bkd.full((1,), t + 1.0)

        bc = DirichletBC(bkd, left_idx, g)

        state = bkd.zeros((npts,))
        residual = bkd.zeros((npts,))

        # At t=0: g(0) = 1, res = 0 - 1 = -1
        residual = bc.apply_to_residual(residual, state, 0.0)
        self.assertAlmostEqual(float(residual[0]), -1.0, places=12)

        # At t=2: g(2) = 3, res = 0 - 3 = -3
        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 2.0)
        self.assertAlmostEqual(float(residual[0]), -3.0, places=12)


class TestNeumannBC(Generic[Array], unittest.TestCase):
    """Base test class for Neumann boundary conditions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_neumann_zero(self):
        """Test zero Neumann BC (du/dn = 0)."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Left boundary: du/dn = 0
        left_idx = mesh.boundary_indices(0)
        D = basis.derivative_matrix()
        D_bndry = bkd.reshape(D[0, :], (1, npts))  # First row for left boundary

        bc = zero_neumann_bc(bkd, left_idx, D_bndry, -1.0)

        # State = x (linear), so du/dx = 1
        nodes = basis.nodes()
        state = nodes

        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 0.0)

        # res[0] = -1 * (D @ u)[0] - 0 = -1 * 1 = -1
        # (since du/dx = 1 for u = x)
        expected_flux = float(-1.0 * (D_bndry @ state)[0])
        self.assertAlmostEqual(float(residual[0]), expected_flux, places=10)

    def test_neumann_jacobian(self):
        """Test Neumann BC Jacobian."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)
        D = basis.derivative_matrix()
        D_bndry = bkd.reshape(D[0, :], (1, npts))

        bc = zero_neumann_bc(bkd, left_idx, D_bndry, -1.0)

        state = bkd.ones((npts,))
        jacobian = bkd.zeros((npts, npts))
        jacobian = bc.apply_to_jacobian(jacobian, state, 0.0)

        # Row 0 should be -1 * D[0, :]
        expected_row = -1.0 * D[0, :]
        bkd.assert_allclose(jacobian[0, :], expected_row, atol=1e-12)


class TestRobinBC(Generic[Array], unittest.TestCase):
    """Base test class for Robin boundary conditions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_robin_reduces_to_dirichlet(self):
        """Test Robin BC with alpha=1, beta=0 reduces to Dirichlet."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)
        D = basis.derivative_matrix()
        D_bndry = bkd.reshape(D[0, :], (1, npts))

        # Robin with alpha=1, beta=0, g=2: should be u = 2
        normal_op = _LegacyNormalOperator(bkd, D_bndry, -1.0)
        robin_bc = RobinBC(bkd, left_idx, normal_op, 1.0, 0.0, 2.0)
        dirichlet_bc = constant_dirichlet_bc(bkd, left_idx, 2.0)

        state = bkd.ones((npts,)) * 3.0

        res_robin = bkd.zeros((npts,))
        res_dirichlet = bkd.zeros((npts,))
        res_robin = robin_bc.apply_to_residual(res_robin, state, 0.0)
        res_dirichlet = dirichlet_bc.apply_to_residual(res_dirichlet, state, 0.0)

        bkd.assert_allclose(res_robin, res_dirichlet, atol=1e-14)

        jac_robin = bkd.zeros((npts, npts))
        jac_dirichlet = bkd.zeros((npts, npts))
        jac_robin = robin_bc.apply_to_jacobian(jac_robin, state, 0.0)
        jac_dirichlet = dirichlet_bc.apply_to_jacobian(jac_dirichlet, state, 0.0)

        bkd.assert_allclose(jac_robin, jac_dirichlet, atol=1e-14)

    def test_robin_homogeneous(self):
        """Test homogeneous Robin BC."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)
        D = basis.derivative_matrix()
        D_bndry = bkd.reshape(D[0, :], (1, npts))

        # Robin: u + du/dn = 0
        bc = homogeneous_robin_bc(bkd, left_idx, D_bndry, -1.0, 1.0, 1.0)

        state = bkd.ones((npts,))
        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 0.0)

        # res = 1 * u[0] + 1 * (-1 * (D @ u)[0]) - 0
        flux = -1.0 * float((D_bndry @ state)[0])
        expected = 1.0 * state[0] + 1.0 * flux
        self.assertAlmostEqual(float(residual[0]), expected, places=10)


class TestPeriodicBC(Generic[Array], unittest.TestCase):
    """Base test class for periodic boundary conditions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_periodic_residual(self):
        """Test periodic BC residual: value and derivative matching."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        right_idx = umesh.boundary_indices(1)
        D = basis.derivative_matrix()

        bc = PeriodicBC(bkd, left_idx, right_idx, D)

        # State with different values at boundaries
        state = bkd.linspace(0.0, 4.0, npts)  # [0, 1, 2, 3, 4]

        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 0.0)

        # Value matching: res[0] = u[0] - u[4] = 0 - 4 = -4
        self.assertAlmostEqual(float(residual[0]), -4.0, places=12)

        # Derivative matching: res[4] = D[0,:] @ u - D[4,:] @ u
        du_left = float(bkd.dot(D[0, :], state))
        du_right = float(bkd.dot(D[4, :], state))
        self.assertAlmostEqual(
            float(residual[4]), du_left - du_right, places=12
        )

    def test_periodic_jacobian(self):
        """Test periodic BC Jacobian: value and derivative rows."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        right_idx = umesh.boundary_indices(1)
        D = basis.derivative_matrix()

        bc = PeriodicBC(bkd, left_idx, right_idx, D)

        state = bkd.ones((npts,))
        jacobian = bkd.zeros((npts, npts))
        jacobian = bc.apply_to_jacobian(jacobian, state, 0.0)

        # Row 0 (value matching): [1, 0, 0, 0, -1]
        expected_val = bkd.zeros((npts,))
        expected_val = bkd.copy(expected_val)
        expected_val[0] = 1.0
        expected_val[4] = -1.0
        bkd.assert_allclose(jacobian[0, :], expected_val, atol=1e-14)

        # Row 4 (derivative matching): D[0, :] - D[4, :]
        expected_deriv = D[0, :] - D[4, :]
        bkd.assert_allclose(jacobian[4, :], expected_deriv, atol=1e-14)


class TestGradientRobinBC(Generic[Array], unittest.TestCase):
    """Base test class for gradient_robin_bc and gradient_neumann_bc."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_gradient_robin_1d_residual(self):
        """Test gradient Robin BC residual for u = x^2."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        normals = bkd.array([[-1.0]])
        D = basis.derivative_matrix()

        # alpha=1, beta=2, g=5
        # Robin: 1*u + 2*grad(u).n = 5
        bc = gradient_robin_bc(bkd, left_idx, normals, [D], 1.0, 2.0, 5.0)

        # u = x^2 at x=-1: u=1
        # grad(u).n = du/dx * (-1) = 2x*(-1) = -2*(-1)*(-1) = -2. Wait...
        # du/dx = 2x, at x=-1: du/dx=-2, grad(u).n = (-2)*(-1)=2
        # residual = 1*1 + 2*2 - 5 = 1 + 4 - 5 = 0
        nodes = basis.nodes()
        state = nodes * nodes

        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 0.0)
        bkd.assert_allclose(
            bkd.reshape(residual[left_idx], (1,)), bkd.array([0.0]), atol=1e-10
        )

    def test_gradient_robin_1d_jacobian(self):
        """Test gradient Robin BC Jacobian via finite differences."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        normals = bkd.array([[-1.0]])
        D = basis.derivative_matrix()

        bc = gradient_robin_bc(bkd, left_idx, normals, [D], 1.0, 2.0, 5.0)

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(npts))

        jacobian = bkd.zeros((npts, npts))
        jacobian = bc.apply_to_jacobian(jacobian, state, 0.0)

        # Finite difference check on the boundary row
        eps = 1e-7
        res0 = bkd.zeros((npts,))
        res0 = bc.apply_to_residual(res0, state, 0.0)
        row = int(left_idx[0])
        for j in range(npts):
            state_pert = bkd.copy(state)
            state_pert[j] = state_pert[j] + eps
            res1 = bkd.zeros((npts,))
            res1 = bc.apply_to_residual(res1, state_pert, 0.0)
            fd = (res1[row] - res0[row]) / eps
            bkd.assert_allclose(
                bkd.reshape(jacobian[row, j], (1,)),
                bkd.reshape(fd, (1,)),
                atol=1e-5,
            )

    def test_gradient_robin_reduces_to_neumann(self):
        """Test that gradient_robin_bc with alpha=0, beta=1 matches gradient_neumann_bc."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        normals = bkd.array([[-1.0]])
        D = basis.derivative_matrix()

        robin_bc = gradient_robin_bc(bkd, left_idx, normals, [D], 0.0, 1.0, 3.0)
        neumann_bc = gradient_neumann_bc(bkd, left_idx, normals, [D], 3.0)

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(npts))

        res_robin = bkd.zeros((npts,))
        res_robin = robin_bc.apply_to_residual(res_robin, state, 0.0)
        res_neumann = bkd.zeros((npts,))
        res_neumann = neumann_bc.apply_to_residual(res_neumann, state, 0.0)
        bkd.assert_allclose(res_robin, res_neumann, atol=1e-14)

        jac_robin = bkd.zeros((npts, npts))
        jac_robin = robin_bc.apply_to_jacobian(jac_robin, state, 0.0)
        jac_neumann = bkd.zeros((npts, npts))
        jac_neumann = neumann_bc.apply_to_jacobian(jac_neumann, state, 0.0)
        bkd.assert_allclose(jac_robin, jac_neumann, atol=1e-14)


# NumPy backend
class TestDirichletBCNumpy(TestDirichletBC[NDArray[Any]]):
    """NumPy backend tests for Dirichlet BC."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestNeumannBCNumpy(TestNeumannBC[NDArray[Any]]):
    """NumPy backend tests for Neumann BC."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRobinBCNumpy(TestRobinBC[NDArray[Any]]):
    """NumPy backend tests for Robin BC."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPeriodicBCNumpy(TestPeriodicBC[NDArray[Any]]):
    """NumPy backend tests for Periodic BC."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGradientRobinBCNumpy(TestGradientRobinBC[NDArray[Any]]):
    """NumPy backend tests for gradient Robin BC."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# Torch backend
class TestDirichletBCTorch(TestDirichletBC[torch.Tensor]):
    """Torch backend tests for Dirichlet BC."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestNeumannBCTorch(TestNeumannBC[torch.Tensor]):
    """Torch backend tests for Neumann BC."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestRobinBCTorch(TestRobinBC[torch.Tensor]):
    """Torch backend tests for Robin BC."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestPeriodicBCTorch(TestPeriodicBC[torch.Tensor]):
    """Torch backend tests for Periodic BC."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestGradientRobinBCTorch(TestGradientRobinBC[torch.Tensor]):
    """Torch backend tests for gradient Robin BC."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
