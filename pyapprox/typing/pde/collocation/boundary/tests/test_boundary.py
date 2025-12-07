"""Tests for boundary conditions."""

import unittest
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import create_uniform_mesh_1d
from pyapprox.typing.pde.collocation.boundary import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    PeriodicBC,
    constant_dirichlet_bc,
    zero_dirichlet_bc,
    zero_neumann_bc,
    homogeneous_robin_bc,
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
        basis = ChebyshevBasis1D(npts, bkd)
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
        basis = ChebyshevBasis1D(npts, bkd)
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
        basis = ChebyshevBasis1D(npts, bkd)
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
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)
        D = basis.derivative_matrix()
        D_bndry = bkd.reshape(D[0, :], (1, npts))

        # Robin with alpha=1, beta=0, g=2: should be u = 2
        robin_bc = RobinBC(bkd, left_idx, D_bndry, -1.0, 1.0, 0.0, 2.0)
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
        basis = ChebyshevBasis1D(npts, bkd)
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
        """Test periodic BC residual."""
        bkd = self.bkd()
        npts = 5
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)

        bc = PeriodicBC(bkd, left_idx, right_idx)

        # State with different values at boundaries
        state = bkd.linspace(0.0, 4.0, npts)  # [0, 1, 2, 3, 4]

        residual = bkd.zeros((npts,))
        residual = bc.apply_to_residual(residual, state, 0.0)

        # res[0] = u[0] - u[4] = 0 - 4 = -4
        self.assertAlmostEqual(float(residual[0]), -4.0, places=12)

    def test_periodic_jacobian(self):
        """Test periodic BC Jacobian."""
        bkd = self.bkd()
        npts = 5
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)

        bc = PeriodicBC(bkd, left_idx, right_idx)

        state = bkd.ones((npts,))
        jacobian = bkd.zeros((npts, npts))
        jacobian = bc.apply_to_jacobian(jacobian, state, 0.0)

        # Row 0 should be [1, 0, 0, 0, -1]
        expected = bkd.zeros((npts,))
        expected = bkd.copy(expected)
        expected[0] = 1.0
        expected[4] = -1.0
        bkd.assert_allclose(jacobian[0, :], expected, atol=1e-14)


class TestDirichletBCNumpy(TestDirichletBC):
    """NumPy backend tests for Dirichlet BC."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestNeumannBCNumpy(TestNeumannBC):
    """NumPy backend tests for Neumann BC."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestRobinBCNumpy(TestRobinBC):
    """NumPy backend tests for Robin BC."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestPeriodicBCNumpy(TestPeriodicBC):
    """NumPy backend tests for Periodic BC."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
