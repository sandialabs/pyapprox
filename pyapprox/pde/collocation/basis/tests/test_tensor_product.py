"""Tests for tensor product basis."""

import unittest
from typing import Generic

from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
    ChebyshevDerivativeMatrix1D,
    TensorProductBasis,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestTensorProductBasis(Generic[Array], unittest.TestCase):
    """Base test class for tensor product basis."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_1d_properties(self):
        """Test 1D tensor product basis properties."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        self.assertEqual(basis.ndim(), 1)
        self.assertEqual(basis.npts_per_dim(), (npts,))
        self.assertEqual(basis.npts(), npts)
        self.assertEqual(basis.nodes_1d(0).shape, (npts,))
        self.assertEqual(basis.derivative_matrix_1d(0).shape, (npts, npts))

    def test_2d_properties(self):
        """Test 2D tensor product basis properties."""
        bkd = self.bkd()
        npts_x, npts_y = 4, 5
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        self.assertEqual(basis.ndim(), 2)
        self.assertEqual(basis.npts_per_dim(), (npts_x, npts_y))
        self.assertEqual(basis.npts(), npts_x * npts_y)

    def test_3d_properties(self):
        """Test 3D tensor product basis properties."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 3, 4, 5
        mesh = TransformedMesh3D(npts_x, npts_y, npts_z, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        self.assertEqual(basis.ndim(), 3)
        self.assertEqual(basis.npts_per_dim(), (npts_x, npts_y, npts_z))
        self.assertEqual(basis.npts(), npts_x * npts_y * npts_z)

    def test_derivative_matrix_shape_1d(self):
        """Test 1D derivative matrix shape."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        D = basis.derivative_matrix(1, 0)
        self.assertEqual(D.shape, (npts, npts))

    def test_derivative_matrix_shape_2d(self):
        """Test 2D derivative matrix shapes."""
        bkd = self.bkd()
        npts_x, npts_y = 4, 5
        npts = npts_x * npts_y
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        self.assertEqual(Dx.shape, (npts, npts))
        self.assertEqual(Dy.shape, (npts, npts))

    def test_derivative_matrix_shape_3d(self):
        """Test 3D derivative matrix shapes."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 3, 4, 5
        npts = npts_x * npts_y * npts_z
        mesh = TransformedMesh3D(npts_x, npts_y, npts_z, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        Dz = basis.derivative_matrix(1, 2)
        self.assertEqual(Dx.shape, (npts, npts))
        self.assertEqual(Dy.shape, (npts, npts))
        self.assertEqual(Dz.shape, (npts, npts))

    def test_kronecker_structure_2d(self):
        """Test 2D derivative matrices have correct Kronecker structure."""
        bkd = self.bkd()
        npts_x, npts_y = 4, 5
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        # Get full reference matrices (for identity transform, physical = reference)
        Dx = basis.reference_derivative_matrix(1, 0)
        Dy = basis.reference_derivative_matrix(1, 1)

        # Get 1D matrices
        Dx_1d = basis.derivative_matrix_1d(0)
        Dy_1d = basis.derivative_matrix_1d(1)

        # Expected: Dx = kron(I_y, D_x), Dy = kron(D_y, I_x)
        Ix = bkd.eye(npts_x)
        Iy = bkd.eye(npts_y)
        Dx_expected = bkd.kron(Iy, Dx_1d)
        Dy_expected = bkd.kron(Dy_1d, Ix)

        bkd.assert_allclose(Dx, Dx_expected, atol=1e-14)
        bkd.assert_allclose(Dy, Dy_expected, atol=1e-14)

    def test_kronecker_structure_3d(self):
        """Test 3D derivative matrices have correct Kronecker structure."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 3, 4, 3
        mesh = TransformedMesh3D(npts_x, npts_y, npts_z, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        # Get 1D matrices
        Dx_1d = basis.derivative_matrix_1d(0)
        Dy_1d = basis.derivative_matrix_1d(1)
        Dz_1d = basis.derivative_matrix_1d(2)

        # Identity matrices
        Ix = bkd.eye(npts_x)
        Iy = bkd.eye(npts_y)
        Iz = bkd.eye(npts_z)

        # Expected: Dx = kron(I_z, kron(I_y, D_x))
        Dx_expected = bkd.kron(Iz, bkd.kron(Iy, Dx_1d))
        Dy_expected = bkd.kron(Iz, bkd.kron(Dy_1d, Ix))
        Dz_expected = bkd.kron(Dz_1d, bkd.kron(Iy, Ix))

        # Use reference derivative matrices for identity transform
        Dx = basis.reference_derivative_matrix(1, 0)
        Dy = basis.reference_derivative_matrix(1, 1)
        Dz = basis.reference_derivative_matrix(1, 2)

        bkd.assert_allclose(Dx, Dx_expected, atol=1e-14)
        bkd.assert_allclose(Dy, Dy_expected, atol=1e-14)
        bkd.assert_allclose(Dz, Dz_expected, atol=1e-14)

    def test_higher_order_derivative(self):
        """Test second order derivative equals D @ D."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        D1 = basis.derivative_matrix(1, 0)
        D2 = basis.derivative_matrix(2, 0)

        bkd.assert_allclose(D2, D1 @ D1, atol=1e-14)

    def test_derivative_caching(self):
        """Test derivative matrices are cached."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        D1 = basis.derivative_matrix(1, 0)
        D2 = basis.derivative_matrix(1, 0)

        # Should be the same object (cached)
        self.assertIs(D1, D2)

    def test_reference_vs_physical_identity_transform(self):
        """Test reference and physical derivatives match for identity transform."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)
        basis = TensorProductBasis(mesh, deriv_comp, bkd)

        D_ref = basis.reference_derivative_matrix(1, 0)
        D_phys = basis.derivative_matrix(1, 0)

        # For identity transform, they should be equal
        bkd.assert_allclose(D_ref, D_phys, atol=1e-14)


class TestChebyshevBasisWrappers(Generic[Array], unittest.TestCase):
    """Test convenience wrapper classes."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_chebyshev_basis_1d(self):
        """Test ChebyshevBasis1D wrapper."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        self.assertEqual(basis.ndim(), 1)
        self.assertEqual(basis.npts(), npts)
        self.assertEqual(basis.npts_per_dim(), (npts,))
        self.assertEqual(basis.nodes().shape, (npts,))
        self.assertEqual(basis.derivative_matrix().shape, (npts, npts))

    def test_chebyshev_basis_2d(self):
        """Test ChebyshevBasis2D wrapper."""
        bkd = self.bkd()
        npts_x, npts_y = 4, 5
        npts = npts_x * npts_y
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        self.assertEqual(basis.ndim(), 2)
        self.assertEqual(basis.npts(), npts)
        self.assertEqual(basis.npts_per_dim(), (npts_x, npts_y))
        self.assertEqual(basis.nodes_x().shape, (npts_x,))
        self.assertEqual(basis.nodes_y().shape, (npts_y,))
        self.assertEqual(basis.derivative_matrix_x().shape, (npts, npts))
        self.assertEqual(basis.derivative_matrix_y().shape, (npts, npts))

    def test_chebyshev_basis_3d(self):
        """Test ChebyshevBasis3D wrapper."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 3, 4, 5
        npts = npts_x * npts_y * npts_z
        mesh = TransformedMesh3D(npts_x, npts_y, npts_z, bkd)
        basis = ChebyshevBasis3D(mesh, bkd)

        self.assertEqual(basis.ndim(), 3)
        self.assertEqual(basis.npts(), npts)
        self.assertEqual(basis.npts_per_dim(), (npts_x, npts_y, npts_z))
        self.assertEqual(basis.nodes_x().shape, (npts_x,))
        self.assertEqual(basis.nodes_y().shape, (npts_y,))
        self.assertEqual(basis.nodes_z().shape, (npts_z,))
        self.assertEqual(basis.derivative_matrix_x().shape, (npts, npts))
        self.assertEqual(basis.derivative_matrix_y().shape, (npts, npts))
        self.assertEqual(basis.derivative_matrix_z().shape, (npts, npts))

    def test_2d_derivative_on_function(self):
        """Test 2D derivatives on a separable function."""
        bkd = self.bkd()
        npts_x, npts_y = 6, 6
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        # Create mesh of points
        x = basis.nodes_x()
        y = basis.nodes_y()
        # Tensor product ordering: x varies fastest
        # xx[i] = x[i % npts_x], yy[i] = y[i // npts_x]
        xx = bkd.tile(x, (npts_y,))
        yy = bkd.repeat(y, npts_x)

        # f(x,y) = x^2 * y
        # df/dx = 2x * y
        # df/dy = x^2
        f = xx**2 * yy

        Dx = basis.derivative_matrix_x()
        Dy = basis.derivative_matrix_y()

        df_dx = Dx @ f
        df_dy = Dy @ f

        expected_dfdx = 2.0 * xx * yy
        expected_dfdy = xx**2

        bkd.assert_allclose(df_dx, expected_dfdx, atol=1e-10)
        bkd.assert_allclose(df_dy, expected_dfdy, atol=1e-10)


class TestTensorProductBasisNumpy(TestTensorProductBasis):
    """NumPy backend tests for tensor product basis."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestChebyshevBasisWrappersNumpy(TestChebyshevBasisWrappers):
    """NumPy backend tests for Chebyshev basis wrappers."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
