"""
Tests for covariance operators.
"""

import unittest
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd as NumpyBackend
from pyapprox.typing.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    DiagonalCovarianceOperator,
    OperatorBasedCovarianceOperator,
)


class TestDenseCholeskyCovarianceOperator(unittest.TestCase):
    """Tests for DenseCholeskyCovarianceOperator."""

    def setUp(self):
        self.bkd = NumpyBackend()
        # Create a positive definite covariance matrix
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        self.cov = A @ A.T  # Ensure positive definite
        self.op = DenseCholeskyCovarianceOperator(self.cov, self.bkd)

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.op.nvars(), 2)

    def test_apply_identity(self):
        """Test L @ L^{-1} @ x = x."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        Linv_x = self.op.apply_inv(x)
        result = self.op.apply(Linv_x)
        np.testing.assert_array_almost_equal(result, x)

    def test_apply_inv_identity(self):
        """Test L^{-1} @ L @ x = x."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        L_x = self.op.apply(x)
        result = self.op.apply_inv(L_x)
        np.testing.assert_array_almost_equal(result, x)

    def test_cholesky_factorization(self):
        """Test L @ L.T = Cov."""
        L = self.op.cholesky_factor()
        reconstructed = L @ L.T
        np.testing.assert_array_almost_equal(reconstructed, self.cov)

    def test_covariance_inverse(self):
        """Test Cov @ Cov^{-1} = I."""
        cov_inv = self.op.covariance_inverse()
        identity = self.cov @ cov_inv
        np.testing.assert_array_almost_equal(identity, np.eye(2))

    def test_log_determinant(self):
        """Test log determinant computation."""
        log_det = self.op.log_determinant()
        L = self.op.cholesky_factor()
        expected = np.sum(np.log(np.diag(L)))
        self.assertAlmostEqual(log_det, expected)

    def test_apply_transpose(self):
        """Test L.T @ x."""
        x = np.array([[1.0], [2.0]])
        L = self.op.cholesky_factor()
        expected = L.T @ x
        result = self.op.apply_transpose(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_inv_transpose(self):
        """Test L^{-T} @ x."""
        x = np.array([[1.0], [2.0]])
        L_inv = self.op.cholesky_factor_inverse()
        expected = L_inv.T @ x
        result = self.op.apply_inv_transpose(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_shape_validation(self):
        """Test that invalid shapes raise errors."""
        # 1D array should raise
        with self.assertRaises(ValueError):
            self.op.apply(np.array([1.0, 2.0]))

        # Wrong number of rows should raise
        with self.assertRaises(ValueError):
            self.op.apply(np.array([[1.0], [2.0], [3.0]]))


class TestDiagonalCovarianceOperator(unittest.TestCase):
    """Tests for DiagonalCovarianceOperator."""

    def setUp(self):
        self.bkd = NumpyBackend()
        self.variances = np.array([1.0, 4.0, 9.0])
        self.op = DiagonalCovarianceOperator(self.variances, self.bkd)

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.op.nvars(), 3)

    def test_apply(self):
        """Test L @ x = diag(sigma) @ x."""
        x = np.array([[1.0], [1.0], [1.0]])
        result = self.op.apply(x)
        expected = np.array([[1.0], [2.0], [3.0]])  # sqrt of variances
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_inv(self):
        """Test L^{-1} @ x = x / sigma."""
        x = np.array([[1.0], [2.0], [3.0]])
        result = self.op.apply_inv(x)
        expected = np.array([[1.0], [1.0], [1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_identity(self):
        """Test L @ L^{-1} @ x = x."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Linv_x = self.op.apply_inv(x)
        result = self.op.apply(Linv_x)
        np.testing.assert_array_almost_equal(result, x)

    def test_transpose_equals_forward(self):
        """For diagonal, L.T = L."""
        x = np.array([[1.0], [2.0], [3.0]])
        forward = self.op.apply(x)
        transpose = self.op.apply_transpose(x)
        np.testing.assert_array_almost_equal(forward, transpose)

    def test_inv_transpose_equals_inv(self):
        """For diagonal, L^{-T} = L^{-1}."""
        x = np.array([[1.0], [2.0], [3.0]])
        inv = self.op.apply_inv(x)
        inv_transpose = self.op.apply_inv_transpose(x)
        np.testing.assert_array_almost_equal(inv, inv_transpose)

    def test_log_determinant(self):
        """Test log determinant = sum(log(sigma))."""
        log_det = self.op.log_determinant()
        expected = np.sum(np.log(np.sqrt(self.variances)))
        self.assertAlmostEqual(log_det, expected)

    def test_diagonal(self):
        """Test diagonal returns variances."""
        np.testing.assert_array_equal(self.op.diagonal(), self.variances)

    def test_covariance(self):
        """Test covariance returns diagonal matrix."""
        cov = self.op.covariance()
        expected = np.diag(self.variances)
        np.testing.assert_array_equal(cov, expected)

    def test_covariance_inverse(self):
        """Test covariance_inverse returns inverse diagonal."""
        cov_inv = self.op.covariance_inverse()
        expected = np.diag(1.0 / self.variances)
        np.testing.assert_array_equal(cov_inv, expected)

    def test_1d_requirement(self):
        """Test that 2D variances raise error."""
        with self.assertRaises(ValueError):
            DiagonalCovarianceOperator(np.array([[1.0, 2.0]]), self.bkd)


class TestOperatorBasedCovarianceOperator(unittest.TestCase):
    """Tests for OperatorBasedCovarianceOperator."""

    def setUp(self):
        self.bkd = NumpyBackend()
        self.nvars = 3
        self.scale = 2.0

        # Define scaling operator: L = scale * I
        def apply_sqrt(x):
            return self.scale * x

        def apply_sqrt_inv(x):
            return x / self.scale

        # log|L| = nvars * log(scale)
        self.log_det = self.nvars * np.log(self.scale)

        self.op = OperatorBasedCovarianceOperator(
            apply_sqrt=apply_sqrt,
            apply_sqrt_inv=apply_sqrt_inv,
            log_determinant=self.log_det,
            nvars=self.nvars,
            bkd=self.bkd,
        )

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.op.nvars(), 3)

    def test_apply(self):
        """Test apply uses callback."""
        x = np.array([[1.0], [2.0], [3.0]])
        result = self.op.apply(x)
        expected = self.scale * x
        np.testing.assert_array_equal(result, expected)

    def test_apply_inv(self):
        """Test apply_inv uses callback."""
        x = np.array([[2.0], [4.0], [6.0]])
        result = self.op.apply_inv(x)
        expected = x / self.scale
        np.testing.assert_array_equal(result, expected)

    def test_log_determinant(self):
        """Test log_determinant returns provided value."""
        self.assertEqual(self.op.log_determinant(), self.log_det)

    def test_apply_identity(self):
        """Test L @ L^{-1} @ x = x."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Linv_x = self.op.apply_inv(x)
        result = self.op.apply(Linv_x)
        np.testing.assert_array_almost_equal(result, x)

    def test_compute_covariance_diagonal(self):
        """Test diagonal computation via probing."""
        diagonal = self.op.compute_covariance_diagonal()
        # Cov = L @ L.T = scale^2 * I
        expected = np.full(self.nvars, self.scale**2)
        np.testing.assert_array_almost_equal(diagonal, expected)

    def test_compute_covariance_diagonal_batched(self):
        """Test batched diagonal computation."""
        diagonal = self.op.compute_covariance_diagonal(batch_size=2)
        expected = np.full(self.nvars, self.scale**2)
        np.testing.assert_array_almost_equal(diagonal, expected)

    def test_compute_covariance_diagonal_active(self):
        """Test diagonal computation with active indices."""
        active = self.bkd.array([0, 2])
        diagonal = self.op.compute_covariance_diagonal(active_indices=active)
        expected = np.array([self.scale**2, self.scale**2])
        np.testing.assert_array_almost_equal(diagonal, expected)

    def test_non_symmetric_operator(self):
        """Test non-symmetric operator with different transpose callbacks."""
        # L is lower triangular
        L = np.array([[1.0, 0.0], [0.5, 1.0]])
        L_inv = np.linalg.inv(L)

        op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: L @ x,
            apply_sqrt_inv=lambda x: L_inv @ x,
            log_determinant=np.sum(np.log(np.diag(L))),
            nvars=2,
            bkd=self.bkd,
            apply_sqrt_transpose=lambda x: L.T @ x,
            apply_sqrt_inv_transpose=lambda x: L_inv.T @ x,
        )

        x = np.array([[1.0], [2.0]])

        # Test apply
        np.testing.assert_array_almost_equal(op.apply(x), L @ x)

        # Test apply_transpose (should differ from apply)
        np.testing.assert_array_almost_equal(op.apply_transpose(x), L.T @ x)

        # Test that they're different
        self.assertFalse(np.allclose(op.apply(x), op.apply_transpose(x)))


class TestCovarianceOperatorProtocolCompliance(unittest.TestCase):
    """Test that operators satisfy protocol interfaces."""

    def test_dense_satisfies_protocol(self):
        """Test DenseCholeskyCovarianceOperator has required methods."""
        bkd = NumpyBackend()
        cov = np.eye(2)
        op = DenseCholeskyCovarianceOperator(cov, bkd)

        # Check all required protocol methods exist
        self.assertTrue(hasattr(op, "bkd"))
        self.assertTrue(hasattr(op, "nvars"))
        self.assertTrue(hasattr(op, "apply"))
        self.assertTrue(hasattr(op, "apply_transpose"))
        self.assertTrue(hasattr(op, "apply_inv"))
        self.assertTrue(hasattr(op, "apply_inv_transpose"))
        self.assertTrue(hasattr(op, "log_determinant"))
        self.assertTrue(hasattr(op, "covariance"))
        self.assertTrue(hasattr(op, "covariance_inverse"))

    def test_diagonal_satisfies_protocol(self):
        """Test DiagonalCovarianceOperator has required methods."""
        bkd = NumpyBackend()
        variances = np.array([1.0, 2.0])
        op = DiagonalCovarianceOperator(variances, bkd)

        # Check all required protocol methods exist
        self.assertTrue(hasattr(op, "bkd"))
        self.assertTrue(hasattr(op, "nvars"))
        self.assertTrue(hasattr(op, "apply"))
        self.assertTrue(hasattr(op, "apply_transpose"))
        self.assertTrue(hasattr(op, "apply_inv"))
        self.assertTrue(hasattr(op, "apply_inv_transpose"))
        self.assertTrue(hasattr(op, "log_determinant"))
        self.assertTrue(hasattr(op, "diagonal"))

    def test_operator_satisfies_protocol(self):
        """Test OperatorBasedCovarianceOperator has required methods."""
        bkd = NumpyBackend()
        op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: x,
            apply_sqrt_inv=lambda x: x,
            log_determinant=0.0,
            nvars=2,
            bkd=bkd,
        )

        # Check all required protocol methods exist
        self.assertTrue(hasattr(op, "bkd"))
        self.assertTrue(hasattr(op, "nvars"))
        self.assertTrue(hasattr(op, "apply"))
        self.assertTrue(hasattr(op, "apply_transpose"))
        self.assertTrue(hasattr(op, "apply_inv"))
        self.assertTrue(hasattr(op, "apply_inv_transpose"))
        self.assertTrue(hasattr(op, "log_determinant"))


if __name__ == "__main__":
    unittest.main()
