"""
Tests for covariance operators.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    DiagonalCovarianceOperator,
    OperatorBasedCovarianceOperator,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestDenseCholeskyCovarianceOperator(Generic[Array], unittest.TestCase):
    """Tests for DenseCholeskyCovarianceOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # Create a positive definite covariance matrix
        A = self._bkd.asarray([[2.0, 1.0], [1.0, 3.0]])
        self.cov = self._bkd.dot(A, self._bkd.transpose(A))
        self.op = DenseCholeskyCovarianceOperator(self.cov, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.op.nvars(), 2)

    def test_apply_identity(self) -> None:
        """Test L @ L^{-1} @ x = x."""
        x = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        Linv_x = self.op.apply_inv(x)
        result = self.op.apply(Linv_x)
        self.assertTrue(self._bkd.allclose(result, x, rtol=1e-6))

    def test_apply_inv_identity(self) -> None:
        """Test L^{-1} @ L @ x = x."""
        x = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        L_x = self.op.apply(x)
        result = self.op.apply_inv(L_x)
        self.assertTrue(self._bkd.allclose(result, x, rtol=1e-6))

    def test_cholesky_factorization(self) -> None:
        """Test L @ L.T = Cov."""
        L = self.op.cholesky_factor()
        reconstructed = self._bkd.dot(L, self._bkd.transpose(L))
        self.assertTrue(self._bkd.allclose(reconstructed, self.cov, rtol=1e-6))

    def test_covariance_inverse(self) -> None:
        """Test Cov @ Cov^{-1} = I."""
        cov_inv = self.op.covariance_inverse()
        identity = self._bkd.dot(self.cov, cov_inv)
        self.assertTrue(self._bkd.allclose(identity, self._bkd.eye(2), rtol=1e-6))

    def test_log_determinant(self) -> None:
        """Test log determinant computation."""
        log_det = self.op.log_determinant()
        L = self.op.cholesky_factor()
        # Get diagonal and compute expected
        diag_L = self._bkd.asarray([L[i, i] for i in range(L.shape[0])])
        expected = self._bkd.sum(self._bkd.log(diag_L))
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([log_det]),
                self._bkd.asarray([float(expected)]),
                rtol=1e-6,
            )
        )

    def test_apply_transpose(self) -> None:
        """Test L.T @ x."""
        x = self._bkd.asarray([[1.0], [2.0]])
        L = self.op.cholesky_factor()
        expected = self._bkd.dot(self._bkd.transpose(L), x)
        result = self.op.apply_transpose(x)
        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-6))

    def test_apply_inv_transpose(self) -> None:
        """Test L^{-T} @ x."""
        x = self._bkd.asarray([[1.0], [2.0]])
        L_inv = self.op.cholesky_factor_inverse()
        expected = self._bkd.dot(self._bkd.transpose(L_inv), x)
        result = self.op.apply_inv_transpose(x)
        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-6))

    def test_shape_validation(self) -> None:
        """Test that invalid shapes raise errors."""
        # 1D array should raise
        with self.assertRaises(ValueError):
            self.op.apply(self._bkd.asarray([1.0, 2.0]))

        # Wrong number of rows should raise
        with self.assertRaises(ValueError):
            self.op.apply(self._bkd.asarray([[1.0], [2.0], [3.0]]))


class TestDenseCholeskyCovarianceOperatorNumpy(
    TestDenseCholeskyCovarianceOperator[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDenseCholeskyCovarianceOperatorTorch(
    TestDenseCholeskyCovarianceOperator[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestDiagonalCovarianceOperator(Generic[Array], unittest.TestCase):
    """Tests for DiagonalCovarianceOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.variances = self._bkd.asarray([1.0, 4.0, 9.0])
        self.op = DiagonalCovarianceOperator(self.variances, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.op.nvars(), 3)

    def test_apply(self) -> None:
        """Test L @ x = diag(sigma) @ x."""
        x = self._bkd.asarray([[1.0], [1.0], [1.0]])
        result = self.op.apply(x)
        expected = self._bkd.asarray([[1.0], [2.0], [3.0]])  # sqrt of variances
        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-6))

    def test_apply_inv(self) -> None:
        """Test L^{-1} @ x = x / sigma."""
        x = self._bkd.asarray([[1.0], [2.0], [3.0]])
        result = self.op.apply_inv(x)
        expected = self._bkd.asarray([[1.0], [1.0], [1.0]])
        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-6))

    def test_apply_identity(self) -> None:
        """Test L @ L^{-1} @ x = x."""
        x = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Linv_x = self.op.apply_inv(x)
        result = self.op.apply(Linv_x)
        self.assertTrue(self._bkd.allclose(result, x, rtol=1e-6))

    def test_transpose_equals_forward(self) -> None:
        """For diagonal, L.T = L."""
        x = self._bkd.asarray([[1.0], [2.0], [3.0]])
        forward = self.op.apply(x)
        transpose = self.op.apply_transpose(x)
        self.assertTrue(self._bkd.allclose(forward, transpose, rtol=1e-6))

    def test_inv_transpose_equals_inv(self) -> None:
        """For diagonal, L^{-T} = L^{-1}."""
        x = self._bkd.asarray([[1.0], [2.0], [3.0]])
        inv = self.op.apply_inv(x)
        inv_transpose = self.op.apply_inv_transpose(x)
        self.assertTrue(self._bkd.allclose(inv, inv_transpose, rtol=1e-6))

    def test_log_determinant(self) -> None:
        """Test log determinant = sum(log(sigma))."""
        log_det = self.op.log_determinant()
        expected = float(self._bkd.sum(self._bkd.log(self._bkd.sqrt(self.variances))))
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([log_det]),
                self._bkd.asarray([expected]),
                rtol=1e-6,
            )
        )

    def test_diagonal(self) -> None:
        """Test diagonal returns variances."""
        self.assertTrue(self._bkd.allclose(self.op.diagonal(), self.variances))

    def test_covariance(self) -> None:
        """Test covariance returns diagonal matrix."""
        cov = self.op.covariance()
        expected = self._bkd.diag(self.variances)
        self.assertTrue(self._bkd.allclose(cov, expected, rtol=1e-6))

    def test_covariance_inverse(self) -> None:
        """Test covariance_inverse returns inverse diagonal."""
        cov_inv = self.op.covariance_inverse()
        expected = self._bkd.diag(1.0 / self.variances)
        self.assertTrue(self._bkd.allclose(cov_inv, expected, rtol=1e-6))

    def test_1d_requirement(self) -> None:
        """Test that 2D variances raise error."""
        with self.assertRaises(ValueError):
            DiagonalCovarianceOperator(self._bkd.asarray([[1.0, 2.0]]), self._bkd)


class TestDiagonalCovarianceOperatorNumpy(TestDiagonalCovarianceOperator[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDiagonalCovarianceOperatorTorch(TestDiagonalCovarianceOperator[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestOperatorBasedCovarianceOperator(Generic[Array], unittest.TestCase):
    """Tests for OperatorBasedCovarianceOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.nvars = 3
        self.scale = 2.0

        # Define scaling operator: L = scale * I
        def apply_sqrt(x: Array) -> Array:
            return self.scale * x

        def apply_sqrt_inv(x: Array) -> Array:
            return x / self.scale

        # log|L| = nvars * log(scale)
        self.log_det = self.nvars * float(np.log(self.scale))

        self.op = OperatorBasedCovarianceOperator(
            apply_sqrt=apply_sqrt,
            apply_sqrt_inv=apply_sqrt_inv,
            log_determinant=self.log_det,
            nvars=self.nvars,
            bkd=self._bkd,
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.op.nvars(), 3)

    def test_apply(self) -> None:
        """Test apply uses callback."""
        x = self._bkd.asarray([[1.0], [2.0], [3.0]])
        result = self.op.apply(x)
        expected = self.scale * x
        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-6))

    def test_apply_inv(self) -> None:
        """Test apply_inv uses callback."""
        x = self._bkd.asarray([[2.0], [4.0], [6.0]])
        result = self.op.apply_inv(x)
        expected = x / self.scale
        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-6))

    def test_log_determinant(self) -> None:
        """Test log_determinant returns provided value."""
        self.assertAlmostEqual(self.op.log_determinant(), self.log_det)

    def test_apply_identity(self) -> None:
        """Test L @ L^{-1} @ x = x."""
        x = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Linv_x = self.op.apply_inv(x)
        result = self.op.apply(Linv_x)
        self.assertTrue(self._bkd.allclose(result, x, rtol=1e-6))

    def test_compute_covariance_diagonal(self) -> None:
        """Test diagonal computation via probing."""
        diagonal = self.op.compute_covariance_diagonal()
        # Cov = L @ L.T = scale^2 * I
        expected = self._bkd.asarray([self.scale**2] * self.nvars)
        self.assertTrue(self._bkd.allclose(diagonal, expected, rtol=1e-6))

    def test_compute_covariance_diagonal_batched(self) -> None:
        """Test batched diagonal computation."""
        diagonal = self.op.compute_covariance_diagonal(batch_size=2)
        expected = self._bkd.asarray([self.scale**2] * self.nvars)
        self.assertTrue(self._bkd.allclose(diagonal, expected, rtol=1e-6))

    def test_compute_covariance_diagonal_active(self) -> None:
        """Test diagonal computation with active indices."""
        active = self._bkd.asarray([0, 2])
        diagonal = self.op.compute_covariance_diagonal(active_indices=active)
        expected = self._bkd.asarray([self.scale**2, self.scale**2])
        self.assertTrue(self._bkd.allclose(diagonal, expected, rtol=1e-6))


class TestOperatorBasedCovarianceOperatorNumpy(
    TestOperatorBasedCovarianceOperator[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOperatorBasedCovarianceOperatorTorch(
    TestOperatorBasedCovarianceOperator[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestNonSymmetricOperator(unittest.TestCase):
    """Test non-symmetric operator with different transpose callbacks.

    This test uses numpy-specific operations so only runs with NumPy backend.
    """

    def test_non_symmetric_operator(self) -> None:
        """Test non-symmetric operator with different transpose callbacks."""
        bkd = NumpyBkd()
        # L is lower triangular
        L = np.array([[1.0, 0.0], [0.5, 1.0]])
        L_inv = np.linalg.inv(L)

        op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: L @ x,
            apply_sqrt_inv=lambda x: L_inv @ x,
            log_determinant=float(np.sum(np.log(np.diag(L)))),
            nvars=2,
            bkd=bkd,
            apply_sqrt_transpose=lambda x: L.T @ x,
            apply_sqrt_inv_transpose=lambda x: L_inv.T @ x,
        )

        x = np.array([[1.0], [2.0]])

        # Test apply
        self.assertTrue(bkd.allclose(op.apply(x), L @ x, rtol=1e-6))

        # Test apply_transpose (should differ from apply)
        self.assertTrue(bkd.allclose(op.apply_transpose(x), L.T @ x, rtol=1e-6))

        # Test that they're different
        self.assertFalse(bkd.allclose(op.apply(x), op.apply_transpose(x), rtol=1e-6))


class TestCovarianceOperatorProtocolCompliance(unittest.TestCase):
    """Test that operators satisfy protocol interfaces."""

    def test_dense_satisfies_protocol(self) -> None:
        """Test DenseCholeskyCovarianceOperator has required methods."""
        bkd = NumpyBkd()
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

    def test_diagonal_satisfies_protocol(self) -> None:
        """Test DiagonalCovarianceOperator has required methods."""
        bkd = NumpyBkd()
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

    def test_operator_satisfies_protocol(self) -> None:
        """Test OperatorBasedCovarianceOperator has required methods."""
        bkd = NumpyBkd()
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
