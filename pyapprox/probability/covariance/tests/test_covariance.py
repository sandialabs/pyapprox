"""
Tests for covariance operators.
"""

import numpy as np
import pytest

from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    DiagonalCovarianceOperator,
    OperatorBasedCovarianceOperator,
)


class TestDenseCholeskyCovarianceOperator:
    """Tests for DenseCholeskyCovarianceOperator."""

    def _setup(self, bkd):
        # Create a positive definite covariance matrix
        A = bkd.asarray([[2.0, 1.0], [1.0, 3.0]])
        cov = bkd.dot(A, bkd.transpose(A))
        op = DenseCholeskyCovarianceOperator(cov, bkd)
        return cov, op

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, op = self._setup(bkd)
        assert op.nvars() == 2

    def test_apply_identity(self, bkd) -> None:
        """Test L @ L^{-1} @ x = x."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        Linv_x = op.apply_inv(x)
        result = op.apply(Linv_x)
        assert bkd.allclose(result, x, rtol=1e-6)

    def test_apply_inv_identity(self, bkd) -> None:
        """Test L^{-1} @ L @ x = x."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        L_x = op.apply(x)
        result = op.apply_inv(L_x)
        assert bkd.allclose(result, x, rtol=1e-6)

    def test_cholesky_factorization(self, bkd) -> None:
        """Test L @ L.T = Cov."""
        cov, op = self._setup(bkd)
        L = op.cholesky_factor()
        reconstructed = bkd.dot(L, bkd.transpose(L))
        assert bkd.allclose(reconstructed, cov, rtol=1e-6)

    def test_covariance_inverse(self, bkd) -> None:
        """Test Cov @ Cov^{-1} = I."""
        cov, op = self._setup(bkd)
        cov_inv = op.covariance_inverse()
        identity = bkd.dot(cov, cov_inv)
        assert bkd.allclose(identity, bkd.eye(2), rtol=1e-6)

    def test_log_determinant(self, bkd) -> None:
        """Test log determinant computation."""
        _, op = self._setup(bkd)
        log_det = op.log_determinant()
        L = op.cholesky_factor()
        # Get diagonal and compute expected
        diag_L = bkd.asarray([L[i, i] for i in range(L.shape[0])])
        expected = bkd.sum(bkd.log(diag_L))
        assert bkd.allclose(
            bkd.asarray([log_det]),
            bkd.asarray([float(expected)]),
            rtol=1e-6,
        )

    def test_apply_transpose(self, bkd) -> None:
        """Test L.T @ x."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0], [2.0]])
        L = op.cholesky_factor()
        expected = bkd.dot(bkd.transpose(L), x)
        result = op.apply_transpose(x)
        assert bkd.allclose(result, expected, rtol=1e-6)

    def test_apply_inv_transpose(self, bkd) -> None:
        """Test L^{-T} @ x."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0], [2.0]])
        L_inv = op.cholesky_factor_inverse()
        expected = bkd.dot(bkd.transpose(L_inv), x)
        result = op.apply_inv_transpose(x)
        assert bkd.allclose(result, expected, rtol=1e-6)

    def test_shape_validation(self, bkd) -> None:
        """Test that invalid shapes raise errors."""
        _, op = self._setup(bkd)
        # 1D array should raise
        with pytest.raises(ValueError):
            op.apply(bkd.asarray([1.0, 2.0]))

        # Wrong number of rows should raise
        with pytest.raises(ValueError):
            op.apply(bkd.asarray([[1.0], [2.0], [3.0]]))


class TestDiagonalCovarianceOperator:
    """Tests for DiagonalCovarianceOperator."""

    def _setup(self, bkd):
        variances = bkd.asarray([1.0, 4.0, 9.0])
        op = DiagonalCovarianceOperator(variances, bkd)
        return variances, op

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, op = self._setup(bkd)
        assert op.nvars() == 3

    def test_apply(self, bkd) -> None:
        """Test L @ x = diag(sigma) @ x."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0], [1.0], [1.0]])
        result = op.apply(x)
        expected = bkd.asarray([[1.0], [2.0], [3.0]])  # sqrt of variances
        assert bkd.allclose(result, expected, rtol=1e-6)

    def test_apply_inv(self, bkd) -> None:
        """Test L^{-1} @ x = x / sigma."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0], [2.0], [3.0]])
        result = op.apply_inv(x)
        expected = bkd.asarray([[1.0], [1.0], [1.0]])
        assert bkd.allclose(result, expected, rtol=1e-6)

    def test_apply_identity(self, bkd) -> None:
        """Test L @ L^{-1} @ x = x."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Linv_x = op.apply_inv(x)
        result = op.apply(Linv_x)
        assert bkd.allclose(result, x, rtol=1e-6)

    def test_transpose_equals_forward(self, bkd) -> None:
        """For diagonal, L.T = L."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0], [2.0], [3.0]])
        forward = op.apply(x)
        transpose = op.apply_transpose(x)
        assert bkd.allclose(forward, transpose, rtol=1e-6)

    def test_inv_transpose_equals_inv(self, bkd) -> None:
        """For diagonal, L^{-T} = L^{-1}."""
        _, op = self._setup(bkd)
        x = bkd.asarray([[1.0], [2.0], [3.0]])
        inv = op.apply_inv(x)
        inv_transpose = op.apply_inv_transpose(x)
        assert bkd.allclose(inv, inv_transpose, rtol=1e-6)

    def test_log_determinant(self, bkd) -> None:
        """Test log determinant = sum(log(sigma))."""
        variances, op = self._setup(bkd)
        log_det = op.log_determinant()
        expected = float(bkd.sum(bkd.log(bkd.sqrt(variances))))
        assert bkd.allclose(
            bkd.asarray([log_det]),
            bkd.asarray([expected]),
            rtol=1e-6,
        )

    def test_diagonal(self, bkd) -> None:
        """Test diagonal returns variances."""
        variances, op = self._setup(bkd)
        assert bkd.allclose(op.diagonal(), variances)

    def test_covariance(self, bkd) -> None:
        """Test covariance returns diagonal matrix."""
        variances, op = self._setup(bkd)
        cov = op.covariance()
        expected = bkd.diag(variances)
        assert bkd.allclose(cov, expected, rtol=1e-6)

    def test_covariance_inverse(self, bkd) -> None:
        """Test covariance_inverse returns inverse diagonal."""
        variances, op = self._setup(bkd)
        cov_inv = op.covariance_inverse()
        expected = bkd.diag(1.0 / variances)
        assert bkd.allclose(cov_inv, expected, rtol=1e-6)

    def test_1d_requirement(self, bkd) -> None:
        """Test that 2D variances raise error."""
        with pytest.raises(ValueError):
            DiagonalCovarianceOperator(bkd.asarray([[1.0, 2.0]]), bkd)


class TestOperatorBasedCovarianceOperator:
    """Tests for OperatorBasedCovarianceOperator."""

    def _setup(self, bkd):
        nvars = 3
        scale = 2.0

        # Define scaling operator: L = scale * I
        def apply_sqrt(x):
            return scale * x

        def apply_sqrt_inv(x):
            return x / scale

        # log|L| = nvars * log(scale)
        log_det = nvars * float(np.log(scale))

        op = OperatorBasedCovarianceOperator(
            apply_sqrt=apply_sqrt,
            apply_sqrt_inv=apply_sqrt_inv,
            log_determinant=log_det,
            nvars=nvars,
            bkd=bkd,
        )
        return nvars, scale, log_det, op

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, _, _, op = self._setup(bkd)
        assert op.nvars() == 3

    def test_apply(self, bkd) -> None:
        """Test apply uses callback."""
        _, scale, _, op = self._setup(bkd)
        x = bkd.asarray([[1.0], [2.0], [3.0]])
        result = op.apply(x)
        expected = scale * x
        assert bkd.allclose(result, expected, rtol=1e-6)

    def test_apply_inv(self, bkd) -> None:
        """Test apply_inv uses callback."""
        _, scale, _, op = self._setup(bkd)
        x = bkd.asarray([[2.0], [4.0], [6.0]])
        result = op.apply_inv(x)
        expected = x / scale
        assert bkd.allclose(result, expected, rtol=1e-6)

    def test_log_determinant(self, bkd) -> None:
        """Test log_determinant returns provided value."""
        _, _, log_det, op = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([op.log_determinant()]),
            bkd.asarray([log_det]),
        )

    def test_apply_identity(self, bkd) -> None:
        """Test L @ L^{-1} @ x = x."""
        _, _, _, op = self._setup(bkd)
        x = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Linv_x = op.apply_inv(x)
        result = op.apply(Linv_x)
        assert bkd.allclose(result, x, rtol=1e-6)

    def test_compute_covariance_diagonal(self, bkd) -> None:
        """Test diagonal computation via probing."""
        nvars, scale, _, op = self._setup(bkd)
        diagonal = op.compute_covariance_diagonal()
        # Cov = L @ L.T = scale^2 * I
        expected = bkd.asarray([scale**2] * nvars)
        assert bkd.allclose(diagonal, expected, rtol=1e-6)

    def test_compute_covariance_diagonal_batched(self, bkd) -> None:
        """Test batched diagonal computation."""
        nvars, scale, _, op = self._setup(bkd)
        diagonal = op.compute_covariance_diagonal(batch_size=2)
        expected = bkd.asarray([scale**2] * nvars)
        assert bkd.allclose(diagonal, expected, rtol=1e-6)

    def test_compute_covariance_diagonal_active(self, bkd) -> None:
        """Test diagonal computation with active indices."""
        _, scale, _, op = self._setup(bkd)
        active = bkd.asarray([0, 2])
        diagonal = op.compute_covariance_diagonal(active_indices=active)
        expected = bkd.asarray([scale**2, scale**2])
        assert bkd.allclose(diagonal, expected, rtol=1e-6)


class TestNonSymmetricOperator:
    """Test non-symmetric operator with different transpose callbacks.

    This test uses numpy-specific operations so only runs with NumPy backend.
    """

    def test_non_symmetric_operator(self, numpy_bkd) -> None:
        """Test non-symmetric operator with different transpose callbacks."""
        bkd = numpy_bkd
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
        assert bkd.allclose(op.apply(x), L @ x, rtol=1e-6)

        # Test apply_transpose (should differ from apply)
        assert bkd.allclose(op.apply_transpose(x), L.T @ x, rtol=1e-6)

        # Test that they're different
        assert not bkd.allclose(op.apply(x), op.apply_transpose(x), rtol=1e-6)


class TestCovarianceOperatorProtocolCompliance:
    """Test that operators satisfy protocol interfaces."""

    def test_dense_satisfies_protocol(self, numpy_bkd) -> None:
        """Test DenseCholeskyCovarianceOperator has required methods."""
        bkd = numpy_bkd
        cov = np.eye(2)
        op = DenseCholeskyCovarianceOperator(cov, bkd)

        # Check all required protocol methods exist
        assert hasattr(op, "bkd")
        assert hasattr(op, "nvars")
        assert hasattr(op, "apply")
        assert hasattr(op, "apply_transpose")
        assert hasattr(op, "apply_inv")
        assert hasattr(op, "apply_inv_transpose")
        assert hasattr(op, "log_determinant")
        assert hasattr(op, "covariance")
        assert hasattr(op, "covariance_inverse")

    def test_diagonal_satisfies_protocol(self, numpy_bkd) -> None:
        """Test DiagonalCovarianceOperator has required methods."""
        bkd = numpy_bkd
        variances = np.array([1.0, 2.0])
        op = DiagonalCovarianceOperator(variances, bkd)

        # Check all required protocol methods exist
        assert hasattr(op, "bkd")
        assert hasattr(op, "nvars")
        assert hasattr(op, "apply")
        assert hasattr(op, "apply_transpose")
        assert hasattr(op, "apply_inv")
        assert hasattr(op, "apply_inv_transpose")
        assert hasattr(op, "log_determinant")
        assert hasattr(op, "diagonal")

    def test_operator_satisfies_protocol(self, numpy_bkd) -> None:
        """Test OperatorBasedCovarianceOperator has required methods."""
        bkd = numpy_bkd
        op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: x,
            apply_sqrt_inv=lambda x: x,
            log_determinant=0.0,
            nvars=2,
            bkd=bkd,
        )

        # Check all required protocol methods exist
        assert hasattr(op, "bkd")
        assert hasattr(op, "nvars")
        assert hasattr(op, "apply")
        assert hasattr(op, "apply_transpose")
        assert hasattr(op, "apply_inv")
        assert hasattr(op, "apply_inv_transpose")
        assert hasattr(op, "log_determinant")
