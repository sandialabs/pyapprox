import numpy as np
import pytest

from pyapprox.util.linalg.cholesky_factor import CholeskyFactor


class TestCholeskyFactor:
    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_factor(self, bkd) -> None:
        """
        Test the factor method.
        """
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        bkd.assert_allclose(cholesky_factor.factor(), L)

    def test_log_determinant(self, bkd) -> None:
        """
        Test the log determinant computation.
        """
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        log_det = cholesky_factor.log_determinant()
        expected_log_det = 2.0 * bkd.sum(bkd.log(bkd.diag(L)))
        bkd.assert_allclose(log_det, expected_log_det)

    def test_matrix_inverse(self, bkd) -> None:
        """
        Test the matrix inverse computation.
        """
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        A_inv = cholesky_factor.matrix_inverse()
        expected_A_inv = bkd.inv(A)
        bkd.assert_allclose(A_inv, expected_A_inv)

    def test_factor_inverse(self, bkd) -> None:
        """
        Test the factor inverse computation.
        """
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        L_inv = cholesky_factor.factor_inverse()
        expected_L_inv = bkd.solve_triangular(L, bkd.eye(L.shape[0]), lower=True)
        bkd.assert_allclose(L_inv, expected_L_inv)

    def test_solve(self, bkd) -> None:
        """
        Test solving a linear system.
        """
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        rhs = bkd.asarray(np.random.randn(3, 1))
        cholesky_factor = CholeskyFactor(L, bkd)
        x = cholesky_factor.solve(rhs)
        expected_x = bkd.solve(A, rhs)
        bkd.assert_allclose(x, expected_x)

    def test_repr(self, bkd) -> None:
        """
        Test the string representation (__repr__).
        """
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        expected_repr = "{0}(N={1}, backend={2})".format(
            cholesky_factor.__class__.__name__, L.shape, bkd.__class__.__name__
        )
        assert repr(cholesky_factor) == expected_repr
