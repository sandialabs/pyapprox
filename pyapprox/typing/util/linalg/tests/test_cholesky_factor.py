import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.linalg.cholesky_factor import CholeskyFactor


class TestCholeskyFactor(Generic[Array], unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_factor(self) -> None:
        """
        Test the factor method.
        """
        bkd = self.bkd()
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        bkd.assert_allclose(cholesky_factor.factor(), L)

    def test_log_determinant(self) -> None:
        """
        Test the log determinant computation.
        """
        bkd = self.bkd()
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        log_det = cholesky_factor.log_determinant()
        expected_log_det = 2.0 * bkd.sum(bkd.log(bkd.diag(L)))
        bkd.assert_allclose(log_det, expected_log_det)

    def test_matrix_inverse(self) -> None:
        """
        Test the matrix inverse computation.
        """
        bkd = self.bkd()
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        A_inv = cholesky_factor.matrix_inverse()
        expected_A_inv = bkd.inv(A)
        bkd.assert_allclose(A_inv, expected_A_inv)

    def test_factor_inverse(self) -> None:
        """
        Test the factor inverse computation.
        """
        bkd = self.bkd()
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        L_inv = cholesky_factor.factor_inverse()
        expected_L_inv = bkd.solve_triangular(
            L, bkd.eye(L.shape[0]), lower=True
        )
        bkd.assert_allclose(L_inv, expected_L_inv)

    def test_solve(self) -> None:
        """
        Test solving a linear system.
        """
        bkd = self.bkd()
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        rhs = bkd.asarray(np.random.randn(3, 1))
        cholesky_factor = CholeskyFactor(L, bkd)
        x = cholesky_factor.solve(rhs)
        expected_x = bkd.solve(A, rhs)
        bkd.assert_allclose(x, expected_x)

    def test_repr(self) -> None:
        """
        Test the string representation (__repr__).
        """
        bkd = self.bkd()
        A = bkd.asarray(np.random.randn(3, 3))
        A = A @ A.T  # Ensure positive definiteness
        L = bkd.cholesky(A)
        cholesky_factor = CholeskyFactor(L, bkd)
        expected_repr = "{0}(N={1}, backend={2})".format(
            cholesky_factor.__class__.__name__, L.shape, bkd.__class__.__name__
        )
        self.assertEqual(repr(cholesky_factor), expected_repr)


# Derived test class for Np backend
class TestCholeskyFactorNumpy(TestCholeskyFactor[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestCholeskyFactorTorch(TestCholeskyFactor[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class
    ContinuousScipyRandomVariable1D.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestCholeskyFactorNumpy,
        TestCholeskyFactorTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
