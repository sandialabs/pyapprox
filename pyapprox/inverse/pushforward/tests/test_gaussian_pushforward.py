"""
Tests for Gaussian pushforward.
"""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.inverse.pushforward.gaussian import GaussianPushforward


class TestGaussianPushforwardBase(Generic[Array], unittest.TestCase):
    """Base test class for GaussianPushforward."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        # 2D input, 2D output (square transformation for positive definite output)
        self.nvars = 2
        self.nqoi = 2

        # Linear transformation matrix (full rank for positive definite output)
        self.matrix = self.bkd().asarray(
            [[1.0, 0.5], [0.5, 1.0]]
        )

        # Input Gaussian
        self.mean = self.bkd().asarray([[1.0], [2.0]])
        self.cov = self.bkd().asarray([[1.0, 0.3], [0.3, 1.0]])

        # Offset
        self.offset = self.bkd().asarray([[0.1], [0.2]])

        # Create pushforward without offset
        self.pf = GaussianPushforward(
            self.matrix, self.mean, self.cov, self.bkd()
        )

        # Create pushforward with offset
        self.pf_offset = GaussianPushforward(
            self.matrix, self.mean, self.cov, self.bkd(), self.offset
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.pf.nvars(), self.nvars)

    def test_nqoi(self) -> None:
        """Test nqoi returns correct value."""
        self.assertEqual(self.pf.nqoi(), self.nqoi)

    def test_mean_shape(self) -> None:
        """Test mean has correct shape."""
        mean = self.pf.mean()
        self.assertEqual(mean.shape, (self.nqoi, 1))

    def test_covariance_shape(self) -> None:
        """Test covariance has correct shape."""
        cov = self.pf.covariance()
        self.assertEqual(cov.shape, (self.nqoi, self.nqoi))

    def test_covariance_symmetric(self) -> None:
        """Test covariance is symmetric."""
        cov = self.pf.covariance()
        cov_np = self.bkd().to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T)

    def test_covariance_positive_semidefinite(self) -> None:
        """Test covariance is positive semidefinite."""
        cov = self.pf.covariance()
        cov_np = self.bkd().to_numpy(cov)
        eigenvalues = np.linalg.eigvalsh(cov_np)
        self.assertTrue(all(eigenvalues >= -1e-10))  # Allow for numerical error

    def test_pushforward_variable_returns_gaussian(self) -> None:
        """Test pushforward_variable returns a Gaussian distribution."""
        pf_var = self.pf.pushforward_variable()
        self.assertTrue(hasattr(pf_var, 'logpdf'))
        self.assertTrue(hasattr(pf_var, 'rvs'))


class TestGaussianPushforwardAnalytical(Generic[Array], unittest.TestCase):
    """Test against analytical formulas."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_mean_formula(self) -> None:
        """Test pushforward mean = A @ mean + offset."""
        matrix = self.bkd().asarray([[1.0, 2.0], [3.0, 4.0]])
        mean = self.bkd().asarray([[1.0], [1.0]])
        cov = self.bkd().eye(2)
        offset = self.bkd().asarray([[0.5], [1.5]])

        pf = GaussianPushforward(matrix, mean, cov, self.bkd(), offset)

        pf_mean = self.bkd().to_numpy(pf.mean())
        # Expected: [[1*1 + 2*1 + 0.5], [3*1 + 4*1 + 1.5]] = [[3.5], [8.5]]
        expected = np.array([[3.5], [8.5]])
        np.testing.assert_array_almost_equal(pf_mean, expected)

    def test_covariance_formula(self) -> None:
        """Test pushforward cov = A @ cov @ A.T."""
        matrix = self.bkd().asarray([[1.0, 0.0], [0.0, 2.0]])
        mean = self.bkd().zeros((2, 1))
        cov = self.bkd().asarray([[1.0, 0.5], [0.5, 1.0]])

        pf = GaussianPushforward(matrix, mean, cov, self.bkd())

        pf_cov = self.bkd().to_numpy(pf.covariance())
        # A @ cov @ A.T = [[1, 0], [0, 2]] @ [[1, 0.5], [0.5, 1]] @ [[1, 0], [0, 2]]
        #               = [[1, 0.5], [1, 2]] @ [[1, 0], [0, 2]]
        #               = [[1, 1], [1, 4]]
        expected = np.array([[1.0, 1.0], [1.0, 4.0]])
        np.testing.assert_array_almost_equal(pf_cov, expected)

    def test_identity_transform(self) -> None:
        """Test identity transformation preserves distribution."""
        nvars = 3
        matrix = self.bkd().eye(nvars)
        mean = self.bkd().asarray([[1.0], [2.0], [3.0]])
        cov = self.bkd().asarray(
            [[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]
        )

        pf = GaussianPushforward(matrix, mean, cov, self.bkd())

        pf_mean = self.bkd().to_numpy(pf.mean())
        pf_cov = self.bkd().to_numpy(pf.covariance())

        np.testing.assert_array_almost_equal(
            pf_mean, self.bkd().to_numpy(mean)
        )
        np.testing.assert_array_almost_equal(
            pf_cov, self.bkd().to_numpy(cov)
        )

    def test_scalar_output(self) -> None:
        """Test pushforward to 1D output."""
        matrix = self.bkd().asarray([[1.0, 1.0]])  # Sum of inputs
        mean = self.bkd().asarray([[1.0], [2.0]])
        cov = self.bkd().asarray([[1.0, 0.5], [0.5, 1.0]])

        pf = GaussianPushforward(matrix, mean, cov, self.bkd())

        pf_mean = self.bkd().to_numpy(pf.mean())
        pf_var = self.bkd().to_numpy(pf.covariance())

        # Mean should be sum of input means: 1 + 2 = 3
        self.assertAlmostEqual(pf_mean[0, 0], 3.0, places=5)

        # Var([1, 1] @ x) = [1, 1] @ cov @ [1, 1]^T = 1 + 0.5 + 0.5 + 1 = 3
        self.assertAlmostEqual(pf_var[0, 0], 3.0, places=5)


class TestGaussianPushforwardValidation(Generic[Array], unittest.TestCase):
    """Test input validation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_wrong_mean_shape_raises(self) -> None:
        """Test wrong mean shape raises error."""
        matrix = self.bkd().eye(2)
        mean = self.bkd().zeros((3, 1))  # Wrong shape
        cov = self.bkd().eye(2)

        with self.assertRaises(ValueError):
            GaussianPushforward(matrix, mean, cov, self.bkd())

    def test_wrong_cov_shape_raises(self) -> None:
        """Test wrong covariance shape raises error."""
        matrix = self.bkd().eye(2)
        mean = self.bkd().zeros((2, 1))
        cov = self.bkd().eye(3)  # Wrong shape

        with self.assertRaises(ValueError):
            GaussianPushforward(matrix, mean, cov, self.bkd())

    def test_wrong_offset_shape_raises(self) -> None:
        """Test wrong offset shape raises error."""
        matrix = self.bkd().eye(2)
        mean = self.bkd().zeros((2, 1))
        cov = self.bkd().eye(2)
        offset = self.bkd().zeros((3, 1))  # Wrong shape

        with self.assertRaises(ValueError):
            GaussianPushforward(matrix, mean, cov, self.bkd(), offset)


# NumPy backend tests
class TestGaussianPushforwardNumpy(TestGaussianPushforwardBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGaussianPushforwardAnalyticalNumpy(
    TestGaussianPushforwardAnalytical[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGaussianPushforwardValidationNumpy(
    TestGaussianPushforwardValidation[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestGaussianPushforwardTorch(TestGaussianPushforwardBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestGaussianPushforwardAnalyticalTorch(
    TestGaussianPushforwardAnalytical[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestGaussianPushforwardValidationTorch(
    TestGaussianPushforwardValidation[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
