"""
Tests for Gaussian pushforward.
"""

import numpy as np
import pytest

from pyapprox.inverse.pushforward.gaussian import GaussianPushforward


class TestGaussianPushforwardBase:
    """Base test class for GaussianPushforward."""

    def _make_pf(self, bkd):
        """Create pushforward objects for tests."""
        nvars = 2
        nqoi = 2

        # Linear transformation matrix (full rank for positive definite output)
        matrix = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])

        # Input Gaussian
        mean = bkd.asarray([[1.0], [2.0]])
        cov = bkd.asarray([[1.0, 0.3], [0.3, 1.0]])

        # Offset
        offset = bkd.asarray([[0.1], [0.2]])

        # Create pushforward without offset
        pf = GaussianPushforward(matrix, mean, cov, bkd)

        # Create pushforward with offset
        pf_offset = GaussianPushforward(
            matrix, mean, cov, bkd, offset
        )
        return pf, pf_offset, nvars, nqoi

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        pf, _, nvars, _ = self._make_pf(bkd)
        assert pf.nvars() == nvars

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns correct value."""
        pf, _, _, nqoi = self._make_pf(bkd)
        assert pf.nqoi() == nqoi

    def test_mean_shape(self, bkd) -> None:
        """Test mean has correct shape."""
        pf, _, _, nqoi = self._make_pf(bkd)
        mean = pf.mean()
        assert mean.shape == (nqoi, 1)

    def test_covariance_shape(self, bkd) -> None:
        """Test covariance has correct shape."""
        pf, _, _, nqoi = self._make_pf(bkd)
        cov = pf.covariance()
        assert cov.shape == (nqoi, nqoi)

    def test_covariance_symmetric(self, bkd) -> None:
        """Test covariance is symmetric."""
        pf, _, _, _ = self._make_pf(bkd)
        cov = pf.covariance()
        cov_np = bkd.to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T)

    def test_covariance_positive_semidefinite(self, bkd) -> None:
        """Test covariance is positive semidefinite."""
        pf, _, _, _ = self._make_pf(bkd)
        cov = pf.covariance()
        cov_np = bkd.to_numpy(cov)
        eigenvalues = np.linalg.eigvalsh(cov_np)
        assert all(eigenvalues >= -1e-10)  # Allow for numerical error

    def test_pushforward_variable_returns_gaussian(self, bkd) -> None:
        """Test pushforward_variable returns a Gaussian distribution."""
        pf, _, _, _ = self._make_pf(bkd)
        pf_var = pf.pushforward_variable()
        assert hasattr(pf_var, "logpdf")
        assert hasattr(pf_var, "rvs")


class TestGaussianPushforwardAnalytical:
    """Test against analytical formulas."""

    def test_mean_formula(self, bkd) -> None:
        """Test pushforward mean = A @ mean + offset."""
        matrix = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        mean = bkd.asarray([[1.0], [1.0]])
        cov = bkd.eye(2)
        offset = bkd.asarray([[0.5], [1.5]])

        pf = GaussianPushforward(matrix, mean, cov, bkd, offset)

        pf_mean = bkd.to_numpy(pf.mean())
        # Expected: [[1*1 + 2*1 + 0.5], [3*1 + 4*1 + 1.5]] = [[3.5], [8.5]]
        expected = np.array([[3.5], [8.5]])
        np.testing.assert_array_almost_equal(pf_mean, expected)

    def test_covariance_formula(self, bkd) -> None:
        """Test pushforward cov = A @ cov @ A.T."""
        matrix = bkd.asarray([[1.0, 0.0], [0.0, 2.0]])
        mean = bkd.zeros((2, 1))
        cov = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])

        pf = GaussianPushforward(matrix, mean, cov, bkd)

        pf_cov = bkd.to_numpy(pf.covariance())
        # A @ cov @ A.T = [[1, 0], [0, 2]] @ [[1, 0.5], [0.5, 1]] @ [[1, 0], [0, 2]]
        #               = [[1, 0.5], [1, 2]] @ [[1, 0], [0, 2]]
        #               = [[1, 1], [1, 4]]
        expected = np.array([[1.0, 1.0], [1.0, 4.0]])
        np.testing.assert_array_almost_equal(pf_cov, expected)

    def test_identity_transform(self, bkd) -> None:
        """Test identity transformation preserves distribution."""
        nvars = 3
        matrix = bkd.eye(nvars)
        mean = bkd.asarray([[1.0], [2.0], [3.0]])
        cov = bkd.asarray([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])

        pf = GaussianPushforward(matrix, mean, cov, bkd)

        pf_mean = bkd.to_numpy(pf.mean())
        pf_cov = bkd.to_numpy(pf.covariance())

        np.testing.assert_array_almost_equal(pf_mean, bkd.to_numpy(mean))
        np.testing.assert_array_almost_equal(pf_cov, bkd.to_numpy(cov))

    def test_scalar_output(self, bkd) -> None:
        """Test pushforward to 1D output."""
        matrix = bkd.asarray([[1.0, 1.0]])  # Sum of inputs
        mean = bkd.asarray([[1.0], [2.0]])
        cov = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])

        pf = GaussianPushforward(matrix, mean, cov, bkd)

        pf_mean = bkd.to_numpy(pf.mean())
        pf_var = bkd.to_numpy(pf.covariance())

        # Mean should be sum of input means: 1 + 2 = 3
        assert pf_mean[0, 0] == pytest.approx(3.0, abs=1e-5)

        # Var([1, 1] @ x) = [1, 1] @ cov @ [1, 1]^T = 1 + 0.5 + 0.5 + 1 = 3
        assert pf_var[0, 0] == pytest.approx(3.0, abs=1e-5)


class TestGaussianPushforwardValidation:
    """Test input validation."""

    def test_wrong_mean_shape_raises(self, bkd) -> None:
        """Test wrong mean shape raises error."""
        matrix = bkd.eye(2)
        mean = bkd.zeros((3, 1))  # Wrong shape
        cov = bkd.eye(2)

        with pytest.raises(ValueError):
            GaussianPushforward(matrix, mean, cov, bkd)

    def test_wrong_cov_shape_raises(self, bkd) -> None:
        """Test wrong covariance shape raises error."""
        matrix = bkd.eye(2)
        mean = bkd.zeros((2, 1))
        cov = bkd.eye(3)  # Wrong shape

        with pytest.raises(ValueError):
            GaussianPushforward(matrix, mean, cov, bkd)

    def test_wrong_offset_shape_raises(self, bkd) -> None:
        """Test wrong offset shape raises error."""
        matrix = bkd.eye(2)
        mean = bkd.zeros((2, 1))
        cov = bkd.eye(2)
        offset = bkd.zeros((3, 1))  # Wrong shape

        with pytest.raises(ValueError):
            GaussianPushforward(matrix, mean, cov, bkd, offset)
