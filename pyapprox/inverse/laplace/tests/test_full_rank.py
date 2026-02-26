"""
Tests for full-rank Laplace posterior.
"""

import pytest

import numpy as np

from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.laplace.full_rank import DenseLaplacePosterior


class TestDenseLaplacePosteriorBase:
    """Base test class for DenseLaplacePosterior."""

    def _make_laplace(self, bkd):
        """Create Laplace posterior for tests."""
        nvars = 3

        # MAP point
        map_point = bkd.asarray([[1.0], [2.0], [3.0]])

        # Prior precision
        prior_precision = bkd.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # Likelihood Hessian (from some nonlinear model)
        likelihood_hessian = bkd.asarray(
            [
                [2.0, 0.5, 0.1],
                [0.5, 3.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        laplace = DenseLaplacePosterior(
            map_point,
            prior_precision,
            likelihood_hessian,
            bkd,
        )
        return laplace, nvars, map_point

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        laplace, nvars, _ = self._make_laplace(bkd)
        assert laplace.nvars() == nvars

    def test_posterior_mean_is_map(self, bkd) -> None:
        """Test posterior mean is the MAP point."""
        laplace, _, map_point = self._make_laplace(bkd)
        mean = laplace.posterior_mean()
        mean_np = bkd.to_numpy(mean)
        map_np = bkd.to_numpy(map_point)
        np.testing.assert_array_equal(mean_np, map_np)

    def test_posterior_covariance_shape(self, bkd) -> None:
        """Test posterior covariance has correct shape."""
        laplace, nvars, _ = self._make_laplace(bkd)
        laplace.compute()
        cov = laplace.posterior_covariance()
        assert cov.shape == (nvars, nvars)

    def test_posterior_covariance_symmetric(self, bkd) -> None:
        """Test posterior covariance is symmetric."""
        laplace, _, _ = self._make_laplace(bkd)
        laplace.compute()
        cov = laplace.posterior_covariance()
        cov_np = bkd.to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T)

    def test_posterior_covariance_positive_definite(self, bkd) -> None:
        """Test posterior covariance is positive definite."""
        laplace, _, _ = self._make_laplace(bkd)
        laplace.compute()
        cov = laplace.posterior_covariance()
        cov_np = bkd.to_numpy(cov)
        eigenvalues = np.linalg.eigvalsh(cov_np)
        assert all(eigenvalues > 0)

    def test_covariance_diagonal_shape(self, bkd) -> None:
        """Test covariance diagonal has correct shape."""
        laplace, nvars, _ = self._make_laplace(bkd)
        laplace.compute()
        diag = laplace.covariance_diagonal()
        assert diag.shape == (nvars,)

    def test_covariance_diagonal_matches_matrix(self, bkd) -> None:
        """Test diagonal matches full matrix diagonal."""
        laplace, _, _ = self._make_laplace(bkd)
        laplace.compute()
        diag = laplace.covariance_diagonal()
        cov = laplace.posterior_covariance()
        diag_np = bkd.to_numpy(diag)
        cov_diag_np = np.diag(bkd.to_numpy(cov))
        np.testing.assert_array_almost_equal(diag_np, cov_diag_np)

    def test_compute_not_called_raises(self, bkd) -> None:
        """Test accessing covariance before compute raises error."""
        laplace, _, _ = self._make_laplace(bkd)
        with pytest.raises(RuntimeError):
            laplace.posterior_covariance()

    def test_posterior_variable_returns_gaussian(self, bkd) -> None:
        """Test posterior_variable returns a Gaussian distribution."""
        laplace, _, _ = self._make_laplace(bkd)
        laplace.compute()
        post = laplace.posterior_variable()
        assert hasattr(post, "logpdf")
        assert hasattr(post, "rvs")


class TestDenseLaplacePosteriorVsConjugate:
    """Test Laplace matches conjugate for linear Gaussian model."""

    def test_matches_conjugate_for_linear_model(self, bkd) -> None:
        """
        For linear Gaussian model, Laplace should match conjugate solution.

        The log-likelihood Hessian for a linear model y = Ax + noise with
        noise ~ N(0, Sigma) is A^T @ Sigma^{-1} @ A, which is constant.
        """
        nvars = 2
        nobs = 3

        # Linear model matrix
        A = bkd.asarray([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])

        # Prior
        prior_mean = bkd.zeros((nvars, 1))
        prior_cov = bkd.eye(nvars)
        prior_precision = bkd.eye(nvars)

        # Noise
        noise_var = 0.1
        noise_cov = noise_var * bkd.eye(nobs)
        noise_precision = (1.0 / noise_var) * bkd.eye(nobs)

        # Observations
        obs = bkd.asarray([[1.0], [1.5], [2.0]])

        # Conjugate solution
        conjugate = DenseGaussianConjugatePosterior(
            A, prior_mean, prior_cov, noise_cov, bkd
        )
        conjugate.compute(obs)
        bkd.to_numpy(conjugate.posterior_mean())
        conj_cov = bkd.to_numpy(conjugate.posterior_covariance())

        # Laplace solution
        # For linear model, likelihood Hessian = A^T @ noise_precision @ A
        likelihood_hessian = A.T @ noise_precision @ A

        laplace = DenseLaplacePosterior(
            conjugate.posterior_mean(),  # Use conjugate MAP as starting point
            prior_precision,
            likelihood_hessian,
            bkd,
        )
        laplace.compute()
        laplace_cov = bkd.to_numpy(laplace.posterior_covariance())

        # Covariances should match
        np.testing.assert_array_almost_equal(laplace_cov, conj_cov, decimal=5)


class TestDenseLaplacePosteriorValidation:
    """Test input validation."""

    def test_wrong_map_shape_raises(self, bkd) -> None:
        """Test wrong MAP point shape raises error."""
        map_point = bkd.zeros((3, 2))  # Wrong shape
        prior_precision = bkd.eye(3)
        likelihood_hessian = bkd.eye(3)

        with pytest.raises(ValueError):
            DenseLaplacePosterior(
                map_point, prior_precision, likelihood_hessian, bkd
            )

    def test_wrong_prior_precision_shape_raises(self, bkd) -> None:
        """Test wrong prior precision shape raises error."""
        map_point = bkd.zeros((3, 1))
        prior_precision = bkd.eye(2)  # Wrong shape
        likelihood_hessian = bkd.eye(3)

        with pytest.raises(ValueError):
            DenseLaplacePosterior(
                map_point, prior_precision, likelihood_hessian, bkd
            )

    def test_wrong_likelihood_hessian_shape_raises(self, bkd) -> None:
        """Test wrong likelihood Hessian shape raises error."""
        map_point = bkd.zeros((3, 1))
        prior_precision = bkd.eye(3)
        likelihood_hessian = bkd.eye(2)  # Wrong shape

        with pytest.raises(ValueError):
            DenseLaplacePosterior(
                map_point, prior_precision, likelihood_hessian, bkd
            )
