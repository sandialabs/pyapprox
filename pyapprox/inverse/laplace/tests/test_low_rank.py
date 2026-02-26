"""
Tests for low-rank Laplace posterior.
"""

import pytest

import numpy as np

from pyapprox.inverse.laplace.full_rank import DenseLaplacePosterior
from pyapprox.inverse.laplace.low_rank import LowRankLaplacePosterior
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator


class TestLowRankLaplacePosteriorBase:
    """Base test class for LowRankLaplacePosterior."""

    def _make_laplace(self, bkd):
        """Create low-rank Laplace posterior for tests."""
        nvars = 5
        rank = 3

        # MAP point (use numpy float64)
        map_point = bkd.asarray(np.zeros((nvars, 1), dtype=np.float64))

        # Prior covariance (use numpy float64)
        prior_cov = bkd.asarray(np.eye(nvars, dtype=np.float64))
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, bkd)

        # Misfit Hessian (symmetric positive semi-definite)
        # Create a low-rank Hessian for testing
        V_np = np.array(
            [
                [1.0, 0.5, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.3, 0.0, 0.0],
                [0.0, 0.3, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ).T
        D_np = np.diag(np.array([3.0, 2.0, 1.0], dtype=np.float64))
        hess_np = V_np @ D_np @ V_np.T
        hess_matrix = bkd.asarray(hess_np)

        def apply_conditioned_hessian(vecs):
            # L^T @ H @ L @ vecs where L = I for unit prior
            L = bkd.cholesky(prior_cov)
            return L.T @ hess_matrix @ L @ vecs

        laplace = LowRankLaplacePosterior(
            map_point,
            prior_sqrt,
            apply_conditioned_hessian,
            rank,
            bkd,
        )
        return laplace, nvars, rank, map_point, prior_cov, apply_conditioned_hessian

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        laplace, nvars, _, _, _, _ = self._make_laplace(bkd)
        assert laplace.nvars() == nvars

    def test_rank(self, bkd) -> None:
        """Test rank returns correct value."""
        laplace, _, rank, _, _, _ = self._make_laplace(bkd)
        assert laplace.rank() == rank

    def test_posterior_mean_is_map(self, bkd) -> None:
        """Test posterior mean is the MAP point."""
        laplace, _, _, map_point, _, _ = self._make_laplace(bkd)
        mean = laplace.posterior_mean()
        mean_np = bkd.to_numpy(mean)
        map_np = bkd.to_numpy(map_point)
        np.testing.assert_array_equal(mean_np, map_np)

    def test_posterior_covariance_shape(self, bkd) -> None:
        """Test posterior covariance has correct shape."""
        laplace, nvars, _, _, _, _ = self._make_laplace(bkd)
        laplace.compute()
        cov = laplace.posterior_covariance()
        assert cov.shape == (nvars, nvars)

    def test_posterior_covariance_symmetric(self, bkd) -> None:
        """Test posterior covariance is symmetric."""
        laplace, _, _, _, _, _ = self._make_laplace(bkd)
        laplace.compute()
        cov = laplace.posterior_covariance()
        cov_np = bkd.to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T, decimal=5)

    def test_eigenvalues_shape(self, bkd) -> None:
        """Test eigenvalues have correct shape."""
        laplace, _, rank, _, _, _ = self._make_laplace(bkd)
        laplace.compute()
        eigs = laplace.eigenvalues()
        assert eigs.shape == (rank,)

    def test_eigenvalues_nonnegative(self, bkd) -> None:
        """Test eigenvalues are non-negative."""
        laplace, _, _, _, _, _ = self._make_laplace(bkd)
        laplace.compute()
        eigs = laplace.eigenvalues()
        eigs_np = bkd.to_numpy(eigs)
        assert all(eigs_np >= -1e-10)

    def test_eigenvectors_shape(self, bkd) -> None:
        """Test eigenvectors have correct shape."""
        laplace, nvars, rank, _, _, _ = self._make_laplace(bkd)
        laplace.compute()
        vecs = laplace.eigenvectors()
        assert vecs.shape == (nvars, rank)

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs generates samples with correct shape."""
        laplace, nvars, _, _, _, _ = self._make_laplace(bkd)
        laplace.compute()
        nsamples = 10
        samples = laplace.rvs(nsamples)
        assert samples.shape == (nvars, nsamples)

    def test_compute_not_called_raises(self, bkd) -> None:
        """Test accessing results before compute raises error."""
        laplace, _, _, _, _, _ = self._make_laplace(bkd)
        with pytest.raises(RuntimeError):
            laplace.posterior_covariance()

    def test_posterior_variable_returns_gaussian(self, bkd) -> None:
        """Test posterior_variable returns a Gaussian distribution.

        Note: This test uses full rank to ensure the covariance is positive definite.
        Low-rank approximations produce rank-deficient covariance matrices.
        """
        laplace, nvars, _, _, _, apply_conditioned_hessian = self._make_laplace(bkd)
        # Use full rank for this test to get positive definite covariance
        map_point = bkd.asarray(np.zeros((nvars, 1), dtype=np.float64))
        prior_cov = bkd.asarray(np.eye(nvars, dtype=np.float64))
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, bkd)

        full_rank_laplace = LowRankLaplacePosterior(
            map_point,
            prior_sqrt,
            apply_conditioned_hessian,
            nvars,  # Full rank
            bkd,
        )
        full_rank_laplace.compute(noversampling=5)

        post = full_rank_laplace.posterior_variable()
        assert hasattr(post, "logpdf")
        assert hasattr(post, "rvs")


class TestLowRankVsFullRank:
    """Test low-rank converges to full-rank as rank increases."""

    def test_full_rank_matches_dense(self, bkd) -> None:
        """Test that full-rank low-rank approximation matches dense Laplace."""
        nvars = 3

        # MAP point (use numpy for float64)
        map_point = bkd.asarray(np.zeros((nvars, 1), dtype=np.float64))

        # Prior (use numpy for float64)
        prior_cov = bkd.asarray(np.eye(nvars, dtype=np.float64))
        prior_precision = bkd.asarray(np.eye(nvars, dtype=np.float64))
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, bkd)

        # Symmetric positive definite Hessian (use numpy for float64)
        hess_np = np.array(
            [
                [3.0, 0.5, 0.1],
                [0.5, 2.0, 0.3],
                [0.1, 0.3, 1.5],
            ],
            dtype=np.float64,
        )
        hess = bkd.asarray(hess_np)

        # Full-rank dense Laplace
        dense_laplace = DenseLaplacePosterior(
            map_point, prior_precision, hess, bkd
        )
        dense_laplace.compute()
        dense_cov = bkd.to_numpy(dense_laplace.posterior_covariance())

        # Low-rank with full rank (nvars)
        def apply_conditioned_hessian(vecs):
            L = bkd.cholesky(prior_cov)
            return L.T @ hess @ L @ vecs

        low_rank_laplace = LowRankLaplacePosterior(
            map_point,
            prior_sqrt,
            apply_conditioned_hessian,
            nvars,  # Full rank
            bkd,
        )
        low_rank_laplace.compute(noversampling=5, npower_iters=2)
        low_rank_cov = bkd.to_numpy(low_rank_laplace.posterior_covariance())

        # Should match closely
        np.testing.assert_array_almost_equal(low_rank_cov, dense_cov, decimal=3)


class TestLowRankLaplacePosteriorValidation:
    """Test input validation."""

    def test_rank_exceeds_nvars_raises(self, bkd) -> None:
        """Test rank > nvars raises error."""
        nvars = 3
        map_point = bkd.zeros((nvars, 1))
        prior_cov = bkd.eye(nvars)
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, bkd)

        def apply_hess(vecs):
            return vecs

        with pytest.raises(ValueError):
            LowRankLaplacePosterior(
                map_point,
                prior_sqrt,
                apply_hess,
                nvars + 1,  # Rank exceeds nvars
                bkd,
            )

    def test_wrong_map_shape_raises(self, bkd) -> None:
        """Test wrong MAP point shape raises error."""
        nvars = 3
        map_point = bkd.zeros((nvars + 1, 1))  # Wrong shape
        prior_cov = bkd.eye(nvars)
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, bkd)

        def apply_hess(vecs):
            return vecs

        with pytest.raises(ValueError):
            LowRankLaplacePosterior(
                map_point,
                prior_sqrt,
                apply_hess,
                2,
                bkd,
            )
