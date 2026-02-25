"""
Tests for low-rank Laplace posterior.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.inverse.laplace.full_rank import DenseLaplacePosterior
from pyapprox.inverse.laplace.low_rank import LowRankLaplacePosterior
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestLowRankLaplacePosteriorBase(Generic[Array], unittest.TestCase):
    """Base test class for LowRankLaplacePosterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        self.nvars = 5
        self.rank = 3

        # MAP point (use numpy float64)
        self.map_point = self.bkd().asarray(np.zeros((self.nvars, 1), dtype=np.float64))

        # Prior covariance (use numpy float64)
        self.prior_cov = self.bkd().asarray(np.eye(self.nvars, dtype=np.float64))
        self.prior_sqrt = DenseCholeskyCovarianceOperator(self.prior_cov, self.bkd())

        # Misfit Hessian (symmetric positive semi-definite)
        # Create a low-rank Hessian for testing
        # Use numpy array to ensure float64 dtype
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
        self.hess_matrix = self.bkd().asarray(hess_np)

        def apply_conditioned_hessian(vecs: Array) -> Array:
            # L^T @ H @ L @ vecs where L = I for unit prior
            L = self.bkd().cholesky(self.prior_cov)
            return L.T @ self.hess_matrix @ L @ vecs

        self.apply_conditioned_hessian = apply_conditioned_hessian

        self.laplace = LowRankLaplacePosterior(
            self.map_point,
            self.prior_sqrt,
            apply_conditioned_hessian,
            self.rank,
            self.bkd(),
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.laplace.nvars(), self.nvars)

    def test_rank(self) -> None:
        """Test rank returns correct value."""
        self.assertEqual(self.laplace.rank(), self.rank)

    def test_posterior_mean_is_map(self) -> None:
        """Test posterior mean is the MAP point."""
        mean = self.laplace.posterior_mean()
        mean_np = self.bkd().to_numpy(mean)
        map_np = self.bkd().to_numpy(self.map_point)
        np.testing.assert_array_equal(mean_np, map_np)

    def test_posterior_covariance_shape(self) -> None:
        """Test posterior covariance has correct shape."""
        self.laplace.compute()
        cov = self.laplace.posterior_covariance()
        self.assertEqual(cov.shape, (self.nvars, self.nvars))

    def test_posterior_covariance_symmetric(self) -> None:
        """Test posterior covariance is symmetric."""
        self.laplace.compute()
        cov = self.laplace.posterior_covariance()
        cov_np = self.bkd().to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T, decimal=5)

    def test_eigenvalues_shape(self) -> None:
        """Test eigenvalues have correct shape."""
        self.laplace.compute()
        eigs = self.laplace.eigenvalues()
        self.assertEqual(eigs.shape, (self.rank,))

    def test_eigenvalues_nonnegative(self) -> None:
        """Test eigenvalues are non-negative."""
        self.laplace.compute()
        eigs = self.laplace.eigenvalues()
        eigs_np = self.bkd().to_numpy(eigs)
        self.assertTrue(all(eigs_np >= -1e-10))

    def test_eigenvectors_shape(self) -> None:
        """Test eigenvectors have correct shape."""
        self.laplace.compute()
        vecs = self.laplace.eigenvectors()
        self.assertEqual(vecs.shape, (self.nvars, self.rank))

    def test_rvs_shape(self) -> None:
        """Test rvs generates samples with correct shape."""
        self.laplace.compute()
        nsamples = 10
        samples = self.laplace.rvs(nsamples)
        self.assertEqual(samples.shape, (self.nvars, nsamples))

    def test_compute_not_called_raises(self) -> None:
        """Test accessing results before compute raises error."""
        with self.assertRaises(RuntimeError):
            self.laplace.posterior_covariance()

    def test_posterior_variable_returns_gaussian(self) -> None:
        """Test posterior_variable returns a Gaussian distribution.

        Note: This test uses full rank to ensure the covariance is positive definite.
        Low-rank approximations produce rank-deficient covariance matrices.
        """
        # Use full rank for this test to get positive definite covariance
        map_point = self.bkd().asarray(np.zeros((self.nvars, 1), dtype=np.float64))
        prior_cov = self.bkd().asarray(np.eye(self.nvars, dtype=np.float64))
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, self.bkd())

        full_rank_laplace = LowRankLaplacePosterior(
            map_point,
            prior_sqrt,
            self.apply_conditioned_hessian,
            self.nvars,  # Full rank
            self.bkd(),
        )
        full_rank_laplace.compute(noversampling=5)

        post = full_rank_laplace.posterior_variable()
        self.assertTrue(hasattr(post, "logpdf"))
        self.assertTrue(hasattr(post, "rvs"))


class TestLowRankVsFullRank(Generic[Array], unittest.TestCase):
    """Test low-rank converges to full-rank as rank increases."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_full_rank_matches_dense(self) -> None:
        """Test that full-rank low-rank approximation matches dense Laplace."""
        nvars = 3

        # MAP point (use numpy for float64)
        map_point = self.bkd().asarray(np.zeros((nvars, 1), dtype=np.float64))

        # Prior (use numpy for float64)
        prior_cov = self.bkd().asarray(np.eye(nvars, dtype=np.float64))
        prior_precision = self.bkd().asarray(np.eye(nvars, dtype=np.float64))
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, self.bkd())

        # Symmetric positive definite Hessian (use numpy for float64)
        hess_np = np.array(
            [
                [3.0, 0.5, 0.1],
                [0.5, 2.0, 0.3],
                [0.1, 0.3, 1.5],
            ],
            dtype=np.float64,
        )
        hess = self.bkd().asarray(hess_np)

        # Full-rank dense Laplace
        dense_laplace = DenseLaplacePosterior(
            map_point, prior_precision, hess, self.bkd()
        )
        dense_laplace.compute()
        dense_cov = self.bkd().to_numpy(dense_laplace.posterior_covariance())

        # Low-rank with full rank (nvars)
        def apply_conditioned_hessian(vecs: Array) -> Array:
            L = self.bkd().cholesky(prior_cov)
            return L.T @ hess @ L @ vecs

        low_rank_laplace = LowRankLaplacePosterior(
            map_point,
            prior_sqrt,
            apply_conditioned_hessian,
            nvars,  # Full rank
            self.bkd(),
        )
        low_rank_laplace.compute(noversampling=5, npower_iters=2)
        low_rank_cov = self.bkd().to_numpy(low_rank_laplace.posterior_covariance())

        # Should match closely
        np.testing.assert_array_almost_equal(low_rank_cov, dense_cov, decimal=3)


class TestLowRankLaplacePosteriorValidation(Generic[Array], unittest.TestCase):
    """Test input validation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_rank_exceeds_nvars_raises(self) -> None:
        """Test rank > nvars raises error."""
        nvars = 3
        map_point = self.bkd().zeros((nvars, 1))
        prior_cov = self.bkd().eye(nvars)
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, self.bkd())

        def apply_hess(vecs: Array) -> Array:
            return vecs

        with self.assertRaises(ValueError):
            LowRankLaplacePosterior(
                map_point,
                prior_sqrt,
                apply_hess,
                nvars + 1,  # Rank exceeds nvars
                self.bkd(),
            )

    def test_wrong_map_shape_raises(self) -> None:
        """Test wrong MAP point shape raises error."""
        nvars = 3
        map_point = self.bkd().zeros((nvars + 1, 1))  # Wrong shape
        prior_cov = self.bkd().eye(nvars)
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, self.bkd())

        def apply_hess(vecs: Array) -> Array:
            return vecs

        with self.assertRaises(ValueError):
            LowRankLaplacePosterior(
                map_point,
                prior_sqrt,
                apply_hess,
                2,
                self.bkd(),
            )


# NumPy backend tests
class TestLowRankLaplacePosteriorNumpy(TestLowRankLaplacePosteriorBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestLowRankVsFullRankNumpy(TestLowRankVsFullRank[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestLowRankLaplacePosteriorValidationNumpy(
    TestLowRankLaplacePosteriorValidation[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestLowRankLaplacePosteriorTorch(TestLowRankLaplacePosteriorBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestLowRankVsFullRankTorch(TestLowRankVsFullRank[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestLowRankLaplacePosteriorValidationTorch(
    TestLowRankLaplacePosteriorValidation[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
