"""Tests for KL divergence functions."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.probability.copula.kl_divergence import (
    gaussian_copula_kl_divergence,
)
from pyapprox.probability.gaussian.dense import (
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestGaussianCopulaKL(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._nvars = 3

        # Copula p
        chol_lower_p = self._bkd.asarray([0.5, 0.3, 0.4])
        corr_param_p = CholeskyCorrelationParameterization(
            chol_lower_p, self._nvars, self._bkd
        )
        self._copula_p = GaussianCopula(corr_param_p, self._bkd)

        # Copula q (different)
        chol_lower_q = self._bkd.asarray([0.3, 0.1, 0.2])
        corr_param_q = CholeskyCorrelationParameterization(
            chol_lower_q, self._nvars, self._bkd
        )
        self._copula_q = GaussianCopula(corr_param_q, self._bkd)

    def test_kl_same_copula_is_zero(self) -> None:
        kl = gaussian_copula_kl_divergence(self._copula_p, self._copula_p)
        self._bkd.assert_allclose(
            self._bkd.atleast_1d(kl),
            self._bkd.zeros((1,)),
            atol=1e-10,
        )

    def test_kl_positive(self) -> None:
        kl = gaussian_copula_kl_divergence(self._copula_p, self._copula_q)
        self.assertTrue(float(self._bkd.to_numpy(kl)) > 0.0)

    def test_kl_matches_mvn_kl(self) -> None:
        """Compare copula KL against DenseCholeskyMultivariateGaussian KL.

        For zero-mean Gaussians, the copula KL equals the MVN KL
        when both have the same covariance (= correlation) matrices.
        """
        sigma_p = self._copula_p.correlation_param().correlation_matrix()
        sigma_q = self._copula_q.correlation_param().correlation_matrix()

        mean_zero = self._bkd.zeros((self._nvars, 1))
        mvn_p = DenseCholeskyMultivariateGaussian(mean_zero, sigma_p, self._bkd)
        mvn_q = DenseCholeskyMultivariateGaussian(mean_zero, sigma_q, self._bkd)

        copula_kl = gaussian_copula_kl_divergence(self._copula_p, self._copula_q)
        mvn_kl = mvn_p.kl_divergence(mvn_q)

        self._bkd.assert_allclose(
            self._bkd.atleast_1d(self._bkd.asarray(copula_kl)),
            self._bkd.atleast_1d(self._bkd.asarray(mvn_kl)),
            rtol=1e-8,
        )

    def test_kl_identity_vs_correlated(self) -> None:
        """KL from identity to correlated should be a known positive value."""
        # Create identity copula
        nparams = self._nvars * (self._nvars - 1) // 2
        chol_lower_id = self._bkd.zeros((nparams,))
        corr_param_id = CholeskyCorrelationParameterization(
            chol_lower_id, self._nvars, self._bkd
        )
        copula_id = GaussianCopula(corr_param_id, self._bkd)

        # KL(identity || correlated) should be positive
        kl = gaussian_copula_kl_divergence(copula_id, self._copula_p)
        self.assertTrue(float(self._bkd.to_numpy(kl)) > 0.0)

        # KL(correlated || identity) should also be positive
        kl_rev = gaussian_copula_kl_divergence(self._copula_p, copula_id)
        self.assertTrue(float(self._bkd.to_numpy(kl_rev)) > 0.0)

        # Verify against analytical formula with identity as q:
        # KL(p || I) = 0.5 * (tr(I * Sigma_p) - d + 0 - log|Sigma_p|)
        #            = 0.5 * (tr(Sigma_p) - d - log|Sigma_p|)
        sigma_p = self._copula_p.correlation_param().correlation_matrix()
        log_det_p = self._copula_p.correlation_param().log_det()
        0.5 * (self._bkd.sum(self._bkd.get_diagonal(sigma_p)) - self._nvars - log_det_p)
        # For correlation matrix, tr(Sigma_p) = d, so this simplifies to
        # KL(p || I) = 0.5 * (d - d - log|Sigma_p|) = -0.5 * log|Sigma_p|
        expected_simple = -0.5 * log_det_p
        self._bkd.assert_allclose(
            self._bkd.atleast_1d(kl_rev),
            self._bkd.atleast_1d(expected_simple),
            rtol=1e-10,
        )

    def test_kl_vs_mvn_with_scaled_covariance(self) -> None:
        """Cross-validate: copula KL matches MVN KL with Cov = D @ Sigma @ D.

        For Gaussian copula + Gaussian marginals with shared marginals,
        gaussian_copula_kl_divergence(c_p, c_q) should match the MVN KL
        computed from full covariances D @ Sigma_p @ D and D @ Sigma_q @ D
        (since the mean-dependent terms cancel when means are equal).
        """
        nvars = 2
        stdevs = [2.0, 0.5]
        means = [1.0, -1.0]

        rho_p = 0.6
        rho_q = 0.3

        chol_p = self._bkd.asarray([rho_p])
        corr_p = CholeskyCorrelationParameterization(chol_p, nvars, self._bkd)
        cop_p = GaussianCopula(corr_p, self._bkd)

        chol_q = self._bkd.asarray([rho_q])
        corr_q = CholeskyCorrelationParameterization(chol_q, nvars, self._bkd)
        cop_q = GaussianCopula(corr_q, self._bkd)

        copula_kl = gaussian_copula_kl_divergence(cop_p, cop_q)

        # Build MVNs with same mean and Cov = D @ Sigma @ D
        sigma_p_np = self._bkd.to_numpy(corr_p.correlation_matrix())
        sigma_q_np = self._bkd.to_numpy(corr_q.correlation_matrix())
        D = np.diag(stdevs)
        cov_p = D @ sigma_p_np @ D
        cov_q = D @ sigma_q_np @ D
        mean_arr = self._bkd.asarray(np.array(means).reshape(nvars, 1))

        mvn_p = DenseCholeskyMultivariateGaussian(
            mean_arr, self._bkd.asarray(cov_p), self._bkd
        )
        mvn_q = DenseCholeskyMultivariateGaussian(
            mean_arr, self._bkd.asarray(cov_q), self._bkd
        )
        mvn_kl = mvn_p.kl_divergence(mvn_q)

        self._bkd.assert_allclose(
            self._bkd.atleast_1d(self._bkd.asarray(copula_kl)),
            self._bkd.atleast_1d(self._bkd.asarray(mvn_kl)),
            rtol=1e-8,
        )


class TestGaussianCopulaKLNumpy(TestGaussianCopulaKL[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianCopulaKLTorch(TestGaussianCopulaKL[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
