"""Tests for KL divergence functions."""

import numpy as np

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

# TODO: use bkd.to_float instead of float(bkd.to_numpy

class TestGaussianCopulaKL:

    def _make_copulas(self, bkd, nvars=3):
        # Copula p
        chol_lower_p = bkd.asarray([0.5, 0.3, 0.4])
        corr_param_p = CholeskyCorrelationParameterization(
            chol_lower_p, nvars, bkd
        )
        copula_p = GaussianCopula(corr_param_p, bkd)

        # Copula q (different)
        chol_lower_q = bkd.asarray([0.3, 0.1, 0.2])
        corr_param_q = CholeskyCorrelationParameterization(
            chol_lower_q, nvars, bkd
        )
        copula_q = GaussianCopula(corr_param_q, bkd)

        return copula_p, copula_q

    def test_kl_same_copula_is_zero(self, bkd) -> None:
        copula_p, _ = self._make_copulas(bkd)
        kl = gaussian_copula_kl_divergence(copula_p, copula_p)
        bkd.assert_allclose(
            bkd.atleast_1d(kl),
            bkd.zeros((1,)),
            atol=1e-10,
        )

    def test_kl_positive(self, bkd) -> None:
        copula_p, copula_q = self._make_copulas(bkd)
        kl = gaussian_copula_kl_divergence(copula_p, copula_q)
        assert float(bkd.to_numpy(kl)) > 0.0

    def test_kl_matches_mvn_kl(self, bkd) -> None:
        """Compare copula KL against DenseCholeskyMultivariateGaussian KL.

        For zero-mean Gaussians, the copula KL equals the MVN KL
        when both have the same covariance (= correlation) matrices.
        """
        nvars = 3
        copula_p, copula_q = self._make_copulas(bkd, nvars)
        sigma_p = copula_p.correlation_param().correlation_matrix()
        sigma_q = copula_q.correlation_param().correlation_matrix()

        mean_zero = bkd.zeros((nvars, 1))
        mvn_p = DenseCholeskyMultivariateGaussian(mean_zero, sigma_p, bkd)
        mvn_q = DenseCholeskyMultivariateGaussian(mean_zero, sigma_q, bkd)

        copula_kl = gaussian_copula_kl_divergence(copula_p, copula_q)
        mvn_kl = mvn_p.kl_divergence(mvn_q)

        bkd.assert_allclose(
            bkd.atleast_1d(bkd.asarray(copula_kl)),
            bkd.atleast_1d(bkd.asarray(mvn_kl)),
            rtol=1e-8,
        )

    def test_kl_identity_vs_correlated(self, bkd) -> None:
        """KL from identity to correlated should be a known positive value."""
        nvars = 3
        copula_p, _ = self._make_copulas(bkd, nvars)

        # Create identity copula
        nparams = nvars * (nvars - 1) // 2
        chol_lower_id = bkd.zeros((nparams,))
        corr_param_id = CholeskyCorrelationParameterization(
            chol_lower_id, nvars, bkd
        )
        copula_id = GaussianCopula(corr_param_id, bkd)

        # KL(identity || correlated) should be positive
        kl = gaussian_copula_kl_divergence(copula_id, copula_p)
        assert float(bkd.to_numpy(kl)) > 0.0

        # KL(correlated || identity) should also be positive
        kl_rev = gaussian_copula_kl_divergence(copula_p, copula_id)
        assert float(bkd.to_numpy(kl_rev)) > 0.0

        # Verify against analytical formula with identity as q:
        # KL(p || I) = 0.5 * (tr(I * Sigma_p) - d + 0 - log|Sigma_p|)
        #            = 0.5 * (tr(Sigma_p) - d - log|Sigma_p|)
        sigma_p = copula_p.correlation_param().correlation_matrix()
        log_det_p = copula_p.correlation_param().log_det()
        0.5 * (bkd.sum(bkd.get_diagonal(sigma_p)) - nvars - log_det_p)
        # For correlation matrix, tr(Sigma_p) = d, so this simplifies to
        # KL(p || I) = 0.5 * (d - d - log|Sigma_p|) = -0.5 * log|Sigma_p|
        expected_simple = -0.5 * log_det_p
        bkd.assert_allclose(
            bkd.atleast_1d(kl_rev),
            bkd.atleast_1d(expected_simple),
            rtol=1e-10,
        )

    def test_kl_vs_mvn_with_scaled_covariance(self, bkd) -> None:
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

        chol_p = bkd.asarray([rho_p])
        corr_p = CholeskyCorrelationParameterization(chol_p, nvars, bkd)
        cop_p = GaussianCopula(corr_p, bkd)

        chol_q = bkd.asarray([rho_q])
        corr_q = CholeskyCorrelationParameterization(chol_q, nvars, bkd)
        cop_q = GaussianCopula(corr_q, bkd)

        copula_kl = gaussian_copula_kl_divergence(cop_p, cop_q)

        # Build MVNs with same mean and Cov = D @ Sigma @ D
        sigma_p_np = bkd.to_numpy(corr_p.correlation_matrix())
        sigma_q_np = bkd.to_numpy(corr_q.correlation_matrix())
        D = np.diag(stdevs)
        cov_p = D @ sigma_p_np @ D
        cov_q = D @ sigma_q_np @ D
        mean_arr = bkd.asarray(np.array(means).reshape(nvars, 1))

        mvn_p = DenseCholeskyMultivariateGaussian(
            mean_arr, bkd.asarray(cov_p), bkd
        )
        mvn_q = DenseCholeskyMultivariateGaussian(
            mean_arr, bkd.asarray(cov_q), bkd
        )
        mvn_kl = mvn_p.kl_divergence(mvn_q)

        bkd.assert_allclose(
            bkd.atleast_1d(bkd.asarray(copula_kl)),
            bkd.atleast_1d(bkd.asarray(mvn_kl)),
            rtol=1e-8,
        )
