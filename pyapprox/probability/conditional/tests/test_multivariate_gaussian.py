"""
Tests for ConditionalDenseCholGaussian and ConditionalLowRankCholGaussian.
"""

import math

import numpy as np

from pyapprox.probability.conditional.multivariate_gaussian import (
    ConditionalDenseCholGaussian,
    ConditionalLowRankCholGaussian,
)
from pyapprox.probability.gaussian.dense import (
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.surrogates.affine.expansions.pce import (
    create_pce_from_marginals,
)

# TODO: Fix typing issues
# TODO: add short docstring to each test
# TODO: remove use of np. and replace with bkd. except np.random

def _make_expansion(bkd, nvars_in, degree, nqoi, coeff=0.0):
    """Create a BasisExpansion with given params."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars_in)]
    exp = create_pce_from_marginals(marginals, degree, bkd, nqoi=nqoi)
    nterms = exp.nterms()
    coeffs = np.zeros((nterms, nqoi))
    coeffs[0, :] = coeff
    exp.set_coefficients(bkd.asarray(coeffs))
    return exp


def _make_dense_chol(bkd, nvars_in, d, degree, mean=0.0, log_diag=0.0):
    """Create a ConditionalDenseCholGaussian."""
    mean_func = _make_expansion(bkd, nvars_in, degree, nqoi=d, coeff=mean)
    log_chol_diag_func = _make_expansion(
        bkd,
        nvars_in,
        degree,
        nqoi=d,
        coeff=log_diag,
    )
    n_offdiag = d * (d - 1) // 2
    chol_offdiag_func = None
    if d > 1:
        chol_offdiag_func = _make_expansion(
            bkd,
            nvars_in,
            degree,
            nqoi=n_offdiag,
            coeff=0.0,
        )
    return ConditionalDenseCholGaussian(
        mean_func,
        log_chol_diag_func,
        chol_offdiag_func,
        bkd,
    )


def _make_low_rank(bkd, nvars_in, d, rank, degree, mean=0.0, log_diag=0.0):
    """Create a ConditionalLowRankCholGaussian."""
    mean_func = _make_expansion(bkd, nvars_in, degree, nqoi=d, coeff=mean)
    log_diag_func = _make_expansion(
        bkd,
        nvars_in,
        degree,
        nqoi=d,
        coeff=log_diag,
    )
    factor_func = None
    if rank > 0:
        factor_func = _make_expansion(
            bkd,
            nvars_in,
            degree,
            nqoi=d * rank,
            coeff=0.0,
        )
    return ConditionalLowRankCholGaussian(
        mean_func,
        log_diag_func,
        factor_func,
        rank,
        bkd,
    )


class TestDenseCholBase:
    """Tests for ConditionalDenseCholGaussian."""

    def test_basic_properties(self, bkd) -> None:
        cond = _make_dense_chol(bkd, nvars_in=2, d=3, degree=1)
        assert cond.nvars() == 2
        assert cond.nqoi() == 3
        assert hasattr(cond, "hyp_list")

    def test_logpdf_shape(self, bkd) -> None:
        cond = _make_dense_chol(bkd, nvars_in=1, d=2, degree=0)
        x = bkd.zeros((1, 5))
        y = bkd.zeros((2, 5))
        result = cond.logpdf(x, y)
        assert result.shape == (1, 5)

    def test_rvs_shape(self, bkd) -> None:
        cond = _make_dense_chol(bkd, nvars_in=1, d=2, degree=0)
        x = bkd.zeros((1, 10))
        np.random.seed(42)
        y = cond.rvs(x)
        assert y.shape == (2, 10)

    def test_reparameterize_shape(self, bkd) -> None:
        cond = _make_dense_chol(bkd, nvars_in=1, d=2, degree=0)
        x = bkd.zeros((1, 5))
        base = bkd.zeros((2, 5))
        z = cond.reparameterize(x, base)
        assert z.shape == (2, 5)

    def test_kl_divergence_shape(self, bkd) -> None:
        d = 2
        cond = _make_dense_chol(bkd, nvars_in=1, d=d, degree=0)
        prior = DenseCholeskyMultivariateGaussian(
            bkd.zeros((d, 1)),
            bkd.eye(d),
            bkd,
        )
        x = bkd.zeros((1, 5))
        kl = cond.kl_divergence(x, prior)
        assert kl.shape == (1, 5)

    def test_d1_logpdf_matches_univariate(self, bkd) -> None:
        """d=1 with constant params matches 1D Gaussian logpdf."""
        mean_val = 1.5
        log_stdev_val = 0.3
        stdev = math.exp(log_stdev_val)
        cond = _make_dense_chol(
            bkd,
            nvars_in=1,
            d=1,
            degree=0,
            mean=mean_val,
            log_diag=log_stdev_val,
        )
        x = bkd.zeros((1, 3))
        y = bkd.asarray([[0.0, 1.0, 2.0]])
        result = cond.logpdf(x, y)

        # Manual computation
        for i, yi in enumerate([0.0, 1.0, 2.0]):
            z = (yi - mean_val) / stdev
            expected = -0.5 * math.log(2 * math.pi) - log_stdev_val - 0.5 * z**2
            bkd.assert_allclose(
                bkd.asarray([float(result[0, i])]),
                bkd.asarray([expected]),
                rtol=1e-10,
            )

    def test_d2_logpdf_constant_params(self, bkd) -> None:
        """d=2 with constant params matches multivariate_normal logpdf."""
        from scipy.stats import multivariate_normal

        mean = np.array([1.0, -0.5])
        L = np.array([[1.5, 0.0], [0.3, 0.8]])
        cov = L @ L.T

        cond = _make_dense_chol(bkd, nvars_in=1, d=2, degree=0)
        # Set mean coefficients
        cond._mean_func.set_coefficients(bkd.asarray(mean.reshape(1, 2)))
        # Set log_chol_diag coefficients
        log_diag = np.log(np.diag(L))
        cond._log_chol_diag_func.set_coefficients(bkd.asarray(log_diag.reshape(1, 2)))
        # Set offdiag coefficient
        cond._chol_offdiag_func.set_coefficients(bkd.asarray(np.array([[L[1, 0]]])))

        x = bkd.zeros((1, 4))
        y_np = np.array([[0.0, 1.0, 2.0, -1.0], [0.5, -0.5, 0.0, 1.0]])
        y = bkd.asarray(y_np)
        result = cond.logpdf(x, y)

        for i in range(4):
            expected = multivariate_normal.logpdf(y_np[:, i], mean, cov)
            bkd.assert_allclose(
                bkd.asarray([float(result[0, i])]),
                bkd.asarray([expected]),
                rtol=1e-10,
            )

    def test_kl_matches_fixed_gaussian(self, bkd) -> None:
        """KL with constant params matches
        DenseCholeskyMultivariateGaussian.kl_divergence."""
        d = 2
        mean_q = np.array([0.5, -0.3])
        L_q = np.array([[1.2, 0.0], [0.4, 0.9]])
        cov_q = L_q @ L_q.T

        cond = _make_dense_chol(bkd, nvars_in=1, d=d, degree=0)
        cond._mean_func.set_coefficients(bkd.asarray(mean_q.reshape(1, 2)))
        cond._log_chol_diag_func.set_coefficients(
            bkd.asarray(np.log(np.diag(L_q)).reshape(1, 2))
        )
        cond._chol_offdiag_func.set_coefficients(bkd.asarray(np.array([[L_q[1, 0]]])))

        mean_p = np.array([0.0, 0.0])
        cov_p = np.array([[1.0, 0.2], [0.2, 1.5]])
        prior = DenseCholeskyMultivariateGaussian(
            bkd.asarray(mean_p.reshape(2, 1)),
            bkd.asarray(cov_p),
            bkd,
        )

        x = bkd.zeros((1, 3))
        kl = cond.kl_divergence(x, prior)

        # Reference KL
        q_fixed = DenseCholeskyMultivariateGaussian(
            bkd.asarray(mean_q.reshape(2, 1)),
            bkd.asarray(cov_q),
            bkd,
        )
        kl_ref = q_fixed.kl_divergence(prior)

        # All samples should match (constant params)
        for i in range(3):
            bkd.assert_allclose(
                bkd.asarray([float(kl[0, i])]),
                bkd.asarray([float(kl_ref)]),
                rtol=1e-10,
            )

    def test_kl_self_is_zero(self, bkd) -> None:
        """KL(q || q) = 0 when variational matches prior."""
        d = 2
        # Prior = N(0, I)
        prior = DenseCholeskyMultivariateGaussian(
            bkd.zeros((d, 1)),
            bkd.eye(d),
            bkd,
        )
        # Variational with mean=0, L=I (log_diag=0, offdiag=0)
        cond = _make_dense_chol(bkd, nvars_in=1, d=d, degree=0)
        x = bkd.zeros((1, 3))
        kl = cond.kl_divergence(x, prior)

        bkd.assert_allclose(kl, bkd.zeros((1, 3)), atol=1e-12)

    def test_reparameterize_matches_manual(self, bkd) -> None:
        """z = mu + L @ eps for constant params."""
        d = 2
        mean = np.array([1.0, -0.5])
        L = np.array([[1.5, 0.0], [0.3, 0.8]])

        cond = _make_dense_chol(bkd, nvars_in=1, d=d, degree=0)
        cond._mean_func.set_coefficients(bkd.asarray(mean.reshape(1, 2)))
        cond._log_chol_diag_func.set_coefficients(
            bkd.asarray(np.log(np.diag(L)).reshape(1, 2))
        )
        cond._chol_offdiag_func.set_coefficients(bkd.asarray(np.array([[L[1, 0]]])))

        x = bkd.zeros((1, 3))
        eps = bkd.asarray(np.array([[1.0, 0.0, -1.0], [0.5, 1.0, -0.5]]))
        z = cond.reparameterize(x, eps)

        expected = bkd.asarray(mean.reshape(2, 1)) + bkd.asarray(L) @ eps
        bkd.assert_allclose(z, expected, rtol=1e-10)

    def test_base_distribution(self, bkd) -> None:
        cond = _make_dense_chol(bkd, nvars_in=1, d=2, degree=0)
        base = cond.base_distribution()
        assert base.nvars() == 2


class TestLowRankBase:
    """Tests for ConditionalLowRankCholGaussian."""
    
    def test_basic_properties(self, bkd) -> None:
        cond = _make_low_rank(bkd, nvars_in=2, d=3, rank=1, degree=1)
        assert cond.nvars() == 2
        assert cond.nqoi() == 3
        assert cond.rank() == 1
        assert hasattr(cond, "hyp_list")

    def test_shapes(self, bkd) -> None:
        d = 2
        r = 1
        cond = _make_low_rank(bkd, nvars_in=1, d=d, rank=r, degree=0)
        prior = DenseCholeskyMultivariateGaussian(
            bkd.zeros((d, 1)),
            bkd.eye(d),
            bkd,
        )
        x = bkd.zeros((1, 5))
        y = bkd.zeros((d, 5))
        base = bkd.zeros((d, 5))

        assert cond.logpdf(x, y).shape == (1, 5)
        assert cond.reparameterize(x, base).shape == (d, 5)
        assert cond.kl_divergence(x, prior).shape == (1, 5)

        np.random.seed(42)
        assert cond.rvs(x).shape == (d, 5)

    def test_rank0_is_diagonal(self, bkd) -> None:
        """rank=0 gives diagonal covariance: logpdf matches independent Gaussians."""
        d = 2
        mean = np.array([1.0, -0.5])
        log_diag = np.array([0.3, -0.2])
        stdevs = np.exp(log_diag)

        cond = _make_low_rank(bkd, nvars_in=1, d=d, rank=0, degree=0)
        cond._mean_func.set_coefficients(bkd.asarray(mean.reshape(1, 2)))
        cond._log_diag_func.set_coefficients(bkd.asarray(log_diag.reshape(1, 2)))

        x = bkd.zeros((1, 3))
        y = bkd.asarray(np.array([[0.0, 1.0, 2.0], [0.5, -0.5, 0.0]]))
        result = cond.logpdf(x, y)

        # Manual: sum of independent 1D Gaussian logpdfs
        for i in range(3):
            logp = 0.0
            for j in range(d):
                z = (float(y[j, i]) - mean[j]) / stdevs[j]
                logp += -0.5 * math.log(2 * math.pi) - log_diag[j] - 0.5 * z**2
            bkd.assert_allclose(
                bkd.asarray([float(result[0, i])]),
                bkd.asarray([logp]),
                rtol=1e-10,
            )

    def test_rank0_kl_self_zero(self, bkd) -> None:
        """rank=0, q = prior -> KL = 0."""
        d = 2
        prior = DenseCholeskyMultivariateGaussian(
            bkd.zeros((d, 1)),
            bkd.eye(d),
            bkd,
        )
        cond = _make_low_rank(bkd, nvars_in=1, d=d, rank=0, degree=0)
        x = bkd.zeros((1, 3))
        kl = cond.kl_divergence(x, prior)
        bkd.assert_allclose(kl, bkd.zeros((1, 3)), atol=1e-12)

    def test_rankd_recovers_full_covariance_kl(self, bkd) -> None:
        """rank=d can match any target covariance via D^2 + VV^T.

        Set up D and V so that D^2 + VV^T matches a known full covariance,
        then verify KL matches the DenseChol version.
        """
        d = 2
        # Target covariance
        cov_target = np.array([[1.44, 0.36], [0.36, 0.97]])
        # Decompose: cov = L L^T
        L_target = np.linalg.cholesky(cov_target)
        # Choose D = diag(L), then VV^T = cov - D^2 = offdiag part
        D_diag = np.diag(L_target)
        D2 = np.diag(D_diag**2)
        cov_target - D2
        # remainder = V V^T where V is (d, d)
        # Use Cholesky of remainder (it's PSD since cov > D^2 elementwise)
        # Actually remainder may not be PSD. Let's use a different approach.
        # Set D small and V = L_target
        D_diag = np.array([0.01, 0.01])  # small D
        V = L_target  # (d, d)
        # cov = 0.0001*I + L L^T = cov_target

        mean_q = np.array([0.5, -0.3])

        cond = _make_low_rank(bkd, nvars_in=1, d=d, rank=d, degree=0)
        cond._mean_func.set_coefficients(bkd.asarray(mean_q.reshape(1, 2)))
        cond._log_diag_func.set_coefficients(bkd.asarray(np.log(D_diag).reshape(1, 2)))
        # factor_func: nqoi = d*rank = 4, flat V column-major
        # V_batch = reshape(flat_V.T, (N, d, r)) so flat_V is (d*r, N)
        # flat_V[:, n] = V.flatten() in row-major order since reshape is (N, d, r)
        V_flat = V.flatten()  # row-major: [V[0,0], V[0,1], V[1,0], V[1,1]]
        cond._factor_func.set_coefficients(bkd.asarray(V_flat.reshape(1, d * d)))

        mean_p = np.array([0.0, 0.0])
        cov_p = np.array([[1.0, 0.2], [0.2, 1.5]])
        prior = DenseCholeskyMultivariateGaussian(
            bkd.asarray(mean_p.reshape(2, 1)),
            bkd.asarray(cov_p),
            bkd,
        )

        x = bkd.zeros((1, 1))
        kl_lr = cond.kl_divergence(x, prior)

        # Reference: exact cov = D^2 + VV^T
        cov_q_actual = np.diag(D_diag**2) + V @ V.T
        q_fixed = DenseCholeskyMultivariateGaussian(
            bkd.asarray(mean_q.reshape(2, 1)),
            bkd.asarray(cov_q_actual),
            bkd,
        )
        kl_ref = q_fixed.kl_divergence(prior)

        bkd.assert_allclose(
            bkd.asarray([float(kl_lr[0, 0])]),
            bkd.asarray([float(kl_ref)]),
            rtol=1e-8,
        )

    def test_reparameterize_rank0(self, bkd) -> None:
        """rank=0: z = mu + D * eps."""
        d = 2
        mean = np.array([1.0, -0.5])
        log_diag = np.array([0.3, -0.2])
        stdevs = np.exp(log_diag)

        cond = _make_low_rank(bkd, nvars_in=1, d=d, rank=0, degree=0)
        cond._mean_func.set_coefficients(bkd.asarray(mean.reshape(1, 2)))
        cond._log_diag_func.set_coefficients(bkd.asarray(log_diag.reshape(1, 2)))

        x = bkd.zeros((1, 3))
        eps = bkd.asarray(np.array([[1.0, 0.0, -1.0], [0.5, 1.0, -0.5]]))
        z = cond.reparameterize(x, eps)

        expected = bkd.asarray(mean.reshape(2, 1)) + bkd.asarray(np.diag(stdevs)) @ eps
        bkd.assert_allclose(z, expected, rtol=1e-10)

    def test_base_distribution(self, bkd) -> None:
        cond = _make_low_rank(bkd, nvars_in=1, d=2, rank=1, degree=0)
        base = cond.base_distribution()
        assert base.nvars() == 2  # d, not d + rank
