"""
Tests for multivariate Gaussian distributions.
"""

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    OperatorBasedCovarianceOperator,
)
from pyapprox.probability.gaussian import (
    DenseCholeskyMultivariateGaussian,
    DiagonalMultivariateGaussian,
    GaussianCanonicalForm,
    GaussianLogPDFCore,
    OperatorBasedMultivariateGaussian,
)


class TestGaussianLogPDFCore:
    """Tests for GaussianLogPDFCore."""

    def _setup(self, bkd):
        cov = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        cov_op = DenseCholeskyCovarianceOperator(cov, bkd)
        core = GaussianLogPDFCore(cov_op, bkd)
        return cov, cov_op, core

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, _, core = self._setup(bkd)
        assert core.nvars() == 2

    def test_compute_at_zero(self, bkd) -> None:
        """Test log-PDF at zero residuals."""
        _, _, core = self._setup(bkd)
        residuals = bkd.zeros((2, 1))
        logpdf = core.compute(residuals)
        expected = core.log_normalization_constant()
        assert bkd.allclose(
            bkd.asarray([logpdf[0]]),
            bkd.asarray([expected]),
            rtol=1e-6,
        )

    def test_compute_shape(self, bkd) -> None:
        """Test compute() returns (1, nsamples) not (nsamples,)."""
        _, _, core = self._setup(bkd)
        residuals = bkd.zeros((2, 5))
        logpdf = core.compute(residuals)
        assert logpdf.shape == (1, 5)

    def test_compute_shape_single_sample(self, bkd) -> None:
        """Test compute() returns (1, 1) for single sample."""
        _, _, core = self._setup(bkd)
        residuals = bkd.zeros((2, 1))
        logpdf = core.compute(residuals)
        assert logpdf.shape == (1, 1)

    def test_compute_gradient_at_zero(self, bkd) -> None:
        """Test gradient at zero residuals is zero."""
        _, _, core = self._setup(bkd)
        residuals = bkd.zeros((2, 3))
        gradient = core.compute_gradient(residuals)
        expected = bkd.zeros((2, 3))
        assert bkd.allclose(gradient, expected, atol=1e-10)


class TestDenseCholeskyMultivariateGaussian:
    """Tests for DenseCholeskyMultivariateGaussian."""

    def _setup(self, bkd):
        mean = bkd.asarray([[1.0], [2.0]])
        cov = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        dist = DenseCholeskyMultivariateGaussian(mean, cov, bkd)
        return mean, cov, dist

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, _, dist = self._setup(bkd)
        assert dist.nvars() == 2

    def test_mean(self, bkd) -> None:
        """Test mean returns correct value."""
        mean, _, dist = self._setup(bkd)
        assert bkd.allclose(dist.mean(), mean, atol=1e-10)

    def test_covariance(self, bkd) -> None:
        """Test covariance returns correct value."""
        _, cov, dist = self._setup(bkd)
        assert bkd.allclose(dist.covariance(), cov, atol=1e-10)

    def test_covariance_inverse(self, bkd) -> None:
        """Test covariance_inverse is inverse."""
        _, cov, dist = self._setup(bkd)
        identity = cov @ dist.covariance_inverse()
        expected = bkd.eye(2)
        assert bkd.allclose(identity, expected, atol=1e-10)

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, _, dist = self._setup(bkd)
        samples = dist.rvs(100)
        assert samples.shape == (2, 100)

    def test_logpdf_vs_scipy(self, bkd) -> None:
        """Test logpdf matches scipy.stats."""
        mean, cov, dist = self._setup(bkd)
        scipy_dist = stats.multivariate_normal(
            mean=bkd.to_numpy(mean).flatten(),
            cov=bkd.to_numpy(cov),
        )
        samples = bkd.asarray([[1.0, 0.5, 2.0], [2.0, 1.5, 3.0]])
        logpdf_ours = dist.logpdf(samples)
        logpdf_scipy = bkd.asarray(
            scipy_dist.logpdf(bkd.to_numpy(samples).T)
        )
        assert bkd.allclose(logpdf_ours, logpdf_scipy, rtol=1e-6)

    def test_pdf_exp_logpdf(self, bkd) -> None:
        """Test pdf = exp(logpdf)."""
        _, _, dist = self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0], [2.0, 3.0]])
        pdf_vals = dist.pdf(samples)
        logpdf_vals = dist.logpdf(samples)
        assert bkd.allclose(pdf_vals, bkd.exp(logpdf_vals), rtol=1e-6)

    def test_logpdf_gradient_finite_diff(self, bkd) -> None:
        """Test logpdf gradient against finite differences."""
        _, _, dist = self._setup(bkd)
        sample = bkd.asarray([[1.5], [2.5]])
        grad = dist.logpdf_gradient(sample)

        eps = 1e-6
        grad_fd = bkd.zeros((2, 1))
        for i in range(2):
            sample_plus = bkd.copy(sample)
            sample_plus[i] = sample_plus[i] + eps
            sample_minus = bkd.copy(sample)
            sample_minus[i] = sample_minus[i] - eps
            diff = (
                float(dist.logpdf(sample_plus)[0])
                - float(dist.logpdf(sample_minus)[0])
            ) / (2 * eps)
            grad_fd[i, 0] = diff

        assert bkd.allclose(grad, grad_fd, rtol=1e-5)

    def test_kl_divergence_same_dist(self, bkd) -> None:
        """Test KL divergence of distribution with itself is zero."""
        _, _, dist = self._setup(bkd)
        kl = dist.kl_divergence(dist)
        assert bkd.allclose(
            bkd.asarray([kl]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_kl_divergence_positive(self, bkd) -> None:
        """Test KL divergence is positive for different distributions."""
        _, cov, dist = self._setup(bkd)
        other = DenseCholeskyMultivariateGaussian(
            bkd.asarray([[0.0], [0.0]]), cov, bkd
        )
        kl = dist.kl_divergence(other)
        assert float(kl) > 0.0

    def test_invalid_mean_shape(self, bkd) -> None:
        """Test invalid mean shape raises error."""
        _, cov, _ = self._setup(bkd)
        with pytest.raises(ValueError):
            DenseCholeskyMultivariateGaussian(
                bkd.asarray([1.0, 2.0]), cov, bkd
            )

    def test_invalid_cov_shape(self, bkd) -> None:
        """Test invalid covariance shape raises error."""
        mean, _, _ = self._setup(bkd)
        with pytest.raises(ValueError):
            DenseCholeskyMultivariateGaussian(mean, bkd.eye(3), bkd)

    def test_logpdf_shape(self, bkd) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        _, _, dist = self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        result = dist.logpdf(samples)
        assert result.shape == (1, 3)

    def test_pdf_shape(self, bkd) -> None:
        """Test pdf returns (1, nsamples) not (nsamples,)."""
        _, _, dist = self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        result = dist.pdf(samples)
        assert result.shape == (1, 3)


class TestDiagonalMultivariateGaussian:
    """Tests for DiagonalMultivariateGaussian."""

    def _setup(self, bkd):
        mean = bkd.asarray([[1.0], [2.0], [3.0]])
        variances = bkd.asarray([1.0, 4.0, 9.0])
        dist = DiagonalMultivariateGaussian(mean, variances, bkd)
        return mean, variances, dist

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, _, dist = self._setup(bkd)
        assert dist.nvars() == 3

    def test_variances(self, bkd) -> None:
        """Test variances returns correct value."""
        _, variances, dist = self._setup(bkd)
        assert bkd.allclose(dist.variances(), variances, atol=1e-10)

    def test_standard_deviations(self, bkd) -> None:
        """Test standard_deviations."""
        _, variances, dist = self._setup(bkd)
        expected = bkd.sqrt(variances)
        assert bkd.allclose(dist.standard_deviations(), expected, atol=1e-10)

    def test_covariance(self, bkd) -> None:
        """Test covariance returns diagonal matrix."""
        _, variances, dist = self._setup(bkd)
        cov = dist.covariance()
        expected = bkd.diag(variances)
        assert bkd.allclose(cov, expected, atol=1e-10)

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, _, dist = self._setup(bkd)
        samples = dist.rvs(100)
        assert samples.shape == (3, 100)

    def test_logpdf_vs_product_of_univariates(self, bkd) -> None:
        """Test logpdf equals sum of univariate log-pdfs."""
        mean, variances, dist = self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

        logpdf_ours = dist.logpdf(samples)

        # Compute as product of independent univariates
        samples_np = bkd.to_numpy(samples)
        mean_np = bkd.to_numpy(mean)
        var_np = bkd.to_numpy(variances)
        logpdf_expected = np.zeros(2)
        for i in range(3):
            univariate = stats.norm(mean_np[i, 0], np.sqrt(var_np[i]))
            logpdf_expected += univariate.logpdf(samples_np[i])

        assert bkd.allclose(
            logpdf_ours, bkd.asarray(logpdf_expected), rtol=1e-6
        )

    def test_kl_divergence_same_dist(self, bkd) -> None:
        """Test KL divergence of distribution with itself is zero."""
        _, _, dist = self._setup(bkd)
        kl = dist.kl_divergence(dist)
        assert bkd.allclose(
            bkd.asarray([kl]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_logpdf_shape(self, bkd) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        _, _, dist = self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        result = dist.logpdf(samples)
        assert result.shape == (1, 2)

    def test_pdf_shape(self, bkd) -> None:
        """Test pdf returns (1, nsamples) not (nsamples,)."""
        _, _, dist = self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        result = dist.pdf(samples)
        assert result.shape == (1, 2)


class TestOperatorBasedMultivariateGaussian:
    """Tests for OperatorBasedMultivariateGaussian."""

    def _setup(self, bkd):
        nvars = 3
        scale = 2.0
        mean = bkd.zeros((nvars, 1))

        # Create scaling operator: L = scale * I
        cov_op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: scale * x,
            apply_sqrt_inv=lambda x: x / scale,
            log_determinant=nvars * float(np.log(scale)),
            nvars=nvars,
            bkd=bkd,
        )
        dist = OperatorBasedMultivariateGaussian(mean, cov_op, bkd)
        return nvars, scale, mean, cov_op, dist

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, _, _, _, dist = self._setup(bkd)
        assert dist.nvars() == 3

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, _, _, _, dist = self._setup(bkd)
        samples = dist.rvs(100)
        assert samples.shape == (3, 100)

    def test_logpdf_at_mean(self, bkd) -> None:
        """Test logpdf at mean is maximum."""
        _, _, _, _, dist = self._setup(bkd)
        samples = bkd.asarray(
            [[0.0, 1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, -1.0]]
        )
        logpdf = dist.logpdf(samples)
        # logpdf has shape (1, nsamples), at mean (column 0) should be maximum
        assert float(logpdf[0, 0]) > float(logpdf[0, 1])
        assert float(logpdf[0, 0]) > float(logpdf[0, 2])

    def test_covariance_diagonal(self, bkd) -> None:
        """Test covariance diagonal computation."""
        nvars, scale, _, _, dist = self._setup(bkd)
        diagonal = dist.covariance_diagonal()
        expected = bkd.full((nvars,), scale**2)
        assert bkd.allclose(diagonal, expected, rtol=1e-6)

    def test_invalid_mean_cov_mismatch(self, bkd) -> None:
        """Test mismatched mean and cov_op dimensions."""
        _, _, _, cov_op, _ = self._setup(bkd)
        wrong_mean = bkd.zeros((5, 1))
        with pytest.raises(ValueError):
            OperatorBasedMultivariateGaussian(wrong_mean, cov_op, bkd)

    def test_logpdf_shape(self, bkd) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        _, _, _, _, dist = self._setup(bkd)
        samples = bkd.asarray(
            [[0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5]]
        )
        result = dist.logpdf(samples)
        assert result.shape == (1, 4)

    def test_pdf_shape(self, bkd) -> None:
        """Test pdf returns (1, nsamples) not (nsamples,)."""
        _, _, _, _, dist = self._setup(bkd)
        samples = bkd.asarray(
            [[0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5]]
        )
        result = dist.pdf(samples)
        assert result.shape == (1, 4)


class TestCrossValidation:
    """Cross-validation tests between implementations."""

    def _setup(self, bkd):
        mean = bkd.asarray([[1.0], [2.0]])
        cov = bkd.asarray([[4.0, 0.0], [0.0, 9.0]])  # Diagonal
        return mean, cov

    def test_dense_vs_diagonal_for_diagonal_cov(self, bkd) -> None:
        """Test dense and diagonal implementations agree for diagonal cov."""
        mean, cov = self._setup(bkd)
        dense = DenseCholeskyMultivariateGaussian(mean, cov, bkd)
        diagonal = DiagonalMultivariateGaussian(
            mean, bkd.get_diagonal(cov), bkd
        )

        samples = bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

        logpdf_dense = dense.logpdf(samples)
        logpdf_diagonal = diagonal.logpdf(samples)

        assert bkd.allclose(logpdf_dense, logpdf_diagonal, rtol=1e-6)

    def test_dense_vs_operator_for_identity_cov(self, bkd) -> None:
        """Test dense and operator implementations agree for identity cov."""
        mean = bkd.zeros((3, 1))
        cov = bkd.eye(3)

        dense = DenseCholeskyMultivariateGaussian(mean, cov, bkd)
        cov_op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: x,
            apply_sqrt_inv=lambda x: x,
            log_determinant=0.0,
            nvars=3,
            bkd=bkd,
        )
        operator = OperatorBasedMultivariateGaussian(mean, cov_op, bkd)

        # Use deterministic samples
        samples = bkd.asarray(
            [
                [0.1, -0.5, 1.2, 0.3, -0.8],
                [0.2, 0.7, -0.3, 0.9, 0.1],
                [-0.4, 0.2, 0.5, -0.6, 0.4],
            ]
        )

        logpdf_dense = dense.logpdf(samples)
        logpdf_operator = operator.logpdf(samples)

        assert bkd.allclose(logpdf_dense, logpdf_operator, rtol=1e-6)


class TestGaussianCanonicalForm:
    """Tests for GaussianCanonicalForm."""

    def _setup(self, bkd):
        mean = bkd.asarray([1.0, 2.0])
        cov = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        canonical = GaussianCanonicalForm.from_moments(
            mean, cov, bkd
        )
        return mean, cov, canonical

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, _, canonical = self._setup(bkd)
        assert canonical.nvars() == 2

    def test_roundtrip_moments(self, bkd) -> None:
        """Test from_moments -> to_moments roundtrip."""
        mean, cov, canonical = self._setup(bkd)
        mean_recovered, cov_recovered = canonical.to_moments()
        assert bkd.allclose(mean_recovered, mean, rtol=1e-6)
        assert bkd.allclose(cov_recovered, cov, rtol=1e-6)

    def test_logpdf_vs_dense(self, bkd) -> None:
        """Test logpdf matches DenseCholeskyMultivariateGaussian."""
        mean, cov, canonical = self._setup(bkd)
        mean_2d = bkd.reshape(mean, (2, 1))
        dense = DenseCholeskyMultivariateGaussian(mean_2d, cov, bkd)
        samples = bkd.asarray([[1.0, 0.5, 2.0], [2.0, 1.5, 3.0]])

        logpdf_canonical = canonical.logpdf(samples)
        logpdf_dense = dense.logpdf(samples)

        assert bkd.allclose(logpdf_canonical, logpdf_dense, rtol=1e-6)

    def test_multiply_same(self, bkd) -> None:
        """Test multiplying distribution with itself."""
        _, _, canonical = self._setup(bkd)
        product = canonical.multiply(canonical)

        # Precision and shift should double
        assert bkd.allclose(
            product.precision(), 2 * canonical.precision(), rtol=1e-6
        )
        assert bkd.allclose(product.shift(), 2 * canonical.shift(), rtol=1e-6)

    def test_multiply_normalized(self, bkd) -> None:
        """Test normalized product matches analytical result."""
        _, _, canonical = self._setup(bkd)
        # Product of two Gaussians is still Gaussian
        product = canonical.multiply(canonical).normalize()

        # Should be a valid Gaussian
        assert product.nvars() == 2

        # Test it integrates to 1 (approximately)
        samples = product.rvs(1000)
        assert samples.shape == (2, 1000)

    def test_condition(self, bkd) -> None:
        """Test conditioning on observed variable."""
        _, _, canonical = self._setup(bkd)
        # Condition x_1 on observed value
        fixed_indices = bkd.asarray([1])
        values = bkd.asarray([2.5])

        conditional = canonical.condition(fixed_indices, values)

        # Should have one fewer variable
        assert conditional.nvars() == 1

        # Verify by converting to moments
        mean_cond, cov_cond = conditional.to_moments()
        assert mean_cond.shape == (1,)
        assert cov_cond.shape == (1, 1)

    def test_marginalize(self, bkd) -> None:
        """Test marginalizing out a variable."""
        mean, cov, canonical = self._setup(bkd)
        # Marginalize out x_1
        marg_indices = bkd.asarray([1])

        marginal = canonical.marginalize(marg_indices)

        # Should have one fewer variable
        assert marginal.nvars() == 1

        # Verify by converting to moments
        mean_marg, cov_marg = marginal.to_moments()

        # Mean should be mean[0]
        assert bkd.allclose(
            bkd.asarray([mean_marg[0]]),
            bkd.asarray([mean[0]]),
            rtol=1e-5,
        )

        # Covariance should be cov[0, 0]
        assert bkd.allclose(
            bkd.asarray([cov_marg[0, 0]]),
            bkd.asarray([cov[0, 0]]),
            rtol=1e-5,
        )

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, _, canonical = self._setup(bkd)
        samples = canonical.rvs(100)
        assert samples.shape == (2, 100)

    def test_logpdf_shape(self, bkd) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        _, _, canonical = self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        result = canonical.logpdf(samples)
        assert result.shape == (1, 3)
