"""
Tests for multivariate Gaussian distributions.
"""

import unittest
import numpy as np
from scipy import stats

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.probability.gaussian import (
    GaussianLogPDFCore,
    DenseCholeskyMultivariateGaussian,
    DiagonalMultivariateGaussian,
    OperatorBasedMultivariateGaussian,
)
from pyapprox.typing.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    OperatorBasedCovarianceOperator,
)


class TestGaussianLogPDFCore(unittest.TestCase):
    """Tests for GaussianLogPDFCore."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.cov_op = DenseCholeskyCovarianceOperator(self.cov, self.bkd)
        self.core = GaussianLogPDFCore(self.cov_op, self.bkd)

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.core.nvars(), 2)

    def test_compute_at_zero(self):
        """Test log-PDF at zero residuals."""
        residuals = np.zeros((2, 1))
        logpdf = self.core.compute(residuals)
        # At mean, logpdf = log_const
        expected = self.core.log_normalization_constant()
        self.assertAlmostEqual(logpdf[0], expected)

    def test_compute_gradient_at_zero(self):
        """Test gradient at zero residuals is zero."""
        residuals = np.zeros((2, 3))
        gradient = self.core.compute_gradient(residuals)
        np.testing.assert_array_almost_equal(gradient, np.zeros((2, 3)))


class TestDenseCholeskyMultivariateGaussian(unittest.TestCase):
    """Tests for DenseCholeskyMultivariateGaussian."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.mean = np.array([[1.0], [2.0]])
        self.cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.dist = DenseCholeskyMultivariateGaussian(
            self.mean, self.cov, self.bkd
        )

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.dist.nvars(), 2)

    def test_mean(self):
        """Test mean returns correct value."""
        np.testing.assert_array_equal(self.dist.mean(), self.mean)

    def test_covariance(self):
        """Test covariance returns correct value."""
        np.testing.assert_array_equal(self.dist.covariance(), self.cov)

    def test_covariance_inverse(self):
        """Test covariance_inverse is inverse."""
        identity = self.cov @ self.dist.covariance_inverse()
        np.testing.assert_array_almost_equal(identity, np.eye(2))

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (2, 100))

    def test_logpdf_vs_scipy(self):
        """Test logpdf matches scipy.stats."""
        scipy_dist = stats.multivariate_normal(
            mean=self.mean.flatten(), cov=self.cov
        )
        samples = np.array([[1.0, 0.5, 2.0], [2.0, 1.5, 3.0]])
        logpdf_ours = self.dist.logpdf(samples)
        logpdf_scipy = scipy_dist.logpdf(samples.T)
        np.testing.assert_array_almost_equal(logpdf_ours, logpdf_scipy)

    def test_pdf_exp_logpdf(self):
        """Test pdf = exp(logpdf)."""
        samples = np.array([[1.0, 2.0], [2.0, 3.0]])
        pdf_vals = self.dist.pdf(samples)
        logpdf_vals = self.dist.logpdf(samples)
        np.testing.assert_array_almost_equal(pdf_vals, np.exp(logpdf_vals))

    def test_logpdf_gradient_finite_diff(self):
        """Test logpdf gradient against finite differences."""
        sample = np.array([[1.5], [2.5]])
        grad = self.dist.logpdf_gradient(sample)

        eps = 1e-6
        grad_fd = np.zeros((2, 1))
        for i in range(2):
            sample_plus = sample.copy()
            sample_plus[i] += eps
            sample_minus = sample.copy()
            sample_minus[i] -= eps
            grad_fd[i] = (
                self.dist.logpdf(sample_plus)[0]
                - self.dist.logpdf(sample_minus)[0]
            ) / (2 * eps)

        np.testing.assert_array_almost_equal(grad, grad_fd, decimal=5)

    def test_kl_divergence_same_dist(self):
        """Test KL divergence of distribution with itself is zero."""
        kl = self.dist.kl_divergence(self.dist)
        self.assertAlmostEqual(kl, 0.0)

    def test_kl_divergence_positive(self):
        """Test KL divergence is positive for different distributions."""
        other = DenseCholeskyMultivariateGaussian(
            np.array([[0.0], [0.0]]), self.cov, self.bkd
        )
        kl = self.dist.kl_divergence(other)
        self.assertGreater(kl, 0.0)

    def test_invalid_mean_shape(self):
        """Test invalid mean shape raises error."""
        with self.assertRaises(ValueError):
            DenseCholeskyMultivariateGaussian(
                np.array([1.0, 2.0]), self.cov, self.bkd
            )

    def test_invalid_cov_shape(self):
        """Test invalid covariance shape raises error."""
        with self.assertRaises(ValueError):
            DenseCholeskyMultivariateGaussian(
                self.mean, np.eye(3), self.bkd
            )


class TestDiagonalMultivariateGaussian(unittest.TestCase):
    """Tests for DiagonalMultivariateGaussian."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.mean = np.array([[1.0], [2.0], [3.0]])
        self.variances = np.array([1.0, 4.0, 9.0])
        self.dist = DiagonalMultivariateGaussian(
            self.mean, self.variances, self.bkd
        )

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.dist.nvars(), 3)

    def test_variances(self):
        """Test variances returns correct value."""
        np.testing.assert_array_equal(self.dist.variances(), self.variances)

    def test_standard_deviations(self):
        """Test standard_deviations."""
        expected = np.sqrt(self.variances)
        np.testing.assert_array_almost_equal(
            self.dist.standard_deviations(), expected
        )

    def test_covariance(self):
        """Test covariance returns diagonal matrix."""
        cov = self.dist.covariance()
        expected = np.diag(self.variances)
        np.testing.assert_array_equal(cov, expected)

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

    def test_logpdf_vs_product_of_univariates(self):
        """Test logpdf equals sum of univariate log-pdfs."""
        samples = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

        logpdf_ours = self.dist.logpdf(samples)

        # Compute as product of independent univariates
        logpdf_expected = np.zeros(2)
        for i in range(3):
            univariate = stats.norm(self.mean[i, 0], np.sqrt(self.variances[i]))
            logpdf_expected += univariate.logpdf(samples[i])

        np.testing.assert_array_almost_equal(logpdf_ours, logpdf_expected)

    def test_kl_divergence_same_dist(self):
        """Test KL divergence of distribution with itself is zero."""
        kl = self.dist.kl_divergence(self.dist)
        self.assertAlmostEqual(kl, 0.0)


class TestOperatorBasedMultivariateGaussian(unittest.TestCase):
    """Tests for OperatorBasedMultivariateGaussian."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.nvars = 3
        self.scale = 2.0
        self.mean = np.zeros((self.nvars, 1))

        # Create scaling operator: L = scale * I
        self.cov_op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: self.scale * x,
            apply_sqrt_inv=lambda x: x / self.scale,
            log_determinant=self.nvars * np.log(self.scale),
            nvars=self.nvars,
            bkd=self.bkd,
        )
        self.dist = OperatorBasedMultivariateGaussian(
            self.mean, self.cov_op, self.bkd
        )

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.dist.nvars(), 3)

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

    def test_logpdf_at_mean(self):
        """Test logpdf at mean is maximum."""
        samples = np.array([[0.0, 1.0, -1.0],
                           [0.0, 1.0, -1.0],
                           [0.0, 1.0, -1.0]])
        logpdf = self.dist.logpdf(samples)
        # At mean (column 0), logpdf should be maximum
        self.assertGreater(logpdf[0], logpdf[1])
        self.assertGreater(logpdf[0], logpdf[2])

    def test_covariance_diagonal(self):
        """Test covariance diagonal computation."""
        diagonal = self.dist.covariance_diagonal()
        expected = np.full(self.nvars, self.scale**2)
        np.testing.assert_array_almost_equal(diagonal, expected)

    def test_invalid_mean_cov_mismatch(self):
        """Test mismatched mean and cov_op dimensions."""
        wrong_mean = np.zeros((5, 1))
        with self.assertRaises(ValueError):
            OperatorBasedMultivariateGaussian(
                wrong_mean, self.cov_op, self.bkd
            )


class TestCrossValidation(unittest.TestCase):
    """Cross-validation tests between implementations."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.mean = np.array([[1.0], [2.0]])
        self.cov = np.array([[4.0, 0.0], [0.0, 9.0]])  # Diagonal

    def test_dense_vs_diagonal_for_diagonal_cov(self):
        """Test dense and diagonal implementations agree for diagonal cov."""
        dense = DenseCholeskyMultivariateGaussian(
            self.mean, self.cov, self.bkd
        )
        diagonal = DiagonalMultivariateGaussian(
            self.mean, np.diag(self.cov), self.bkd
        )

        samples = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

        logpdf_dense = dense.logpdf(samples)
        logpdf_diagonal = diagonal.logpdf(samples)

        np.testing.assert_array_almost_equal(logpdf_dense, logpdf_diagonal)

    def test_dense_vs_operator_for_identity_cov(self):
        """Test dense and operator implementations agree for identity cov."""
        mean = np.zeros((3, 1))
        cov = np.eye(3)

        dense = DenseCholeskyMultivariateGaussian(mean, cov, self.bkd)
        cov_op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: x,
            apply_sqrt_inv=lambda x: x,
            log_determinant=0.0,
            nvars=3,
            bkd=self.bkd,
        )
        operator = OperatorBasedMultivariateGaussian(mean, cov_op, self.bkd)

        samples = np.random.randn(3, 5)

        logpdf_dense = dense.logpdf(samples)
        logpdf_operator = operator.logpdf(samples)

        np.testing.assert_array_almost_equal(logpdf_dense, logpdf_operator)


if __name__ == "__main__":
    unittest.main()
