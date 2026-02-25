"""
Tests for multivariate Gaussian distributions.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestGaussianLogPDFCore(Generic[Array], unittest.TestCase):
    """Tests for GaussianLogPDFCore."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.cov = self._bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        self.cov_op = DenseCholeskyCovarianceOperator(self.cov, self._bkd)
        self.core = GaussianLogPDFCore(self.cov_op, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.core.nvars(), 2)

    def test_compute_at_zero(self) -> None:
        """Test log-PDF at zero residuals."""
        residuals = self._bkd.zeros((2, 1))
        logpdf = self.core.compute(residuals)
        expected = self.core.log_normalization_constant()
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([logpdf[0]]),
                self._bkd.asarray([expected]),
                rtol=1e-6,
            )
        )

    def test_compute_shape(self) -> None:
        """Test compute() returns (1, nsamples) not (nsamples,)."""
        residuals = self._bkd.zeros((2, 5))
        logpdf = self.core.compute(residuals)
        self.assertEqual(logpdf.shape, (1, 5))

    def test_compute_shape_single_sample(self) -> None:
        """Test compute() returns (1, 1) for single sample."""
        residuals = self._bkd.zeros((2, 1))
        logpdf = self.core.compute(residuals)
        self.assertEqual(logpdf.shape, (1, 1))

    def test_compute_gradient_at_zero(self) -> None:
        """Test gradient at zero residuals is zero."""
        residuals = self._bkd.zeros((2, 3))
        gradient = self.core.compute_gradient(residuals)
        expected = self._bkd.zeros((2, 3))
        self.assertTrue(self._bkd.allclose(gradient, expected, atol=1e-10))


class TestGaussianLogPDFCoreNumpy(TestGaussianLogPDFCore[NDArray[Any]]):
    """NumPy backend tests for GaussianLogPDFCore."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianLogPDFCoreTorch(TestGaussianLogPDFCore[torch.Tensor]):
    """PyTorch backend tests for GaussianLogPDFCore."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestDenseCholeskyMultivariateGaussian(Generic[Array], unittest.TestCase):
    """Tests for DenseCholeskyMultivariateGaussian."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.mean = self._bkd.asarray([[1.0], [2.0]])
        self.cov = self._bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        self.dist = DenseCholeskyMultivariateGaussian(self.mean, self.cov, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.dist.nvars(), 2)

    def test_mean(self) -> None:
        """Test mean returns correct value."""
        self.assertTrue(self._bkd.allclose(self.dist.mean(), self.mean, atol=1e-10))

    def test_covariance(self) -> None:
        """Test covariance returns correct value."""
        self.assertTrue(
            self._bkd.allclose(self.dist.covariance(), self.cov, atol=1e-10)
        )

    def test_covariance_inverse(self) -> None:
        """Test covariance_inverse is inverse."""
        identity = self.cov @ self.dist.covariance_inverse()
        expected = self._bkd.eye(2)
        self.assertTrue(self._bkd.allclose(identity, expected, atol=1e-10))

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (2, 100))

    def test_logpdf_vs_scipy(self) -> None:
        """Test logpdf matches scipy.stats."""
        scipy_dist = stats.multivariate_normal(
            mean=self._bkd.to_numpy(self.mean).flatten(),
            cov=self._bkd.to_numpy(self.cov),
        )
        samples = self._bkd.asarray([[1.0, 0.5, 2.0], [2.0, 1.5, 3.0]])
        logpdf_ours = self.dist.logpdf(samples)
        logpdf_scipy = self._bkd.asarray(
            scipy_dist.logpdf(self._bkd.to_numpy(samples).T)
        )
        self.assertTrue(self._bkd.allclose(logpdf_ours, logpdf_scipy, rtol=1e-6))

    def test_pdf_exp_logpdf(self) -> None:
        """Test pdf = exp(logpdf)."""
        samples = self._bkd.asarray([[1.0, 2.0], [2.0, 3.0]])
        pdf_vals = self.dist.pdf(samples)
        logpdf_vals = self.dist.logpdf(samples)
        self.assertTrue(
            self._bkd.allclose(pdf_vals, self._bkd.exp(logpdf_vals), rtol=1e-6)
        )

    def test_logpdf_gradient_finite_diff(self) -> None:
        """Test logpdf gradient against finite differences."""
        sample = self._bkd.asarray([[1.5], [2.5]])
        grad = self.dist.logpdf_gradient(sample)

        eps = 1e-6
        grad_fd = self._bkd.zeros((2, 1))
        for i in range(2):
            sample_plus = self._bkd.copy(sample)
            sample_plus[i] = sample_plus[i] + eps
            sample_minus = self._bkd.copy(sample)
            sample_minus[i] = sample_minus[i] - eps
            diff = (
                float(self.dist.logpdf(sample_plus)[0])
                - float(self.dist.logpdf(sample_minus)[0])
            ) / (2 * eps)
            grad_fd[i, 0] = diff

        self.assertTrue(self._bkd.allclose(grad, grad_fd, rtol=1e-5))

    def test_kl_divergence_same_dist(self) -> None:
        """Test KL divergence of distribution with itself is zero."""
        kl = self.dist.kl_divergence(self.dist)
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([kl]),
                self._bkd.asarray([0.0]),
                atol=1e-10,
            )
        )

    def test_kl_divergence_positive(self) -> None:
        """Test KL divergence is positive for different distributions."""
        other = DenseCholeskyMultivariateGaussian(
            self._bkd.asarray([[0.0], [0.0]]), self.cov, self._bkd
        )
        kl = self.dist.kl_divergence(other)
        self.assertGreater(float(kl), 0.0)

    def test_invalid_mean_shape(self) -> None:
        """Test invalid mean shape raises error."""
        with self.assertRaises(ValueError):
            DenseCholeskyMultivariateGaussian(
                self._bkd.asarray([1.0, 2.0]), self.cov, self._bkd
            )

    def test_invalid_cov_shape(self) -> None:
        """Test invalid covariance shape raises error."""
        with self.assertRaises(ValueError):
            DenseCholeskyMultivariateGaussian(self.mean, self._bkd.eye(3), self._bkd)

    def test_logpdf_shape(self) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        samples = self._bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        result = self.dist.logpdf(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_pdf_shape(self) -> None:
        """Test pdf returns (1, nsamples) not (nsamples,)."""
        samples = self._bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        result = self.dist.pdf(samples)
        self.assertEqual(result.shape, (1, 3))


class TestDenseCholeskyMultivariateGaussianNumpy(
    TestDenseCholeskyMultivariateGaussian[NDArray[Any]]
):
    """NumPy backend tests for DenseCholeskyMultivariateGaussian."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDenseCholeskyMultivariateGaussianTorch(
    TestDenseCholeskyMultivariateGaussian[torch.Tensor]
):
    """PyTorch backend tests for DenseCholeskyMultivariateGaussian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestDiagonalMultivariateGaussian(Generic[Array], unittest.TestCase):
    """Tests for DiagonalMultivariateGaussian."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.mean = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.variances = self._bkd.asarray([1.0, 4.0, 9.0])
        self.dist = DiagonalMultivariateGaussian(self.mean, self.variances, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.dist.nvars(), 3)

    def test_variances(self) -> None:
        """Test variances returns correct value."""
        self.assertTrue(
            self._bkd.allclose(self.dist.variances(), self.variances, atol=1e-10)
        )

    def test_standard_deviations(self) -> None:
        """Test standard_deviations."""
        expected = self._bkd.sqrt(self.variances)
        self.assertTrue(
            self._bkd.allclose(self.dist.standard_deviations(), expected, atol=1e-10)
        )

    def test_covariance(self) -> None:
        """Test covariance returns diagonal matrix."""
        cov = self.dist.covariance()
        expected = self._bkd.diag(self.variances)
        self.assertTrue(self._bkd.allclose(cov, expected, atol=1e-10))

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

    def test_logpdf_vs_product_of_univariates(self) -> None:
        """Test logpdf equals sum of univariate log-pdfs."""
        samples = self._bkd.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

        logpdf_ours = self.dist.logpdf(samples)

        # Compute as product of independent univariates
        samples_np = self._bkd.to_numpy(samples)
        mean_np = self._bkd.to_numpy(self.mean)
        var_np = self._bkd.to_numpy(self.variances)
        logpdf_expected = np.zeros(2)
        for i in range(3):
            univariate = stats.norm(mean_np[i, 0], np.sqrt(var_np[i]))
            logpdf_expected += univariate.logpdf(samples_np[i])

        self.assertTrue(
            self._bkd.allclose(
                logpdf_ours, self._bkd.asarray(logpdf_expected), rtol=1e-6
            )
        )

    def test_kl_divergence_same_dist(self) -> None:
        """Test KL divergence of distribution with itself is zero."""
        kl = self.dist.kl_divergence(self.dist)
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([kl]),
                self._bkd.asarray([0.0]),
                atol=1e-10,
            )
        )

    def test_logpdf_shape(self) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        samples = self._bkd.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        result = self.dist.logpdf(samples)
        self.assertEqual(result.shape, (1, 2))

    def test_pdf_shape(self) -> None:
        """Test pdf returns (1, nsamples) not (nsamples,)."""
        samples = self._bkd.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        result = self.dist.pdf(samples)
        self.assertEqual(result.shape, (1, 2))


class TestDiagonalMultivariateGaussianNumpy(
    TestDiagonalMultivariateGaussian[NDArray[Any]]
):
    """NumPy backend tests for DiagonalMultivariateGaussian."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDiagonalMultivariateGaussianTorch(
    TestDiagonalMultivariateGaussian[torch.Tensor]
):
    """PyTorch backend tests for DiagonalMultivariateGaussian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestOperatorBasedMultivariateGaussian(Generic[Array], unittest.TestCase):
    """Tests for OperatorBasedMultivariateGaussian."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.nvars = 3
        self.scale = 2.0
        self.mean = self._bkd.zeros((self.nvars, 1))

        # Create scaling operator: L = scale * I
        self.cov_op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: self.scale * x,
            apply_sqrt_inv=lambda x: x / self.scale,
            log_determinant=self.nvars * float(np.log(self.scale)),
            nvars=self.nvars,
            bkd=self._bkd,
        )
        self.dist = OperatorBasedMultivariateGaussian(self.mean, self.cov_op, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.dist.nvars(), 3)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

    def test_logpdf_at_mean(self) -> None:
        """Test logpdf at mean is maximum."""
        samples = self._bkd.asarray(
            [[0.0, 1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, -1.0]]
        )
        logpdf = self.dist.logpdf(samples)
        # logpdf has shape (1, nsamples), at mean (column 0) should be maximum
        self.assertGreater(float(logpdf[0, 0]), float(logpdf[0, 1]))
        self.assertGreater(float(logpdf[0, 0]), float(logpdf[0, 2]))

    def test_covariance_diagonal(self) -> None:
        """Test covariance diagonal computation."""
        diagonal = self.dist.covariance_diagonal()
        expected = self._bkd.full((self.nvars,), self.scale**2)
        self.assertTrue(self._bkd.allclose(diagonal, expected, rtol=1e-6))

    def test_invalid_mean_cov_mismatch(self) -> None:
        """Test mismatched mean and cov_op dimensions."""
        wrong_mean = self._bkd.zeros((5, 1))
        with self.assertRaises(ValueError):
            OperatorBasedMultivariateGaussian(wrong_mean, self.cov_op, self._bkd)

    def test_logpdf_shape(self) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        samples = self._bkd.asarray(
            [[0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5]]
        )
        result = self.dist.logpdf(samples)
        self.assertEqual(result.shape, (1, 4))

    def test_pdf_shape(self) -> None:
        """Test pdf returns (1, nsamples) not (nsamples,)."""
        samples = self._bkd.asarray(
            [[0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5], [0.0, 1.0, -1.0, 0.5]]
        )
        result = self.dist.pdf(samples)
        self.assertEqual(result.shape, (1, 4))


class TestOperatorBasedMultivariateGaussianNumpy(
    TestOperatorBasedMultivariateGaussian[NDArray[Any]]
):
    """NumPy backend tests for OperatorBasedMultivariateGaussian."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOperatorBasedMultivariateGaussianTorch(
    TestOperatorBasedMultivariateGaussian[torch.Tensor]
):
    """PyTorch backend tests for OperatorBasedMultivariateGaussian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCrossValidation(Generic[Array], unittest.TestCase):
    """Cross-validation tests between implementations."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.mean = self._bkd.asarray([[1.0], [2.0]])
        self.cov = self._bkd.asarray([[4.0, 0.0], [0.0, 9.0]])  # Diagonal

    def test_dense_vs_diagonal_for_diagonal_cov(self) -> None:
        """Test dense and diagonal implementations agree for diagonal cov."""
        dense = DenseCholeskyMultivariateGaussian(self.mean, self.cov, self._bkd)
        diagonal = DiagonalMultivariateGaussian(
            self.mean, self._bkd.get_diagonal(self.cov), self._bkd
        )

        samples = self._bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

        logpdf_dense = dense.logpdf(samples)
        logpdf_diagonal = diagonal.logpdf(samples)

        self.assertTrue(self._bkd.allclose(logpdf_dense, logpdf_diagonal, rtol=1e-6))

    def test_dense_vs_operator_for_identity_cov(self) -> None:
        """Test dense and operator implementations agree for identity cov."""
        mean = self._bkd.zeros((3, 1))
        cov = self._bkd.eye(3)

        dense = DenseCholeskyMultivariateGaussian(mean, cov, self._bkd)
        cov_op = OperatorBasedCovarianceOperator(
            apply_sqrt=lambda x: x,
            apply_sqrt_inv=lambda x: x,
            log_determinant=0.0,
            nvars=3,
            bkd=self._bkd,
        )
        operator = OperatorBasedMultivariateGaussian(mean, cov_op, self._bkd)

        # Use deterministic samples
        samples = self._bkd.asarray(
            [
                [0.1, -0.5, 1.2, 0.3, -0.8],
                [0.2, 0.7, -0.3, 0.9, 0.1],
                [-0.4, 0.2, 0.5, -0.6, 0.4],
            ]
        )

        logpdf_dense = dense.logpdf(samples)
        logpdf_operator = operator.logpdf(samples)

        self.assertTrue(self._bkd.allclose(logpdf_dense, logpdf_operator, rtol=1e-6))


class TestCrossValidationNumpy(TestCrossValidation[NDArray[Any]]):
    """NumPy backend tests for cross-validation."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCrossValidationTorch(TestCrossValidation[torch.Tensor]):
    """PyTorch backend tests for cross-validation."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGaussianCanonicalForm(Generic[Array], unittest.TestCase):
    """Tests for GaussianCanonicalForm."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.mean = self._bkd.asarray([1.0, 2.0])
        self.cov = self._bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        self.canonical = GaussianCanonicalForm.from_moments(
            self.mean, self.cov, self._bkd
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.canonical.nvars(), 2)

    def test_roundtrip_moments(self) -> None:
        """Test from_moments -> to_moments roundtrip."""
        mean_recovered, cov_recovered = self.canonical.to_moments()
        self.assertTrue(self._bkd.allclose(mean_recovered, self.mean, rtol=1e-6))
        self.assertTrue(self._bkd.allclose(cov_recovered, self.cov, rtol=1e-6))

    def test_logpdf_vs_dense(self) -> None:
        """Test logpdf matches DenseCholeskyMultivariateGaussian."""
        mean_2d = self._bkd.reshape(self.mean, (2, 1))
        dense = DenseCholeskyMultivariateGaussian(mean_2d, self.cov, self._bkd)
        samples = self._bkd.asarray([[1.0, 0.5, 2.0], [2.0, 1.5, 3.0]])

        logpdf_canonical = self.canonical.logpdf(samples)
        logpdf_dense = dense.logpdf(samples)

        self.assertTrue(self._bkd.allclose(logpdf_canonical, logpdf_dense, rtol=1e-6))

    def test_multiply_same(self) -> None:
        """Test multiplying distribution with itself."""
        product = self.canonical.multiply(self.canonical)

        # Precision and shift should double
        self.assertTrue(
            self._bkd.allclose(
                product.precision(), 2 * self.canonical.precision(), rtol=1e-6
            )
        )
        self.assertTrue(
            self._bkd.allclose(product.shift(), 2 * self.canonical.shift(), rtol=1e-6)
        )

    def test_multiply_normalized(self) -> None:
        """Test normalized product matches analytical result."""
        # Product of two Gaussians is still Gaussian
        product = self.canonical.multiply(self.canonical).normalize()

        # Should be a valid Gaussian
        self.assertEqual(product.nvars(), 2)

        # Test it integrates to 1 (approximately)
        samples = product.rvs(1000)
        self.assertEqual(samples.shape, (2, 1000))

    def test_condition(self) -> None:
        """Test conditioning on observed variable."""
        # Condition x_1 on observed value
        fixed_indices = self._bkd.asarray([1])
        values = self._bkd.asarray([2.5])

        conditional = self.canonical.condition(fixed_indices, values)

        # Should have one fewer variable
        self.assertEqual(conditional.nvars(), 1)

        # Verify by converting to moments
        mean_cond, cov_cond = conditional.to_moments()
        self.assertEqual(mean_cond.shape, (1,))
        self.assertEqual(cov_cond.shape, (1, 1))

    def test_marginalize(self) -> None:
        """Test marginalizing out a variable."""
        # Marginalize out x_1
        marg_indices = self._bkd.asarray([1])

        marginal = self.canonical.marginalize(marg_indices)

        # Should have one fewer variable
        self.assertEqual(marginal.nvars(), 1)

        # Verify by converting to moments
        mean_marg, cov_marg = marginal.to_moments()

        # Mean should be mean[0]
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([mean_marg[0]]),
                self._bkd.asarray([self.mean[0]]),
                rtol=1e-5,
            )
        )

        # Covariance should be cov[0, 0]
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([cov_marg[0, 0]]),
                self._bkd.asarray([self.cov[0, 0]]),
                rtol=1e-5,
            )
        )

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.canonical.rvs(100)
        self.assertEqual(samples.shape, (2, 100))

    def test_logpdf_shape(self) -> None:
        """Test logpdf returns (1, nsamples) not (nsamples,)."""
        samples = self._bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        result = self.canonical.logpdf(samples)
        self.assertEqual(result.shape, (1, 3))


class TestGaussianCanonicalFormNumpy(TestGaussianCanonicalForm[NDArray[Any]]):
    """NumPy backend tests for GaussianCanonicalForm."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianCanonicalFormTorch(TestGaussianCanonicalForm[torch.Tensor]):
    """PyTorch backend tests for GaussianCanonicalForm."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
