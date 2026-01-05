"""
Tests for joint probability distributions.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.probability.univariate import (
    ScipyContinuousMarginal,
    GaussianMarginal,
)
from pyapprox.typing.probability.joint import IndependentJoint


class TestIndependentJoint(Generic[Array], unittest.TestCase):
    """Tests for IndependentJoint."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # Create marginals: standard normal, beta, uniform
        self.marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.joint.nvars(), 3)

    def test_marginals(self) -> None:
        """Test marginals returns list of marginals."""
        marginals = self.joint.marginals()
        self.assertEqual(len(marginals), 3)

    def test_marginal(self) -> None:
        """Test accessing individual marginal."""
        marginal = self.joint.marginal(1)
        self.assertEqual(marginal.name, "beta")

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.joint.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

    def test_logpdf_sum_of_marginals(self) -> None:
        """Test logpdf equals sum of marginal logpdfs."""
        samples = self._bkd.asarray(
            [[0.0, 0.5, -1.0], [0.3, 0.5, 0.2], [0.5, 0.2, 0.8]]
        )

        logpdf_joint = self.joint.logpdf(samples)

        # Compute manually as sum - marginals expect 2D input (1, nsamples)
        logpdf_expected = self._bkd.zeros((1, 3))
        for i, marginal in enumerate(self.marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            logpdf_expected = logpdf_expected + marginal.logpdf(row_2d)
        logpdf_expected = self._bkd.flatten(logpdf_expected)

        self.assertTrue(
            self._bkd.allclose(logpdf_joint, logpdf_expected, rtol=1e-6)
        )

    def test_pdf_exp_logpdf(self) -> None:
        """Test pdf = exp(logpdf)."""
        samples = self._bkd.asarray([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])
        pdf_vals = self.joint.pdf(samples)
        logpdf_vals = self.joint.logpdf(samples)
        self.assertTrue(
            self._bkd.allclose(pdf_vals, self._bkd.exp(logpdf_vals), rtol=1e-6)
        )

    def test_cdf_product_of_marginals(self) -> None:
        """Test cdf equals product of marginal cdfs."""
        samples = self._bkd.asarray([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])

        cdf_joint = self.joint.cdf(samples)

        # Compute manually as product - marginals expect 2D input (1, nsamples)
        cdf_expected = self._bkd.ones((1, 2))
        for i, marginal in enumerate(self.marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            cdf_expected = cdf_expected * marginal.cdf(row_2d)
        cdf_expected = self._bkd.flatten(cdf_expected)

        self.assertTrue(self._bkd.allclose(cdf_joint, cdf_expected, rtol=1e-6))

    def test_invcdf_component_wise(self) -> None:
        """Test invcdf applies to each component."""
        probs = self._bkd.asarray([[0.5, 0.25], [0.5, 0.75], [0.5, 0.1]])

        quantiles = self.joint.invcdf(probs)

        # Verify each component - marginals expect 2D input (1, nsamples)
        for i, marginal in enumerate(self.marginals):
            row_2d = self._bkd.reshape(probs[i], (1, -1))
            expected = self._bkd.flatten(marginal.invcdf(row_2d))
            self.assertTrue(self._bkd.allclose(quantiles[i], expected, rtol=1e-6))

    def test_correlation_matrix_identity(self) -> None:
        """Test correlation matrix is identity for independent marginals."""
        corr = self.joint.correlation_matrix()
        expected = self._bkd.eye(3)
        self.assertTrue(self._bkd.allclose(corr, expected, rtol=1e-6))

    def test_covariance_diagonal(self) -> None:
        """Test covariance is diagonal for independent marginals."""
        cov = self.joint.covariance()
        # Check diagonal
        diag = self._bkd.get_diagonal(cov)
        self.assertTrue(self._bkd.all_bool(diag > 0))
        # Check off-diagonal is zero
        off_diag = cov - self._bkd.diag(diag)
        expected = self._bkd.zeros((3, 3))
        self.assertTrue(self._bkd.allclose(off_diag, expected, atol=1e-10))

    def test_mean(self) -> None:
        """Test mean computation."""
        mean = self.joint.mean()
        self.assertEqual(mean.shape, (3,))
        # Check first marginal (standard normal)
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([mean[0]]),
                self._bkd.asarray([0.0]),
                atol=0.1,
            )
        )

    def test_variance(self) -> None:
        """Test variance computation."""
        var = self.joint.variance()
        self.assertEqual(var.shape, (3,))
        # Check first marginal (standard normal, var=1)
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([var[0]]),
                self._bkd.asarray([1.0]),
                atol=0.1,
            )
        )

    def test_bounds(self) -> None:
        """Test bounds computation."""
        bounds = self.joint.bounds()
        self.assertEqual(bounds.shape, (2, 3))
        # Uniform has bounds [0, 1]
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([bounds[0, 2]]),
                self._bkd.asarray([0.0]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([bounds[1, 2]]),
                self._bkd.asarray([1.0]),
                atol=1e-10,
            )
        )

    def test_empty_marginals_raises(self) -> None:
        """Test empty marginals raises error."""
        with self.assertRaises(ValueError):
            IndependentJoint([], self._bkd)


class TestIndependentJointNumpy(TestIndependentJoint[NDArray[Any]]):
    """NumPy backend tests for IndependentJoint."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointTorch(TestIndependentJoint[torch.Tensor]):
    """PyTorch backend tests for IndependentJoint."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointGaussian(Generic[Array], unittest.TestCase):
    """Tests for IndependentJoint with Gaussian marginals."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.means = [0.0, 1.0, 2.0]
        self.stds = [1.0, 2.0, 0.5]
        self.marginals = [
            GaussianMarginal(m, s, self._bkd)
            for m, s in zip(self.means, self.stds)
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.joint.nvars(), 3)

    def test_logpdf_vs_multivariate_normal(self) -> None:
        """Test logpdf matches multivariate normal with diagonal covariance."""
        from scipy.stats import multivariate_normal

        cov = np.diag([s**2 for s in self.stds])
        scipy_dist = multivariate_normal(self.means, cov)

        samples = self._bkd.asarray(
            [[0.0, 0.5, -1.0], [1.0, 0.5, 2.0], [2.0, 2.5, 1.5]]
        )

        logpdf_ours = self.joint.logpdf(samples)
        samples_np = self._bkd.to_numpy(samples)
        logpdf_scipy = self._bkd.asarray(scipy_dist.logpdf(samples_np.T))

        self.assertTrue(
            self._bkd.allclose(logpdf_ours, logpdf_scipy, rtol=1e-6)
        )

    def test_mean_matches_marginal_means(self) -> None:
        """Test mean matches marginal means."""
        mean = self.joint.mean()
        expected = self._bkd.asarray(self.means)
        self.assertTrue(self._bkd.allclose(mean, expected, rtol=1e-6))

    def test_variance_matches_marginal_variances(self) -> None:
        """Test variance matches marginal variances."""
        var = self.joint.variance()
        expected = self._bkd.asarray([s**2 for s in self.stds])
        self.assertTrue(self._bkd.allclose(var, expected, rtol=1e-6))


class TestIndependentJointGaussianNumpy(
    TestIndependentJointGaussian[NDArray[Any]]
):
    """NumPy backend tests for IndependentJoint with Gaussian marginals."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointGaussianTorch(
    TestIndependentJointGaussian[torch.Tensor]
):
    """PyTorch backend tests for IndependentJoint with Gaussian marginals."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointSingleVariable(Generic[Array], unittest.TestCase):
    """Tests for IndependentJoint with single variable."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginal = ScipyContinuousMarginal(stats.norm(0, 1), self._bkd)
        self.joint = IndependentJoint([self.marginal], self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self.joint.nvars(), 1)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.joint.rvs(50)
        self.assertEqual(samples.shape, (1, 50))

    def test_logpdf_matches_marginal(self) -> None:
        """Test logpdf matches marginal logpdf."""
        samples = self._bkd.asarray([[0.0, 1.0, -1.0]])

        logpdf_joint = self.joint.logpdf(samples)
        # Marginal expects 2D input (1, nsamples), joint samples is already (1, nsamples)
        logpdf_marginal = self._bkd.flatten(self.marginal.logpdf(samples))

        self.assertTrue(
            self._bkd.allclose(logpdf_joint, logpdf_marginal, rtol=1e-6)
        )


class TestIndependentJointSingleVariableNumpy(
    TestIndependentJointSingleVariable[NDArray[Any]]
):
    """NumPy backend tests for IndependentJoint with single variable."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointSingleVariableTorch(
    TestIndependentJointSingleVariable[torch.Tensor]
):
    """PyTorch backend tests for IndependentJoint with single variable."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointProtocol(Generic[Array], unittest.TestCase):
    """Tests for JointDistributionProtocol compliance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_has_bkd(self) -> None:
        """Test has bkd method."""
        self.assertIsNotNone(self.joint.bkd())

    def test_has_nvars(self) -> None:
        """Test has nvars method."""
        self.assertEqual(self.joint.nvars(), 2)

    def test_has_rvs(self) -> None:
        """Test has rvs method."""
        samples = self.joint.rvs(10)
        self.assertEqual(samples.shape, (2, 10))

    def test_has_logpdf(self) -> None:
        """Test has logpdf method."""
        samples = self._bkd.asarray([[0.0, 0.5], [0.5, 0.8]])
        logpdf = self.joint.logpdf(samples)
        # Joint logpdf returns (1, nsamples) = (1, 2)
        self.assertEqual(logpdf.shape, (1, 2))

    def test_has_marginals(self) -> None:
        """Test has marginals method."""
        marginals = self.joint.marginals()
        self.assertEqual(len(marginals), 2)

    def test_has_correlation_matrix(self) -> None:
        """Test has correlation_matrix method."""
        corr = self.joint.correlation_matrix()
        self.assertEqual(corr.shape, (2, 2))


class TestIndependentJointProtocolNumpy(
    TestIndependentJointProtocol[NDArray[Any]]
):
    """NumPy backend tests for JointDistributionProtocol compliance."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointProtocolTorch(
    TestIndependentJointProtocol[torch.Tensor]
):
    """PyTorch backend tests for JointDistributionProtocol compliance."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
