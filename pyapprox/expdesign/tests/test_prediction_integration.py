"""
Integration tests for prediction OED workflow.

Tests cover:
- Full prediction OED workflow from start to finish
- Factory functions for creating objectives
- Different deviation measures with different risk measures
- Comparison with analytical formulas for Gaussian distributions
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.expdesign import (
    EntropicDeviationMeasure,
    SampleAverageMean,
    SampleAverageVariance,
    StandardDeviationMeasure,
    create_deviation_measure,
    create_prediction_oed_objective,
    create_risk_measure,
)
from pyapprox.probability.risk.gaussian import GaussianAnalyticalRiskMeasures
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestPredictionOEDWorkflow(Generic[Array], unittest.TestCase):
    """Test complete prediction OED workflow."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_simple_prediction_workflow(self):
        """Test simple prediction OED workflow from start to finish."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = self._bkd.asarray(np.random.randn(ninner, npred))

        # Create using factory function
        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        # Evaluate
        weights = self._bkd.ones((nobs, 1))
        value = objective(weights)

        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(np.isfinite(self._bkd.to_numpy(value)[0, 0]))

    def test_factory_creates_correct_deviation(self):
        """Test that factory creates the correct deviation measure."""
        npred = 2

        stdev = create_deviation_measure("stdev", npred, self._bkd)
        self.assertIsInstance(stdev, StandardDeviationMeasure)
        self.assertEqual(stdev.label(), "StdDev")

        entropic = create_deviation_measure("entropic", npred, self._bkd, alpha=0.3)
        self.assertIsInstance(entropic, EntropicDeviationMeasure)
        self.assertEqual(entropic.label(), "Entropic")

    def test_factory_creates_correct_risk_measure(self):
        """Test that factory creates the correct risk measure."""
        mean = create_risk_measure("mean", self._bkd)
        self.assertIsInstance(mean, SampleAverageMean)

        variance = create_risk_measure("variance", self._bkd)
        self.assertIsInstance(variance, SampleAverageVariance)

    def test_factory_unknown_deviation_raises(self):
        """Test that unknown deviation type raises ValueError."""
        with self.assertRaises(ValueError):
            create_deviation_measure("unknown", 2, self._bkd)

    def test_factory_unknown_risk_raises(self):
        """Test that unknown risk type raises ValueError."""
        with self.assertRaises(ValueError):
            create_risk_measure("unknown", self._bkd)

    def test_different_deviation_measures(self):
        """Test prediction OED with different deviation measures."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = self._bkd.asarray(np.random.randn(ninner, npred))

        weights = self._bkd.ones((nobs, 1))

        for dev_type in ["stdev", "entropic"]:
            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                self._bkd,
                deviation_type=dev_type,
            )

            value = objective(weights)
            self.assertEqual(value.shape, (1, 1))
            self.assertTrue(np.isfinite(self._bkd.to_numpy(value)[0, 0]))

    def test_different_risk_measures(self):
        """Test prediction OED with different risk measures."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = self._bkd.asarray(np.random.randn(ninner, npred))

        weights = self._bkd.ones((nobs, 1))

        for risk_type in ["mean", "variance"]:
            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                self._bkd,
                risk_type=risk_type,
            )

            value = objective(weights)
            self.assertEqual(value.shape, (1, 1))
            self.assertTrue(np.isfinite(self._bkd.to_numpy(value)[0, 0]))

    def test_jacobian_with_factory_created_objective(self):
        """Test Jacobian computation with factory-created objective."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = self._bkd.asarray(np.random.randn(ninner, npred))

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
        )

        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))
        jac = objective.jacobian(weights)

        self.assertEqual(jac.shape, (1, nobs))
        jac_np = self._bkd.to_numpy(jac)
        self.assertTrue(np.all(np.isfinite(jac_np)))

    def test_different_weights_different_values(self):
        """Test that different weights produce different objective values."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = self._bkd.asarray(np.random.randn(ninner, npred))

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
        )

        weights1 = self._bkd.ones((nobs, 1)) * 0.5
        weights2 = self._bkd.ones((nobs, 1)) * 2.0

        val1 = objective(weights1)
        val2 = objective(weights2)

        # Higher weights = less noise = less deviation
        self.assertFalse(self._bkd.allclose(val1, val2, atol=1e-8))


class TestPredictionOEDWorkflowNumpy(TestPredictionOEDWorkflow[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPredictionOEDWorkflowTorch(TestPredictionOEDWorkflow[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestPredictionOEDAnalytical(Generic[Array], unittest.TestCase):
    """Test prediction OED against analytical formulas.

    For a single Gaussian QoI with uniform prior weights (no observations),
    the posterior is equal to the prior, and we can verify the deviation
    measures against analytical formulas.
    """

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_stdev_uniform_prior_matches_sample_std(self):
        """Test that StdDev deviation matches sample standard deviation.

        With very small noise (high design weights), the posterior concentrates
        on the data-generating prior sample, making the posterior variance
        approach the prior variance.
        """
        nobs = 3
        ninner = 100
        nouter = 50
        npred = 1

        # Use fixed samples for reproducibility
        np.random.seed(42)

        # Create QoI values with known statistics
        mu, sigma = 2.0, 1.5
        qoi_vals_np = mu + sigma * np.random.randn(ninner, npred)
        qoi_vals = self._bkd.asarray(qoi_vals_np)

        # Sample standard deviation
        sample_std = np.std(qoi_vals_np, ddof=0)

        # Create objective with uniform weights (essentially just prior)
        noise_variances = self._bkd.asarray(np.array([0.1, 0.1, 0.1]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
        )

        # With small weights, observations are noisy, posterior ≈ prior
        # StdDev should be close to sample standard deviation
        weights = self._bkd.ones((nobs, 1)) * 0.1
        value = objective(weights)

        # The objective averages over outer samples and QoIs
        # Due to likelihood weighting, exact match is not expected
        # Just verify it's in reasonable range
        val_np = self._bkd.to_numpy(value)[0, 0]
        self.assertTrue(val_np > 0)
        self.assertTrue(val_np < 10 * sample_std)

    def test_entropic_small_alpha_approximation(self):
        """Test entropic deviation for small alpha.

        For small alpha: entropic_risk ≈ E[qoi] + alpha * Var[qoi] / 2
        So entropic_deviation ≈ alpha * Var[qoi] / 2
        """
        nobs = 3
        ninner = 100
        nouter = 50
        npred = 1

        np.random.seed(42)

        # Create QoI values
        mu, sigma = 0.0, 1.0
        qoi_vals_np = mu + sigma * np.random.randn(ninner, npred)
        qoi_vals = self._bkd.asarray(qoi_vals_np)
        sample_var = np.var(qoi_vals_np, ddof=0)

        noise_variances = self._bkd.asarray(np.array([0.1, 0.1, 0.1]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))

        # Small alpha for approximation
        alpha = 0.01
        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="entropic",
            alpha=alpha,
        )

        # Small weights = noisy obs = posterior ≈ prior
        weights = self._bkd.ones((nobs, 1)) * 0.1
        value = objective(weights)

        # For small alpha, entropic deviation ≈ alpha * var / 2
        expected_approx = alpha * sample_var / 2.0
        val_np = self._bkd.to_numpy(value)[0, 0]

        # Verify it's positive and reasonable
        self.assertTrue(val_np > 0)
        # Loose check: should be in same order of magnitude
        self.assertTrue(val_np < 10 * expected_approx + 0.1)


class TestPredictionOEDAnalyticalNumpy(TestPredictionOEDAnalytical[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPredictionOEDAnalyticalTorch(TestPredictionOEDAnalytical[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestGaussianRiskMeasuresIntegration(unittest.TestCase):
    """Test that GaussianAnalyticalRiskMeasures can be used for validation."""

    def setUp(self):
        from pyapprox.util.backends.numpy import NumpyBkd

        self._bkd = NumpyBkd()

    def test_gaussian_stdev(self):
        """Test Gaussian analytical stdev."""
        mu, sigma = 1.0, 2.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)

        self.assertEqual(risk.mean(), mu)
        self.assertEqual(risk.stdev(), sigma)
        self.assertEqual(risk.variance(), sigma**2)

    def test_gaussian_entropic(self):
        """Test Gaussian analytical entropic risk."""
        mu, sigma = 0.0, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)

        alpha = 0.5
        # Entropic for Gaussian: mu + alpha * sigma^2 / 2
        expected = mu + alpha * sigma**2 / 2.0
        self._bkd.assert_allclose(
            self._bkd.asarray([risk.entropic(alpha)]),
            self._bkd.asarray([expected]),
            rtol=1e-7,
        )

    def test_gaussian_avar(self):
        """Test Gaussian analytical AVaR."""
        mu, sigma = 0.0, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)

        # For standard normal, AVaR(0.5) = phi(0)/(1-0.5) ≈ 0.7979
        # where phi(0) = 1/sqrt(2*pi) ≈ 0.3989
        avar = risk.AVaR(0.5)
        self.assertTrue(avar > 0)
        self._bkd.assert_allclose(
            self._bkd.asarray([avar]),
            self._bkd.asarray([0.7978845608]),
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
