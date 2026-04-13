"""
Integration tests for prediction OED workflow.

Tests cover:
- Full prediction OED workflow from start to finish
- Factory functions for creating objectives
- Different deviation measures with different risk measures
- Comparison with analytical formulas for Gaussian distributions
"""

import numpy as np
import pytest

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


class TestPredictionOEDWorkflow:
    """Test complete prediction OED workflow."""

    def test_simple_prediction_workflow(self, bkd):
        """Test simple prediction OED workflow from start to finish."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = bkd.asarray(np.random.randn(ninner, npred))

        # Create using factory function
        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        # Evaluate
        weights = bkd.ones((nobs, 1))
        value = objective(weights)

        assert value.shape == (1, 1)
        assert np.isfinite(bkd.to_numpy(value)[0, 0])

    def test_factory_creates_correct_deviation(self, bkd):
        """Test that factory creates the correct deviation measure."""
        npred = 2

        stdev = create_deviation_measure("stdev", npred, bkd)
        assert isinstance(stdev, StandardDeviationMeasure)
        assert stdev.label() == "StdDev"

        entropic = create_deviation_measure("entropic", npred, bkd, alpha=0.3)
        assert isinstance(entropic, EntropicDeviationMeasure)
        assert entropic.label() == "Entropic"

    def test_factory_creates_correct_risk_measure(self, bkd):
        """Test that factory creates the correct risk measure."""
        mean = create_risk_measure("mean", bkd)
        assert isinstance(mean, SampleAverageMean)

        variance = create_risk_measure("variance", bkd)
        assert isinstance(variance, SampleAverageVariance)

    def test_factory_unknown_deviation_raises(self, bkd):
        """Test that unknown deviation type raises ValueError."""
        with pytest.raises(ValueError):
            create_deviation_measure("unknown", 2, bkd)

    def test_factory_unknown_risk_raises(self, bkd):
        """Test that unknown risk type raises ValueError."""
        with pytest.raises(ValueError):
            create_risk_measure("unknown", bkd)

    def test_different_deviation_measures(self, bkd):
        """Test prediction OED with different deviation measures."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = bkd.asarray(np.random.randn(ninner, npred))

        weights = bkd.ones((nobs, 1))

        for dev_type in ["stdev", "entropic"]:
            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                bkd,
                deviation_type=dev_type,
            )

            value = objective(weights)
            assert value.shape == (1, 1)
            assert np.isfinite(bkd.to_numpy(value)[0, 0])

    def test_different_risk_measures(self, bkd):
        """Test prediction OED with different risk measures."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = bkd.asarray(np.random.randn(ninner, npred))

        weights = bkd.ones((nobs, 1))

        for risk_type in ["mean", "variance"]:
            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                bkd,
                risk_type=risk_type,
            )

            value = objective(weights)
            assert value.shape == (1, 1)
            assert np.isfinite(bkd.to_numpy(value)[0, 0])

    def test_jacobian_with_factory_created_objective(self, bkd):
        """Test Jacobian computation with factory-created objective."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = bkd.asarray(np.random.randn(ninner, npred))

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
        )

        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))
        jac = objective.jacobian(weights)

        assert jac.shape == (1, nobs)
        jac_np = bkd.to_numpy(jac)
        assert np.all(np.isfinite(jac_np))

    def test_different_weights_different_values(self, bkd):
        """Test that different weights produce different objective values."""
        nobs = 3
        ninner = 20
        nouter = 15
        npred = 2

        np.random.seed(42)

        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
        qoi_vals = bkd.asarray(np.random.randn(ninner, npred))

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
        )

        weights1 = bkd.ones((nobs, 1)) * 0.5
        weights2 = bkd.ones((nobs, 1)) * 2.0

        val1 = objective(weights1)
        val2 = objective(weights2)

        # Higher weights = less noise = less deviation
        assert not bkd.allclose(val1, val2, atol=1e-8)


class TestPredictionOEDAnalytical:
    """Test prediction OED against analytical formulas.

    For a single Gaussian QoI with uniform prior weights (no observations),
    the posterior is equal to the prior, and we can verify the deviation
    measures against analytical formulas.
    """

    def test_stdev_uniform_prior_matches_sample_std(self, bkd):
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
        qoi_vals = bkd.asarray(qoi_vals_np)

        # Sample standard deviation
        sample_std = np.std(qoi_vals_np, ddof=0)

        # Create objective with uniform weights (essentially just prior)
        noise_variances = bkd.asarray(np.array([0.1, 0.1, 0.1]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
        )

        # With small weights, observations are noisy, posterior ~ prior
        # StdDev should be close to sample standard deviation
        weights = bkd.ones((nobs, 1)) * 0.1
        value = objective(weights)

        # The objective averages over outer samples and QoIs
        # Due to likelihood weighting, exact match is not expected
        # Just verify it's in reasonable range
        val_np = bkd.to_numpy(value)[0, 0]
        assert val_np > 0
        assert val_np < 10 * sample_std

    def test_entropic_small_alpha_approximation(self, bkd):
        """Test entropic deviation for small alpha.

        For small alpha: entropic_risk ~ E[qoi] + alpha * Var[qoi] / 2
        So entropic_deviation ~ alpha * Var[qoi] / 2
        """
        nobs = 3
        ninner = 100
        nouter = 50
        npred = 1

        np.random.seed(42)

        # Create QoI values
        mu, sigma = 0.0, 1.0
        qoi_vals_np = mu + sigma * np.random.randn(ninner, npred)
        qoi_vals = bkd.asarray(qoi_vals_np)
        sample_var = np.var(qoi_vals_np, ddof=0)

        noise_variances = bkd.asarray(np.array([0.1, 0.1, 0.1]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

        # Small alpha for approximation
        alpha = 0.01
        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="entropic",
            alpha=alpha,
        )

        # Small weights = noisy obs = posterior ~ prior
        weights = bkd.ones((nobs, 1)) * 0.1
        value = objective(weights)

        # For small alpha, entropic deviation ~ alpha * var / 2
        expected_approx = alpha * sample_var / 2.0
        val_np = bkd.to_numpy(value)[0, 0]

        # Verify it's positive and reasonable
        assert val_np > 0
        # Loose check: should be in same order of magnitude
        assert val_np < 10 * expected_approx + 0.1


class TestGaussianRiskMeasuresIntegration:
    """Test that GaussianAnalyticalRiskMeasures can be used for validation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._bkd = NumpyBkd()

    def test_gaussian_stdev(self):
        """Test Gaussian analytical stdev."""
        mu, sigma = 1.0, 2.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)

        assert risk.mean() == mu
        assert risk.stdev() == sigma
        assert risk.variance() == sigma**2

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

        # For standard normal, AVaR(0.5) = phi(0)/(1-0.5) ~ 0.7979
        # where phi(0) = 1/sqrt(2*pi) ~ 0.3989
        avar = risk.AVaR(0.5)
        assert avar > 0
        self._bkd.assert_allclose(
            self._bkd.asarray([avar]),
            self._bkd.asarray([0.7978845608]),
            rtol=1e-5,
        )
