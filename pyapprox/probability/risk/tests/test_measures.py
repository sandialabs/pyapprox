"""Tests for sample-based risk measures."""

import numpy as np
import pytest

from pyapprox.probability.risk.gaussian import (
    GaussianAnalyticalRiskMeasures,
)
from pyapprox.probability.risk.measures import (
    AverageValueAtRisk,
    DisutilitySSD,
    EntropicRisk,
    SafetyMarginRiskMeasure,
    UtilitySSD,
    ValueAtRisk,
)


class TestRiskMeasures:
    """Tests for sample-based risk measures."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_safety_margin_mean_std(self, bkd) -> None:
        """SafetyMarginRiskMeasure computes mean + strength * std."""
        # Create samples with known mean and std
        nsamples = 10000
        mu, sigma = 2.0, 0.5
        samples = bkd.asarray(np.random.normal(mu, sigma, (1, nsamples)))

        strength = 1.5
        risk = SafetyMarginRiskMeasure(bkd, strength)
        risk.set_samples(samples)
        result = risk()

        # Expected: mu + strength * sigma
        expected = mu + strength * sigma
        bkd.assert_allclose(
            bkd.asarray([float(result)]),
            bkd.asarray([expected]),
            rtol=0.05,  # Monte Carlo tolerance
        )

    def test_safety_margin_strength_accessor(self, bkd) -> None:
        """SafetyMarginRiskMeasure.strength() returns correct value."""
        risk = SafetyMarginRiskMeasure(bkd, 2.5)
        assert risk.strength() == pytest.approx(2.5)

    def test_value_at_risk_uniform(self, bkd) -> None:
        """ValueAtRisk finds correct quantile for uniform samples."""
        # Uniform samples on [0, 1]
        nsamples = 1000
        samples = bkd.asarray(np.linspace(0, 1, nsamples)[None, :])

        risk = ValueAtRisk(bkd, beta=0.9)
        risk.set_samples(samples)
        result = risk()

        # VaR_0.9 of Uniform[0,1] should be ~0.9
        bkd.assert_allclose(
            bkd.asarray([float(result)]),
            bkd.asarray([0.9]),
            atol=0.01,
        )

    def test_value_at_risk_beta_accessor(self, bkd) -> None:
        """ValueAtRisk.beta() returns correct value."""
        risk = ValueAtRisk(bkd, beta=0.75)
        assert risk.beta() == pytest.approx(0.75)

    def test_value_at_risk_invalid_beta(self, bkd) -> None:
        """ValueAtRisk raises for invalid beta."""
        with pytest.raises(ValueError):
            ValueAtRisk(bkd, beta=1.0)
        with pytest.raises(ValueError):
            ValueAtRisk(bkd, beta=-0.1)

    def test_average_value_at_risk_gaussian(self, bkd) -> None:
        """AverageValueAtRisk matches analytical formula for Gaussian."""
        mu, sigma = 1.0, 2.0
        nsamples = 50000
        samples = bkd.asarray(np.random.normal(mu, sigma, (1, nsamples)))

        beta = 0.8
        risk = AverageValueAtRisk(bkd, beta)
        risk.set_samples(samples)
        result = risk()

        # Compare with analytical formula
        analytical = GaussianAnalyticalRiskMeasures(mu, sigma)
        expected = analytical.AVaR(beta)

        bkd.assert_allclose(
            bkd.asarray([float(result)]),
            bkd.asarray([expected]),
            rtol=0.05,  # Monte Carlo tolerance
        )

    def test_average_value_at_risk_beta_accessor(self, bkd) -> None:
        """AverageValueAtRisk.beta() returns correct value."""
        risk = AverageValueAtRisk(bkd, beta=0.95)
        assert risk.beta() == pytest.approx(0.95)

    def test_entropic_risk_gaussian(self, bkd) -> None:
        """EntropicRisk matches analytical formula for Gaussian."""
        mu, sigma = 0.5, 1.5
        nsamples = 50000
        samples = bkd.asarray(np.random.normal(mu, sigma, (1, nsamples)))

        beta = 1.0
        risk = EntropicRisk(bkd, beta)
        risk.set_samples(samples)
        result = risk()

        # Compare with analytical formula: mu + beta * sigma^2 / 2
        analytical = GaussianAnalyticalRiskMeasures(mu, sigma)
        expected = analytical.entropic(beta)

        bkd.assert_allclose(
            bkd.asarray([float(result)]),
            bkd.asarray([expected]),
            rtol=0.05,  # Monte Carlo tolerance
        )

    def test_entropic_risk_beta_accessor(self, bkd) -> None:
        """EntropicRisk.beta() returns correct value."""
        risk = EntropicRisk(bkd, beta=2.0)
        assert risk.beta() == pytest.approx(2.0)

    def test_utility_ssd_nonnegative(self, bkd) -> None:
        """UtilitySSD is always non-negative."""
        nsamples = 100
        samples = bkd.asarray(np.random.randn(1, nsamples))

        risk = UtilitySSD(bkd)
        risk.set_samples(samples)
        risk.set_eta(bkd.asarray(np.linspace(-2, 2, 10)))
        result = risk()

        # All values should be >= 0
        assert float(bkd.min(result)) >= -1e-10

    def test_disutility_ssd_nonnegative(self, bkd) -> None:
        """DisutilitySSD is always non-negative."""
        nsamples = 100
        samples = bkd.asarray(np.random.randn(1, nsamples))

        risk = DisutilitySSD(bkd)
        risk.set_samples(samples)
        risk.set_eta(bkd.asarray(np.linspace(-2, 2, 10)))
        result = risk()

        # All values should be >= 0
        assert float(bkd.min(result)) >= -1e-10

    def test_samples_shape_validation(self, bkd) -> None:
        """Risk measures raise for invalid samples shape."""
        risk = SafetyMarginRiskMeasure(bkd, 1.0)

        # 1D array should fail
        with pytest.raises(ValueError):
            risk.set_samples(bkd.asarray(np.random.randn(10)))

        # Shape (2, n) should fail
        with pytest.raises(ValueError):
            risk.set_samples(bkd.asarray(np.random.randn(2, 10)))

    def test_weights_shape_validation(self, bkd) -> None:
        """Risk measures raise for invalid weights shape."""
        risk = SafetyMarginRiskMeasure(bkd, 1.0)
        samples = bkd.asarray(np.random.randn(1, 10))

        # 2D weights should fail
        with pytest.raises(ValueError):
            risk.set_samples(samples, bkd.asarray(np.ones((10, 1))))

        # Wrong length should fail
        with pytest.raises(ValueError):
            risk.set_samples(samples, bkd.asarray(np.ones(5)))

    def test_call_before_set_samples_raises(self, bkd) -> None:
        """Calling risk measure before set_samples raises RuntimeError."""
        risk = SafetyMarginRiskMeasure(bkd, 1.0)
        with pytest.raises(RuntimeError):
            risk()

    def test_custom_weights(self, bkd) -> None:
        """Risk measures work with custom weights."""
        # Two samples with weights
        samples = bkd.asarray([[1.0, 3.0]])
        weights = bkd.asarray([0.25, 0.75])  # Weighted mean = 0.25*1 + 0.75*3 = 2.5

        risk = SafetyMarginRiskMeasure(bkd, 0.0)  # strength=0 gives just mean
        risk.set_samples(samples, weights)
        result = risk()

        bkd.assert_allclose(
            bkd.asarray([float(result)]),
            bkd.asarray([2.5]),
        )
