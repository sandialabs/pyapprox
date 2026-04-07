"""Tests for ported risk measures: ValueAtRisk, ExactAVaR, UtilitySSD, DisutilitySSD.

Tests cover:
- Value correctness against analytical formulas (Gaussian)
- Multi-QoI support
- Shape validation
- Cross-validation: ExactAVaR vs SampleAverageSmoothedAVaR convergence
- Dual-backend testing (NumPy and PyTorch)
"""

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.risk import GaussianAnalyticalRiskMeasures
from pyapprox.risk import (
    DisutilitySSD,
    ExactAVaR,
    SampleAverageSmoothedAVaR,
    UtilitySSD,
    ValueAtRisk,
)


class TestValueAtRisk:
    """Tests for ValueAtRisk statistic."""

    def test_uniform_quantile(self, bkd) -> None:
        """VaR finds correct quantile for uniform samples."""
        nsamples = 1000
        values = bkd.asarray(np.linspace(0, 1, nsamples)[None, :])
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = ValueAtRisk(0.9, bkd)
        result = stat(values, weights)

        assert result.shape == (1, 1)
        bkd.assert_allclose(result, bkd.asarray([[0.9]]), atol=0.01)

    def test_multi_qoi(self, bkd) -> None:
        """VaR works with multiple QoIs."""
        np.random.seed(42)
        nsamples = 100
        nqoi = 3
        values = bkd.asarray(np.random.randn(nqoi, nsamples))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = ValueAtRisk(0.5, bkd)
        result = stat(values, weights)

        assert result.shape == (nqoi, 1)

    def test_invalid_beta(self, bkd) -> None:
        """VaR raises for invalid beta."""
        with pytest.raises(ValueError):
            ValueAtRisk(1.0, bkd)
        with pytest.raises(ValueError):
            ValueAtRisk(-0.1, bkd)

    def test_gaussian_quantile(self, bkd) -> None:
        """VaR matches scipy quantile for Gaussian."""
        mu, sigma = 1.0, 2.0
        beta = 0.9
        nsamples = 50000
        np.random.seed(42)
        values = bkd.asarray(np.random.normal(mu, sigma, (1, nsamples)))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = ValueAtRisk(beta, bkd)
        result = stat(values, weights)

        expected = stats.norm(mu, sigma).ppf(beta)
        bkd.assert_allclose(result, bkd.asarray([[expected]]), rtol=0.05)


class TestExactAVaR:
    """Tests for ExactAVaR statistic."""

    def test_gaussian_avar(self, bkd) -> None:
        """ExactAVaR matches analytical Gaussian AVaR."""
        mu, sigma = 1.0, 2.0
        beta = 0.8
        np.random.seed(42)
        nsamples = 50000
        values = bkd.asarray(np.random.normal(mu, sigma, (1, nsamples)))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = ExactAVaR(beta, bkd)
        result = stat(values, weights)

        analytical = GaussianAnalyticalRiskMeasures(mu, sigma)
        expected = analytical.AVaR(beta)

        assert result.shape == (1, 1)
        bkd.assert_allclose(result, bkd.asarray([[expected]]), rtol=0.05)

    def test_multi_qoi(self, bkd) -> None:
        """ExactAVaR works with multiple QoIs."""
        np.random.seed(42)
        nsamples = 100
        nqoi = 3
        values = bkd.asarray(np.random.randn(nqoi, nsamples))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = ExactAVaR(0.5, bkd)
        result = stat(values, weights)

        assert result.shape == (nqoi, 1)

    def test_invalid_beta(self, bkd) -> None:
        """ExactAVaR raises for invalid beta."""
        with pytest.raises(ValueError):
            ExactAVaR(1.0, bkd)
        with pytest.raises(ValueError):
            ExactAVaR(-0.1, bkd)

    def test_convergence_to_smoothed_avar(self, bkd) -> None:
        """ExactAVaR and SmoothedAVaR converge for large delta."""
        mu, sigma, beta = 0, 1, 0.5
        rv = stats.norm(mu, sigma)
        nsamples = 10000
        np.random.seed(42)
        samples_np = rv.ppf(np.linspace(1e-6, 1 - 1e-6, nsamples))
        values = bkd.asarray(samples_np)[None, :]
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        exact = ExactAVaR(beta, bkd)
        exact_val = exact(values, weights)

        smoothed = SampleAverageSmoothedAVaR(beta, bkd, delta=100000)
        smoothed_val = smoothed(values, weights)

        bkd.assert_allclose(exact_val, smoothed_val, rtol=1e-3)


class TestUtilitySSD:
    """Tests for UtilitySSD statistic."""

    def test_nonnegative(self, bkd) -> None:
        """UtilitySSD is always non-negative."""
        np.random.seed(42)
        nsamples = 100
        values = bkd.asarray(np.random.randn(1, nsamples))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        eta = bkd.asarray(np.linspace(-2, 2, 10))
        stat = UtilitySSD(eta, bkd)
        result = stat(values, weights)

        assert result.shape == (10, 1)
        assert float(bkd.min(result)) >= -1e-10

    def test_invalid_eta(self, bkd) -> None:
        """UtilitySSD raises for non-1D eta."""
        with pytest.raises(ValueError):
            UtilitySSD(bkd.asarray([[1.0, 2.0]]), bkd)


class TestDisutilitySSD:
    """Tests for DisutilitySSD statistic."""

    def test_nonnegative(self, bkd) -> None:
        """DisutilitySSD is always non-negative."""
        np.random.seed(42)
        nsamples = 100
        values = bkd.asarray(np.random.randn(1, nsamples))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        eta = bkd.asarray(np.linspace(-2, 2, 10))
        stat = DisutilitySSD(eta, bkd)
        result = stat(values, weights)

        assert result.shape == (10, 1)
        assert float(bkd.min(result)) >= -1e-10

    def test_invalid_eta(self, bkd) -> None:
        """DisutilitySSD raises for non-1D eta."""
        with pytest.raises(ValueError):
            DisutilitySSD(bkd.asarray([[1.0, 2.0]]), bkd)
