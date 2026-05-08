"""
Tests for Gaussian analytical risk measures.
"""

import pytest
from scipy import stats

from pyapprox.probability.risk import GaussianAnalyticalRiskMeasures


class TestGaussianAnalyticalRiskMeasures:
    """Tests for GaussianAnalyticalRiskMeasures."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mu = 0.5
        self.sigma = 2.0
        self.risk = GaussianAnalyticalRiskMeasures(self.mu, self.sigma)

    def test_mean(self):
        """Test mean returns mu."""
        assert self.risk.mean() == self.mu

    def test_variance(self):
        """Test variance returns sigma^2."""
        assert self.risk.variance() == pytest.approx(self.sigma**2)

    def test_stdev(self):
        """Test stdev returns sigma."""
        assert self.risk.stdev() == self.sigma

    def test_mean_plus_stddev(self):
        """Test mean_plus_stddev formula."""
        alpha = 1.5
        expected = self.mu + alpha * self.sigma
        assert self.risk.mean_plus_stddev(alpha) == pytest.approx(expected)

    def test_entropic(self):
        """Test entropic risk formula."""
        alpha = 0.5
        expected = self.mu + alpha * self.sigma**2 / 2.0
        assert self.risk.entropic(alpha) == pytest.approx(expected)

    def test_avar_standard_normal(self):
        """Test AVaR for standard normal against known values."""
        # For standard normal, AVaR_0.5 = phi(0) / 0.5 ~ 0.7979
        risk = GaussianAnalyticalRiskMeasures(0.0, 1.0)
        avar_05 = risk.AVaR(0.5)
        expected = stats.norm.pdf(0) / 0.5
        assert avar_05 == pytest.approx(expected, abs=1e-10)

    def test_avar_shifted_scaled(self):
        """Test AVaR respects location-scale."""
        beta = 0.75
        # AVaR_beta(mu + sigma*Z) = mu + sigma * AVaR_beta(Z)
        risk_std = GaussianAnalyticalRiskMeasures(0.0, 1.0)
        avar_std = risk_std.AVaR(beta)
        expected = self.mu + self.sigma * avar_std
        assert self.risk.AVaR(beta) == pytest.approx(expected, abs=1e-10)

    def test_avar_greater_than_mean(self):
        """Test AVaR is always >= mean for beta > 0."""
        for beta in [0.1, 0.5, 0.9, 0.95]:
            avar = self.risk.AVaR(beta)
            assert avar >= self.mu

    def test_avar_increases_with_beta(self):
        """Test AVaR is increasing in beta."""
        betas = [0.1, 0.3, 0.5, 0.7, 0.9]
        avars = [self.risk.AVaR(b) for b in betas]
        for i in range(len(avars) - 1):
            assert avars[i] < avars[i + 1]
