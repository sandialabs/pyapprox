"""
Tests for Gaussian analytical risk measures.
"""

import unittest
import numpy as np
from scipy import stats

from pyapprox.typing.probability.risk import GaussianAnalyticalRiskMeasures


class TestGaussianAnalyticalRiskMeasures(unittest.TestCase):
    """Tests for GaussianAnalyticalRiskMeasures."""

    def setUp(self):
        self.mu = 0.5
        self.sigma = 2.0
        self.risk = GaussianAnalyticalRiskMeasures(self.mu, self.sigma)

    def test_mean(self):
        """Test mean returns mu."""
        self.assertEqual(self.risk.mean(), self.mu)

    def test_variance(self):
        """Test variance returns sigma^2."""
        self.assertAlmostEqual(self.risk.variance(), self.sigma ** 2)

    def test_stdev(self):
        """Test stdev returns sigma."""
        self.assertEqual(self.risk.stdev(), self.sigma)

    def test_mean_plus_stddev(self):
        """Test mean_plus_stddev formula."""
        alpha = 1.5
        expected = self.mu + alpha * self.sigma
        self.assertAlmostEqual(self.risk.mean_plus_stddev(alpha), expected)

    def test_entropic(self):
        """Test entropic risk formula."""
        alpha = 0.5
        expected = self.mu + alpha * self.sigma ** 2 / 2.0
        self.assertAlmostEqual(self.risk.entropic(alpha), expected)

    def test_avar_standard_normal(self):
        """Test AVaR for standard normal against known values."""
        # For standard normal, AVaR_0.5 = phi(0) / 0.5 ≈ 0.7979
        risk = GaussianAnalyticalRiskMeasures(0.0, 1.0)
        avar_05 = risk.AVaR(0.5)
        expected = stats.norm.pdf(0) / 0.5
        self.assertAlmostEqual(avar_05, expected, places=10)

    def test_avar_shifted_scaled(self):
        """Test AVaR respects location-scale."""
        beta = 0.75
        # AVaR_beta(mu + sigma*Z) = mu + sigma * AVaR_beta(Z)
        risk_std = GaussianAnalyticalRiskMeasures(0.0, 1.0)
        avar_std = risk_std.AVaR(beta)
        expected = self.mu + self.sigma * avar_std
        self.assertAlmostEqual(self.risk.AVaR(beta), expected, places=10)

    def test_avar_greater_than_mean(self):
        """Test AVaR is always >= mean for beta > 0."""
        for beta in [0.1, 0.5, 0.9, 0.95]:
            avar = self.risk.AVaR(beta)
            self.assertGreaterEqual(avar, self.mu)

    def test_avar_increases_with_beta(self):
        """Test AVaR is increasing in beta."""
        betas = [0.1, 0.3, 0.5, 0.7, 0.9]
        avars = [self.risk.AVaR(b) for b in betas]
        for i in range(len(avars) - 1):
            self.assertLess(avars[i], avars[i + 1])


if __name__ == "__main__":
    unittest.main()
