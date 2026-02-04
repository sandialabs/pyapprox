"""
Legacy comparison tests for LogNormal analytical risk measures.

TODO: Delete after legacy removed.
"""

import unittest
import numpy as np


class TestLogNormalRiskMeasuresLegacyComparison(unittest.TestCase):
    """Verify typing LogNormalAnalyticalRiskMeasures matches legacy."""

    def test_mean_matches_legacy(self):
        """Test mean matches legacy."""
        mu, sigma = 0.5, 1.0

        # Legacy
        from pyapprox.optimization.risk import (
            LogNormalAnalyticalRiskMeasures as LegacyLogNormal,
        )
        legacy = LegacyLogNormal(mu, sigma)
        legacy_result = legacy.mean()

        # Typing
        from pyapprox.typing.probability.risk import (
            LogNormalAnalyticalRiskMeasures as TypingLogNormal,
        )
        typing_risk = TypingLogNormal(mu, sigma)
        typing_result = typing_risk.mean()

        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_std_matches_legacy(self):
        """Test std matches legacy."""
        mu, sigma = 0.5, 1.0

        # Legacy
        from pyapprox.optimization.risk import (
            LogNormalAnalyticalRiskMeasures as LegacyLogNormal,
        )
        legacy = LegacyLogNormal(mu, sigma)
        legacy_result = legacy.std()

        # Typing
        from pyapprox.typing.probability.risk import (
            LogNormalAnalyticalRiskMeasures as TypingLogNormal,
        )
        typing_risk = TypingLogNormal(mu, sigma)
        typing_result = typing_risk.std()

        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_var_matches_legacy(self):
        """Test VaR matches legacy."""
        mu, sigma = 0.5, 1.0

        # Legacy
        from pyapprox.optimization.risk import (
            LogNormalAnalyticalRiskMeasures as LegacyLogNormal,
        )
        legacy = LegacyLogNormal(mu, sigma)

        # Typing
        from pyapprox.typing.probability.risk import (
            LogNormalAnalyticalRiskMeasures as TypingLogNormal,
        )
        typing_risk = TypingLogNormal(mu, sigma)

        for beta in [0.1, 0.5, 0.75, 0.9, 0.95]:
            legacy_result = legacy.VaR(beta)
            typing_result = typing_risk.VaR(beta)
            np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_avar_matches_legacy(self):
        """Test AVaR matches legacy."""
        mu, sigma = 0.5, 1.0

        # Legacy
        from pyapprox.optimization.risk import (
            LogNormalAnalyticalRiskMeasures as LegacyLogNormal,
        )
        legacy = LegacyLogNormal(mu, sigma)

        # Typing
        from pyapprox.typing.probability.risk import (
            LogNormalAnalyticalRiskMeasures as TypingLogNormal,
        )
        typing_risk = TypingLogNormal(mu, sigma)

        for beta in [0.0, 0.1, 0.5, 0.75, 0.9]:
            legacy_result = legacy.AVaR(beta)
            typing_result = typing_risk.AVaR(beta)
            np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_utility_ssd_matches_legacy(self):
        """Test utility_SSD matches legacy."""
        mu, sigma = 0.5, 1.0
        eta = np.array([0.5, 1.0, 2.0, 5.0])

        # Legacy
        from pyapprox.optimization.risk import (
            LogNormalAnalyticalRiskMeasures as LegacyLogNormal,
        )
        legacy = LegacyLogNormal(mu, sigma)
        legacy_result = legacy.utility_SSD(eta)

        # Typing
        from pyapprox.typing.probability.risk import (
            LogNormalAnalyticalRiskMeasures as TypingLogNormal,
        )
        typing_risk = TypingLogNormal(mu, sigma)
        typing_result = typing_risk.utility_SSD(eta)

        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_disutility_ssd_matches_legacy(self):
        """Test disutility_SSD matches legacy."""
        mu, sigma = 0.5, 1.0
        eta = np.array([0.5, 1.0, 2.0, 5.0])

        # Legacy
        from pyapprox.optimization.risk import (
            LogNormalAnalyticalRiskMeasures as LegacyLogNormal,
        )
        legacy = LegacyLogNormal(mu, sigma)
        legacy_result = legacy.disutility_SSD(eta)

        # Typing
        from pyapprox.typing.probability.risk import (
            LogNormalAnalyticalRiskMeasures as TypingLogNormal,
        )
        typing_risk = TypingLogNormal(mu, sigma)
        typing_result = typing_risk.disutility_SSD(eta)

        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)

    def test_kl_divergence_matches_legacy(self):
        """Test kl_divergence matches legacy."""
        mu1, sigma1 = 0.5, 1.0
        mu2, sigma2 = 1.0, 2.0

        # Legacy
        from pyapprox.optimization.risk import (
            LogNormalAnalyticalRiskMeasures as LegacyLogNormal,
        )
        legacy = LegacyLogNormal(mu1, sigma1)
        legacy_result = legacy.kl_divergence(mu2, sigma2)

        # Typing
        from pyapprox.typing.probability.risk import (
            LogNormalAnalyticalRiskMeasures as TypingLogNormal,
        )
        typing_risk = TypingLogNormal(mu1, sigma1)
        typing_result = typing_risk.kl_divergence(mu2, sigma2)

        np.testing.assert_allclose(typing_result, legacy_result, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
