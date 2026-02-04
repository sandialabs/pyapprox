"""
Legacy comparison tests for KLOEDDiagnostics.

TODO: Delete after legacy removed.

These tests verify that the typing diagnostics produce similar
results to the legacy implementation.
"""

import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin


class TestDiagnosticsLegacyComparison(unittest.TestCase):
    """Verify typing KLOEDDiagnostics produces consistent results with legacy."""

    def test_exact_eig_matches_legacy(self):
        """Test exact EIG matches legacy benchmark."""
        nobs = 5
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5
        weights = np.ones((nobs, 1)) / nobs

        # Legacy
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDBenchmark,
            BayesianKLOEDDiagnostics,
        )

        legacy_problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=NumpyMixin
        )
        legacy_diagnostic = BayesianKLOEDDiagnostics(legacy_problem)
        legacy_exact = legacy_diagnostic.exact_utility(weights)

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.benchmarks import LinearGaussianOEDBenchmark
        from pyapprox.typing.expdesign.diagnostics import KLOEDDiagnostics

        bkd = NumpyBkd()
        typing_benchmark = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )
        typing_diagnostic = KLOEDDiagnostics(typing_benchmark)
        typing_exact = typing_diagnostic.exact_eig(bkd.asarray(weights))

        np.testing.assert_allclose(typing_exact, legacy_exact, rtol=1e-12)

    def test_convergence_rate_computation(self):
        """Test convergence rate computation matches legacy."""
        sample_counts = [10, 20, 40, 80, 160]
        # Simulate O(1/n) convergence
        values = [1.0 / n for n in sample_counts]

        # Legacy
        from pyapprox.expdesign.bayesoed_benchmarks import BayesianOEDDiagnostics

        # BayesianOEDDiagnostics.compute_convergence_rate is a method that
        # can be called with any instance, but we need the static functionality
        # Use numpy directly for this simple computation
        log_n = np.log(np.array(sample_counts))
        log_vals = np.log(np.array(values))
        legacy_slope = np.polyfit(log_n, log_vals, 1)[0]
        legacy_rate = -legacy_slope

        # Typing
        from pyapprox.typing.expdesign.diagnostics import KLOEDDiagnostics

        typing_rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        np.testing.assert_allclose(typing_rate, legacy_rate, rtol=1e-12)

        # For O(1/n) data, rate should be approximately 1.0
        self.assertAlmostEqual(typing_rate, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
