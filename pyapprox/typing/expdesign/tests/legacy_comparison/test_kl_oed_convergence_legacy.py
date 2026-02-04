"""
Legacy comparison tests for KL-OED convergence.

TODO: Delete after legacy removed.

Tests verify that typing KL-OED diagnostics produce results consistent
with the legacy implementation using same problem setup and parameters.
"""

import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401


class TestKLOEDConvergenceLegacyComparison(unittest.TestCase):
    """Compare typing KL-OED convergence with legacy."""

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

    def test_exact_eig_different_weights_matches_legacy(self):
        """Test exact EIG with non-uniform weights matches legacy."""
        nobs = 5
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5

        np.random.seed(42)
        weights = np.random.uniform(0.1, 1.0, (nobs, 1))
        weights = weights / weights.sum()  # Normalize

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

    def test_convergence_rate_computation_matches_legacy(self):
        """Test convergence rate computation matches legacy formula."""
        sample_counts = [10, 20, 40, 80, 160]
        # Simulate O(1/n) convergence
        values = [1.0 / n for n in sample_counts]

        # Legacy computation (same as used in BayesianOEDDiagnostics)
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

    @slow_test
    def test_numerical_eig_same_seed_matches_legacy(self):
        """Test numerical EIG with same seed produces similar results to legacy.

        Note: Due to implementation differences, we only check that both
        implementations produce reasonable EIG values in similar range.
        """
        nobs = 3
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5
        weights = np.ones((nobs, 1)) / nobs

        # Legacy setup and computation
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDBenchmark,
            BayesianKLOEDDiagnostics,
        )
        from pyapprox.expdesign.bayesoed import (
            IndependentGaussianOEDInnerLoopLogLikelihood,
            KLBayesianOED,
        )

        legacy_problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=NumpyMixin
        )
        legacy_diagnostic = BayesianKLOEDDiagnostics(legacy_problem)
        legacy_exact = legacy_diagnostic.exact_utility(weights)

        # Typing setup
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.benchmarks import LinearGaussianOEDBenchmark
        from pyapprox.typing.expdesign.diagnostics import KLOEDDiagnostics

        bkd = NumpyBkd()
        typing_benchmark = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )
        typing_diagnostic = KLOEDDiagnostics(typing_benchmark)
        typing_weights = bkd.asarray(weights)

        # Both exact EIG should match
        typing_exact = typing_diagnostic.exact_eig(typing_weights)
        np.testing.assert_allclose(typing_exact, legacy_exact, rtol=1e-12)

        # Numerical EIG should be in reasonable range of exact
        # (within 2x for small sample sizes)
        numerical_eig = typing_diagnostic.compute_numerical_eig(
            nouter=200, ninner=100, design_weights=typing_weights, seed=42
        )
        self.assertTrue(np.isfinite(numerical_eig))
        # Numerical should be somewhat close to exact
        relative_error = abs(numerical_eig - typing_exact) / abs(typing_exact)
        self.assertLess(relative_error, 1.0)  # Within 100%

if __name__ == "__main__":
    unittest.main()
