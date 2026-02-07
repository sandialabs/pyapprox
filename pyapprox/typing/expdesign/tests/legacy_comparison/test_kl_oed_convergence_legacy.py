"""
Legacy comparison tests for KL-OED convergence.

TODO: Delete after legacy removed.

Tests verify that typing KL-OED diagnostics produce results consistent
with the legacy implementation using same problem setup and parameters.
"""

import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestKLOEDConvergenceLegacyComparison(unittest.TestCase):
    """Compare typing KL-OED convergence with legacy."""

    def setUp(self):
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        self._bkd = NumpyBkd()

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

        self._bkd.assert_allclose(
            self._bkd.asarray([typing_exact]),
            self._bkd.asarray([legacy_exact]),
            rtol=1e-12,
        )

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

        self._bkd.assert_allclose(
            self._bkd.asarray([typing_exact]),
            self._bkd.asarray([legacy_exact]),
            rtol=1e-12,
        )

    def test_convergence_rate_computation_matches_legacy(self):
        """Test convergence rate computation matches legacy."""
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDBenchmark,
            BayesianKLOEDDiagnostics,
        )
        from pyapprox.typing.expdesign.diagnostics import KLOEDDiagnostics

        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / n for n in sample_counts]

        # Legacy
        nobs = 3
        legacy_problem = LinearGaussianBayesianOEDBenchmark(
            nobs, 0, 2, 0.5, 0.5, backend=NumpyMixin
        )
        legacy_diagnostic = BayesianKLOEDDiagnostics(legacy_problem)
        legacy_rate = legacy_diagnostic.compute_convergence_rate(
            sample_counts, values
        )

        # Typing
        typing_rate = KLOEDDiagnostics.compute_convergence_rate(
            sample_counts, values
        )

        self._bkd.assert_allclose(
            self._bkd.asarray([typing_rate]),
            self._bkd.asarray([legacy_rate]),
            rtol=1e-12,
        )

if __name__ == "__main__":
    unittest.main()
