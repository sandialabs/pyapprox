"""
Legacy comparison tests for LinearGaussianOEDBenchmark.

TODO: Delete after legacy removed.

These tests verify that the new typing module benchmark produces
identical analytical EIG values to the legacy implementation.
"""

import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin


class TestBenchmarkLegacyComparison(unittest.TestCase):
    """Verify typing LinearGaussianOEDBenchmark matches legacy."""

    def setUp(self):
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        self._bkd = NumpyBkd()

    def test_exact_eig_matches_legacy(self):
        """Test exact EIG matches legacy implementation."""
        nobs = 5
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5
        weights = np.ones((nobs, 1)) / nobs

        # Legacy
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDBenchmark,
        )

        legacy_problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=NumpyMixin
        )
        legacy_eig = legacy_problem.exact_expected_information_gain(weights)

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.benchmarks import (
            LinearGaussianOEDBenchmark,
        )

        bkd = NumpyBkd()
        typing_problem = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )
        typing_eig = typing_problem.exact_eig(bkd.asarray(weights))

        self._bkd.assert_allclose(
            self._bkd.asarray([typing_eig]),
            self._bkd.asarray([legacy_eig]),
            rtol=1e-12,
        )

    def test_exact_eig_different_weights(self):
        """Test exact EIG with non-uniform weights."""
        np.random.seed(123)
        nobs = 4
        min_degree = 0
        degree = 1
        noise_std = 0.3
        prior_std = 0.8
        weights = np.random.dirichlet(np.ones(nobs))[:, None]

        # Legacy
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDBenchmark,
        )

        legacy_problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=NumpyMixin
        )
        legacy_eig = legacy_problem.exact_expected_information_gain(weights)

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.benchmarks import (
            LinearGaussianOEDBenchmark,
        )

        bkd = NumpyBkd()
        typing_problem = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )
        typing_eig = typing_problem.exact_eig(bkd.asarray(weights))

        self._bkd.assert_allclose(
            self._bkd.asarray([typing_eig]),
            self._bkd.asarray([legacy_eig]),
            rtol=1e-12,
        )

    def test_design_matrix_matches_legacy(self):
        """Test design matrix matches legacy."""
        nobs = 5
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5

        # Legacy
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDBenchmark,
        )

        legacy_problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=NumpyMixin
        )
        legacy_matrix = legacy_problem.get_observation_model().matrix()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.benchmarks import (
            LinearGaussianOEDBenchmark,
        )

        bkd = NumpyBkd()
        typing_problem = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )
        typing_matrix = typing_problem.design_matrix()

        self._bkd.assert_allclose(
            typing_matrix,
            self._bkd.asarray(NumpyMixin.to_numpy(legacy_matrix)),
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
