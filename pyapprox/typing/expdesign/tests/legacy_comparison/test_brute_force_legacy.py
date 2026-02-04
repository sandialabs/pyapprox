"""
Legacy comparison tests for brute-force D-optimal OED.

TODO: Delete after legacy removed.

Replicates legacy test_brute_force_d_optimal_oed from test_bayesoed.py:575-649.
Verifies that:
1. Symmetric design pairs have equal utility values
2. Optimal k=3 design for 5-observation polynomial regression is [0, 2, 4]
"""

import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.misc import hash_array
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401


class TestBruteForceLegacyComparison(unittest.TestCase):
    """Compare brute-force OED results with legacy implementation."""

    @slow_test
    def test_brute_force_utility_symmetry(self):
        """Test that symmetric design pairs have equal utility values.

        Replicates legacy test_brute_force_d_optimal_oed at test_bayesoed.py:625-647.
        """
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianDOptimalOEDBenchmark,
            BayesianKLOEDDiagnostics,
        )
        from pyapprox.expdesign.bayesoed import (
            IndependentGaussianOEDInnerLoopLogLikelihood,
            BruteForceKLBayesianOED,
        )

        bkd = NumpyMixin
        nobs = 5
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5

        outerloop_quadtype = "gauss"
        nouterloop_samples = 100000
        innerloop_quadtype = "gauss"
        ninnerloop_samples = 1000

        # Initialize problem
        problem = LinearGaussianBayesianDOptimalOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd
        )

        # Initialize diagnostic
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        brute_oed = BruteForceKLBayesianOED(innerloop_loglike)

        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.data_generator().prepare_simulation_inputs(
            brute_oed,
            problem.get_prior(),
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )

        brute_oed.set_data_from_model(
            problem.get_observation_model(),
            problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )
        opt_design = brute_oed.compute(3)

        # Check utility vals are symmetric
        pairs = bkd.asarray(
            [
                [[0, 1, 2], [2, 3, 4]],
                [[0, 1, 3], [1, 3, 4]],
                [[0, 1, 4], [0, 3, 4]],
                [[1, 2, 4], [0, 2, 3]],
            ]
        )
        pairs_dict = {
            hash_array(index, bkd=bkd): ii
            for ii, index in enumerate(brute_oed._design_indices)
        }
        for pair in pairs:
            self.assertTrue(
                bkd.allclose(
                    brute_oed._utility_vals[
                        pairs_dict[hash_array(pair[0], bkd=bkd)]
                    ],
                    brute_oed._utility_vals[
                        pairs_dict[hash_array(pair[1], bkd=bkd)]
                    ],
                )
            )

        # Check optimal design is [0, 2, 4]
        np.testing.assert_array_equal(
            np.asarray(opt_design), np.array([0, 2, 4])
        )

    @slow_test
    def test_brute_force_optimal_design(self):
        """Test that optimal k=3 design is [0, 2, 4].

        Replicates legacy test_brute_force_d_optimal_oed at test_bayesoed.py:649.
        """
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianDOptimalOEDBenchmark,
            BayesianKLOEDDiagnostics,
        )
        from pyapprox.expdesign.bayesoed import (
            IndependentGaussianOEDInnerLoopLogLikelihood,
            BruteForceKLBayesianOED,
        )

        bkd = NumpyMixin
        nobs = 5
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5

        outerloop_quadtype = "gauss"
        nouterloop_samples = 100000
        innerloop_quadtype = "gauss"
        ninnerloop_samples = 1000

        # Initialize problem
        problem = LinearGaussianBayesianDOptimalOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd
        )

        # Initialize diagnostic
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        brute_oed = BruteForceKLBayesianOED(innerloop_loglike)

        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.data_generator().prepare_simulation_inputs(
            brute_oed,
            problem.get_prior(),
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )

        brute_oed.set_data_from_model(
            problem.get_observation_model(),
            problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )
        opt_design = brute_oed.compute(3)

        # Check optimal design is [0, 2, 4]
        np.testing.assert_array_equal(
            np.asarray(opt_design), np.array([0, 2, 4])
        )


if __name__ == "__main__":
    unittest.main()
