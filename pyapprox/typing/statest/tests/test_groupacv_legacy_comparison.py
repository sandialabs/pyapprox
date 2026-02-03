"""Legacy comparison tests for GroupACV and MLBLUE estimators.

These tests compare the typing implementation against the legacy
implementation to ensure numerical equivalence. These tests should
be removed once the legacy module is deleted.

NOTE: These tests only run with NumPy backend since the legacy code
uses a different backend interface.
"""

import unittest

import numpy as np

from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

# Typing imports
from pyapprox.typing.statest.groupacv import (
    GroupACVEstimatorIS as TypingGroupACVIS,
    GroupACVEstimatorNested as TypingGroupACVNested,
    get_model_subsets as typing_get_subsets,
)
from pyapprox.typing.statest.statistics import MultiOutputMean
from pyapprox.typing.util.backends.numpy import NumpyBkd

# Legacy imports
from pyapprox.multifidelity.groupacv import (
    GroupACVEstimator as LegacyGroupACV,
    get_model_subsets as legacy_get_subsets,
)
from pyapprox.multifidelity.factory import multioutput_stats
from pyapprox.util.backends.numpy import NumpyMixin as LegacyBkd


class TestGroupACVLegacyComparison(unittest.TestCase):
    """Compare typing GroupACV implementation against legacy."""

    def setUp(self):
        self.legacy_bkd = LegacyBkd
        self.typing_bkd = NumpyBkd()
        np.random.seed(1)

    def test_get_model_subsets(self):
        """Test that get_model_subsets produces identical results."""
        nmodels = 3
        legacy_subsets = legacy_get_subsets(nmodels, self.legacy_bkd)
        typing_subsets = typing_get_subsets(nmodels, self.typing_bkd)

        self.assertEqual(len(legacy_subsets), len(typing_subsets))
        for ls, ts in zip(legacy_subsets, typing_subsets):
            np.testing.assert_allclose(ls, ts)

    def test_groupacv_basic_properties(self):
        """Test that GroupACVEstimator has identical basic properties."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACVIS(
            typing_stat, self.typing_bkd.array(costs)
        )

        self.assertEqual(legacy_est.nmodels(), typing_est.nmodels())
        self.assertEqual(legacy_est.nsubsets(), typing_est.nsubsets())
        self.assertEqual(legacy_est.npartitions(), typing_est.npartitions())

    def test_compute_nsamples_per_model(self):
        """Test that _compute_nsamples_per_model produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACVIS(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_nsamples = legacy_est._compute_nsamples_per_model(
            npartition_samples_np
        )
        typing_nsamples = typing_est._compute_nsamples_per_model(
            npartition_samples
        )

        np.testing.assert_allclose(legacy_nsamples, typing_nsamples)

    def test_estimator_cost(self):
        """Test that _estimator_cost produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACVIS(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_cost = legacy_est._estimator_cost(npartition_samples_np)
        typing_cost = typing_est._estimator_cost(npartition_samples)

        np.testing.assert_allclose(legacy_cost, typing_cost)

    def test_nintersect_samples(self):
        """Test that _nintersect_samples produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACVIS(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_intersect = legacy_est._nintersect_samples(npartition_samples_np)
        typing_intersect = typing_est._nintersect_samples(npartition_samples)

        np.testing.assert_allclose(legacy_intersect, typing_intersect)

    def test_sigma_matrix(self):
        """Test that _sigma produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACVIS(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_sigma = legacy_est._sigma(npartition_samples_np)
        typing_sigma = typing_est._sigma(npartition_samples)

        np.testing.assert_allclose(legacy_sigma, typing_sigma)

    def test_covariance_from_npartition_samples(self):
        """Test that _covariance_from_npartition_samples produces identical results."""
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs)

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACVIS(
            typing_stat, self.typing_bkd.array(costs)
        )

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_cov = legacy_est._covariance_from_npartition_samples(
            npartition_samples_np
        )
        typing_cov = typing_est._covariance_from_npartition_samples(
            npartition_samples
        )

        np.testing.assert_allclose(legacy_cov, typing_cov)

    def test_nested_estimation(self):
        """Test nested estimation type produces identical results."""
        nmodels = 3
        np.random.seed(1)
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1, dtype=float)

        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = LegacyGroupACV(legacy_stat, costs, est_type="nested")

        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov))
        typing_est = TypingGroupACVNested(
            typing_stat, self.typing_bkd.array(costs)
        )

        self.assertEqual(legacy_est.nsubsets(), typing_est.nsubsets())

        npartition_samples_np = np.arange(
            2.0, 2 + legacy_est.nsubsets(), dtype=float
        )
        npartition_samples = self.typing_bkd.array(npartition_samples_np)

        legacy_nsamples = legacy_est._compute_nsamples_per_model(
            npartition_samples_np
        )
        typing_nsamples = typing_est._compute_nsamples_per_model(
            npartition_samples
        )
        np.testing.assert_allclose(legacy_nsamples, typing_nsamples)


class TestMLBLUEAllocationLegacyComparison(unittest.TestCase):
    """Compare typing MLBLUE allocation against legacy."""

    def setUp(self):
        self.legacy_bkd = LegacyBkd
        self.typing_bkd = NumpyBkd()
        np.random.seed(0)

    def _covariance_to_correlation(self, cov):
        """Convert covariance to correlation matrix."""
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)

    def test_mlblue_allocation_nmodels_2(self):
        """Test MLBLUE allocation matches legacy for nmodels=2."""
        from pyapprox.multifidelity.groupacv import (
            MLBLUEEstimator as LegacyMLBLUE,
            GroupACVGradientOptimizer,
        )
        from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
        from pyapprox.typing.statest.groupacv import (
            MLBLUEEstimator as TypingMLBLUE,
        )

        nmodels = 2
        min_nhf_samples = 11
        target_cost = 100

        # Create covariance matrix
        np.random.seed(0)
        cov_np = np.random.normal(0, 1, (nmodels, nmodels))
        cov_np = cov_np.T @ cov_np
        cov_np = self._covariance_to_correlation(cov_np)

        # Costs
        costs_np = np.flip(np.logspace(-nmodels + 1, 0, nmodels))

        # Legacy estimator
        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov_np)
        legacy_est = LegacyMLBLUE(legacy_stat, costs_np, reg_blue=1e-10)
        legacy_opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        legacy_opt.set_estimator(legacy_est)
        legacy_est.set_optimizer(legacy_opt)
        legacy_iterate = legacy_est._init_guess(target_cost)
        legacy_est.allocate_samples(
            target_cost, min_nhf_samples, iterate=legacy_iterate
        )

        # Typing estimator
        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov_np))
        typing_est = TypingMLBLUE(
            typing_stat, self.typing_bkd.array(costs_np), reg_blue=1e-10
        )

        # Use same allocation as legacy (skip optimization for now)
        typing_est._set_optimized_params(
            self.typing_bkd.array(legacy_est._rounded_npartition_samples),
            round_nsamples=False,
        )

        # Compare results
        print(f"\nLegacy npartition_samples: {legacy_est._rounded_npartition_samples}")
        print(f"Legacy nsamples_per_model: {legacy_est._rounded_nsamples_per_model}")
        print(f"Legacy covariance: {legacy_est._optimized_covariance}")

        print(f"\nTyping npartition_samples: {typing_est._rounded_npartition_samples}")
        print(f"Typing nsamples_per_model: {typing_est._rounded_nsamples_per_model}")

        # Check that with same allocation, we get same nsamples_per_model
        np.testing.assert_allclose(
            legacy_est._rounded_nsamples_per_model,
            self.typing_bkd.to_numpy(typing_est._rounded_nsamples_per_model),
        )

        # Check covariance computation matches
        typing_cov = typing_est._covariance_from_npartition_samples(
            typing_est._rounded_npartition_samples
        )
        np.testing.assert_allclose(
            legacy_est._optimized_covariance,
            self.typing_bkd.to_numpy(typing_cov),
            rtol=1e-10,
        )

    def test_mlblue_optimization_comparison(self):
        """Test MLBLUE optimization produces similar results to legacy."""
        from pyapprox.multifidelity.groupacv import (
            MLBLUEEstimator as LegacyMLBLUE,
            GroupACVGradientOptimizer,
        )
        from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
        from pyapprox.typing.statest.groupacv import (
            MLBLUEEstimator as TypingMLBLUE,
            MLBLUEObjective,
            GroupACVCostConstraint,
        )
        from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.typing.optimization.minimize.scipy.diffevol import (
            ScipyDifferentialEvolutionOptimizer,
        )

        nmodels = 2
        min_nhf_samples = 11
        target_cost = 100

        # Create covariance matrix
        np.random.seed(0)
        cov_np = np.random.normal(0, 1, (nmodels, nmodels))
        cov_np = cov_np.T @ cov_np
        cov_np = self._covariance_to_correlation(cov_np)

        # Costs
        costs_np = np.flip(np.logspace(-nmodels + 1, 0, nmodels))

        # Legacy estimator and optimization
        legacy_stat = multioutput_stats["mean"](1, backend=self.legacy_bkd)
        legacy_stat.set_pilot_quantities(cov_np)
        legacy_est = LegacyMLBLUE(legacy_stat, costs_np, reg_blue=1e-10)
        legacy_opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        legacy_opt.set_estimator(legacy_est)
        legacy_est.set_optimizer(legacy_opt)
        legacy_iterate = legacy_est._init_guess(target_cost)
        legacy_est.allocate_samples(
            target_cost, min_nhf_samples, iterate=legacy_iterate
        )

        print(f"\n=== Legacy Results ===")
        print(f"npartition_samples: {legacy_est._rounded_npartition_samples}")
        print(f"nsamples_per_model: {legacy_est._rounded_nsamples_per_model}")
        print(f"covariance: {legacy_est._optimized_covariance}")

        # Typing estimator - manually set up optimization
        typing_stat = MultiOutputMean(1, self.typing_bkd)
        typing_stat.set_pilot_quantities(self.typing_bkd.array(cov_np))
        typing_est = TypingMLBLUE(
            typing_stat, self.typing_bkd.array(costs_np), reg_blue=1e-10
        )

        # Set up objective and constraint (use MLBLUEObjective for analytical jacobian)
        objective = MLBLUEObjective(self.typing_bkd)
        objective.set_estimator(typing_est)

        constraint = GroupACVCostConstraint(self.typing_bkd)
        constraint.set_estimator(typing_est)
        constraint.set_budget(target_cost, min_nhf_samples)

        # Set up bounds
        max_npartition_samples = target_cost / float(costs_np.min()) + 1
        bounds = self.typing_bkd.array(
            [[0.0, max_npartition_samples]] * typing_est.npartitions()
        )

        # Create optimizer
        global_opt = ScipyDifferentialEvolutionOptimizer(
            maxiter=3,
            polish=False,
            seed=1,
            tol=1e-8,
            raise_on_failure=False,
        )
        local_opt = ScipyTrustConstrOptimizer(
            gtol=1e-6,
            maxiter=2000,
        )
        optimizer = ChainedOptimizer(global_opt, local_opt)
        optimizer.bind(objective, bounds, [constraint])

        # Run optimization
        typing_iterate = typing_est._init_guess(target_cost)
        result = optimizer.minimize(typing_iterate)

        print(f"\n=== Typing Results ===")
        print(f"success: {result.success()}")
        print(f"optima: {result.optima().flatten()}")
        print(f"objective: {result.fun()}")

        # Apply the result
        typing_est._set_optimized_params(result.optima()[:, 0], round_nsamples=True)

        print(f"npartition_samples: {typing_est._rounded_npartition_samples}")
        print(f"nsamples_per_model: {typing_est._rounded_nsamples_per_model}")

        typing_cov = typing_est._covariance_from_npartition_samples(
            typing_est._rounded_npartition_samples
        )
        print(f"covariance: {typing_cov}")

        # The optimizations may not match exactly due to different solvers,
        # but the covariance should be similar
        np.testing.assert_allclose(
            legacy_est._optimized_covariance,
            self.typing_bkd.to_numpy(typing_cov),
            rtol=0.1,  # Allow 10% tolerance since optimizers differ
        )


if __name__ == "__main__":
    unittest.main()
