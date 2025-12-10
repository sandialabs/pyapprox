"""Tests for factory module."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.factory import (
    get_estimator,
    BestEstimator,
    compare_estimators,
)
from pyapprox.typing.stats.factory.estimator_factory import (
    list_estimators,
    register_estimator,
)
from pyapprox.typing.stats.factory.comparison import (
    rank_estimators,
    variance_reduction,
)
from pyapprox.typing.stats.protocols import EstimatorProtocol


class TestGetEstimator(unittest.TestCase):
    """Tests for get_estimator factory function."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        self.stat.set_pilot_quantities(cov)
        self.costs = self.bkd.asarray([10.0, 1.0, 0.1])

    def test_get_mc(self):
        """Test creating MC estimator."""
        estimator = get_estimator("mc", self.stat, self.costs)
        self.assertIsInstance(estimator, EstimatorProtocol)
        self.assertEqual(estimator.nmodels(), 1)

    def test_get_cv(self):
        """Test creating CV estimator."""
        # CV requires exactly 2 models
        stat_2 = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov_2 = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat_2.set_pilot_quantities(cov_2)
        costs_2 = self.bkd.asarray([10.0, 1.0])
        estimator = get_estimator("cv", stat_2, costs_2)
        self.assertIsInstance(estimator, EstimatorProtocol)
        self.assertEqual(estimator.nmodels(), 2)

    def test_get_mfmc(self):
        """Test creating MFMC estimator."""
        estimator = get_estimator("mfmc", self.stat, self.costs)
        self.assertIsInstance(estimator, EstimatorProtocol)
        self.assertEqual(estimator.nmodels(), 3)

    def test_get_mlmc(self):
        """Test creating MLMC estimator."""
        estimator = get_estimator("mlmc", self.stat, self.costs)
        self.assertIsInstance(estimator, EstimatorProtocol)
        self.assertEqual(estimator.nmodels(), 3)

    def test_get_acv_variants(self):
        """Test creating ACV variant estimators."""
        for name in ["acv", "gmf", "grd", "gis"]:
            estimator = get_estimator(name, self.stat, self.costs)
            self.assertIsInstance(estimator, EstimatorProtocol)

    def test_get_group_estimators(self):
        """Test creating group estimators."""
        for name in ["groupacv", "group_acv", "mlblue"]:
            estimator = get_estimator(name, self.stat, self.costs)
            self.assertIsInstance(estimator, EstimatorProtocol)

    def test_case_insensitive(self):
        """Test that estimator names are case-insensitive."""
        for name in ["MC", "Mc", "mC", "MFMC", "MfMc"]:
            estimator = get_estimator(name, self.stat, self.costs)
            self.assertIsInstance(estimator, EstimatorProtocol)

    def test_unknown_estimator(self):
        """Test error for unknown estimator."""
        with self.assertRaises(ValueError) as ctx:
            get_estimator("unknown", self.stat, self.costs)
        self.assertIn("Unknown estimator", str(ctx.exception))

    def test_list_estimators(self):
        """Test list_estimators function."""
        estimators = list_estimators()
        self.assertIn("mc", estimators)
        self.assertIn("mfmc", estimators)
        self.assertIn("mlblue", estimators)

    def test_register_estimator(self):
        """Test registering custom estimator."""
        class CustomEstimator:
            def __init__(self, stat, costs, bkd):
                self.stat = stat
                self.costs = costs
                self.bkd = bkd

        register_estimator("custom_test", CustomEstimator)
        self.assertIn("custom_test", list_estimators())


class TestBestEstimator(unittest.TestCase):
    """Tests for BestEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([
            [1.0, 0.95, 0.9],
            [0.95, 1.0, 0.95],
            [0.9, 0.95, 1.0],
        ])
        self.stat.set_pilot_quantities(cov)
        self.costs = self.bkd.asarray([10.0, 1.0, 0.1])

    def test_basic_usage(self):
        """Test basic BestEstimator usage."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)

        # Should have found a best estimator
        estimator = best.best_estimator()
        self.assertIsInstance(estimator, EstimatorProtocol)

    def test_best_estimator_type(self):
        """Test that best_estimator_type returns valid type."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)

        est_type = best.best_estimator_type()
        self.assertIsInstance(est_type, str)

    def test_best_models(self):
        """Test that best_models returns valid models."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)

        models = best.best_models()
        self.assertIsInstance(models, list)
        # Should always include HF model
        self.assertIn(0, models)

    def test_comparison_results(self):
        """Test comparison_results returns results."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)

        results = best.comparison_results()
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

    def test_custom_estimator_types(self):
        """Test with custom estimator types."""
        best = BestEstimator(
            self.stat, self.costs, self.bkd,
            estimator_types=["mfmc", "mlmc"]
        )
        best.allocate_samples(target_cost=100.0)

        est_type = best.best_estimator_type()
        self.assertIn(est_type, ["mfmc", "mlmc"])

    def test_max_nmodels(self):
        """Test max_nmodels parameter."""
        best = BestEstimator(
            self.stat, self.costs, self.bkd,
            max_nmodels=2
        )
        best.allocate_samples(target_cost=100.0)

        models = best.best_models()
        self.assertLessEqual(len(models), 2)

    def test_nsamples_per_model(self):
        """Test nsamples_per_model passthrough."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)

        nsamples = best.nsamples_per_model()
        self.assertTrue(np.all(self.bkd.to_numpy(nsamples) > 0))

    def test_optimized_covariance(self):
        """Test optimized_covariance passthrough."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)

        cov = best.optimized_covariance()
        cov_np = self.bkd.to_numpy(cov)
        self.assertTrue(np.all(np.isfinite(cov_np)))

    def test_estimate(self):
        """Test estimate via __call__."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)

        # Generate values for best models
        nsamples = self.bkd.to_numpy(best.nsamples_per_model())
        values = []
        for n in nsamples:
            n = int(n)
            values.append(self.bkd.asarray(np.random.randn(n, 1)))

        estimate = best(values)
        self.assertEqual(estimate.shape, (1,))

    def test_repr_before_allocate(self):
        """Test repr before allocation."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        repr_str = repr(best)
        self.assertIn("not allocated", repr_str)

    def test_repr_after_allocate(self):
        """Test repr after allocation."""
        best = BestEstimator(self.stat, self.costs, self.bkd)
        best.allocate_samples(target_cost=100.0)
        repr_str = repr(best)
        self.assertIn("BestEstimator", repr_str)


class TestCompareEstimators(unittest.TestCase):
    """Tests for compare_estimators function."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        self.stat.set_pilot_quantities(cov)
        self.costs = self.bkd.asarray([10.0, 1.0, 0.1])

    def test_basic_comparison(self):
        """Test basic estimator comparison."""
        results = compare_estimators(
            self.stat, self.costs, self.bkd,
            target_cost=100.0
        )

        self.assertIsInstance(results, dict)
        self.assertIn("mc", results)

    def test_specific_estimator_types(self):
        """Test with specific estimator types."""
        results = compare_estimators(
            self.stat, self.costs, self.bkd,
            target_cost=100.0,
            estimator_types=["mc", "mfmc"]
        )

        self.assertIn("mc", results)
        self.assertIn("mfmc", results)

    def test_result_structure(self):
        """Test result structure for successful estimator."""
        results = compare_estimators(
            self.stat, self.costs, self.bkd,
            target_cost=100.0,
            estimator_types=["mc"]
        )

        mc_result = results["mc"]
        self.assertIn("estimator", mc_result)
        self.assertIn("variance", mc_result)
        self.assertIn("nsamples", mc_result)
        self.assertIn("total_cost", mc_result)

    def test_rank_estimators(self):
        """Test ranking estimators."""
        results = compare_estimators(
            self.stat, self.costs, self.bkd,
            target_cost=100.0,
            estimator_types=["mc", "mfmc", "mlmc"]
        )

        ranking = rank_estimators(results)
        self.assertIsInstance(ranking, list)
        self.assertGreater(len(ranking), 0)

    def test_variance_reduction(self):
        """Test variance reduction computation."""
        results = compare_estimators(
            self.stat, self.costs, self.bkd,
            target_cost=100.0,
            estimator_types=["mc", "mfmc"]
        )

        reductions = variance_reduction(results)
        self.assertIn("mc", reductions)
        self.assertAlmostEqual(reductions["mc"], 1.0)  # MC vs MC = 1

        # MFMC should generally have reduction > 1
        if "mfmc" in reductions:
            self.assertGreater(reductions["mfmc"], 0)


if __name__ == "__main__":
    unittest.main()
