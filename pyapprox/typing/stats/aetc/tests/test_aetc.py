"""Tests for AETC module."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.aetc import AETCEstimator, AETCBLUEEstimator


class TestAETCEstimator(unittest.TestCase):
    """Tests for AETCEstimator."""

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

    def test_allocate_samples(self):
        """Test AETC sample allocation."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        aetc.allocate_samples(target_variance=0.1)

        nsamples = aetc.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        self.assertTrue(np.all(nsamples_np > 0))

    def test_achieved_variance(self):
        """Test that achieved variance is computed."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        aetc.allocate_samples(target_variance=0.1)

        variance = aetc.achieved_variance()
        self.assertGreater(variance, 0)

    def test_total_cost(self):
        """Test that total cost is computed."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        aetc.allocate_samples(target_variance=0.1)

        cost = aetc.total_cost()
        self.assertGreater(cost, 0)

    def test_iteration_history(self):
        """Test that iteration history is recorded."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        aetc.allocate_samples(target_variance=0.1)

        history = aetc.iteration_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

    def test_allocate_for_target_cost(self):
        """Test allocate_for_target_cost method."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        aetc.allocate_for_target_cost(target_cost=100.0)

        nsamples = aetc.nsamples_per_model()
        self.assertTrue(np.all(self.bkd.to_numpy(nsamples) > 0))

    def test_estimate(self):
        """Test AETC estimation."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        aetc.allocate_for_target_cost(target_cost=100.0)

        nsamples = aetc.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        np.random.seed(42)
        values = []
        for n in nsamples_np:
            n = int(n)
            values.append(self.bkd.asarray(np.random.randn(n, 1)))

        estimate = aetc(values)
        self.assertEqual(estimate.shape, (1,))

    def test_different_base_estimators(self):
        """Test with different base estimators."""
        for base in ["mfmc", "mlmc", "acv"]:
            aetc = AETCEstimator(
                self.stat, self.costs, self.bkd, base_estimator=base
            )
            aetc.allocate_for_target_cost(target_cost=100.0)
            self.assertIsNotNone(aetc.nsamples_per_model())

    def test_repr_before_allocation(self):
        """Test repr before allocation."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        repr_str = repr(aetc)
        self.assertIn("not allocated", repr_str)

    def test_repr_after_allocation(self):
        """Test repr after allocation."""
        aetc = AETCEstimator(self.stat, self.costs, self.bkd)
        aetc.allocate_for_target_cost(target_cost=100.0)
        repr_str = repr(aetc)
        self.assertIn("AETCEstimator", repr_str)


class TestAETCBLUEEstimator(unittest.TestCase):
    """Tests for AETCBLUEEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # MLMC-like covariance
        cov = self.bkd.asarray([
            [1.0, 0.99, 0.9],
            [0.99, 1.0, 0.99],
            [0.9, 0.99, 1.0],
        ])
        self.stat.set_pilot_quantities(cov)
        self.costs = self.bkd.asarray([1.0, 10.0, 100.0])

    def test_allocate_samples(self):
        """Test AETC-BLUE sample allocation."""
        aetc_blue = AETCBLUEEstimator(self.stat, self.costs, self.bkd)
        aetc_blue.allocate_samples(target_variance=0.1)

        nsamples = aetc_blue.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        self.assertTrue(np.all(nsamples_np > 0))

    def test_blue_weights(self):
        """Test that BLUE weights are computed."""
        aetc_blue = AETCBLUEEstimator(self.stat, self.costs, self.bkd)
        aetc_blue.allocate_samples(target_variance=0.1)

        weights = aetc_blue.blue_weights()
        weights_np = self.bkd.to_numpy(weights)

        # BLUE weights should sum to 1
        np.testing.assert_allclose(np.sum(weights_np), 1.0, rtol=1e-5)

    def test_iteration_history_has_weights(self):
        """Test that iteration history includes BLUE weights."""
        aetc_blue = AETCBLUEEstimator(self.stat, self.costs, self.bkd)
        aetc_blue.allocate_samples(target_variance=0.1)

        history = aetc_blue.iteration_history()
        self.assertIn("blue_weights", history[0])

    def test_estimate(self):
        """Test AETC-BLUE estimation."""
        aetc_blue = AETCBLUEEstimator(self.stat, self.costs, self.bkd)
        aetc_blue.allocate_samples(target_variance=0.1)

        nsamples = aetc_blue.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        np.random.seed(42)
        values = []
        for n in nsamples_np:
            n = int(n)
            values.append(self.bkd.asarray(np.random.randn(n, 1)))

        estimate = aetc_blue(values)
        self.assertEqual(estimate.shape, (1,))

    def test_repr(self):
        """Test repr."""
        aetc_blue = AETCBLUEEstimator(self.stat, self.costs, self.bkd)
        repr_str = repr(aetc_blue)
        self.assertIn("not allocated", repr_str)

        aetc_blue.allocate_samples(target_variance=0.1)
        repr_str = repr(aetc_blue)
        self.assertIn("AETCBLUEEstimator", repr_str)


if __name__ == "__main__":
    unittest.main()
