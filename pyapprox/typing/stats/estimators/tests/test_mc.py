"""Tests for MCEstimator."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.estimators.mc import MCEstimator
from pyapprox.typing.stats.protocols import EstimatorProtocol


class TestMCEstimator(unittest.TestCase):
    """Tests for MCEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_protocol_compliance(self):
        """Test that MCEstimator satisfies EstimatorProtocol."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        self.assertIsInstance(mc, EstimatorProtocol)

    def test_nmodels(self):
        """Test nmodels returns 1."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        self.assertEqual(mc.nmodels(), 1)

    def test_allocate_samples(self):
        """Test sample allocation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        mc.allocate_samples(target_cost=100.0)

        nsamples = mc.nsamples_per_model()
        # 100 / 10 = 10 samples
        self.assertEqual(int(self.bkd.to_numpy(nsamples)[0]), 10)

    def test_allocate_samples_min(self):
        """Test allocation respects minimum samples."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([100.0])

        mc = MCEstimator(stat, costs, self.bkd)
        # Very small budget
        mc.allocate_samples(target_cost=10.0)

        nsamples = mc.nsamples_per_model()
        # Should be at least min_nsamples = 1
        self.assertGreaterEqual(int(self.bkd.to_numpy(nsamples)[0]), 1)

    def test_generate_samples(self):
        """Test sample generation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        mc.allocate_samples(target_cost=100.0)

        np.random.seed(42)
        def rvs(n):
            return self.bkd.asarray(np.random.randn(n, 2))

        samples = mc.generate_samples_per_model(rvs)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape[0], 10)
        self.assertEqual(samples[0].shape[1], 2)

    def test_estimate(self):
        """Test Monte Carlo estimation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        mc.allocate_samples(target_cost=100.0)

        # Create sample values
        values = [self.bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0],
                                    [6.0], [7.0], [8.0], [9.0], [10.0]])]

        estimate = mc(values)

        # Mean of 1-10 = 5.5
        np.testing.assert_allclose(
            self.bkd.to_numpy(estimate)[0], 5.5, rtol=1e-10
        )

    def test_optimized_covariance(self):
        """Test covariance computation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        variance = 4.0
        stat.set_pilot_quantities(self.bkd.asarray([[variance]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        mc.allocate_samples(target_cost=100.0)

        cov = mc.optimized_covariance()

        # Var(mean) = variance / n = 4 / 10 = 0.4
        np.testing.assert_allclose(
            self.bkd.to_numpy(cov)[0, 0], 0.4, rtol=1e-10
        )

    def test_total_cost(self):
        """Test total cost computation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        mc.allocate_samples(target_cost=100.0)

        total_cost = mc.total_cost()
        # 10 samples * 10.0 cost = 100.0
        self.assertAlmostEqual(total_cost, 100.0, places=5)

    def test_repr(self):
        """Test string representation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)

        # Before allocation
        repr_str = repr(mc)
        self.assertIn("MCEstimator", repr_str)
        self.assertIn("not allocated", repr_str)

        # After allocation
        mc.allocate_samples(target_cost=100.0)
        repr_str = repr(mc)
        self.assertIn("10", repr_str)


class TestMCEstimatorMultiQoI(unittest.TestCase):
    """Tests for MCEstimator with multiple QoIs."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_multi_qoi_estimate(self):
        """Test estimation with multiple QoIs."""
        stat = MultiOutputMean(nqoi=2, bkd=self.bkd)
        cov = self.bkd.asarray([
            [1.0, 0.5],
            [0.5, 2.0],
        ])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0])

        mc = MCEstimator(stat, costs, self.bkd)
        mc.allocate_samples(target_cost=100.0)

        # Create sample values
        values = [self.bkd.asarray([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
            [8.0, 80.0],
            [9.0, 90.0],
            [10.0, 100.0],
        ])]

        estimate = mc(values)

        # Mean = [5.5, 55.0]
        np.testing.assert_allclose(
            self.bkd.to_numpy(estimate), [5.5, 55.0], rtol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
