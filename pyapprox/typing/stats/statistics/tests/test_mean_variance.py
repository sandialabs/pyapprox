"""Tests for MultiOutputMeanAndVariance statistic."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean_variance import MultiOutputMeanAndVariance
from pyapprox.typing.stats.protocols import (
    StatisticProtocol,
    StatisticWithCovarianceProtocol,
    StatisticWithDiscrepancyProtocol,
)


class TestMultiOutputMeanAndVariance(unittest.TestCase):
    """Tests for MultiOutputMeanAndVariance statistic."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_protocol_compliance(self):
        """Test that MultiOutputMeanAndVariance satisfies all protocol levels."""
        stat = MultiOutputMeanAndVariance(nqoi=2, bkd=self.bkd)

        self.assertIsInstance(stat, StatisticProtocol)
        self.assertIsInstance(stat, StatisticWithCovarianceProtocol)
        self.assertIsInstance(stat, StatisticWithDiscrepancyProtocol)

    def test_nstats(self):
        """Test nstats returns 2*nqoi for mean+variance."""
        stat = MultiOutputMeanAndVariance(nqoi=3, bkd=self.bkd)
        self.assertEqual(stat.nstats(), 6)

    def test_min_nsamples(self):
        """Test min_nsamples returns 2."""
        stat = MultiOutputMeanAndVariance(nqoi=2, bkd=self.bkd)
        self.assertEqual(stat.min_nsamples(), 2)

    def test_sample_estimate_single_qoi(self):
        """Test sample mean+variance computation for single QoI."""
        stat = MultiOutputMeanAndVariance(nqoi=1, bkd=self.bkd)
        # Values: 1, 2, 3, 4, 5 -> mean=3, var=2.5
        values = self.bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = stat.sample_estimate(values)

        expected = np.array([3.0, 2.5])  # [mean, var]
        np.testing.assert_allclose(
            self.bkd.to_numpy(result), expected, rtol=1e-10
        )

    def test_sample_estimate_multi_qoi(self):
        """Test sample mean+variance computation for multiple QoIs."""
        stat = MultiOutputMeanAndVariance(nqoi=2, bkd=self.bkd)
        values = self.bkd.asarray([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ])
        result = stat.sample_estimate(values)

        # mean([1,2,3]) = 2, var([1,2,3]) = 1
        # mean([10,20,30]) = 20, var([10,20,30]) = 100
        expected = np.array([2.0, 20.0, 1.0, 100.0])
        np.testing.assert_allclose(
            self.bkd.to_numpy(result), expected, rtol=1e-10
        )

    def test_compute_pilot_quantities(self):
        """Test pilot quantities computation."""
        stat = MultiOutputMeanAndVariance(nqoi=1, bkd=self.bkd)

        np.random.seed(42)
        n_pilot = 100
        x = np.random.randn(n_pilot)
        model0 = x + np.random.randn(n_pilot) * 0.1
        model1 = 0.8 * x + np.random.randn(n_pilot) * 0.2

        pilot_values = [
            self.bkd.asarray(model0.reshape(-1, 1)),
            self.bkd.asarray(model1.reshape(-1, 1)),
        ]

        cov, means, variances = stat.compute_pilot_quantities(pilot_values)

        # Check shapes
        # 2 models * 2 stats (mean + var) * 1 qoi = 4
        self.assertEqual(cov.shape, (4, 4))
        self.assertEqual(means.shape, (2, 1))
        self.assertEqual(variances.shape, (2, 1))

        # Check covariance is symmetric
        cov_np = self.bkd.to_numpy(cov)
        np.testing.assert_allclose(cov_np, cov_np.T, rtol=1e-10)

    def test_set_pilot_quantities(self):
        """Test setting pilot quantities directly."""
        stat = MultiOutputMeanAndVariance(nqoi=1, bkd=self.bkd)

        # 2 models * 2 stats = 4x4 covariance
        cov = self.bkd.asarray([
            [1.0, 0.5, 0.1, 0.05],
            [0.5, 1.0, 0.05, 0.1],
            [0.1, 0.05, 0.5, 0.25],
            [0.05, 0.1, 0.25, 0.5],
        ])
        stat.set_pilot_quantities(cov)

        result_cov = stat.cov()
        np.testing.assert_allclose(
            self.bkd.to_numpy(result_cov),
            self.bkd.to_numpy(cov),
            rtol=1e-10
        )

    def test_repr(self):
        """Test string representation."""
        stat = MultiOutputMeanAndVariance(nqoi=3, bkd=self.bkd)
        repr_str = repr(stat)

        self.assertIn("MultiOutputMeanAndVariance", repr_str)
        self.assertIn("nqoi=3", repr_str)


class TestMultiOutputMeanAndVarianceMultiQoI(unittest.TestCase):
    """Tests for MultiOutputMeanAndVariance with multiple QoIs."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_pilot_quantities_multi_qoi(self):
        """Test pilot quantities with multiple QoIs."""
        nqoi = 2
        stat = MultiOutputMeanAndVariance(nqoi=nqoi, bkd=self.bkd)

        np.random.seed(123)
        n_pilot = 200

        # Model 0: correlated QoIs
        x = np.random.randn(n_pilot)
        model0 = np.column_stack([
            x + np.random.randn(n_pilot) * 0.1,
            2 * x + np.random.randn(n_pilot) * 0.2,
        ])

        # Model 1: also correlated with model 0
        model1 = np.column_stack([
            0.9 * x + np.random.randn(n_pilot) * 0.15,
            1.8 * x + np.random.randn(n_pilot) * 0.25,
        ])

        pilot_values = [
            self.bkd.asarray(model0),
            self.bkd.asarray(model1),
        ]

        cov, means, variances = stat.compute_pilot_quantities(pilot_values)

        # Check shapes
        # 2 models * 2 stats * 2 qoi = 8
        expected_cov_shape = (2 * 2 * nqoi, 2 * 2 * nqoi)
        self.assertEqual(cov.shape, expected_cov_shape)
        self.assertEqual(means.shape, (2, nqoi))
        self.assertEqual(variances.shape, (2, nqoi))

        # Verify covariance is symmetric
        cov_np = self.bkd.to_numpy(cov)
        np.testing.assert_allclose(cov_np, cov_np.T, rtol=1e-10)


class TestMeanAndVarianceNumerical(unittest.TestCase):
    """Numerical tests for mean+variance estimation."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_known_distribution(self):
        """Test estimation for known normal distribution."""
        stat = MultiOutputMeanAndVariance(nqoi=1, bkd=self.bkd)

        np.random.seed(789)
        mu = 5.0
        sigma = 2.0
        n = 10000

        samples = mu + sigma * np.random.randn(n, 1)
        values = self.bkd.asarray(samples)

        result = stat.sample_estimate(values)
        result_np = self.bkd.to_numpy(result)

        # Mean should be close to mu
        np.testing.assert_allclose(result_np[0], mu, rtol=0.02)
        # Variance should be close to sigma^2
        np.testing.assert_allclose(result_np[1], sigma**2, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
