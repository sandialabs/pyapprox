"""Tests for MultiOutputVariance statistic."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.variance import MultiOutputVariance
from pyapprox.typing.stats.protocols import (
    StatisticProtocol,
    StatisticWithCovarianceProtocol,
    StatisticWithDiscrepancyProtocol,
)


class TestMultiOutputVariance(unittest.TestCase):
    """Tests for MultiOutputVariance statistic."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_protocol_compliance(self):
        """Test that MultiOutputVariance satisfies all protocol levels."""
        stat = MultiOutputVariance(nqoi=2, bkd=self.bkd)

        self.assertIsInstance(stat, StatisticProtocol)
        self.assertIsInstance(stat, StatisticWithCovarianceProtocol)
        self.assertIsInstance(stat, StatisticWithDiscrepancyProtocol)

    def test_nstats(self):
        """Test nstats returns nqoi for variance."""
        stat = MultiOutputVariance(nqoi=3, bkd=self.bkd)
        self.assertEqual(stat.nstats(), 3)

    def test_min_nsamples(self):
        """Test min_nsamples returns 2 for variance."""
        stat = MultiOutputVariance(nqoi=2, bkd=self.bkd)
        self.assertEqual(stat.min_nsamples(), 2)

    def test_sample_estimate_single_qoi(self):
        """Test sample variance computation for single QoI."""
        stat = MultiOutputVariance(nqoi=1, bkd=self.bkd)
        # Values: 1, 2, 3, 4, 5 -> mean=3, var=2.5 (with Bessel)
        values = self.bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = stat.sample_estimate(values)

        expected = np.array([2.5])
        np.testing.assert_allclose(
            self.bkd.to_numpy(result), expected, rtol=1e-10
        )

    def test_sample_estimate_multi_qoi(self):
        """Test sample variance computation for multiple QoIs."""
        stat = MultiOutputVariance(nqoi=2, bkd=self.bkd)
        values = self.bkd.asarray([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ])
        result = stat.sample_estimate(values)

        # Var([1,2,3]) = 1.0, Var([10,20,30]) = 100.0
        expected = np.array([1.0, 100.0])
        np.testing.assert_allclose(
            self.bkd.to_numpy(result), expected, rtol=1e-10
        )

    def test_compute_pilot_quantities(self):
        """Test pilot quantities computation."""
        stat = MultiOutputVariance(nqoi=1, bkd=self.bkd)

        np.random.seed(42)
        n_pilot = 100
        x = np.random.randn(n_pilot)
        model0 = x + np.random.randn(n_pilot) * 0.1
        model1 = 0.8 * x + np.random.randn(n_pilot) * 0.2

        pilot_values = [
            self.bkd.asarray(model0.reshape(-1, 1)),
            self.bkd.asarray(model1.reshape(-1, 1)),
        ]

        cov, W = stat.compute_pilot_quantities(pilot_values)

        # Check shapes
        # cov shape: (nmodels*nqoi, nmodels*nqoi) = (2, 2)
        self.assertEqual(cov.shape, (2, 2))
        # W shape: (nmodels*nqoi^2, nmodels*nqoi^2) = (2, 2) for nqoi=1
        self.assertEqual(W.shape, (2, 2))

        # Check covariance is positive semi-definite
        cov_np = self.bkd.to_numpy(cov)
        eigvals = np.linalg.eigvalsh(cov_np)
        self.assertTrue(np.all(eigvals >= -1e-10))

    def test_set_pilot_quantities(self):
        """Test setting pilot quantities directly."""
        stat = MultiOutputVariance(nqoi=1, bkd=self.bkd)

        cov = self.bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        stat.set_pilot_quantities(cov)

        result_cov = stat.cov()
        np.testing.assert_allclose(
            self.bkd.to_numpy(result_cov),
            self.bkd.to_numpy(cov),
            rtol=1e-10
        )

    def test_validate_values_insufficient_samples(self):
        """Test validation catches insufficient samples."""
        stat = MultiOutputVariance(nqoi=1, bkd=self.bkd)

        # Only 1 sample should fail (need 2 for variance)
        values = self.bkd.asarray([[1.0]])
        with self.assertRaises(ValueError):
            stat.sample_estimate(values)

    def test_repr(self):
        """Test string representation."""
        stat = MultiOutputVariance(nqoi=3, bkd=self.bkd)
        repr_str = repr(stat)

        self.assertIn("MultiOutputVariance", repr_str)
        self.assertIn("nqoi=3", repr_str)


class TestMultiOutputVarianceNumerical(unittest.TestCase):
    """Numerical tests for variance estimation."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_variance_known_distribution(self):
        """Test variance estimation for known distribution."""
        stat = MultiOutputVariance(nqoi=1, bkd=self.bkd)

        np.random.seed(123)
        # Standard normal: true variance = 1.0
        n = 10000
        samples = np.random.randn(n, 1)
        values = self.bkd.asarray(samples)

        result = stat.sample_estimate(values)
        # Should be close to 1.0
        np.testing.assert_allclose(
            self.bkd.to_numpy(result)[0], 1.0, rtol=0.05
        )

    def test_variance_scaled_distribution(self):
        """Test variance for scaled distribution."""
        stat = MultiOutputVariance(nqoi=1, bkd=self.bkd)

        np.random.seed(456)
        sigma = 3.0
        n = 10000
        samples = sigma * np.random.randn(n, 1)
        values = self.bkd.asarray(samples)

        result = stat.sample_estimate(values)
        # Should be close to sigma^2 = 9.0
        np.testing.assert_allclose(
            self.bkd.to_numpy(result)[0], sigma**2, rtol=0.05
        )


if __name__ == "__main__":
    unittest.main()
