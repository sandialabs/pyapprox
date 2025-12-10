"""Tests for MultiOutputMean statistic."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.protocols import (
    StatisticProtocol,
    StatisticWithCovarianceProtocol,
    StatisticWithDiscrepancyProtocol,
)


class TestMultiOutputMean(unittest.TestCase):
    """Tests for MultiOutputMean statistic."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_protocol_compliance(self):
        """Test that MultiOutputMean satisfies all protocol levels."""
        stat = MultiOutputMean(nqoi=2, bkd=self.bkd)

        self.assertIsInstance(stat, StatisticProtocol)
        self.assertIsInstance(stat, StatisticWithCovarianceProtocol)
        self.assertIsInstance(stat, StatisticWithDiscrepancyProtocol)

    def test_nstats(self):
        """Test nstats returns nqoi for mean."""
        stat = MultiOutputMean(nqoi=3, bkd=self.bkd)
        self.assertEqual(stat.nstats(), 3)

    def test_min_nsamples(self):
        """Test min_nsamples returns 1 for mean."""
        stat = MultiOutputMean(nqoi=2, bkd=self.bkd)
        self.assertEqual(stat.min_nsamples(), 1)

    def test_sample_estimate_single_qoi(self):
        """Test sample mean computation for single QoI."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        values = self.bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = stat.sample_estimate(values)

        expected = np.array([3.0])
        np.testing.assert_allclose(
            self.bkd.to_numpy(result), expected, rtol=1e-10
        )

    def test_sample_estimate_multi_qoi(self):
        """Test sample mean computation for multiple QoIs."""
        stat = MultiOutputMean(nqoi=2, bkd=self.bkd)
        values = self.bkd.asarray([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ])
        result = stat.sample_estimate(values)

        expected = np.array([2.0, 20.0])
        np.testing.assert_allclose(
            self.bkd.to_numpy(result), expected, rtol=1e-10
        )

    def test_compute_pilot_quantities(self):
        """Test pilot covariance computation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)

        # Create correlated pilot samples for 2 models
        np.random.seed(42)
        n_pilot = 100
        x = np.random.randn(n_pilot)
        model0 = x + np.random.randn(n_pilot) * 0.1
        model1 = 0.8 * x + np.random.randn(n_pilot) * 0.2

        pilot_values = [
            self.bkd.asarray(model0.reshape(-1, 1)),
            self.bkd.asarray(model1.reshape(-1, 1)),
        ]

        cov, means = stat.compute_pilot_quantities(pilot_values)

        # Check shapes
        self.assertEqual(cov.shape, (2, 2))
        self.assertEqual(means.shape, (2, 1))

        # Check covariance is positive semi-definite
        cov_np = self.bkd.to_numpy(cov)
        eigvals = np.linalg.eigvalsh(cov_np)
        self.assertTrue(np.all(eigvals >= -1e-10))

        # Check means are reasonable
        means_np = self.bkd.to_numpy(means)
        np.testing.assert_allclose(
            means_np[0, 0], np.mean(model0), rtol=1e-10
        )
        np.testing.assert_allclose(
            means_np[1, 0], np.mean(model1), rtol=1e-10
        )

    def test_set_pilot_quantities(self):
        """Test setting pilot quantities directly."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)

        cov = self.bkd.asarray([[1.0, 0.8], [0.8, 1.0]])
        stat.set_pilot_quantities(cov)

        result_cov = stat.cov()
        np.testing.assert_allclose(
            self.bkd.to_numpy(result_cov),
            self.bkd.to_numpy(cov),
            rtol=1e-10
        )

    def test_cov_raises_without_pilot(self):
        """Test cov() raises error if pilot not set."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)

        with self.assertRaises(ValueError):
            stat.cov()

    def test_high_fidelity_estimator_covariance(self):
        """Test HF estimator covariance computation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)

        # Set pilot covariance
        variance = 4.0
        cov = self.bkd.asarray([[variance]])
        stat.set_pilot_quantities(cov)

        # Covariance of mean should be Var/n
        nhf = 100
        result = stat.high_fidelity_estimator_covariance(nhf)

        expected = variance / nhf
        np.testing.assert_allclose(
            self.bkd.to_numpy(result)[0, 0], expected, rtol=1e-10
        )

    def test_validate_values_wrong_shape(self):
        """Test validation catches wrong value shapes."""
        stat = MultiOutputMean(nqoi=2, bkd=self.bkd)

        # 1D array should fail
        values_1d = self.bkd.asarray([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            stat.sample_estimate(values_1d)

        # Wrong number of columns should fail
        values_wrong_cols = self.bkd.asarray([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            stat.sample_estimate(values_wrong_cols)

    def test_cv_discrepancy_covariances_basic(self):
        """Test CV discrepancy covariance computation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)

        # Set up 2-model covariance
        cov = self.bkd.asarray([
            [1.0, 0.8],
            [0.8, 1.0],
        ])
        stat.set_pilot_quantities(cov)

        # 2 partitions: HF only (10), shared (20)
        npartition = self.bkd.asarray([10, 20])

        CF, cf = stat.get_cv_discrepancy_covariances(npartition)

        # CF = Cov(Q0, Q1) / n_shared
        expected_CF = 0.8 / 20
        np.testing.assert_allclose(
            self.bkd.to_numpy(CF)[0, 0], expected_CF, rtol=1e-10
        )

        # cf = Var(Q1) / n_shared
        expected_cf = 1.0 / 20
        np.testing.assert_allclose(
            self.bkd.to_numpy(cf)[0, 0], expected_cf, rtol=1e-10
        )

    def test_repr(self):
        """Test string representation."""
        stat = MultiOutputMean(nqoi=3, bkd=self.bkd)
        repr_str = repr(stat)

        self.assertIn("MultiOutputMean", repr_str)
        self.assertIn("nqoi=3", repr_str)
        self.assertIn("has_pilot_cov=False", repr_str)


class TestMultiOutputMeanMultiQoI(unittest.TestCase):
    """Tests for MultiOutputMean with multiple QoIs."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_pilot_quantities_multi_qoi(self):
        """Test pilot covariance with multiple QoIs."""
        nqoi = 2
        stat = MultiOutputMean(nqoi=nqoi, bkd=self.bkd)

        # Create pilot samples for 2 models, 2 QoIs
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

        cov, means = stat.compute_pilot_quantities(pilot_values)

        # Check shapes: (nmodels * nqoi, nmodels * nqoi)
        expected_cov_shape = (2 * nqoi, 2 * nqoi)
        self.assertEqual(cov.shape, expected_cov_shape)

        # Check means shape
        self.assertEqual(means.shape, (2, nqoi))

        # Verify covariance is symmetric
        cov_np = self.bkd.to_numpy(cov)
        np.testing.assert_allclose(cov_np, cov_np.T, rtol=1e-10)

    def test_hf_covariance_multi_qoi(self):
        """Test HF estimator covariance extracts correct block."""
        nqoi = 2
        stat = MultiOutputMean(nqoi=nqoi, bkd=self.bkd)

        # Set up 2-model covariance (4x4 for 2 models, 2 QoIs)
        cov = self.bkd.asarray([
            [1.0, 0.5, 0.8, 0.4],
            [0.5, 2.0, 0.4, 0.6],
            [0.8, 0.4, 1.1, 0.55],
            [0.4, 0.6, 0.55, 2.2],
        ])
        stat.set_pilot_quantities(cov)

        nhf = 50
        hf_cov = stat.high_fidelity_estimator_covariance(nhf)

        # Should extract upper-left 2x2 block and divide by n
        expected = np.array([
            [1.0 / nhf, 0.5 / nhf],
            [0.5 / nhf, 2.0 / nhf],
        ])
        np.testing.assert_allclose(
            self.bkd.to_numpy(hf_cov), expected, rtol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
