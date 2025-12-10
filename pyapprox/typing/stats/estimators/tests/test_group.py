"""Tests for Group ACV estimators."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.estimators.group import (
    GroupACVEstimator,
    MLBLUEEstimator,
)
from pyapprox.typing.stats.protocols import EstimatorProtocol


class TestGroupACVEstimator(unittest.TestCase):
    """Tests for GroupACVEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def _create_stat(self, nmodels=3):
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])[:nmodels, :nmodels]
        stat.set_pilot_quantities(self.bkd.asarray(cov))
        return stat

    def test_protocol_compliance(self):
        """Test that GroupACVEstimator satisfies EstimatorProtocol."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        gacv = GroupACVEstimator(stat, costs, self.bkd)
        self.assertIsInstance(gacv, EstimatorProtocol)

    def test_custom_groups(self):
        """Test custom group specification."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        groups = [[0, 1], [0, 2]]
        gacv = GroupACVEstimator(stat, costs, self.bkd, groups=groups)

        self.assertEqual(gacv.ngroups(), 2)
        self.assertEqual(gacv.groups(), groups)

    def test_invalid_groups(self):
        """Test error for invalid groups."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])

        # Group with invalid model index
        with self.assertRaises(ValueError):
            GroupACVEstimator(stat, costs, self.bkd, groups=[[0, 5]])

        # Groups without HF model
        with self.assertRaises(ValueError):
            GroupACVEstimator(stat, costs, self.bkd, groups=[[1, 2]])

    def test_allocate_samples(self):
        """Test sample allocation with groups."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        gacv = GroupACVEstimator(stat, costs, self.bkd)
        gacv.allocate_samples(target_cost=100.0)

        nsamples = gacv.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # All models should have samples
        self.assertTrue(np.all(nsamples_np > 0))

    def test_group_samples(self):
        """Test group samples computation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        groups = [[0, 1], [2]]
        gacv = GroupACVEstimator(stat, costs, self.bkd, groups=groups)
        gacv.allocate_samples(target_cost=100.0)

        group_samples = gacv.group_samples()

        self.assertEqual(len(group_samples), 2)
        self.assertTrue(all(n >= 0 for n in group_samples))

    def test_estimate(self):
        """Test Group ACV estimation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        gacv = GroupACVEstimator(stat, costs, self.bkd)
        gacv.allocate_samples(target_cost=100.0)

        # Generate samples
        np.random.seed(42)
        def rvs(n):
            return self.bkd.asarray(np.random.randn(n, 1))

        samples = gacv.generate_samples_per_model(rvs)

        # Create values
        nsamples = gacv.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        values = []
        for m in range(3):
            n = int(nsamples_np[m])
            values.append(self.bkd.asarray(np.random.randn(n, 1)))

        estimate = gacv(values)
        self.assertEqual(estimate.shape, (1,))

    def test_repr(self):
        """Test string representation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        gacv = GroupACVEstimator(stat, costs, self.bkd)

        repr_str = repr(gacv)
        self.assertIn("GroupACVEstimator", repr_str)
        self.assertIn("nmodels=3", repr_str)


class TestMLBLUEEstimator(unittest.TestCase):
    """Tests for MLBLUEEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # MLMC-like covariance
        cov = self.bkd.asarray([
            [1.0, 0.99, 0.9],
            [0.99, 1.0, 0.99],
            [0.9, 0.99, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_protocol_compliance(self):
        """Test that MLBLUEEstimator satisfies EstimatorProtocol."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlblue = MLBLUEEstimator(stat, costs, self.bkd)
        self.assertIsInstance(mlblue, EstimatorProtocol)

    def test_default_groups(self):
        """Test default MLMC-style groups."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlblue = MLBLUEEstimator(stat, costs, self.bkd)

        groups = mlblue.groups()

        # Should have MLMC-style groups
        # [[0], [0,1], [1,2]]
        self.assertEqual(len(groups), 3)
        self.assertIn(0, groups[0])

    def test_blue_weights(self):
        """Test BLUE weights sum to 1."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlblue = MLBLUEEstimator(stat, costs, self.bkd)
        mlblue.allocate_samples(target_cost=1000.0)

        weights = mlblue.blue_weights()
        weights_np = self.bkd.to_numpy(weights)

        # BLUE weights should sum to 1 (unbiasedness constraint)
        np.testing.assert_allclose(np.sum(weights_np), 1.0, rtol=1e-5)

    def test_allocate_samples(self):
        """Test MLBLUE sample allocation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlblue = MLBLUEEstimator(stat, costs, self.bkd)
        mlblue.allocate_samples(target_cost=1000.0)

        nsamples = mlblue.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # All levels should have samples
        self.assertTrue(np.all(nsamples_np > 0))

    def test_estimate(self):
        """Test MLBLUE estimation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlblue = MLBLUEEstimator(stat, costs, self.bkd)
        mlblue.allocate_samples(target_cost=1000.0)

        nsamples = mlblue.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # Create values
        np.random.seed(42)
        values = []
        for m in range(3):
            n = int(nsamples_np[m])
            values.append(self.bkd.asarray(np.random.randn(n, 1)))

        estimate = mlblue(values)
        self.assertEqual(estimate.shape, (1,))

    def test_optimized_covariance(self):
        """Test optimized covariance computation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlblue = MLBLUEEstimator(stat, costs, self.bkd)
        mlblue.allocate_samples(target_cost=1000.0)

        cov = mlblue.optimized_covariance()
        cov_np = self.bkd.to_numpy(cov)

        # Covariance should be finite and non-negative (can be diagonal)
        self.assertTrue(np.all(np.isfinite(cov_np)))
        # Diagonal entries should be non-negative
        self.assertTrue(np.all(np.diag(cov_np) >= 0))

    def test_repr(self):
        """Test string representation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlblue = MLBLUEEstimator(stat, costs, self.bkd)

        repr_str = repr(mlblue)
        self.assertIn("MLBLUEEstimator", repr_str)
        self.assertIn("nlevels=3", repr_str)


if __name__ == "__main__":
    unittest.main()
