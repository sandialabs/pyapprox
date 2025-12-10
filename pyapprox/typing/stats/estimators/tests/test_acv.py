"""Tests for ACV estimator family."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.estimators.acv import (
    ACVEstimator,
    GMFEstimator,
    GRDEstimator,
    GISEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.typing.stats.protocols import EstimatorProtocol


class TestACVEstimator(unittest.TestCase):
    """Tests for base ACVEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def _create_stat(self, nmodels=3, rho=0.9):
        """Create a MultiOutputMean with correlated models."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # Create covariance matrix with specified correlation
        cov = np.ones((nmodels, nmodels))
        for i in range(nmodels):
            for j in range(nmodels):
                cov[i, j] = rho ** abs(i - j)
        stat.set_pilot_quantities(self.bkd.asarray(cov))
        return stat

    def test_protocol_compliance(self):
        """Test that ACVEstimator satisfies EstimatorProtocol."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self.bkd)
        self.assertIsInstance(acv, EstimatorProtocol)

    def test_nmodels(self):
        """Test nmodels returns correct value."""
        stat = self._create_stat(nmodels=4)
        costs = self.bkd.asarray([10.0, 1.0, 0.1, 0.01])
        acv = ACVEstimator(stat, costs, self.bkd)
        self.assertEqual(acv.nmodels(), 4)

    def test_allocate_samples(self):
        """Test sample allocation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self.bkd)
        acv.allocate_samples(target_cost=100.0)

        nsamples = acv.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # LF models should have more samples than HF
        self.assertGreater(nsamples_np[1], nsamples_np[0])
        self.assertGreater(nsamples_np[2], nsamples_np[1])

    def test_get_allocation_matrix(self):
        """Test allocation matrix generation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self.bkd)

        A = acv.get_allocation_matrix()
        A_np = self.bkd.to_numpy(A)

        # Shape should be (nmodels, npartitions)
        self.assertEqual(A_np.shape[0], 3)
        self.assertEqual(A_np.shape[1], 5)  # 2*(3-1)+1 = 5

    def test_repr(self):
        """Test string representation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self.bkd)

        repr_str = repr(acv)
        self.assertIn("ACVEstimator", repr_str)
        self.assertIn("nmodels=3", repr_str)


class TestGMFEstimator(unittest.TestCase):
    """Tests for GMFEstimator."""

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

    def test_recursion_index_is_mfmc(self):
        """Test that GMF uses MFMC recursion."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        gmf = GMFEstimator(stat, costs, self.bkd)

        ridx = gmf.recursion_index()
        expected = np.array([0, 0])
        np.testing.assert_array_equal(ridx, expected)

    def test_allocate_and_estimate(self):
        """Test allocation and estimation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        gmf = GMFEstimator(stat, costs, self.bkd)
        gmf.allocate_samples(target_cost=100.0)

        # Generate samples
        np.random.seed(42)
        def rvs(n):
            return self.bkd.asarray(np.random.randn(n, 1))

        samples = gmf.generate_samples_per_model(rvs)

        # Create values (correlated)
        nsamples = gmf.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        values = []
        for m in range(3):
            n = int(nsamples_np[m])
            values.append(self.bkd.asarray(np.random.randn(n, 1)))

        estimate = gmf(values)
        self.assertEqual(estimate.shape, (1,))


class TestMFMCEstimator(unittest.TestCase):
    """Tests for MFMCEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # Ordered by decreasing correlation
        cov = self.bkd.asarray([
            [1.0, 0.95, 0.8],
            [0.95, 1.0, 0.9],
            [0.8, 0.9, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_mfmc_allocation(self):
        """Test MFMC uses analytical allocation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([100.0, 10.0, 1.0])
        mfmc = MFMCEstimator(stat, costs, self.bkd)
        mfmc.allocate_samples(target_cost=1000.0)

        nsamples = mfmc.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # Samples should increase with decreasing cost
        self.assertGreater(nsamples_np[1], nsamples_np[0])
        self.assertGreater(nsamples_np[2], nsamples_np[1])

    def test_mfmc_weights(self):
        """Test MFMC computes optimal weights."""
        stat = self._create_stat()
        costs = self.bkd.asarray([100.0, 10.0, 1.0])
        mfmc = MFMCEstimator(stat, costs, self.bkd)
        mfmc.allocate_samples(target_cost=1000.0)

        weights = mfmc.weights()
        weights_np = self.bkd.to_numpy(weights)

        # Weights should be positive for positively correlated models
        self.assertTrue(np.all(weights_np > 0))


class TestMLMCEstimator(unittest.TestCase):
    """Tests for MLMCEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # MLMC-like covariance (strong adjacent correlation)
        cov = self.bkd.asarray([
            [1.0, 0.99, 0.9],
            [0.99, 1.0, 0.99],
            [0.9, 0.99, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_mlmc_recursion(self):
        """Test MLMC uses successive recursion."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlmc = MLMCEstimator(stat, costs, self.bkd)

        ridx = mlmc.recursion_index()
        expected = np.array([0, 1])
        np.testing.assert_array_equal(ridx, expected)

    def test_mlmc_allocation(self):
        """Test MLMC sample allocation."""
        stat = self._create_stat()
        costs = self.bkd.asarray([1.0, 10.0, 100.0])
        mlmc = MLMCEstimator(stat, costs, self.bkd)
        mlmc.allocate_samples(target_cost=1000.0)

        nsamples = mlmc.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # Coarse levels should have more samples
        self.assertGreater(nsamples_np[0], nsamples_np[2])


class TestGRDEstimator(unittest.TestCase):
    """Tests for GRDEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([
            [1.0, 0.95, 0.8],
            [0.95, 1.0, 0.95],
            [0.8, 0.95, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_grd_recursion(self):
        """Test GRD uses MLMC recursion."""
        stat = self._create_stat()
        costs = self.bkd.asarray([100.0, 10.0, 1.0])
        grd = GRDEstimator(stat, costs, self.bkd)

        ridx = grd.recursion_index()
        expected = np.array([0, 1])
        np.testing.assert_array_equal(ridx, expected)


class TestGISEstimator(unittest.TestCase):
    """Tests for GISEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_gis_independent_samples(self):
        """Test GIS generates independent samples."""
        stat = self._create_stat()
        costs = self.bkd.asarray([10.0, 1.0, 0.1])
        gis = GISEstimator(stat, costs, self.bkd)
        gis.allocate_samples(target_cost=100.0)

        np.random.seed(42)
        call_count = [0]
        def rvs(n):
            call_count[0] += 1
            return self.bkd.asarray(np.random.randn(n, 1))

        samples = gis.generate_samples_per_model(rvs)

        # Should call rvs once per model
        self.assertEqual(call_count[0], 3)


if __name__ == "__main__":
    unittest.main()
