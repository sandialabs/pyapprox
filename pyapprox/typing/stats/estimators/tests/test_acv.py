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


class TestBootstrapEstimator(unittest.TestCase):
    """Tests for bootstrap variance estimation."""

    def setUp(self):
        self.bkd = NumpyBkd()
        np.random.seed(42)

    def _generate_correlated_values(self, nsamples_per_model, cov):
        """Generate correlated model outputs from multivariate normal."""
        nmodels = len(nsamples_per_model)
        # Generate correlated samples for the maximum sample size
        max_n = max(nsamples_per_model)
        # Use Cholesky to generate correlated samples
        L = np.linalg.cholesky(cov)
        z = np.random.randn(max_n, nmodels)
        correlated = z @ L.T

        values = []
        for m in range(nmodels):
            n = nsamples_per_model[m]
            values.append(self.bkd.asarray(correlated[:n, m:m+1]))
        return values

    def test_bootstrap_mc(self):
        """Test bootstrap for MC estimator."""
        from pyapprox.typing.stats.estimators.mc import MCEstimator

        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        stat.set_pilot_quantities(self.bkd.asarray([[1.0]]))
        costs = self.bkd.asarray([1.0])

        mc = MCEstimator(stat, costs, self.bkd)
        mc.allocate_samples(target_cost=100.0)

        # Generate values
        nhf = int(self.bkd.to_numpy(mc.nsamples_per_model())[0])
        values = [self.bkd.asarray(np.random.randn(nhf, 1))]

        # Bootstrap
        boot_mean, boot_cov = mc.bootstrap(values, nbootstraps=500)

        # Bootstrap should return correct shapes
        self.assertEqual(boot_mean.shape, (1,))

        # Bootstrap variance should be positive
        self.assertTrue(float(self.bkd.to_numpy(boot_cov)[0]) > 0)

    def test_bootstrap_cv(self):
        """Test bootstrap for CV estimator."""
        from pyapprox.typing.stats.estimators.cv import CVEstimator

        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        cv.allocate_samples(target_cost=100.0)

        nsamples = self.bkd.to_numpy(cv.nsamples_per_model())
        nsamples_list = [int(n) for n in nsamples]

        # Generate correlated values
        values = self._generate_correlated_values(
            nsamples_list, self.bkd.to_numpy(cov)
        )

        # Bootstrap
        boot_mean, boot_cov = cv.bootstrap(values, nbootstraps=500)

        # Bootstrap should return correct shapes
        self.assertEqual(boot_mean.shape, (1,))

        # Bootstrap variance should be positive
        self.assertTrue(float(self.bkd.to_numpy(boot_cov)[0]) > 0)

    def test_bootstrap_mfmc(self):
        """Test bootstrap for MFMC estimator."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = np.array([
            [1.0, 0.95, 0.8],
            [0.95, 1.0, 0.9],
            [0.8, 0.9, 1.0],
        ])
        stat.set_pilot_quantities(self.bkd.asarray(cov))
        costs = self.bkd.asarray([100.0, 10.0, 1.0])

        mfmc = MFMCEstimator(stat, costs, self.bkd)
        mfmc.allocate_samples(target_cost=1000.0)

        nsamples = self.bkd.to_numpy(mfmc.nsamples_per_model())
        nsamples_list = [int(n) for n in nsamples]

        # Generate correlated values
        values = self._generate_correlated_values(nsamples_list, cov)

        # Bootstrap
        boot_mean, boot_cov = mfmc.bootstrap(values, nbootstraps=500)

        # Bootstrap should return correct shapes
        self.assertEqual(boot_mean.shape, (1,))

        # Bootstrap variance should be positive
        self.assertTrue(float(self.bkd.to_numpy(boot_cov)[0]) > 0)


class TestInsertPilotSamples(unittest.TestCase):
    """Tests for pilot sample insertion."""

    def setUp(self):
        self.bkd = NumpyBkd()
        np.random.seed(42)

    def test_insert_pilot_values_acv(self):
        """Test pilot sample insertion for ACV estimator."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0, 0.1])

        acv = ACVEstimator(stat, costs, self.bkd)
        acv.allocate_samples(target_cost=100.0)

        # Get allocation info
        nsamples = self.bkd.to_numpy(acv.nsamples_per_model())
        alloc_mat = self.bkd.to_numpy(acv.get_allocation_matrix())

        # Generate pilot values
        npilot = 5
        pilot_values = [
            self.bkd.asarray(np.random.randn(npilot, 1))
            for _ in range(3)
        ]

        # Generate values for non-pilot samples
        # Only models using partition 0 have pilot samples subtracted
        values = []
        for m in range(3):
            uses_p0 = alloc_mat[m, 0] == 1
            n_remaining = int(nsamples[m]) - (npilot if uses_p0 else 0)
            values.append(self.bkd.asarray(np.random.randn(n_remaining, 1)))

        # Insert pilot values
        combined = acv.insert_pilot_values(pilot_values, values)

        # Check shapes - combined should have full sample count
        for m in range(3):
            expected_n = int(nsamples[m])
            self.assertEqual(combined[m].shape[0], expected_n)

        # Check pilot values are at the beginning for models using partition 0
        for m in range(3):
            if alloc_mat[m, 0] == 1:
                np.testing.assert_array_equal(
                    self.bkd.to_numpy(combined[m][:npilot]),
                    self.bkd.to_numpy(pilot_values[m])
                )

    def test_insert_pilot_values_hf_only(self):
        """Test that only HF model (using partition 0) gets pilot values."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        acv = ACVEstimator(stat, costs, self.bkd)
        acv.allocate_samples(target_cost=100.0)

        nsamples = self.bkd.to_numpy(acv.nsamples_per_model())
        alloc_mat = self.bkd.to_numpy(acv.get_allocation_matrix())

        # Pilot values
        npilot = 3
        pilot_values = [
            self.bkd.asarray(np.random.randn(npilot, 1)),
            self.bkd.asarray(np.random.randn(npilot, 1)),
        ]

        # Remaining values - subtract pilot only from models using partition 0
        remaining = []
        for m in range(2):
            uses_p0 = alloc_mat[m, 0] == 1
            n_remaining = int(nsamples[m]) - (npilot if uses_p0 else 0)
            remaining.append(self.bkd.asarray(np.random.randn(n_remaining, 1)))

        # Insert pilot values
        combined = acv.insert_pilot_values(pilot_values, remaining)

        # Check combined has correct total size
        for m in range(2):
            self.assertEqual(combined[m].shape[0], int(nsamples[m]))

        # Verify HF (model 0) has pilot values prepended
        if alloc_mat[0, 0] == 1:
            np.testing.assert_array_equal(
                self.bkd.to_numpy(combined[0][:npilot]),
                self.bkd.to_numpy(pilot_values[0])
            )


class TestNumericalOptimization(unittest.TestCase):
    """Tests for ACV sample allocation optimization.

    Note: Jacobian tests require PyTorch backend for autograd support.
    """

    def setUp(self):
        # Use torch backend for autograd support
        try:
            import torch
            from pyapprox.typing.util.backends.torch import TorchBkd
            torch.set_default_dtype(torch.float64)
            self.bkd = TorchBkd()
            self.has_torch = True
        except ImportError:
            self.bkd = NumpyBkd()
            self.has_torch = False
        np.random.seed(42)

    def _create_estimator(self, nmodels=3):
        """Create an ACV estimator for testing."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # Create covariance matrix with decreasing correlation
        cov = np.ones((nmodels, nmodels))
        for i in range(nmodels):
            for j in range(nmodels):
                cov[i, j] = 0.9 ** abs(i - j)
        stat.set_pilot_quantities(self.bkd.asarray(cov))

        costs = self.bkd.asarray([10.0 / (2 ** i) for i in range(nmodels)])
        return MFMCEstimator(stat, costs, self.bkd)

    def test_objective_jacobian(self):
        """Test ACVLogDeterminantObjective Jacobian with finite differences."""
        if not self.has_torch:
            self.skipTest("Jacobian tests require PyTorch")

        from pyapprox.typing.stats.estimators.acv.optimization import (
            ACVLogDeterminantObjective,
        )
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        est = self._create_estimator(nmodels=3)
        target_cost = 100.0

        objective = ACVLogDeterminantObjective()
        objective.set_estimator(est)
        objective.set_target_cost(target_cost)

        # Test at a sample point - partition ratios > 1
        # nvars = npartitions - 1 = 4 for 3 models (5 partitions)
        nvars = objective.nvars()
        sample = self.bkd.asarray([[2.0 + i] for i in range(nvars)])  # shape (nvars, 1)

        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)

        # Error ratio should be close to machine precision for good Jacobian
        # We use a relaxed tolerance since autograd may have some numerical noise
        error_ratio = float(self.bkd.to_numpy(checker.error_ratio(errors[0])))
        self.assertLess(error_ratio, 1e-4)

    def test_constraint_jacobian(self):
        """Test ACVPartitionConstraint Jacobian with finite differences."""
        if not self.has_torch:
            self.skipTest("Jacobian tests require PyTorch")

        from pyapprox.typing.stats.estimators.acv.optimization import (
            ACVPartitionConstraint,
        )
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        est = self._create_estimator(nmodels=3)
        target_cost = 100.0
        est.allocate_samples(target_cost)  # Need to allocate first

        constraint = ACVPartitionConstraint(est, target_cost)

        # Test at a sample point
        # nvars = npartitions - 1 = 4 for 3 models (5 partitions)
        nvars = constraint.nvars()
        sample = self.bkd.asarray([[2.0 + i] for i in range(nvars)])  # shape (nvars, 1)

        checker = DerivativeChecker(constraint)
        errors = checker.check_derivatives(sample, verbosity=0)

        # Error ratio should indicate good Jacobian
        error_ratio = float(self.bkd.to_numpy(checker.error_ratio(errors[0])))
        self.assertLess(error_ratio, 1e-4)

    def test_objective_value_computation(self):
        """Test that objective computes sensible values."""
        from pyapprox.typing.stats.estimators.acv.optimization import (
            ACVLogDeterminantObjective,
        )

        est = self._create_estimator(nmodels=3)
        target_cost = 100.0

        objective = ACVLogDeterminantObjective()
        objective.set_estimator(est)
        objective.set_target_cost(target_cost)

        # Evaluate at a point - nvars = npartitions - 1
        nvars = objective.nvars()
        sample = self.bkd.asarray([[2.0 + i] for i in range(nvars)])
        value = objective(sample)

        # Should return shape (1, 1)
        self.assertEqual(value.shape, (1, 1))

        # Value should be finite
        self.assertTrue(np.isfinite(float(self.bkd.to_numpy(value))))

    def test_constraint_value_computation(self):
        """Test that constraint computes sensible values."""
        from pyapprox.typing.stats.estimators.acv.optimization import (
            ACVPartitionConstraint,
        )

        est = self._create_estimator(nmodels=3)
        target_cost = 100.0
        est.allocate_samples(target_cost)

        constraint = ACVPartitionConstraint(est, target_cost)

        # Evaluate at a point with positive ratios - nvars = npartitions - 1
        nvars = constraint.nvars()
        sample = self.bkd.asarray([[2.0 + i] for i in range(nvars)])
        value = constraint(sample)

        # Should return shape (npartitions, 1)
        npartitions = est.npartitions()
        self.assertEqual(value.shape, (npartitions, 1))

        # Values should be finite
        self.assertTrue(np.all(np.isfinite(self.bkd.to_numpy(value))))


if __name__ == "__main__":
    unittest.main()
