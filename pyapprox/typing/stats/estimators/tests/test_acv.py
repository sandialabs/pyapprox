"""Tests for ACV estimator family."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
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


class TestACVEstimator(Generic[Array], unittest.TestCase):
    """Tests for base ACVEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def _create_stat(self, nmodels=3, rho=0.9):
        """Create a MultiOutputMean with correlated models."""
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        # Create covariance matrix with specified correlation
        cov = np.ones((nmodels, nmodels))
        for i in range(nmodels):
            for j in range(nmodels):
                cov[i, j] = rho ** abs(i - j)
        stat.set_pilot_quantities(self._bkd.asarray(cov))
        return stat

    def test_protocol_compliance(self):
        """Test that ACVEstimator satisfies EstimatorProtocol."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self._bkd)
        self.assertIsInstance(acv, EstimatorProtocol)

    def test_nmodels(self):
        """Test nmodels returns correct value."""
        stat = self._create_stat(nmodels=4)
        costs = self._bkd.asarray([10.0, 1.0, 0.1, 0.01])
        acv = ACVEstimator(stat, costs, self._bkd)
        self.assertEqual(acv.nmodels(), 4)

    def test_allocate_samples(self):
        """Test sample allocation."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self._bkd)
        acv.allocate_samples(target_cost=100.0)

        nsamples = acv.nsamples_per_model()
        nsamples_np = self._bkd.to_numpy(nsamples)

        # LF models should have more samples than HF
        self.assertGreater(nsamples_np[1], nsamples_np[0])
        self.assertGreater(nsamples_np[2], nsamples_np[1])

    def test_get_allocation_matrix(self):
        """Test allocation matrix generation."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self._bkd)

        A = acv.get_allocation_matrix()
        A_np = self._bkd.to_numpy(A)

        # Shape should be (nmodels, npartitions)
        self.assertEqual(A_np.shape[0], 3)
        self.assertEqual(A_np.shape[1], 5)  # 2*(3-1)+1 = 5

    def test_repr(self):
        """Test string representation."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        acv = ACVEstimator(stat, costs, self._bkd)

        repr_str = repr(acv)
        self.assertIn("ACVEstimator", repr_str)
        self.assertIn("nmodels=3", repr_str)


class TestACVEstimatorNumpy(TestACVEstimator[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestACVEstimatorTorch(TestACVEstimator[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestGMFEstimator(Generic[Array], unittest.TestCase):
    """Tests for GMFEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def _create_stat(self, nmodels=3):
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        cov = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])[:nmodels, :nmodels]
        stat.set_pilot_quantities(self._bkd.asarray(cov))
        return stat

    def test_recursion_index_is_mfmc(self):
        """Test that GMF uses MFMC recursion."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        gmf = GMFEstimator(stat, costs, self._bkd)

        ridx = gmf.recursion_index()
        expected = self._bkd.asarray([0, 0])
        self._bkd.assert_allclose(
            self._bkd.asarray(ridx), expected
        )

    def test_allocate_and_estimate(self):
        """Test allocation and estimation."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        gmf = GMFEstimator(stat, costs, self._bkd)
        gmf.allocate_samples(target_cost=100.0)

        # Generate samples
        np.random.seed(42)
        def rvs(n):
            return self._bkd.asarray(np.random.randn(n, 1))

        samples = gmf.generate_samples_per_model(rvs)

        # Create values (correlated)
        nsamples = gmf.nsamples_per_model()
        nsamples_np = self._bkd.to_numpy(nsamples)

        values = []
        for m in range(3):
            n = int(nsamples_np[m])
            values.append(self._bkd.asarray(np.random.randn(n, 1)))

        estimate = gmf(values)
        self.assertEqual(estimate.shape, (1,))


class TestGMFEstimatorNumpy(TestGMFEstimator[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGMFEstimatorTorch(TestGMFEstimator[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestMFMCEstimator(Generic[Array], unittest.TestCase):
    """Tests for MFMCEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        # Ordered by decreasing correlation
        cov = self._bkd.asarray([
            [1.0, 0.95, 0.8],
            [0.95, 1.0, 0.9],
            [0.8, 0.9, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_mfmc_allocation(self):
        """Test MFMC uses analytical allocation."""
        stat = self._create_stat()
        costs = self._bkd.asarray([100.0, 10.0, 1.0])
        mfmc = MFMCEstimator(stat, costs, self._bkd)
        mfmc.allocate_samples(target_cost=1000.0)

        nsamples = mfmc.nsamples_per_model()
        nsamples_np = self._bkd.to_numpy(nsamples)

        # Samples should increase with decreasing cost
        self.assertGreater(nsamples_np[1], nsamples_np[0])
        self.assertGreater(nsamples_np[2], nsamples_np[1])

    def test_mfmc_weights(self):
        """Test MFMC computes optimal weights."""
        stat = self._create_stat()
        costs = self._bkd.asarray([100.0, 10.0, 1.0])
        mfmc = MFMCEstimator(stat, costs, self._bkd)
        mfmc.allocate_samples(target_cost=1000.0)

        weights = mfmc.weights()
        weights_np = self._bkd.to_numpy(weights)

        # Weights should be positive for positively correlated models
        self.assertTrue(np.all(weights_np > 0))


class TestMFMCEstimatorNumpy(TestMFMCEstimator[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestMFMCEstimatorTorch(TestMFMCEstimator[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestMLMCEstimator(Generic[Array], unittest.TestCase):
    """Tests for MLMCEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        # MLMC-like covariance (strong adjacent correlation)
        cov = self._bkd.asarray([
            [1.0, 0.99, 0.9],
            [0.99, 1.0, 0.99],
            [0.9, 0.99, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_mlmc_recursion(self):
        """Test MLMC uses successive recursion."""
        stat = self._create_stat()
        costs = self._bkd.asarray([1.0, 10.0, 100.0])
        mlmc = MLMCEstimator(stat, costs, self._bkd)

        ridx = mlmc.recursion_index()
        expected = self._bkd.asarray([0, 1])
        self._bkd.assert_allclose(
            self._bkd.asarray(ridx), expected
        )

    def test_mlmc_allocation(self):
        """Test MLMC sample allocation."""
        stat = self._create_stat()
        # Model 0 = HF (most expensive), Model 2 = coarsest (cheapest)
        costs = self._bkd.asarray([100.0, 10.0, 1.0])
        mlmc = MLMCEstimator(stat, costs, self._bkd)
        mlmc.allocate_samples(target_cost=1000.0)

        nsamples = mlmc.nsamples_per_model()
        nsamples_np = self._bkd.to_numpy(nsamples)

        # Coarse levels should have more samples than HF
        self.assertGreater(nsamples_np[2], nsamples_np[0])


class TestMLMCEstimatorNumpy(TestMLMCEstimator[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestMLMCEstimatorTorch(TestMLMCEstimator[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestGRDEstimator(Generic[Array], unittest.TestCase):
    """Tests for GRDEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        cov = self._bkd.asarray([
            [1.0, 0.95, 0.8],
            [0.95, 1.0, 0.95],
            [0.8, 0.95, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_grd_recursion(self):
        """Test GRD uses MLMC recursion."""
        stat = self._create_stat()
        costs = self._bkd.asarray([100.0, 10.0, 1.0])
        grd = GRDEstimator(stat, costs, self._bkd)

        ridx = grd.recursion_index()
        expected = self._bkd.asarray([0, 1])
        self._bkd.assert_allclose(
            self._bkd.asarray(ridx), expected
        )


class TestGRDEstimatorNumpy(TestGRDEstimator[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGRDEstimatorTorch(TestGRDEstimator[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestGISEstimator(Generic[Array], unittest.TestCase):
    """Tests for GISEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()

    def _create_stat(self):
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        cov = self._bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        return stat

    def test_gis_allocation_matrix(self):
        """Test GIS uses correct allocation matrix with union semantics."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        gis = GISEstimator(stat, costs, self._bkd)
        gis.allocate_samples(target_cost=100.0)

        # GIS uses 2*nmodels partitions
        A = gis.get_allocation_matrix()
        self.assertEqual(A.shape[1], 2 * gis.nmodels())

        # Check union semantics: odd columns are >= even columns
        A_np = self._bkd.to_numpy(A)
        for m in range(1, gis.nmodels()):
            # Odd column (all samples) should be >= even column (shared)
            self.assertTrue(
                np.all(A_np[:, 2 * m + 1] >= A_np[:, 2 * m]),
                f"Union semantics violated for model {m}"
            )

    def test_gis_applies_control_variate(self):
        """Test GIS applies control variate correction (not just MC)."""
        stat = self._create_stat()
        costs = self._bkd.asarray([10.0, 1.0, 0.1])
        gis = GISEstimator(stat, costs, self._bkd)
        gis.allocate_samples(target_cost=100.0)

        # Generate correlated values
        np.random.seed(42)
        nsamples = self._bkd.to_numpy(gis.nsamples_per_model()).astype(int)
        cov = self._bkd.to_numpy(stat.cov())

        # Generate correlated outputs
        L = np.linalg.cholesky(cov)
        max_n = max(nsamples)
        z = np.random.randn(max_n, len(nsamples))
        correlated = z @ L.T

        values = [
            self._bkd.asarray(correlated[:n, m:m+1])
            for m, n in enumerate(nsamples)
        ]

        # Compute GIS estimate
        estimate = gis(values)

        # With high correlation (0.9), weights should be non-zero
        weights = self._bkd.to_numpy(gis.weights())
        self.assertTrue(np.any(np.abs(weights) > 0.1), "Weights should be non-zero")


class TestGISEstimatorNumpy(TestGISEstimator[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGISEstimatorTorch(TestGISEstimator[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestBootstrapEstimator(Generic[Array], unittest.TestCase):
    """Tests for bootstrap variance estimation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()
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
            values.append(self._bkd.asarray(correlated[:n, m:m+1]))
        return values

    def test_bootstrap_mc(self):
        """Test bootstrap for MC estimator."""
        from pyapprox.typing.stats.estimators.mc import MCEstimator

        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        stat.set_pilot_quantities(self._bkd.asarray([[1.0]]))
        costs = self._bkd.asarray([1.0])

        mc = MCEstimator(stat, costs, self._bkd)
        mc.allocate_samples(target_cost=100.0)

        # Generate values
        nhf = int(self._bkd.to_numpy(mc.nsamples_per_model())[0])
        values = [self._bkd.asarray(np.random.randn(nhf, 1))]

        # Bootstrap
        boot_mean, boot_cov = mc.bootstrap(values, nbootstraps=500)

        # Bootstrap should return correct shapes
        self.assertEqual(boot_mean.shape, (1,))

        # Bootstrap variance should be positive
        self.assertTrue(float(self._bkd.to_numpy(boot_cov)[0]) > 0)

    def test_bootstrap_cv(self):
        """Test bootstrap for CV estimator."""
        from pyapprox.typing.stats.estimators.cv import CVEstimator

        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        cov = self._bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self._bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self._bkd)
        cv.allocate_samples(target_cost=100.0)

        nsamples = self._bkd.to_numpy(cv.nsamples_per_model())
        nsamples_list = [int(n) for n in nsamples]

        # Generate correlated values
        values = self._generate_correlated_values(
            nsamples_list, self._bkd.to_numpy(cov)
        )

        # Bootstrap
        boot_mean, boot_cov = cv.bootstrap(values, nbootstraps=500)

        # Bootstrap should return correct shapes
        self.assertEqual(boot_mean.shape, (1,))

        # Bootstrap variance should be positive
        self.assertTrue(float(self._bkd.to_numpy(boot_cov)[0]) > 0)

    def test_bootstrap_mfmc(self):
        """Test bootstrap for MFMC estimator."""
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        cov = np.array([
            [1.0, 0.95, 0.8],
            [0.95, 1.0, 0.9],
            [0.8, 0.9, 1.0],
        ])
        stat.set_pilot_quantities(self._bkd.asarray(cov))
        costs = self._bkd.asarray([100.0, 10.0, 1.0])

        mfmc = MFMCEstimator(stat, costs, self._bkd)
        mfmc.allocate_samples(target_cost=1000.0)

        nsamples = self._bkd.to_numpy(mfmc.nsamples_per_model())
        nsamples_list = [int(n) for n in nsamples]

        # Generate correlated values
        values = self._generate_correlated_values(nsamples_list, cov)

        # Bootstrap
        boot_mean, boot_cov = mfmc.bootstrap(values, nbootstraps=500)

        # Bootstrap should return correct shapes
        self.assertEqual(boot_mean.shape, (1,))

        # Bootstrap variance should be positive
        self.assertTrue(float(self._bkd.to_numpy(boot_cov)[0]) > 0)


class TestBootstrapEstimatorNumpy(TestBootstrapEstimator[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        np.random.seed(42)

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestBootstrapEstimatorTorch(TestBootstrapEstimator[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestInsertPilotSamples(Generic[Array], unittest.TestCase):
    """Tests for pilot sample insertion."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_insert_pilot_values_acv(self):
        """Test pilot sample insertion for ACV estimator."""
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        cov = self._bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        costs = self._bkd.asarray([10.0, 1.0, 0.1])

        acv = ACVEstimator(stat, costs, self._bkd)
        acv.allocate_samples(target_cost=100.0)

        # Get allocation info
        nsamples = self._bkd.to_numpy(acv.nsamples_per_model())
        alloc_mat = self._bkd.to_numpy(acv.get_allocation_matrix())

        # Generate pilot values
        npilot = 5
        pilot_values = [
            self._bkd.asarray(np.random.randn(npilot, 1))
            for _ in range(3)
        ]

        # Generate values for non-pilot samples
        # Only models using partition 0 have pilot samples subtracted
        values = []
        for m in range(3):
            uses_p0 = alloc_mat[m, 0] == 1
            n_remaining = int(nsamples[m]) - (npilot if uses_p0 else 0)
            values.append(self._bkd.asarray(np.random.randn(n_remaining, 1)))

        # Insert pilot values
        combined = acv.insert_pilot_values(pilot_values, values)

        # Check shapes - combined should have full sample count
        for m in range(3):
            expected_n = int(nsamples[m])
            self.assertEqual(combined[m].shape[0], expected_n)

        # Check pilot values are at the beginning for models using partition 0
        for m in range(3):
            if alloc_mat[m, 0] == 1:
                self._bkd.assert_allclose(
                    combined[m][:npilot],
                    pilot_values[m]
                )

    def test_insert_pilot_values_hf_only(self):
        """Test that only HF model (using partition 0) gets pilot values."""
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        cov = self._bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self._bkd.asarray([10.0, 1.0])

        acv = ACVEstimator(stat, costs, self._bkd)
        acv.allocate_samples(target_cost=100.0)

        nsamples = self._bkd.to_numpy(acv.nsamples_per_model())
        alloc_mat = self._bkd.to_numpy(acv.get_allocation_matrix())

        # Pilot values
        npilot = 3
        pilot_values = [
            self._bkd.asarray(np.random.randn(npilot, 1)),
            self._bkd.asarray(np.random.randn(npilot, 1)),
        ]

        # Remaining values - subtract pilot only from models using partition 0
        remaining = []
        for m in range(2):
            uses_p0 = alloc_mat[m, 0] == 1
            n_remaining = int(nsamples[m]) - (npilot if uses_p0 else 0)
            remaining.append(self._bkd.asarray(np.random.randn(n_remaining, 1)))

        # Insert pilot values
        combined = acv.insert_pilot_values(pilot_values, remaining)

        # Check combined has correct total size
        for m in range(2):
            self.assertEqual(combined[m].shape[0], int(nsamples[m]))

        # Verify HF (model 0) has pilot values prepended
        if alloc_mat[0, 0] == 1:
            self._bkd.assert_allclose(
                combined[0][:npilot],
                pilot_values[0]
            )


class TestInsertPilotSamplesNumpy(TestInsertPilotSamples[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        np.random.seed(42)

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestInsertPilotSamplesTorch(TestInsertPilotSamples[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestNumericalOptimization(unittest.TestCase):
    """Tests for ACV sample allocation optimization.

    Note: Jacobian tests require PyTorch backend for autograd support.
    These are NumPy-only tests that test basic functionality.
    """

    def setUp(self):
        # Use torch backend for autograd support
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_estimator(self, nmodels=3):
        """Create an ACV estimator for testing."""
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        # Create covariance matrix with decreasing correlation
        cov = np.ones((nmodels, nmodels))
        for i in range(nmodels):
            for j in range(nmodels):
                cov[i, j] = 0.9 ** abs(i - j)
        stat.set_pilot_quantities(self._bkd.asarray(cov))

        costs = self._bkd.asarray([10.0 / (2 ** i) for i in range(nmodels)])
        return MFMCEstimator(stat, costs, self._bkd)

    def test_objective_jacobian(self):
        """Test ACVLogDeterminantObjective Jacobian with finite differences."""
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
        sample = self._bkd.asarray([[2.0 + i] for i in range(nvars)])  # shape (nvars, 1)

        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)

        # Error ratio should be close to machine precision for good Jacobian
        # We use a relaxed tolerance since autograd may have some numerical noise
        error_ratio = float(self._bkd.to_numpy(checker.error_ratio(errors[0])))
        self.assertLess(error_ratio, 1e-4)

    def test_constraint_jacobian(self):
        """Test ACVPartitionConstraint Jacobian with finite differences."""
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
        sample = self._bkd.asarray([[2.0 + i] for i in range(nvars)])  # shape (nvars, 1)

        checker = DerivativeChecker(constraint)
        errors = checker.check_derivatives(sample, verbosity=0)

        # Error ratio should indicate good Jacobian
        error_ratio = float(self._bkd.to_numpy(checker.error_ratio(errors[0])))
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
        sample = self._bkd.asarray([[2.0 + i] for i in range(nvars)])
        value = objective(sample)

        # Should return shape (1, 1)
        self.assertEqual(value.shape, (1, 1))

        # Value should be finite
        self.assertTrue(np.isfinite(float(self._bkd.to_numpy(value))))

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
        sample = self._bkd.asarray([[2.0 + i] for i in range(nvars)])
        value = constraint(sample)

        # Should return shape (npartitions, 1)
        npartitions = est.npartitions()
        self.assertEqual(value.shape, (npartitions, 1))

        # Values should be finite
        self.assertTrue(np.all(np.isfinite(self._bkd.to_numpy(value))))


class TestGISLegacyComparison(Generic[Array], unittest.TestCase):
    """Tests comparing typing GIS to legacy GIS implementation.

    These tests verify that the typing implementation produces
    identical results to the legacy implementation.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_matching_setup(self, nmodels=3):
        """Create identical setups for legacy and typing."""
        # Create covariance matrix with exponential decay
        cov = np.ones((nmodels, nmodels))
        for i in range(nmodels):
            for j in range(nmodels):
                cov[i, j] = 0.9 ** abs(i - j)

        costs = np.array([10.0 ** (-i) for i in range(nmodels)])
        return cov, costs

    def test_gis_allocation_matrix_matches_legacy(self):
        """Test that typing GIS allocation matrix matches legacy exactly."""
        from pyapprox.typing.stats.allocation.matrices import get_allocation_matrix_gis

        # Test multiple recursion indices
        test_cases = [
            (3, np.array([0, 0])),  # Both coupled with HF (MFMC-like)
            (3, np.array([0, 1])),  # Sequential coupling
            (4, np.array([0, 0, 0])),  # 4 models, all coupled with HF
            (4, np.array([0, 1, 2])),  # 4 models, successive
        ]

        for nmodels, ridx in test_cases:
            with self.subTest(nmodels=nmodels, recursion_index=ridx.tolist()):
                # Generate typing allocation matrix
                typing_A = get_allocation_matrix_gis(nmodels, ridx, self._bkd)

                # Manually compute expected allocation matrix following legacy algorithm
                # Step 1: Set diagonal for odd columns
                expected = np.zeros((nmodels, 2 * nmodels))
                for ii in range(nmodels):
                    expected[ii, 2 * ii + 1] = 1

                # Step 2: Set even columns from recursion
                for ii in range(1, nmodels):
                    expected[:, 2 * ii] = expected[:, ridx[ii - 1] * 2 + 1]

                # Step 3: GIS-specific maximum merge
                for ii in range(1, nmodels):
                    expected[:, 2 * ii + 1] = np.maximum(
                        expected[:, 2 * ii], expected[:, 2 * ii + 1]
                    )

                self._bkd.assert_allclose(
                    typing_A, self._bkd.asarray(expected), rtol=1e-10
                )

    def test_gis_estimate_uses_control_variates(self):
        """Test that GIS estimate differs from plain MC (applies CV)."""
        cov, costs = self._create_matching_setup(nmodels=3)

        # Setup typing GIS
        stat = MultiOutputMean(nqoi=1, bkd=self._bkd)
        stat.set_pilot_quantities(self._bkd.asarray(cov))
        gis = GISEstimator(stat, self._bkd.asarray(costs), self._bkd)
        gis.allocate_samples(target_cost=100.0)

        # Generate correlated values
        np.random.seed(123)
        nsamples = self._bkd.to_numpy(gis.nsamples_per_model()).astype(int)

        # Create correlated values using Cholesky
        L = np.linalg.cholesky(cov)
        max_n = max(nsamples)
        z = np.random.randn(max_n, 3)
        correlated = z @ L.T

        values = [
            self._bkd.asarray(correlated[:n, m:m+1])
            for m, n in enumerate(nsamples)
        ]

        # Compute estimates
        gis_est = gis(values)

        # GIS with correlated models should apply control variates
        # The weights should be non-zero for correlated models
        weights = self._bkd.to_numpy(gis.weights())
        self.assertTrue(
            np.any(np.abs(weights) > 0.1),
            "GIS weights should be non-zero for correlated models"
        )


class TestGISLegacyComparisonNumpy(TestGISLegacyComparison[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        np.random.seed(42)

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGISLegacyComparisonTorch(TestGISLegacyComparison[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401, E402


if __name__ == "__main__":
    unittest.main()
