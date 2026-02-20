"""Tests for ACV allocation module."""

from typing import Any, Generic
import unittest

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.statest.acv.allocation import (
    ACVAllocationResult,
    ACVAllocator,
    AnalyticalAllocator,
    default_allocator_factory,
)
from pyapprox.typing.statest.statistics import MultiOutputMean
from pyapprox.typing.statest.acv.variants import (
    GMFEstimator,
    MFMCEstimator,
    MLMCEstimator,
)


class TestACVAllocationResult(Generic[Array], unittest.TestCase):
    """Tests for ACVAllocationResult dataclass."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_allocation_result_creation(self):
        """Test dataclass instantiation."""
        result = ACVAllocationResult(
            partition_ratios=self._bkd.array([1.0, 2.0]),
            continuous_npartition_samples=self._bkd.array([10.0, 10.0, 20.0]),
            objective_value=self._bkd.array([-5.0]),
            npartition_samples=self._bkd.array([10, 10, 20], dtype=int),
            nsamples_per_model=self._bkd.array([10, 20, 30], dtype=int),
            target_cost=100.0,
            actual_cost=95.0,
            success=True,
            message="",
        )
        self.assertTrue(result.success)
        self._bkd.assert_allclose(
            self._bkd.asarray([result.target_cost]),
            self._bkd.asarray([100.0]),
        )

    def test_allocation_result_frozen(self):
        """Test immutability."""
        result = ACVAllocationResult(
            partition_ratios=self._bkd.array([1.0, 2.0]),
            continuous_npartition_samples=self._bkd.array([10.0, 10.0, 20.0]),
            objective_value=self._bkd.array([-5.0]),
            npartition_samples=self._bkd.array([10, 10, 20], dtype=int),
            nsamples_per_model=self._bkd.array([10, 20, 30], dtype=int),
            target_cost=100.0,
            actual_cost=95.0,
            success=True,
            message="",
        )
        with self.assertRaises(AttributeError):
            result.success = False  # type: ignore

    def test_allocation_result_objective_value_is_array(self):
        """Test that objective_value is an Array, not float."""
        result = ACVAllocationResult(
            partition_ratios=self._bkd.array([1.0, 2.0]),
            continuous_npartition_samples=self._bkd.array([10.0, 10.0, 20.0]),
            objective_value=self._bkd.array([-5.0]),
            npartition_samples=self._bkd.array([10, 10, 20], dtype=int),
            nsamples_per_model=self._bkd.array([10, 20, 30], dtype=int),
            target_cost=100.0,
            actual_cost=95.0,
            success=True,
            message="",
        )
        # objective_value should be an array with shape (1,)
        self.assertEqual(result.objective_value.shape, (1,))
        self._bkd.assert_allclose(result.objective_value, self._bkd.array([-5.0]))


class TestACVAllocationResultNumpy(TestACVAllocationResult[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestACVAllocationResultTorch(TestACVAllocationResult[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestACVAllocator(Generic[Array], unittest.TestCase):
    """Tests for ACVAllocator with both backends.

    NumpyBkd estimators are automatically cloned to TorchBkd for optimization,
    with results converted back.
    """

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_stat_and_costs(self, nmodels: int = 3, nqoi: int = 1):
        """Helper to create test statistic and costs."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(self._bkd.array(cov))

        costs = self._bkd.array([10.0 ** (nmodels - 1 - i) for i in range(nmodels)])
        return stat, costs

    def test_acv_allocator_success(self):
        """Basic allocation works."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        self.assertTrue(result.success)
        self.assertLessEqual(result.actual_cost, result.target_cost)

    def test_acv_allocator_budget_too_small(self):
        """Returns failure when budget too small."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=0.1)  # Too small
        self.assertFalse(result.success)
        self.assertIn("Budget too small", result.message)

    def test_acv_allocator_respects_budget(self):
        """actual_cost <= target_cost."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        target_cost = 500.0
        result = allocator.allocate(target_cost=target_cost)
        if result.success:
            self.assertLessEqual(result.actual_cost, target_cost)

    def test_acv_allocator_objective_value_is_array(self):
        """objective_value should be Array, not float."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        if result.success:
            self.assertEqual(result.objective_value.shape, (1,))

    def test_acv_allocator_failure_objective_value_is_array(self):
        """Even failed allocations should have Array objective_value."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=0.1)  # Will fail
        self.assertFalse(result.success)
        self.assertEqual(result.objective_value.shape, (1,))
        self.assertTrue(float(result.objective_value[0]) == float("inf"))

    def test_acv_allocator_result_backend_matches(self):
        """Result arrays are in the estimator's backend type."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        self.assertTrue(result.success)
        # All array fields should be the same type as the backend's arrays
        expected_type = type(self._bkd.array([0.0]))
        self.assertIsInstance(result.partition_ratios, expected_type)
        self.assertIsInstance(result.continuous_npartition_samples, expected_type)
        self.assertIsInstance(result.objective_value, expected_type)
        self.assertIsInstance(result.npartition_samples, expected_type)
        self.assertIsInstance(result.nsamples_per_model, expected_type)


class TestACVAllocatorNumpy(TestACVAllocator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestACVAllocatorTorch(TestACVAllocator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestAnalyticalAllocator(Generic[Array], unittest.TestCase):
    """Tests for AnalyticalAllocator."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_mfmc_stat_and_costs(self, nmodels: int = 3, nqoi: int = 1):
        """Create stat and costs that satisfy MFMC hierarchy requirements."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.95 ** max(i, j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(self._bkd.array(cov))

        costs = self._bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def _create_mlmc_stat_and_costs(self, nmodels: int = 3, nqoi: int = 1):
        """Create stat and costs that satisfy MLMC requirements (decreasing cost)."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(self._bkd.array(cov))

        costs = self._bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_default_allocator_factory_gmf(self):
        """Returns ACVAllocator for GMF (no analytical method)."""
        stat, costs = self._create_mfmc_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = default_allocator_factory(est)
        self.assertIsInstance(allocator, ACVAllocator)

    def test_default_allocator_factory_mfmc_returns_analytical(self):
        """Returns AnalyticalAllocator for MFMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_mfmc_stat_and_costs()
        est = MFMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        self.assertIsInstance(allocator, AnalyticalAllocator)

    def test_default_allocator_factory_mlmc_returns_analytical(self):
        """Returns AnalyticalAllocator for MLMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_mlmc_stat_and_costs()
        est = MLMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        self.assertIsInstance(allocator, AnalyticalAllocator)

    def test_analytical_allocator_mfmc_success(self):
        """AnalyticalAllocator works with MFMC."""
        stat, costs = self._create_mfmc_stat_and_costs()
        est = MFMCEstimator(stat, costs)
        allocator = AnalyticalAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        self.assertTrue(result.success)
        self.assertLessEqual(result.actual_cost, result.target_cost)
        self.assertEqual(result.objective_value.shape, (1,))

    def test_analytical_allocator_mlmc_success(self):
        """AnalyticalAllocator works with MLMC."""
        stat, costs = self._create_mlmc_stat_and_costs()
        est = MLMCEstimator(stat, costs)
        allocator = AnalyticalAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        self.assertTrue(result.success)
        self.assertLessEqual(result.actual_cost, result.target_cost)
        self.assertEqual(result.objective_value.shape, (1,))

    def test_analytical_allocator_budget_too_small(self):
        """AnalyticalAllocator returns failure when budget too small."""
        stat, costs = self._create_mfmc_stat_and_costs()
        est = MFMCEstimator(stat, costs)
        allocator = AnalyticalAllocator(est)
        result = allocator.allocate(target_cost=0.1)  # Too small
        self.assertFalse(result.success)
        self.assertIn("Budget too small", result.message)


class TestAnalyticalAllocatorNumpy(TestAnalyticalAllocator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAnalyticalAllocatorTorch(TestAnalyticalAllocator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestAllocatorFactory(Generic[Array], unittest.TestCase):
    """Tests for default_allocator_factory."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_stat_and_costs(self, nmodels: int = 3, nqoi: int = 1):
        """Helper to create test statistic and costs."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(self._bkd.array(cov))
        costs = self._bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_factory_returns_acv_for_gmf(self):
        """Returns ACVAllocator for GMF."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = default_allocator_factory(est)
        self.assertIsInstance(allocator, ACVAllocator)

    def test_factory_returns_analytical_for_mfmc(self):
        """Returns AnalyticalAllocator for MFMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_stat_and_costs()
        est = MFMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        self.assertIsInstance(allocator, AnalyticalAllocator)

    def test_factory_returns_analytical_for_mlmc(self):
        """Returns AnalyticalAllocator for MLMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_stat_and_costs()
        est = MLMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        self.assertIsInstance(allocator, AnalyticalAllocator)


class TestAllocatorFactoryNumpy(TestAllocatorFactory[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAllocatorFactoryTorch(TestAllocatorFactory[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestEstimatorAllocationAPI(Generic[Array], unittest.TestCase):
    """Tests for ACVEstimator allocation management methods."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_stat_and_costs(self, nmodels: int = 3, nqoi: int = 1):
        """Helper to create test statistic and costs."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(self._bkd.array(cov))
        costs = self._bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_estimator_has_allocation_initially_false(self):
        """has_allocation returns False before allocation is set."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        self.assertFalse(est.has_allocation)

    def test_estimator_allocation_not_set_raises(self):
        """allocation() raises RuntimeError when not set."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        with self.assertRaises(RuntimeError) as ctx:
            est.allocation()
        self.assertIn("No allocation set", str(ctx.exception))

    def test_estimator_set_allocation(self):
        """set_allocation() stores allocation and updates internal state."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        self.assertTrue(result.success)

        est.set_allocation(result)

        self.assertTrue(est.has_allocation)
        self.assertIs(est.allocation(), result)

        self._bkd.assert_allclose(
            est._rounded_npartition_samples, result.npartition_samples
        )
        self._bkd.assert_allclose(
            est._rounded_nsamples_per_model, result.nsamples_per_model
        )

    def test_estimator_set_failed_allocation_raises(self):
        """set_allocation() raises ValueError for failed allocation."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=0.1)
        self.assertFalse(result.success)

        with self.assertRaises(ValueError) as ctx:
            est.set_allocation(result)
        self.assertIn("Cannot set failed allocation", str(ctx.exception))

    def test_estimator_covariance_from_ratios(self):
        """covariance_from_ratios() computes covariance from continuous ratios."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        partition_ratios = self._bkd.array([2.0, 5.0])
        target_cost = 500.0

        cov = est.covariance_from_ratios(target_cost, partition_ratios)

        self.assertEqual(len(cov.shape), 2)
        self.assertEqual(cov.shape[0], cov.shape[1])

    def test_estimator_npartition_samples_from_ratios(self):
        """npartition_samples_from_ratios() computes sample counts from ratios."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        partition_ratios = self._bkd.array([2.0, 5.0])
        target_cost = 500.0

        npartition_samples = est.npartition_samples_from_ratios(
            target_cost, partition_ratios
        )

        self.assertEqual(len(npartition_samples), est._npartitions)
        self.assertTrue(all(float(n) > 0 for n in npartition_samples))

    def test_estimator_covariance_requires_allocation(self):
        """covariance() raises RuntimeError without allocation."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        with self.assertRaises(RuntimeError) as ctx:
            est.covariance()
        self.assertIn("No allocation set", str(ctx.exception))

    def test_estimator_covariance_with_allocation(self):
        """covariance() returns covariance matrix when allocation is set."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        est.set_allocation(result)

        cov = est.covariance()

        self.assertEqual(len(cov.shape), 2)
        self.assertEqual(cov.shape[0], cov.shape[1])

    def test_allocate_samples_delegates_to_allocator(self):
        """allocate_samples() produces the same result as ACVAllocator."""
        stat, costs = self._create_stat_and_costs()
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        est.allocate_samples(target_cost=1000.0)

        self.assertTrue(est.has_allocation)
        alloc = est.allocation()
        self.assertTrue(alloc.success)
        self.assertLessEqual(alloc.actual_cost, 1000.0)


class TestEstimatorAllocationAPINumpy(TestEstimatorAllocationAPI[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEstimatorAllocationAPITorch(TestEstimatorAllocationAPI[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
