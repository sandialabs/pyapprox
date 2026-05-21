"""Tests for ACV allocation module."""

import numpy as np
import pytest

from pyapprox.statest.acv.allocation import (
    ACVAllocator,
    AnalyticalAllocator,
    default_allocator_factory,
)
from pyapprox.statest.acv.base import FittedACVEstimator
from pyapprox.statest.acv.result import ACVAllocationResult
from pyapprox.statest.acv.variants import (
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.statistics import MultiOutputMean
from tests._helpers.markers import slow_test


class TestACVAllocationResult:
    """Tests for ACVAllocationResult dataclass."""

    def test_allocation_result_creation(self, bkd):
        """Test dataclass instantiation."""
        result = ACVAllocationResult(
            partition_ratios=bkd.array([1.0, 2.0]),
            continuous_npartition_samples=bkd.array([10.0, 10.0, 20.0]),
            objective_value=bkd.array([-5.0]),
            npartition_samples=bkd.array([10, 10, 20], dtype=int),
            nsamples_per_model=bkd.array([10, 20, 30], dtype=int),
            target_cost=100.0,
            actual_cost=95.0,
            success=True,
            message="",
        )
        assert result.success
        bkd.assert_allclose(
            bkd.asarray([result.target_cost]),
            bkd.asarray([100.0]),
        )

    def test_allocation_result_frozen(self, bkd):
        """Test immutability."""
        result = ACVAllocationResult(
            partition_ratios=bkd.array([1.0, 2.0]),
            continuous_npartition_samples=bkd.array([10.0, 10.0, 20.0]),
            objective_value=bkd.array([-5.0]),
            npartition_samples=bkd.array([10, 10, 20], dtype=int),
            nsamples_per_model=bkd.array([10, 20, 30], dtype=int),
            target_cost=100.0,
            actual_cost=95.0,
            success=True,
            message="",
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore

    def test_allocation_result_objective_value_is_array(self, bkd):
        """Test that objective_value is an Array, not float."""
        result = ACVAllocationResult(
            partition_ratios=bkd.array([1.0, 2.0]),
            continuous_npartition_samples=bkd.array([10.0, 10.0, 20.0]),
            objective_value=bkd.array([-5.0]),
            npartition_samples=bkd.array([10, 10, 20], dtype=int),
            nsamples_per_model=bkd.array([10, 20, 30], dtype=int),
            target_cost=100.0,
            actual_cost=95.0,
            success=True,
            message="",
        )
        # objective_value should be an array with shape (1,)
        assert result.objective_value.shape == (1,)
        bkd.assert_allclose(result.objective_value, bkd.array([-5.0]))


class TestACVAllocator:
    """Tests for ACVAllocator with both backends.

    NumpyBkd estimators are automatically cloned to TorchBkd for optimization,
    with results converted back.
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_stat_and_costs(self, bkd, nmodels: int = 3, nqoi: int = 1):
        """Helper to create test statistic and costs."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(bkd.array(cov))

        costs = bkd.array([10.0 ** (nmodels - 1 - i) for i in range(nmodels)])
        return stat, costs

    def test_acv_allocator_success(self, bkd):
        """Basic allocation works."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        assert result.success
        assert result.actual_cost <= result.target_cost

    def test_acv_allocator_budget_too_small(self, bkd):
        """Returns failure when budget too small."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=0.1)  # Too small
        assert not result.success
        assert "Budget too small" in result.message

    def test_acv_allocator_respects_budget(self, bkd):
        """actual_cost <= target_cost."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        target_cost = 500.0
        result = allocator.allocate(target_cost=target_cost)
        if result.success:
            assert result.actual_cost <= target_cost

    def test_acv_allocator_objective_value_is_array(self, bkd):
        """objective_value should be Array, not float."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        if result.success:
            assert result.objective_value.shape == (1,)

    def test_acv_allocator_failure_objective_value_is_array(self, bkd):
        """Even failed allocations should have Array objective_value."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=0.1)  # Will fail
        assert not result.success
        assert result.objective_value.shape == (1,)
        assert float(result.objective_value[0]) == float("inf")

    def test_acv_allocator_result_backend_matches(self, bkd):
        """Result arrays are in the estimator's backend type."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        assert result.success
        # All array fields should be the same type as the backend's arrays
        expected_type = type(bkd.array([0.0]))
        assert isinstance(result.partition_ratios, expected_type)
        assert isinstance(result.continuous_npartition_samples, expected_type)
        assert isinstance(result.objective_value, expected_type)
        assert isinstance(result.npartition_samples, expected_type)
        assert isinstance(result.nsamples_per_model, expected_type)


class TestAnalyticalAllocator:
    """Tests for AnalyticalAllocator."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_mfmc_stat_and_costs(self, bkd, nmodels: int = 3, nqoi: int = 1):
        """Create stat and costs that satisfy MFMC hierarchy requirements."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.95 ** max(i, j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(bkd.array(cov))

        costs = bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def _create_mlmc_stat_and_costs(self, bkd, nmodels: int = 3, nqoi: int = 1):
        """Create stat and costs that satisfy MLMC requirements (decreasing cost)."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(bkd.array(cov))

        costs = bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_default_allocator_factory_gmf_chain(self, bkd):
        """Returns ACVAllocator for GMF with chain index."""
        stat, costs = self._create_mfmc_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, ACVAllocator)

    def test_default_allocator_factory_gmf_non_chain(self, bkd):
        """Returns ACVAllocator for GMF with non-chain index."""
        stat, costs = self._create_mfmc_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 0], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, ACVAllocator)

    def test_default_allocator_factory_mfmc_returns_analytical(self, bkd):
        """Returns AnalyticalAllocator for MFMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_mfmc_stat_and_costs(bkd)
        est = MFMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, AnalyticalAllocator)

    def test_default_allocator_factory_mlmc_returns_analytical(self, bkd):
        """Returns AnalyticalAllocator for MLMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_mlmc_stat_and_costs(bkd)
        est = MLMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, AnalyticalAllocator)

    def test_analytical_allocator_mfmc_success(self, bkd):
        """AnalyticalAllocator works with MFMC."""
        stat, costs = self._create_mfmc_stat_and_costs(bkd)
        est = MFMCEstimator(stat, costs)
        allocator = AnalyticalAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        assert result.success
        assert result.actual_cost <= result.target_cost
        assert result.objective_value.shape == (1,)

    def test_analytical_allocator_mlmc_success(self, bkd):
        """AnalyticalAllocator works with MLMC."""
        stat, costs = self._create_mlmc_stat_and_costs(bkd)
        est = MLMCEstimator(stat, costs)
        allocator = AnalyticalAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        assert result.success
        assert result.actual_cost <= result.target_cost
        assert result.objective_value.shape == (1,)

    def test_analytical_allocator_budget_too_small(self, bkd):
        """AnalyticalAllocator returns failure when budget too small."""
        stat, costs = self._create_mfmc_stat_and_costs(bkd)
        est = MFMCEstimator(stat, costs)
        allocator = AnalyticalAllocator(est)
        result = allocator.allocate(target_cost=0.1)  # Too small
        assert not result.success
        assert "Budget too small" in result.message


class TestAllocatorFactory:
    """Tests for default_allocator_factory."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_stat_and_costs(self, bkd, nmodels: int = 3, nqoi: int = 1):
        """Helper to create test statistic and costs."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(bkd.array(cov))
        costs = bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_factory_returns_acv_for_gmf_chain(self, bkd):
        """Returns ACVAllocator for GMF with chain index."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, ACVAllocator)

    def test_factory_returns_acv_for_gmf_non_chain(self, bkd):
        """Returns ACVAllocator for GMF with non-chain index."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 0], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, ACVAllocator)

    def test_factory_returns_analytical_for_mfmc(self, bkd):
        """Returns AnalyticalAllocator for MFMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_stat_and_costs(bkd)
        est = MFMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, AnalyticalAllocator)

    def test_factory_returns_analytical_for_mlmc(self, bkd):
        """Returns AnalyticalAllocator for MLMC (has _allocate_samples_analytical)."""
        stat, costs = self._create_stat_and_costs(bkd)
        est = MLMCEstimator(stat, costs)
        allocator = default_allocator_factory(est)
        assert isinstance(allocator, AnalyticalAllocator)


class TestFittedACVEstimator:
    """Tests for FittedACVEstimator."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_stat_and_costs(self, bkd, nmodels: int = 3, nqoi: int = 1):
        """Helper to create test statistic and costs."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(bkd.array(cov))
        costs = bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_fitted_from_successful_allocation(self, bkd):
        """FittedACVEstimator can be created from successful allocation."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        assert result.success

        fitted = FittedACVEstimator(est, result)
        cov = fitted.covariance()
        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1]

    def test_fitted_from_failed_allocation_raises(self, bkd):
        """FittedACVEstimator raises ValueError for failed allocation."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=0.1)
        assert not result.success

        with pytest.raises(ValueError, match="failed allocation"):
            FittedACVEstimator(est, result)

    def test_covariance_at(self, bkd):
        """covariance_at() computes covariance from continuous ratios."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        partition_ratios = bkd.array([2.0, 5.0])
        target_cost = 500.0

        cov = est.covariance_at(target_cost, partition_ratios)

        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1]

    def test_npartition_samples_at(self, bkd):
        """npartition_samples_at() computes sample counts from ratios."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        partition_ratios = bkd.array([2.0, 5.0])
        target_cost = 500.0

        npartition_samples = est.npartition_samples_at(
            target_cost, partition_ratios
        )

        assert len(npartition_samples) == est._npartitions
        assert all(float(n) > 0 for n in npartition_samples)

    def test_covariance_consistency(self, bkd):
        """FittedACVEstimator.covariance() equals fresh computation from discrete counts."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        assert result.success

        fitted = FittedACVEstimator(est, result)
        fresh_cov = est._covariance_from_npartition_samples(result.npartition_samples)
        bkd.assert_allclose(fitted.covariance(), fresh_cov)

    def test_template_unchanged_after_fitting(self, bkd):
        """Template estimator is not mutated by FittedACVEstimator creation."""
        stat, costs = self._create_stat_and_costs(bkd)
        recursion_index = bkd.array([0, 1], dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)

        allocator = ACVAllocator(est)
        result = allocator.allocate(target_cost=1000.0)
        assert result.success

        repr_before = repr(est)
        FittedACVEstimator(est, result)
        repr_after = repr(est)
        assert repr_before == repr_after


class TestAnalyticalVsNumerical:
    """Compare analytical MFMC/MLMC allocation with numerical optimization."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_mfmc_stat_and_costs(self, bkd, nmodels: int = 3, nqoi: int = 1):
        """Create stat/costs satisfying MFMC hierarchy (decreasing correlation)."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.95 ** max(i, j)
                        cov[q * nmodels + i, q * nmodels + j] = corr
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(bkd.array(cov))
        costs = bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def _create_mlmc_stat_and_costs(self, bkd, nmodels: int = 3, nqoi: int = 1):
        """Create stat/costs satisfying MLMC requirements (decreasing cost)."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(bkd.array(cov))
        costs = bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    @slow_test
    def test_mfmc_analytical_vs_numerical(self, bkd):
        """Analytical MFMC allocation matches numerical optimization."""
        stat, costs = self._create_mfmc_stat_and_costs(bkd)
        target_cost = 1000.0

        # Analytical
        mfmc_est = MFMCEstimator(stat, costs)
        analytical_allocator = AnalyticalAllocator(mfmc_est)
        analytical_result = analytical_allocator.allocate(target_cost)
        assert analytical_result.success

        # Numerical (force optimization by using ACVAllocator directly)
        numerical_allocator = ACVAllocator(mfmc_est)
        numerical_result = numerical_allocator.allocate(target_cost)
        assert numerical_result.success

        # rtol=1e-5 accounts for optimizer convergence tolerance
        bkd.assert_allclose(
            analytical_result.partition_ratios,
            numerical_result.partition_ratios,
            rtol=1e-5,
        )
        bkd.assert_allclose(
            analytical_result.objective_value,
            numerical_result.objective_value,
            rtol=1e-5,
        )

    @slow_test
    def test_mlmc_analytical_vs_numerical(self, bkd):
        """Analytical MLMC allocation matches numerical optimization."""
        stat, costs = self._create_mlmc_stat_and_costs(bkd)
        target_cost = 1000.0

        # Analytical
        mlmc_est = MLMCEstimator(stat, costs)
        analytical_allocator = AnalyticalAllocator(mlmc_est)
        analytical_result = analytical_allocator.allocate(target_cost)
        assert analytical_result.success

        # Numerical (MLMCEstimator uses -1 weights, same objective)
        numerical_allocator = ACVAllocator(mlmc_est)
        numerical_result = numerical_allocator.allocate(target_cost)
        assert numerical_result.success

        # rtol=1e-5 accounts for optimizer convergence tolerance
        bkd.assert_allclose(
            analytical_result.partition_ratios,
            numerical_result.partition_ratios,
            rtol=1e-5,
        )
        bkd.assert_allclose(
            analytical_result.objective_value,
            numerical_result.objective_value,
            rtol=1e-5,
        )
