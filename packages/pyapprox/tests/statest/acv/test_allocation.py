"""Tests for ACV allocation module.

Allocation through ACVAllocatorViaAutodiff, which optimizes a copy of the
estimator on an autodiff backend whatever its backend, is tested in
test_allocation_autodiff.py.
"""

from typing import Any, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.statest.acv.allocation import (
    ACVAllocator,
    AnalyticalAllocator,
    AnalyticallyAllocatable,
    default_allocator_factory,
)
from pyapprox.statest.acv.result import ACVAllocationResult
from pyapprox.statest.acv.variants import (
    GMFEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


def _stat_and_costs(
    bkd: Backend[Array], nmodels: int = 3, nqoi: int = 1
) -> Tuple[MultiOutputMean[Array], Array]:
    """Statistic with correlation 0.9**|i-j| between models, and costs
    decreasing by factors of ten. Satisfies the MLMC requirements."""
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


def _mfmc_stat_and_costs(
    bkd: Backend[Array], nmodels: int = 3, nqoi: int = 1
) -> Tuple[MultiOutputMean[Array], Array]:
    """Statistic and costs satisfying the MFMC hierarchy requirements
    (correlation with the high-fidelity model decreasing)."""
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
    costs = bkd.array([10.0 ** (nmodels - 1 - i) for i in range(nmodels)])
    return stat, costs


def _gmf_estimator(
    bkd: Backend[Array], recursion_index: Tuple[int, ...] = (0, 1)
) -> GMFEstimator[Array]:
    stat, costs = _stat_and_costs(bkd)
    return GMFEstimator(
        stat,
        costs,
        recursion_index=bkd.asarray(recursion_index, dtype=bkd.int64_dtype()),
    )


def _allocation_result(bkd: Backend[Array]) -> ACVAllocationResult[Array]:
    return ACVAllocationResult(
        partition_ratios=bkd.array([1.0, 2.0]),
        continuous_npartition_samples=bkd.array([10.0, 10.0, 20.0]),
        objective_value=bkd.array([-5.0]),
        npartition_samples=bkd.array([10, 10, 20], dtype=bkd.int64_dtype()),
        nsamples_per_model=bkd.array([10, 20, 30], dtype=bkd.int64_dtype()),
        target_cost=100.0,
        actual_cost=95.0,
        success=True,
        message="",
    )


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(42)


class TestACVAllocationResult:
    """Tests for ACVAllocationResult dataclass."""

    def test_allocation_result_creation(self, bkd: Backend[Array]) -> None:
        """Test dataclass instantiation."""
        result = _allocation_result(bkd)
        assert result.success
        bkd.assert_allclose(
            bkd.asarray([result.target_cost]),
            bkd.asarray([100.0]),
        )

    def test_allocation_result_frozen(self, bkd: Backend[Array]) -> None:
        """Test immutability."""
        result = _allocation_result(bkd)
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_allocation_result_objective_value_is_array(
        self, bkd: Backend[Array]
    ) -> None:
        """Test that objective_value is an Array, not float."""
        result = _allocation_result(bkd)
        # objective_value should be an array with shape (1,)
        assert result.objective_value.shape == (1,)
        bkd.assert_allclose(result.objective_value, bkd.array([-5.0]))


class TestACVAllocatorOnOwnBackend:
    """ACVAllocator optimizes on the estimator's backend, never another."""

    def test_autodiff_estimator_with_autograd(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """On an autodiff backend the default optimizer gets jacobians."""
        result = ACVAllocator(_gmf_estimator(autodiff_bkd)).allocate(1000.0)
        assert result.success
        assert isinstance(result.partition_ratios, type(autodiff_bkd.zeros((1,))))

    def test_numpy_estimator_with_finite_differences(self, numpy_bkd: NumpyBkd) -> None:
        """Without autodiff, an optimizer that approximates jacobians works.

        SLSQP falls back to finite differences when the objective and
        constraint declare no jacobian, so no torch copy is needed.
        test_allocation_autodiff.py checks the allocation matches the one
        optimized with autograd.
        """
        est = _gmf_estimator(numpy_bkd)
        result = ACVAllocator(
            est, optimizer=ScipySLSQPOptimizer[NDArray[Any]](maxiter=200)
        ).allocate(1000.0)
        assert result.success
        assert isinstance(result.partition_ratios, np.ndarray)


class TestAnalyticalAllocator:
    """Tests for AnalyticalAllocator."""

    def test_analytical_allocator_mfmc_success(self, bkd: Backend[Array]) -> None:
        """AnalyticalAllocator works with MFMC."""
        stat, costs = _mfmc_stat_and_costs(bkd)
        allocator = AnalyticalAllocator(MFMCEstimator(stat, costs))
        result = allocator.allocate(target_cost=1000.0)
        assert result.success
        assert result.actual_cost <= result.target_cost
        assert result.objective_value.shape == (1,)

    def test_analytical_allocator_mlmc_success(self, bkd: Backend[Array]) -> None:
        """AnalyticalAllocator works with MLMC."""
        stat, costs = _stat_and_costs(bkd)
        allocator = AnalyticalAllocator(MLMCEstimator(stat, costs))
        result = allocator.allocate(target_cost=1000.0)
        assert result.success
        assert result.actual_cost <= result.target_cost
        assert result.objective_value.shape == (1,)

    def test_analytical_allocator_budget_too_small(self, bkd: Backend[Array]) -> None:
        """AnalyticalAllocator returns failure when budget too small."""
        stat, costs = _mfmc_stat_and_costs(bkd)
        allocator = AnalyticalAllocator(MFMCEstimator(stat, costs))
        result = allocator.allocate(target_cost=0.1)  # Too small
        assert not result.success
        assert "Budget too small" in result.message

    def test_estimator_without_closed_form_refused(self, bkd: Backend[Array]) -> None:
        """An estimator with no closed-form allocation is refused at once."""
        est = _gmf_estimator(bkd)
        assert not isinstance(est, AnalyticallyAllocatable)
        with pytest.raises(TypeError, match="AnalyticallyAllocatable"):
            AnalyticalAllocator(est)


class TestAllocatorFactory:
    """Tests for default_allocator_factory."""

    @pytest.mark.parametrize("recursion_index", [(0, 1), (0, 0)])
    def test_factory_returns_acv_allocator_for_gmf(
        self, bkd: Backend[Array], recursion_index: Tuple[int, ...]
    ) -> None:
        """Returns ACVAllocator for GMF with chain and non-chain recursion
        indices."""
        allocator = default_allocator_factory(_gmf_estimator(bkd, recursion_index))
        assert isinstance(allocator, ACVAllocator)

    def test_factory_allocates_gmf_on_own_backend(self, bkd: Backend[Array]) -> None:
        """The default allocation succeeds on every backend, autograd or
        not, and returns the estimator's array type."""
        result = default_allocator_factory(_gmf_estimator(bkd)).allocate(1000.0)
        assert result.success
        assert result.actual_cost <= result.target_cost
        assert isinstance(result.partition_ratios, type(bkd.zeros((1,))))

    def test_factory_returns_analytical_for_mfmc(self, bkd: Backend[Array]) -> None:
        """Returns AnalyticalAllocator for MFMC (satisfies AnalyticallyAllocatable)."""
        stat, costs = _mfmc_stat_and_costs(bkd)
        allocator = default_allocator_factory(MFMCEstimator(stat, costs))
        assert isinstance(allocator, AnalyticalAllocator)

    def test_factory_returns_analytical_for_mlmc(self, bkd: Backend[Array]) -> None:
        """Returns AnalyticalAllocator for MLMC (satisfies AnalyticallyAllocatable)."""
        stat, costs = _stat_and_costs(bkd)
        allocator = default_allocator_factory(MLMCEstimator(stat, costs))
        assert isinstance(allocator, AnalyticalAllocator)


class TestACVEstimatorAtRatios:
    """Covariance and sample counts at given partition ratios."""

    def test_covariance_at(self, bkd: Backend[Array]) -> None:
        """covariance_at() computes covariance from continuous ratios."""
        est = _gmf_estimator(bkd)
        cov = est.covariance_at(500.0, bkd.array([2.0, 5.0]))
        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1]

    def test_npartition_samples_at(self, bkd: Backend[Array]) -> None:
        """npartition_samples_at() computes sample counts from ratios."""
        est = _gmf_estimator(bkd)
        npartition_samples = est.npartition_samples_at(500.0, bkd.array([2.0, 5.0]))
        assert len(npartition_samples) == est._npartitions
        assert all(float(n) > 0 for n in npartition_samples)
