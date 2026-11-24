"""Tests for ACVAllocatorViaAutodiff and ViaAutodiffAllocatorFactory.

The allocator optimizes a copy of the estimator on an autodiff backend
whatever the estimator's backend, so each test takes ``autodiff_bkd`` for
the optimization and ``bkd`` for the estimator.
"""

from typing import Any, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.statest.acv.allocation import (
    ACVAllocator,
    ACVAllocatorViaAutodiff,
    AnalyticalAllocator,
    ViaAutodiffAllocatorFactory,
)
from pyapprox.statest.acv.base import FittedACVEstimator
from pyapprox.statest.acv.optimization import ACVLogDeterminantObjective
from pyapprox.statest.acv.variants import (
    GMFEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.protocols import TargetArray
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test


def _stat_and_costs(
    bkd: Backend[Array], corr_exponent: str = "abs"
) -> Tuple[MultiOutputMean[Array], Array]:
    """Three-model statistic and costs ``[100, 10, 1]``.

    ``corr_exponent="abs"`` gives correlation ``0.9**|i-j|`` (satisfies
    the MLMC requirements); ``"max"`` gives ``0.95**max(i, j)`` (satisfies
    the MFMC hierarchy).
    """
    nmodels = 3
    cov = np.eye(nmodels)
    for i in range(nmodels):
        for j in range(nmodels):
            if i != j:
                cov[i, j] = (
                    0.9 ** abs(i - j) if corr_exponent == "abs" else 0.95 ** max(i, j)
                )
    stat = MultiOutputMean(1, bkd)
    stat.set_pilot_quantities(bkd.array(cov))
    return stat, bkd.array([100.0, 10.0, 1.0])


def _gmf_estimator(bkd: Backend[Array]) -> GMFEstimator[Array]:
    stat, costs = _stat_and_costs(bkd)
    return GMFEstimator(
        stat, costs, recursion_index=bkd.asarray([0, 1], dtype=bkd.index_dtype())
    )


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(42)


class TestACVAllocatorViaAutodiff:
    """Estimators on any backend are optimized on an autodiff-backend copy,
    with results converted back."""

    def test_acv_allocator_success(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """Basic allocation works."""
        result = ACVAllocatorViaAutodiff(_gmf_estimator(bkd), autodiff_bkd).allocate(
            target_cost=1000.0
        )
        assert result.success
        assert result.actual_cost <= result.target_cost

    def test_acv_allocator_budget_too_small(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """Returns failure when budget too small."""
        result = ACVAllocatorViaAutodiff(_gmf_estimator(bkd), autodiff_bkd).allocate(
            target_cost=0.1
        )
        assert not result.success
        assert "Budget too small" in result.message

    def test_acv_allocator_respects_budget(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """actual_cost <= target_cost."""
        target_cost = 500.0
        result = ACVAllocatorViaAutodiff(_gmf_estimator(bkd), autodiff_bkd).allocate(
            target_cost=target_cost
        )
        if result.success:
            assert result.actual_cost <= target_cost

    def test_acv_allocator_objective_value_is_array(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """objective_value should be Array, not float."""
        result = ACVAllocatorViaAutodiff(_gmf_estimator(bkd), autodiff_bkd).allocate(
            target_cost=1000.0
        )
        if result.success:
            assert result.objective_value.shape == (1,)

    def test_acv_allocator_failure_objective_value_is_array(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """Even failed allocations should have Array objective_value."""
        result = ACVAllocatorViaAutodiff(_gmf_estimator(bkd), autodiff_bkd).allocate(
            target_cost=0.1
        )  # Will fail
        assert not result.success
        assert result.objective_value.shape == (1,)
        assert float(result.objective_value[0]) == float("inf")

    def test_acv_allocator_result_backend_matches(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """Result arrays are in the estimator's backend type."""
        result = ACVAllocatorViaAutodiff(_gmf_estimator(bkd), autodiff_bkd).allocate(
            target_cost=1000.0
        )
        assert result.success
        # All array fields should be the same type as the backend's arrays
        expected_type = type(bkd.array([0.0]))
        assert isinstance(result.partition_ratios, expected_type)
        assert isinstance(result.continuous_npartition_samples, expected_type)
        assert isinstance(result.objective_value, expected_type)
        assert isinstance(result.npartition_samples, expected_type)
        assert isinstance(result.nsamples_per_model, expected_type)

    def test_custom_objective_is_used(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """The objective given is the one optimized, on the autodiff copy."""
        objective = ACVLogDeterminantObjective[TargetArray]()
        result = ACVAllocatorViaAutodiff(
            _gmf_estimator(bkd), autodiff_bkd, objective=objective
        ).allocate(target_cost=1000.0)
        assert result.success
        assert isinstance(objective.bkd(), type(autodiff_bkd))

    def test_non_autodiff_backend_refused(self, numpy_bkd: NumpyBkd) -> None:
        """A backend without autodiff is refused at construction."""
        with pytest.raises(TypeError, match="AutodiffBackend"):
            ACVAllocatorViaAutodiff(
                _gmf_estimator(numpy_bkd),
                numpy_bkd,  # type: ignore[arg-type]
            )

    def test_matches_numpy_finite_differences(
        self, numpy_bkd: NumpyBkd, autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """ACVAllocator on NumPy, with SLSQP's finite-difference jacobians,
        matches the allocation optimized with autograd."""
        est = _gmf_estimator(numpy_bkd)
        result = ACVAllocator(
            est, optimizer=ScipySLSQPOptimizer[NDArray[Any]](maxiter=200)
        ).allocate(1000.0)
        via_autodiff = ACVAllocatorViaAutodiff(
            est, autodiff_bkd, optimizer=ScipySLSQPOptimizer[TargetArray](maxiter=200)
        ).allocate(1000.0)
        numpy_bkd.assert_allclose(
            result.objective_value, via_autodiff.objective_value, rtol=1e-4
        )


class TestViaAutodiffAllocatorFactory:
    """The factory dispatches like default_allocator_factory, but optimizes
    on the autodiff backend."""

    def test_returns_via_autodiff_for_gmf(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        allocator = ViaAutodiffAllocatorFactory(autodiff_bkd)(_gmf_estimator(bkd))
        assert isinstance(allocator, ACVAllocatorViaAutodiff)
        assert allocator.allocate(1000.0).success

    @pytest.mark.parametrize(
        "estimator_class,corr_exponent",
        [(MFMCEstimator, "max"), (MLMCEstimator, "abs")],
    )
    def test_returns_analytical_for_closed_form(
        self,
        bkd: Backend[Array],
        autodiff_bkd: AutodiffBackend[TargetArray],
        estimator_class: type,
        corr_exponent: str,
    ) -> None:
        stat, costs = _stat_and_costs(bkd, corr_exponent)
        allocator = ViaAutodiffAllocatorFactory(autodiff_bkd)(
            estimator_class(stat, costs)
        )
        assert isinstance(allocator, AnalyticalAllocator)

    def test_non_autodiff_backend_refused(self, numpy_bkd: NumpyBkd) -> None:
        """A backend without autodiff is refused at construction."""
        with pytest.raises(TypeError, match="AutodiffBackend"):
            ViaAutodiffAllocatorFactory(numpy_bkd)  # type: ignore[arg-type]


class TestFittedACVEstimator:
    """Tests for FittedACVEstimator built from an autodiff allocation."""

    def test_fitted_from_successful_allocation(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """FittedACVEstimator can be created from successful allocation."""
        est = _gmf_estimator(bkd)
        result = ACVAllocatorViaAutodiff(est, autodiff_bkd).allocate(target_cost=1000.0)
        assert result.success

        fitted = FittedACVEstimator(est, result)
        cov = fitted.covariance()
        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1]

    def test_fitted_from_failed_allocation_raises(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """FittedACVEstimator raises ValueError for failed allocation."""
        est = _gmf_estimator(bkd)
        result = ACVAllocatorViaAutodiff(est, autodiff_bkd).allocate(target_cost=0.1)
        assert not result.success

        with pytest.raises(ValueError, match="failed allocation"):
            FittedACVEstimator(est, result)

    def test_covariance_consistency(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """FittedACVEstimator.covariance() equals a fresh computation.

        The fresh computation uses the same discrete counts.
        """
        est = _gmf_estimator(bkd)
        result = ACVAllocatorViaAutodiff(est, autodiff_bkd).allocate(target_cost=1000.0)
        assert result.success

        fitted = FittedACVEstimator(est, result)
        # The template evaluates the continuous relaxation, so the discrete
        # allocation must be cast to float at the boundary, exactly as
        # FittedACVEstimator does internally.
        fresh_cov = est.covariance_at_npartition_samples(
            bkd.asarray(result.npartition_samples, dtype=bkd.double_dtype())
        )
        bkd.assert_allclose(fitted.covariance(), fresh_cov)

    def test_template_unchanged_after_fitting(
        self, bkd: Backend[Array], autodiff_bkd: AutodiffBackend[TargetArray]
    ) -> None:
        """Template estimator is not mutated by FittedACVEstimator creation."""
        est = _gmf_estimator(bkd)
        result = ACVAllocatorViaAutodiff(est, autodiff_bkd).allocate(target_cost=1000.0)
        assert result.success

        repr_before = repr(est)
        FittedACVEstimator(est, result)
        repr_after = repr(est)
        assert repr_before == repr_after


class TestAnalyticalVsNumerical:
    """Compare analytical MFMC/MLMC allocation with numerical optimization."""

    @slow_test
    @pytest.mark.parametrize(
        "estimator_class,corr_exponent",
        [(MFMCEstimator, "max"), (MLMCEstimator, "abs")],
    )
    def test_analytical_vs_numerical(
        self,
        bkd: Backend[Array],
        autodiff_bkd: AutodiffBackend[TargetArray],
        estimator_class: type,
        corr_exponent: str,
    ) -> None:
        """Analytical MFMC and MLMC allocations match numerical optimization
        (MLMCEstimator uses -1 weights, same objective)."""
        stat, costs = _stat_and_costs(bkd, corr_exponent)
        target_cost = 1000.0
        est = estimator_class(stat, costs)

        analytical_result = AnalyticalAllocator(est).allocate(target_cost)
        assert analytical_result.success

        # Force optimization by using ACVAllocatorViaAutodiff directly
        numerical_result = ACVAllocatorViaAutodiff(est, autodiff_bkd).allocate(
            target_cost
        )
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
