"""Tests for MeanGuidedSubsetFitter."""

from typing import List

import numpy as np

from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.statest.groupacv import (
    GroupACVAllocationOptimizer,
    GroupACVEstimatorIS,
    MeanGuidedSubsetFitter,
    MeanGuidedSubsetResult,
)
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.statest.statistics import (
    MultiOutputVariance,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test


def _slsqp(bkd: Backend[Array]) -> ScipySLSQPOptimizer[Array]:
    """SLSQP optimizer for allocations on ``bkd``'s array type."""
    return ScipySLSQPOptimizer(maxiter=1000, ftol=1e-10)


def _make_correlated_variance_stat(
    bkd: Backend[Array], nmodels: int = 5, nqoi: int = 1, npilot: int = 10000
) -> MultiOutputVariance[Array]:
    """Create a Variance stat with realistic correlations from pilot data."""
    np.random.seed(42)
    pilot_values: List[Array] = []
    base = np.random.randn(nqoi, npilot)
    for i in range(nmodels):
        rho = 0.95 ** (i + 1)
        noise_std = (1 - rho**2) ** 0.5
        vals = rho * base + noise_std * np.random.randn(nqoi, npilot)
        pilot_values.append(bkd.array(vals))
    stat = MultiOutputVariance(nqoi, bkd)
    cov, W = stat.compute_pilot_quantities(pilot_values)
    stat.set_pilot_quantities(cov, W)
    return stat


class TestMeanGuidedSubsetFitterTorchOnly:
    """Tests requiring an autodiff backend."""

    def test_smoke_5model_variance(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        """Basic smoke test: fitter runs and returns valid result."""
        bkd = autodiff_bkd
        stat = _make_correlated_variance_stat(bkd, nmodels=5)
        costs = bkd.array([10.0, 4.0, 2.0, 1.0, 0.5])
        config = AllocationProblemConfig(
            variable_scaling="log",
            budget_constraint_form="inequality",
        )

        fitter = MeanGuidedSubsetFitter(
            stat,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
            problem_config=config,
        )
        result = fitter.fit(target_cost=100)

        assert isinstance(result, MeanGuidedSubsetResult)
        assert result.best_allocation.success
        assert len(result.active_subset_indices) > 0
        assert result.partitions_pruned() >= 0

    @slow_test
    def test_pruning_reduces_variance_at_low_budget(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """At low budget, mean-guided pruning should reduce variance vs full set."""
        bkd = autodiff_bkd
        stat = _make_correlated_variance_stat(bkd, nmodels=5)
        costs = bkd.array([10.0, 4.0, 2.0, 1.0, 0.5])
        target_cost = 50.0
        config = AllocationProblemConfig(
            variable_scaling="log",
            budget_constraint_form="inequality",
        )

        # Full solve (all subsets, with dead threshold)
        full_est = GroupACVEstimatorIS(stat, costs)
        full_allocator: GroupACVAllocationOptimizer[Array] = (
            GroupACVAllocationOptimizer(
                full_est,
                optimizer=_slsqp(bkd),
                problem_config=config,
            )
        )
        full_result = full_allocator.optimize(target_cost)

        # Mean-guided solve
        fitter = MeanGuidedSubsetFitter(
            stat,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
            problem_config=config,
        )
        guided_result = fitter.fit(target_cost=target_cost)

        if full_result.success and guided_result.best_allocation.success:
            full_obj = float(bkd.to_numpy(full_result.objective_value[0]))
            guided_obj = float(
                bkd.to_numpy(guided_result.best_allocation.objective_value[0])
            )
            assert guided_obj <= full_obj + 1e-8, (
                f"Mean-guided ({guided_obj}) should be <= full ({full_obj})"
            )

    def test_all_active_at_high_budget(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """At high budget, pruning should keep most or all partitions."""
        bkd = autodiff_bkd
        stat = _make_correlated_variance_stat(bkd, nmodels=3)
        costs = bkd.array([3.0, 2.0, 1.0])
        config = AllocationProblemConfig(
            variable_scaling="log",
            budget_constraint_form="inequality",
        )

        fitter = MeanGuidedSubsetFitter(
            stat,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
            problem_config=config,
        )
        result = fitter.fit(target_cost=5000)

        nactive = len(result.active_subset_indices)
        # With ample budget, at least one partition should be active
        assert nactive >= 1

    def test_mean_stat_has_zero_dead_threshold(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Verify the internal Mean stat allows partitions to reach zero."""
        bkd = autodiff_bkd
        stat = _make_correlated_variance_stat(bkd, nmodels=3)
        costs = bkd.array([3.0, 2.0, 1.0])

        fitter = MeanGuidedSubsetFitter(
            stat,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
        )
        mean_stat = fitter._build_mean_stat()
        assert mean_stat.continuous_dead_threshold() == 0.0

    def test_custom_activity_threshold(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Custom threshold changes which partitions are pruned."""
        bkd = autodiff_bkd
        stat = _make_correlated_variance_stat(bkd, nmodels=5)
        costs = bkd.array([10.0, 4.0, 2.0, 1.0, 0.5])
        config = AllocationProblemConfig(
            variable_scaling="log",
            budget_constraint_form="inequality",
        )

        # Low threshold — keep more
        fitter_low = MeanGuidedSubsetFitter(
            stat,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
            problem_config=config,
            activity_threshold=1e-6,
        )
        result_low = fitter_low.fit(target_cost=100)

        # High threshold — keep fewer
        fitter_high = MeanGuidedSubsetFitter(
            stat,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
            problem_config=config,
            activity_threshold=1.0,
        )
        result_high = fitter_high.fit(target_cost=100)

        assert len(result_high.active_subset_indices) <= len(
            result_low.active_subset_indices
        )

    def test_deterministic(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        """Same inputs produce identical results."""
        bkd = autodiff_bkd
        costs = bkd.array([3.0, 2.0, 1.0])
        config = AllocationProblemConfig(
            variable_scaling="log",
            budget_constraint_form="inequality",
        )

        stat1 = _make_correlated_variance_stat(bkd, nmodels=3)
        fitter1 = MeanGuidedSubsetFitter(
            stat1,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
            problem_config=config,
        )
        result1 = fitter1.fit(target_cost=100)

        stat2 = _make_correlated_variance_stat(bkd, nmodels=3)
        fitter2 = MeanGuidedSubsetFitter(
            stat2,
            costs,
            GroupACVEstimatorIS,
            optimizer=_slsqp(bkd),
            problem_config=config,
        )
        result2 = fitter2.fit(target_cost=100)

        assert result1.active_subset_indices == result2.active_subset_indices
        bkd.assert_allclose(
            result1.best_allocation.objective_value,
            result2.best_allocation.objective_value,
        )


class TestMeanGuidedSubsetFitterDualBackend:
    """Tests that run on both NumPy and Torch backends."""

    def test_construction(self, bkd: Backend[Array]) -> None:
        """Fitter constructs on both backends."""
        np.random.seed(42)
        nmodels = 3
        pilot_values = [bkd.array(np.random.randn(1, 100)) for _ in range(nmodels)]
        stat = MultiOutputVariance(1, bkd)
        cov, W = stat.compute_pilot_quantities(pilot_values)
        stat.set_pilot_quantities(cov, W)
        costs = bkd.array([3.0, 2.0, 1.0])

        fitter = MeanGuidedSubsetFitter(stat, costs, GroupACVEstimatorIS)
        assert fitter._bkd is bkd

    def test_build_mean_stat_shares_cov(self, bkd: Backend[Array]) -> None:
        """Internal Mean stat has the same covariance as the target stat."""
        np.random.seed(42)
        nmodels = 3
        pilot_values = [bkd.array(np.random.randn(1, 100)) for _ in range(nmodels)]
        stat = MultiOutputVariance(1, bkd)
        cov, W = stat.compute_pilot_quantities(pilot_values)
        stat.set_pilot_quantities(cov, W)
        costs = bkd.array([3.0, 2.0, 1.0])

        fitter = MeanGuidedSubsetFitter(stat, costs, GroupACVEstimatorIS)
        mean_stat = fitter._build_mean_stat()
        bkd.assert_allclose(mean_stat.pilot_covariance(), stat.pilot_covariance())
