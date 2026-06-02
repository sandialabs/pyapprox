"""Tests for BruteForceSubsetFitter (D1-D11 from plan)."""

import numpy as np
import pytest
from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.statest.groupacv import (
    BruteForceSubsetFitter,
    BruteForceSubsetResult,
    GroupACVAllocationOptimizer,
    GroupACVEstimatorIS,
    GroupACVEstimatorNested,
    get_model_subsets,
)
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.util.backends.torch import TorchBkd

from tests._helpers.markers import slow_test


def _slsqp():
    return ScipySLSQPOptimizer(maxiter=1000, ftol=1e-8)


def _make_mean_stat(bkd, nmodels, nqoi=1, seed=1):
    np.random.seed(seed)
    cov_size = nmodels * nqoi
    cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    cov = cov.T @ cov
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    return stat


def _make_variance_stat(bkd, nmodels, nqoi=1, seed=1):
    np.random.seed(seed)
    cov_size = nmodels * nqoi
    cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    cov = cov.T @ cov
    W = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    W = W.T @ W
    stat = MultiOutputVariance(nqoi, bkd)
    stat.set_pilot_quantities(cov, W)
    return stat


def _make_mean_and_variance_stat(bkd, nmodels, nqoi=1, seed=1):
    np.random.seed(seed)
    cov_size = nmodels * nqoi
    cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    cov = cov.T @ cov
    W = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    W = W.T @ W
    B = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    B = B.T @ B
    stat = MultiOutputMeanAndVariance(nqoi, bkd)
    stat.set_pilot_quantities(cov, W, B)
    return stat


class TestBruteForceSubsetFitterTorchOnly:
    """Tests requiring Torch for autograd-based optimization."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def test_d1_smoke_2model_mean_is(self):
        """D1: Smoke test — 2-model Mean IS, K=3 subsets."""
        bkd = self._bkd
        stat = _make_mean_stat(bkd, nmodels=2)
        costs = bkd.array([2.0, 1.0])
        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorIS, optimizer=_slsqp(),
        )
        result = fitter.fit(target_cost=100)

        assert isinstance(result, BruteForceSubsetResult)
        assert result.best_allocation.success
        assert len(result.best_active_indices) > 0
        assert result.patterns_evaluated() > 0
        assert result.patterns_successful() > 0

    def test_d2_uncorrelated_sparse_optimum(self):
        """D2: Uncorrelated models → optimal pattern uses only model-0 subsets."""
        bkd = self._bkd
        nmodels = 3
        cov = bkd.eye(nmodels, dtype=bkd.double_dtype())
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        costs = bkd.array([3.0, 2.0, 1.0])

        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorIS, optimizer=_slsqp(),
        )
        result = fitter.fit(target_cost=100)
        assert result.best_allocation.success

        # Best pattern should only use subsets containing model 0 alone
        # since LF models provide no correlation benefit
        candidate_subsets = result.candidate_subsets
        for i in result.best_active_indices:
            np_subset = bkd.to_numpy(candidate_subsets[i])
            assert 0 in np_subset

    @slow_test
    def test_d3a_threshold_preserves_active_set(self):
        """D3a: Threshold doesn't change which subsets are optimally active."""
        bkd = self._bkd
        nmodels = 3
        np.random.seed(42)
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        # Weaken model 2 correlation
        cov_np = bkd.to_numpy(cov)
        cov_np[0, 2] *= 0.01
        cov_np[2, 0] *= 0.01
        cov_np[1, 2] *= 0.01
        cov_np[2, 1] *= 0.01
        cov = bkd.array(cov_np)

        stat_thresh = MultiOutputMean(1, bkd)
        stat_thresh.set_pilot_quantities(cov)
        costs = bkd.array([3.0, 2.0, 1.0])

        config_thresh = AllocationProblemConfig(bounds_lb="dead_threshold")
        fitter_thresh = BruteForceSubsetFitter(
            stat_thresh, costs, GroupACVEstimatorIS,
            optimizer=_slsqp(), problem_config=config_thresh,
        )
        result_thresh = fitter_thresh.fit(target_cost=100)

        stat_nothresh = MultiOutputMean(1, bkd)
        stat_nothresh.set_pilot_quantities(cov)
        config_nothresh = AllocationProblemConfig(bounds_lb=1e-8)
        fitter_nothresh = BruteForceSubsetFitter(
            stat_nothresh, costs, GroupACVEstimatorIS,
            optimizer=_slsqp(), problem_config=config_nothresh,
        )
        result_nothresh = fitter_nothresh.fit(target_cost=100)

        assert result_thresh.best_active_indices == (
            result_nothresh.best_active_indices
        )

    @slow_test
    def test_d3b_brute_force_beats_standard(self):
        """D3b: Brute force objective <= standard allocator objective."""
        bkd = self._bkd
        stat = _make_mean_stat(bkd, nmodels=3)
        costs = bkd.array([3.0, 2.0, 1.0])

        config = AllocationProblemConfig(bounds_lb="dead_threshold")

        # Standard allocator on full subset list
        est_std = GroupACVEstimatorIS(stat, costs)
        allocator_std = GroupACVAllocationOptimizer(
            est_std, optimizer=_slsqp(), problem_config=config
        )
        result_std = allocator_std.optimize(target_cost=100)

        # Brute force
        stat2 = _make_mean_stat(bkd, nmodels=3)
        fitter = BruteForceSubsetFitter(
            stat2, costs, GroupACVEstimatorIS,
            optimizer=_slsqp(), problem_config=config,
        )
        result_bf = fitter.fit(target_cost=100)

        assert result_std.success
        assert result_bf.best_allocation.success
        bf_obj = float(bkd.to_numpy(
            result_bf.best_allocation.objective_value[0]
        ))
        std_obj = float(bkd.to_numpy(result_std.objective_value[0]))
        assert bf_obj <= std_obj + 1e-6

    @slow_test
    def test_d4_all_active_matches_standard(self):
        """D4: All-active pattern matches standard allocator."""
        bkd = self._bkd
        stat = _make_mean_stat(bkd, nmodels=2)
        costs = bkd.array([2.0, 1.0])
        target_cost = 100.0

        config = AllocationProblemConfig(bounds_lb=1e-8)

        est = GroupACVEstimatorIS(stat, costs)
        allocator = GroupACVAllocationOptimizer(
            est, optimizer=_slsqp(), problem_config=config,
        )
        result_std = allocator.optimize(target_cost=target_cost)

        stat2 = _make_mean_stat(bkd, nmodels=2)
        subsets = get_model_subsets(2, bkd)
        all_active = tuple(range(len(subsets)))
        fitter = BruteForceSubsetFitter(
            stat2, costs, GroupACVEstimatorIS,
            optimizer=_slsqp(), problem_config=config,
        )
        # Find the all-active result
        result_bf = fitter.fit(target_cost=target_cost)
        all_active_result = None
        for idx, alloc in result_bf.all_allocations:
            if idx == all_active:
                all_active_result = alloc
                break

        assert all_active_result is not None
        assert all_active_result.success
        assert result_std.success
        bkd.assert_allclose(
            all_active_result.npartition_samples,
            result_std.npartition_samples,
        )

    @slow_test
    def test_d6_nested_estimator(self):
        """D6: Nested estimator skips [0]-only patterns."""
        bkd = self._bkd
        stat = _make_mean_stat(bkd, nmodels=3)
        costs = bkd.array([3.0, 2.0, 1.0])

        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorNested, optimizer=_slsqp(),
        )
        result = fitter.fit(target_cost=100)

        # Verify no pattern has only [0] subsets
        subsets = result.candidate_subsets
        for idx, _ in result.all_allocations:
            all_singleton_0 = all(
                len(bkd.to_numpy(subsets[i])) == 1
                and int(bkd.to_numpy(subsets[i])[0]) == 0
                for i in idx
            )
            assert not all_singleton_0

        assert result.patterns_successful() > 0

    @pytest.mark.parametrize(
        "stat_factory",
        [_make_mean_stat, _make_variance_stat, _make_mean_and_variance_stat],
        ids=["Mean", "Variance", "MeanAndVariance"],
    )
    def test_d7_all_stat_types(self, stat_factory):
        """D7: Smoke test for each stat type."""
        bkd = self._bkd
        stat = stat_factory(bkd, nmodels=2)
        costs = bkd.array([2.0, 1.0])

        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorIS, optimizer=_slsqp(),
        )
        result = fitter.fit(target_cost=100, allow_failures=True)
        assert result.best_allocation.success

    @slow_test
    def test_d9_pattern_count_is(self):
        """D9: 3 models → K=7 subsets. Verify feasible IS pattern count."""
        bkd = self._bkd
        stat = _make_mean_stat(bkd, nmodels=3)
        costs = bkd.array([3.0, 2.0, 1.0])

        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorIS, optimizer=_slsqp(),
        )
        result = fitter.fit(target_cost=100, allow_failures=True)
        # K=7, patterns with at least one model-0 subset: 2^7-1 - (2^3-1) = 120
        assert result.patterns_evaluated() == 120

    @slow_test
    def test_d9_pattern_count_nested(self):
        """D9: Nested variant filters [0]-only patterns."""
        bkd = self._bkd
        stat = _make_mean_stat(bkd, nmodels=3)
        costs = bkd.array([3.0, 2.0, 1.0])

        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorNested, optimizer=_slsqp(),
        )
        result = fitter.fit(target_cost=100, allow_failures=True)
        # IS: 120, minus the 1 pattern that is only {0}: 119
        assert result.patterns_evaluated() == 119

    def test_d10_allow_failures_false_raises(self):
        """D10: allow_failures=False raises on first failure."""
        bkd = self._bkd
        stat = _make_mean_stat(bkd, nmodels=2)
        # Very tight budget — some patterns will fail
        costs = bkd.array([100.0, 1.0])

        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorIS, optimizer=_slsqp(),
        )
        with pytest.raises(RuntimeError):
            fitter.fit(target_cost=50, allow_failures=False)

    def test_d11_seed_determinism(self):
        """D11: Same pattern twice → identical objective value."""
        bkd = self._bkd
        stat1 = _make_mean_stat(bkd, nmodels=2)
        costs = bkd.array([2.0, 1.0])
        fitter1 = BruteForceSubsetFitter(
            stat1, costs, GroupACVEstimatorIS, optimizer=_slsqp(),
        )
        result1 = fitter1.fit(target_cost=100)

        stat2 = _make_mean_stat(bkd, nmodels=2)
        fitter2 = BruteForceSubsetFitter(
            stat2, costs, GroupACVEstimatorIS, optimizer=_slsqp(),
        )
        result2 = fitter2.fit(target_cost=100)

        bkd.assert_allclose(
            result1.best_allocation.objective_value,
            result2.best_allocation.objective_value,
        )
        assert result1.best_active_indices == result2.best_active_indices


class TestBruteForceSubsetFitterDualBackend:
    """Tests that run on both NumPy and Torch backends."""

    def test_d5_too_many_subsets_raises(self, bkd):
        """D5: K > 16 raises ValueError."""
        stat = _make_mean_stat(bkd, nmodels=2)
        costs = bkd.array([2.0, 1.0])
        fake_subsets = [bkd.array([0]) for _ in range(17)]

        with pytest.raises(ValueError, match="Too many"):
            BruteForceSubsetFitter(
                stat, costs, GroupACVEstimatorIS,
                candidate_subsets=fake_subsets,
            )

    def test_d8_backend_coverage(self, bkd):
        """D8: Smoke test on both backends (fit not called — just construction)."""
        stat = _make_mean_stat(bkd, nmodels=2)
        costs = bkd.array([2.0, 1.0])
        fitter = BruteForceSubsetFitter(
            stat, costs, GroupACVEstimatorIS
        )
        assert fitter._bkd is bkd
        count = sum(1 for _ in fitter._iter_patterns())
        assert count > 0
