"""Tests for mixed-mean ML-BLUE estimator.

Tests that partial known low-fidelity means produce correct restriction,
correction, and variance reduction in MLBLUEEstimator and
GroupACVEstimatorNested.
"""

import numpy as np
import pytest

from pyapprox_benchmarks.statest.multioutput_ensemble import (
    MultiOutputEnsembleBenchmark,
)
from pyapprox_benchmarks.statest.polynomial_ensemble import (
    PolynomialEnsembleBenchmark,
)
from pyapprox.statest.allocation import CVAllocator
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.groupacv import (
    GroupACVEstimatorNested,
    MLBLUEEstimator,
    MLBLUESPDAllocationOptimizer,
)
from pyapprox.statest.groupacv.allocation import GroupACVAllocationResult
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputVariance,
)
from pyapprox.util.optional_deps import package_available

HAS_CVXPY = package_available("cvxpy")


def _poly_benchmark(bkd, nmodels=5):
    return PolynomialEnsembleBenchmark(bkd, nmodels=nmodels)


def _multioutput_benchmark(bkd):
    return MultiOutputEnsembleBenchmark(bkd, psd=True)


def _make_stat(bkd, cov, nqoi=1):
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    return stat


def _known_means_dict(models, means_arr, nqoi=1):
    """Build known_quantities dict from model indices and means array.

    Parameters
    ----------
    models : list of int
        Model indices (1-based).
    means_arr : Array
        Means array, shape (len(models), nqoi).
    nqoi : int
        Number of QoI.
    """
    kq = {}
    for ii, m in enumerate(models):
        kq[(m, "mean")] = means_arr[ii, :nqoi]
    return kq


def _make_mlblue_nqoi1(bkd, nmodels=5, **kwargs):
    bench = _poly_benchmark(bkd, nmodels)
    cov = bench.ensemble_covariance()
    costs = bench.problem().costs()
    stat = _make_stat(bkd, cov, 1)
    est = MLBLUEEstimator(stat, costs, **kwargs)
    return est, bench


def _make_mlblue_multi_qoi(bkd, nqoi=3, **kwargs):
    bench = _multioutput_benchmark(bkd)
    nmodels = 3
    cov = bench.ensemble_covariance()
    costs = bench.problem().costs()
    stat = _make_stat(bkd, cov, nqoi)
    est = MLBLUEEstimator(stat, costs, **kwargs)
    return est, bench, nmodels


def _make_allocation(est, npartition_samples):
    bkd = est.bkd()
    rounded = bkd.floor(npartition_samples + 1e-4)
    nsamples_per_model = est._compute_nsamples_per_model(rounded)
    actual_cost = float(est._estimator_cost(rounded))
    return GroupACVAllocationResult(
        npartition_samples=rounded,
        nsamples_per_model=nsamples_per_model,
        actual_cost=actual_cost,
        objective_value=bkd.array([0.0]),
        success=True,
        message="",
    )


NQOI_VALUES = [1, 3]


class TestKnownQuantitiesValidation:
    """Input validation tests for known_quantities dict."""

    def test_model_0_rejected(self, bkd) -> None:
        bench = _poly_benchmark(bkd, 3)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        stat = _make_stat(bkd, cov)
        with pytest.raises(ValueError, match="Model 0"):
            MLBLUEEstimator(
                stat, costs,
                known_quantities={(0, "mean"): bkd.asarray([0.5])},
            )

    def test_out_of_range_rejected(self, bkd) -> None:
        bench = _poly_benchmark(bkd, 3)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        stat = _make_stat(bkd, cov)
        with pytest.raises(ValueError, match="out of range"):
            MLBLUEEstimator(
                stat, costs,
                known_quantities={(5, "mean"): bkd.asarray([0.5])},
            )

    def test_shape_mismatch_rejected(self, bkd) -> None:
        bench = _poly_benchmark(bkd, 3)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        stat = _make_stat(bkd, cov)
        with pytest.raises(ValueError, match="shape"):
            MLBLUEEstimator(
                stat, costs,
                known_quantities={(1, "mean"): bkd.asarray([0.5, 0.3])},
            )

    def test_nonfinite_rejected(self, bkd) -> None:
        bench = _poly_benchmark(bkd, 3)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        stat = _make_stat(bkd, cov)
        with pytest.raises(ValueError, match="non-finite"):
            MLBLUEEstimator(
                stat, costs,
                known_quantities={
                    (1, "mean"): bkd.asarray([float("nan")])
                },
            )

    def test_variance_only_stat_rejected(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        nmodels = 3
        cov = bkd.eye(nmodels)
        costs = bkd.ones((nmodels,))
        stat = MultiOutputVariance(1, bkd)
        W = bkd.eye(nmodels)
        stat.set_pilot_quantities(cov, W)
        with pytest.raises(NotImplementedError, match="mean estimation"):
            MLBLUEEstimator(
                stat, costs,
                known_quantities={(1, "variance"): bkd.asarray([0.5])},
            )

    def test_invalid_stat_name_rejected(self, bkd) -> None:
        bench = _poly_benchmark(bkd, 3)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        stat = _make_stat(bkd, cov)
        with pytest.raises(ValueError, match="not available"):
            MLBLUEEstimator(
                stat, costs,
                known_quantities={(1, "banana"): bkd.asarray([0.5])},
            )

    def test_variance_on_mean_stat_rejected(self, bkd) -> None:
        bench = _poly_benchmark(bkd, 3)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        stat = _make_stat(bkd, cov)
        with pytest.raises(ValueError, match="not available"):
            MLBLUEEstimator(
                stat, costs,
                known_quantities={(1, "variance"): bkd.asarray([0.5])},
            )


class TestEmptyKBitIdentical:
    """Verify that empty K produces identical results to standard estimator."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize("nqoi", NQOI_VALUES)
    def test_none_vs_standard(self, bkd, nqoi) -> None:
        if nqoi == 1:
            est_std, _ = _make_mlblue_nqoi1(bkd, nmodels=3)
            est_km, _ = _make_mlblue_nqoi1(
                bkd, nmodels=3,
                known_quantities=None,
            )
        else:
            est_std, _, _ = _make_mlblue_multi_qoi(bkd, nqoi=nqoi)
            est_km, _, _ = _make_mlblue_multi_qoi(
                bkd, nqoi=nqoi,
                known_quantities=None,
            )

        nps = bkd.full((est_std.nsubsets(),), 50.0)
        est_std.set_allocation(_make_allocation(est_std, nps))
        est_km.set_allocation(_make_allocation(est_km, nps))

        sigma_std = est_std.optimized_sigma()
        sigma_km = est_km.optimized_sigma()
        bkd.assert_allclose(sigma_std, sigma_km)

        beta_std = est_std._grouped_acv_beta(sigma_std)
        beta_km = est_km._grouped_acv_beta(sigma_km)
        bkd.assert_allclose(beta_std, beta_km)

        cov_std = est_std.optimized_covariance()
        cov_km = est_km.optimized_covariance()
        bkd.assert_allclose(cov_std, cov_km)

    @pytest.mark.parametrize("nqoi", NQOI_VALUES)
    def test_empty_dict_vs_standard(self, bkd, nqoi) -> None:
        if nqoi == 1:
            est_std, _ = _make_mlblue_nqoi1(bkd, nmodels=3)
            est_km, _ = _make_mlblue_nqoi1(
                bkd, nmodels=3,
                known_quantities={},
            )
        else:
            est_std, _, _ = _make_mlblue_multi_qoi(bkd, nqoi=nqoi)
            est_km, _, _ = _make_mlblue_multi_qoi(
                bkd, nqoi=nqoi,
                known_quantities={},
            )

        nps = bkd.full((est_std.nsubsets(),), 50.0)
        est_std.set_allocation(_make_allocation(est_std, nps))
        est_km.set_allocation(_make_allocation(est_km, nps))

        bkd.assert_allclose(
            est_std.optimized_covariance(),
            est_km.optimized_covariance(),
        )


class TestAllKnownMeansRecoversCVEstimator:
    """All-known-means + single group should recover CVEstimator."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize("nqoi", NQOI_VALUES)
    def test_estimator_value(self, bkd, nqoi) -> None:
        if nqoi == 1:
            nmodels = 3
            bench = _poly_benchmark(bkd, nmodels)
        else:
            nmodels = 3
            bench = _multioutput_benchmark(bkd)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        means = bench.ensemble_means()

        kq = _known_means_dict(list(range(1, nmodels)), means[1:], nqoi)

        stat_mlblue = _make_stat(bkd, cov, nqoi)
        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        est_mlblue = MLBLUEEstimator(
            stat_mlblue, costs,
            model_subsets=subsets,
            known_quantities=kq,
        )

        nsamples = 100
        nps = bkd.full((est_mlblue.nsubsets(),), float(nsamples))
        est_mlblue.set_allocation(_make_allocation(est_mlblue, nps))

        stat_cv = _make_stat(bkd, cov, nqoi)
        lowfi_stats = bkd.zeros((nmodels - 1, stat_cv.nstats()))
        for ii in range(nmodels - 1):
            for qq in range(nqoi):
                lowfi_stats[ii, qq] = means[ii + 1, qq]
        est_cv = CVEstimator(stat_cv, costs, lowfi_stats=lowfi_stats)
        fitted_cv = CVAllocator(est_cv).allocate(float(bkd.sum(costs)) * nsamples)

        np.random.seed(123)
        nsamples_int = int(nsamples)
        values_per_model = [
            bkd.asarray(np.random.randn(nqoi, nsamples_int))
            for _ in range(nmodels)
        ]

        est_mlblue_val = est_mlblue(values_per_model)
        est_cv_val = fitted_cv(values_per_model)
        bkd.assert_allclose(est_mlblue_val, est_cv_val, rtol=1e-8)

    def test_all_known_variance_equals_cv_floor(self, bkd) -> None:
        nmodels = 5
        bench = _poly_benchmark(bkd, nmodels)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        means = bench.ensemble_means()

        kq = _known_means_dict(list(range(1, nmodels)), means[1:])

        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        stat = _make_stat(bkd, cov, 1)
        est = MLBLUEEstimator(
            stat, costs, model_subsets=subsets,
            known_quantities=kq,
        )
        nsamples = 100.0
        nps = bkd.full((est.nsubsets(),), nsamples)
        est.set_allocation(_make_allocation(est, nps))

        cov_np = np.asarray(cov)
        sigma0_sq = cov_np[0, 0]
        C_LL = cov_np[1:, 1:]
        C_0L = cov_np[0, 1:]
        R2 = C_0L @ np.linalg.solve(C_LL, C_0L) / sigma0_sq
        expected_var = sigma0_sq * (1 - R2) / nsamples

        actual_var = float(est.optimized_covariance()[0, 0])
        bkd.assert_allclose(
            bkd.asarray([actual_var]),
            bkd.asarray([expected_var]),
            rtol=1e-10,
        )


class TestVarianceMonotonicity:
    """Variance decreases (or stays equal) as |K| grows."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize("nqoi", NQOI_VALUES)
    def test_monotonicity_sweep(self, bkd, nqoi) -> None:
        if nqoi == 1:
            nmodels = 5
            bench = _poly_benchmark(bkd, nmodels)
        else:
            nmodels = 3
            bench = _multioutput_benchmark(bkd)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        means = bench.ensemble_means()

        nsamples = 200.0
        prev_var = None
        for k_size in range(nmodels):
            if k_size > 0:
                kq = _known_means_dict(
                    list(range(1, 1 + k_size)),
                    means[1:1 + k_size],
                    nqoi,
                )
            else:
                kq = None
            stat = _make_stat(bkd, cov, nqoi)
            est = MLBLUEEstimator(
                stat, costs,
                known_quantities=kq,
            )
            nps = bkd.full((est.nsubsets(),), nsamples)
            est.set_allocation(_make_allocation(est, nps))
            var = float(est.optimized_covariance()[0, 0])
            if prev_var is not None:
                assert var <= prev_var + 1e-6, (
                    f"|K|={k_size}: var={var:.2e} > prev={prev_var:.2e}"
                )
            prev_var = var


class TestIntermediateK:
    """Intermediate |K| produces variance between empty-K and all-K."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_intermediate_variance(self, bkd) -> None:
        nmodels = 5
        bench = _poly_benchmark(bkd, nmodels)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        means = bench.ensemble_means()

        nsamples = 200.0

        def _var_for_k(kq):
            stat = _make_stat(bkd, cov)
            est = MLBLUEEstimator(
                stat, costs,
                known_quantities=kq,
            )
            nps = bkd.full((est.nsubsets(),), nsamples)
            est.set_allocation(_make_allocation(est, nps))
            return float(est.optimized_covariance()[0, 0])

        var_empty = _var_for_k(None)
        var_partial = _var_for_k(
            _known_means_dict([2, 3], means[2:4])
        )
        var_all = _var_for_k(
            _known_means_dict(list(range(1, nmodels)), means[1:])
        )

        assert var_all <= var_partial + 1e-6
        assert var_partial <= var_empty + 1e-6


class TestEmpiricalUnbiasedness:
    """Empirical unbiasedness via Monte Carlo trials."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)

    @pytest.mark.slow_on("TorchBkd")
    def test_empirical_bias(self, bkd) -> None:
        nmodels = 5
        bench = _poly_benchmark(bkd, nmodels)
        cov_np = np.array(bench.ensemble_covariance())
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        means = bench.ensemble_means()
        true_mean = float(means[0, 0])

        kq = _known_means_dict([2, 4], means[[2, 4]])

        stat = _make_stat(bkd, cov)
        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        est = MLBLUEEstimator(
            stat, costs,
            model_subsets=subsets,
            known_quantities=kq,
        )
        nps = bkd.full((est.nsubsets(),), 50.0)
        est.set_allocation(_make_allocation(est, nps))

        nsamps_per = np.array(est._compute_nsamples_per_model(nps), dtype=int)
        ntrials = 10000
        estimates = np.empty(ntrials)
        means_np = np.array(means).ravel()

        L = np.linalg.cholesky(cov_np)
        nsamples_int = int(nsamps_per[0])
        assert np.all(nsamps_per == nsamples_int)
        for trial in range(ntrials):
            z = np.random.randn(nmodels, nsamples_int)
            raw = L @ z
            values_per_model = [
                bkd.asarray(raw[ii:ii+1, :] + means_np[ii])
                for ii in range(nmodels)
            ]
            result = est(values_per_model)
            estimates[trial] = float(result[0])

        empirical_mean = np.mean(estimates)
        empirical_se = np.std(estimates) / np.sqrt(ntrials)
        bias = abs(empirical_mean - true_mean)
        assert bias / empirical_se < 3, (
            f"bias/SE={bias/empirical_se:.2f}, "
            f"empirical_mean={empirical_mean:.6f}, "
            f"true_mean={true_mean:.6f}"
        )


class TestAnalyticalVsEmpiricalVariance:
    """Analytical variance matches empirical variance within tolerance."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)

    @pytest.mark.slow_on("TorchBkd")
    def test_variance_match(self, bkd) -> None:
        nmodels = 5
        bench = _poly_benchmark(bkd, nmodels)
        cov_np = np.array(bench.ensemble_covariance())
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        means = bench.ensemble_means()

        kq = _known_means_dict([2, 4], means[[2, 4]])

        stat = _make_stat(bkd, cov)
        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        est = MLBLUEEstimator(
            stat, costs,
            model_subsets=subsets,
            known_quantities=kq,
        )
        nps = bkd.full((est.nsubsets(),), 50.0)
        est.set_allocation(_make_allocation(est, nps))

        analytical_var = float(est.optimized_covariance()[0, 0])

        nsamps_per = np.array(est._compute_nsamples_per_model(nps), dtype=int)
        nsamples_int = int(nsamps_per[0])
        assert np.all(nsamps_per == nsamples_int)
        ntrials = 10000
        estimates = np.empty(ntrials)
        means_np = np.array(means).ravel()
        L = np.linalg.cholesky(cov_np)
        for trial in range(ntrials):
            z = np.random.randn(nmodels, nsamples_int)
            raw = L @ z
            values_per_model = [
                bkd.asarray(raw[ii:ii+1, :] + means_np[ii])
                for ii in range(nmodels)
            ]
            result = est(values_per_model)
            estimates[trial] = float(result[0])

        empirical_var = np.var(estimates)
        rel_err = abs(empirical_var - analytical_var) / analytical_var
        assert rel_err < 0.10, (
            f"rel_err={rel_err:.3f}, "
            f"empirical={empirical_var:.2e}, analytical={analytical_var:.2e}"
        )


class TestNestedVariant:
    """Tests for GroupACVEstimatorNested with known means."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize("nqoi", NQOI_VALUES)
    def test_empty_k_bit_identical(self, bkd, nqoi) -> None:
        if nqoi == 1:
            nmodels = 3
            bench = _poly_benchmark(bkd, nmodels)
        else:
            nmodels = 3
            bench = _multioutput_benchmark(bkd)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        stat_std = _make_stat(bkd, cov, nqoi)
        stat_km = _make_stat(bkd, cov, nqoi)

        est_std = GroupACVEstimatorNested(stat_std, costs)
        est_km = GroupACVEstimatorNested(
            stat_km, costs,
            known_quantities=None,
        )

        nps = bkd.full((est_std.nsubsets(),), 50.0)
        est_std.set_allocation(_make_allocation(est_std, nps))
        est_km.set_allocation(_make_allocation(est_km, nps))

        bkd.assert_allclose(
            est_std.optimized_covariance(),
            est_km.optimized_covariance(),
        )

    def test_monotonicity(self, bkd) -> None:
        nmodels = 4
        bench = _poly_benchmark(bkd, nmodels)
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        means = bench.ensemble_means()

        nsamples = 200.0
        prev_var = None
        for k_size in range(nmodels):
            if k_size > 0:
                kq = _known_means_dict(
                    list(range(1, 1 + k_size)),
                    means[1:1 + k_size],
                )
            else:
                kq = None
            stat = _make_stat(bkd, cov)
            est = GroupACVEstimatorNested(
                stat, costs,
                known_quantities=kq,
            )
            nps = bkd.full((est.nsubsets(),), nsamples)
            est.set_allocation(_make_allocation(est, nps))
            var = float(est.optimized_covariance()[0, 0])
            if prev_var is not None:
                assert var <= prev_var + 1e-6
            prev_var = var


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestSDPSmoke:
    """SDP optimizer smoke test with known means."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_sdp_feasible(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        nmodels = 5
        bench = _poly_benchmark(bkd, nmodels)
        means = bench.ensemble_means()
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()

        stat = _make_stat(bkd, cov, 1)
        kq = _known_means_dict([1], means[1:2])
        est = MLBLUEEstimator(
            stat, costs,
            known_quantities=kq,
        )

        optimizer = MLBLUESPDAllocationOptimizer(est)
        result = optimizer.optimize(target_cost=10.0)
        assert result.success
        nps = result.npartition_samples
        assert bkd.all_bool(nps >= -1e-8)

    def test_sdp_multi_qoi_feasible(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        bench = _multioutput_benchmark(bkd)
        means = bench.ensemble_means()
        cov = bench.ensemble_covariance()
        costs = bench.problem().costs()
        nqoi = 3

        stat = _make_stat(bkd, cov, nqoi)
        kq = _known_means_dict([1], means[1:2], nqoi)
        est = MLBLUEEstimator(
            stat, costs,
            known_quantities=kq,
        )
        with pytest.raises(RuntimeError, match="single outputs"):
            optimizer = MLBLUESPDAllocationOptimizer(est)
            optimizer.optimize(target_cost=10.0)
