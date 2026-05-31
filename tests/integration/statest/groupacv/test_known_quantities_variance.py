"""Tests for known_quantities with variance and mean+variance stats.

Uses GroupACVEstimatorIS (not MLBLUEEstimator which rejects non-mean stats).
All benchmarks use MultiOutputEnsembleBenchmark(psd=True).
"""

from typing import Dict, List, Tuple

import numpy as np
import pytest

from pyapprox_benchmarks.statest.multioutput_ensemble import (
    MultiOutputEnsembleBenchmark,
)
from pyapprox.statest.allocation import CVAllocator
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.groupacv import (
    FittedGroupACVEstimator,
    GroupACVEstimatorIS,
    GroupACVEstimatorNested,
)
from pyapprox.statest.groupacv.allocation import GroupACVAllocationResult
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def _bench(bkd):
    return MultiOutputEnsembleBenchmark(bkd, psd=True)


def _make_allocation(est, npartition_samples):
    bkd = est.bkd()
    rounded = bkd.asarray(
        bkd.floor(npartition_samples + 1e-4), dtype=bkd.int64_dtype()
    )
    nsamples_per_model = bkd.asarray(
        est._compute_nsamples_per_model(
            bkd.asarray(rounded, dtype=bkd.double_dtype())
        ),
        dtype=bkd.int64_dtype(),
    )
    actual_cost = float(
        est._estimator_cost(bkd.asarray(rounded, dtype=bkd.double_dtype()))
    )
    return GroupACVAllocationResult(
        npartition_samples=rounded,
        nsamples_per_model=nsamples_per_model,
        actual_cost=actual_cost,
        objective_value=bkd.array([0.0]),  # Placeholder
        success=True,
        message="",
    )


def _setup_variance_stat(bkd, bench, model_idx, qoi_idx):
    nqoi = len(qoi_idx)
    cov = bench.covariance_subproblem(model_idx, qoi_idx)
    W = bench.covariance_of_centered_values_kronecker_product_subproblem(
        model_idx, qoi_idx
    )
    costs = bench.costs_subproblem(model_idx)
    stat = MultiOutputVariance(nqoi, bkd)
    stat.set_pilot_quantities(cov, W)
    return stat, costs, cov


def _setup_mean_variance_stat(bkd, bench, model_idx, qoi_idx):
    nqoi = len(qoi_idx)
    cov = bench.covariance_subproblem(model_idx, qoi_idx)
    W = bench.covariance_of_centered_values_kronecker_product_subproblem(
        model_idx, qoi_idx
    )
    B = bench.covariance_of_mean_and_variance_estimators_subproblem(
        model_idx, qoi_idx
    )
    costs = bench.costs_subproblem(model_idx)
    stat = MultiOutputMeanAndVariance(nqoi, bkd)
    stat.set_pilot_quantities(cov, W, B)
    return stat, costs, cov


def _compute_known_variance_values(
    bkd: Backend[Array],
    cov: Array,
    model_idx_in_cov: int,
    nqoi: int,
) -> Array:
    """Compute population variance values (lower-tri entries) for a model.

    The population covariance of Q_l is the nqoi x nqoi diagonal block
    of the ensemble covariance. We extract its lower-triangular entries
    in the same flat order as sample_estimate.
    """
    start = model_idx_in_cov * nqoi
    block = cov[start:start + nqoi, start:start + nqoi]
    tril_idx = bkd.tril_indices(nqoi)
    flat_idx = bkd.reshape(
        bkd.arange(nqoi**2, dtype=bkd.int64_dtype()), (nqoi, nqoi)
    )[tril_idx[0], tril_idx[1]]
    return bkd.flatten(block)[flat_idx]


def _compute_known_mean_values(
    bkd: Backend[Array],
    bench,
    model_idx: int,
    qoi_idx: List[int],
) -> Array:
    """Compute population mean values for a model."""
    means = bench.ensemble_means()
    return bkd.asarray([float(means[model_idx, q]) for q in qoi_idx])


def _build_known_quantities_variance(
    bkd, cov, nqoi, lf_model_indices, cov_model_offset=0,
) -> Dict[Tuple[int, str], Array]:
    """Build known_quantities dict for variance stat."""
    kq: Dict[Tuple[int, str], Array] = {}
    for m in lf_model_indices:
        kq[(m, "variance")] = _compute_known_variance_values(
            bkd, cov, m + cov_model_offset, nqoi
        )
    return kq


def _build_known_quantities_mean_variance(
    bkd, bench, cov, nqoi, qoi_idx, lf_model_indices, cov_model_offset=0,
) -> Dict[Tuple[int, str], Array]:
    """Build known_quantities dict for mean+variance stat."""
    kq: Dict[Tuple[int, str], Array] = {}
    for m in lf_model_indices:
        kq[(m, "mean")] = _compute_known_mean_values(
            bkd, bench, m + cov_model_offset, qoi_idx
        )
        kq[(m, "variance")] = _compute_known_variance_values(
            bkd, cov, m + cov_model_offset, nqoi
        )
    return kq


def _build_lowfi_stats_variance(
    bkd, stat, cov, nqoi, nmodels,
) -> Array:
    """Build lowfi_stats array for CVEstimator with variance stat."""
    nstats = stat.nstats()
    lowfi_stats = bkd.zeros((nmodels - 1, nstats))
    for ii in range(nmodels - 1):
        var_vals = _compute_known_variance_values(bkd, cov, ii + 1, nqoi)
        for s in range(nstats):
            lowfi_stats[ii, s] = var_vals[s]
    return lowfi_stats


def _build_lowfi_stats_mean_variance(
    bkd, bench, stat, cov, nqoi, qoi_idx, nmodels,
) -> Array:
    """Build lowfi_stats array for CVEstimator with mean+variance stat."""
    nstats = stat.nstats()
    lowfi_stats = bkd.zeros((nmodels - 1, nstats))
    for ii in range(nmodels - 1):
        mean_vals = _compute_known_mean_values(
            bkd, bench, ii + 1, qoi_idx
        )
        var_vals = _compute_known_variance_values(bkd, cov, ii + 1, nqoi)
        for q in range(nqoi):
            lowfi_stats[ii, q] = mean_vals[q]
        for s in range(var_vals.shape[0]):
            lowfi_stats[ii, nqoi + s] = var_vals[s]
    return lowfi_stats


STAT_CONFIGS = [
    ("variance", [0]),
    ("variance", [0, 1]),
    ("mean_and_variance", [0]),
    ("mean_and_variance", [0, 1]),
]


class TestKnownQuantitiesValidationVariance:
    """Validation tests specific to variance/mean+variance stats."""

    def test_mean_on_variance_stat_rejected(self, bkd) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        stat, costs, cov = _setup_variance_stat(bkd, bench, model_idx, qoi_idx)
        with pytest.raises(ValueError, match="not available"):
            GroupACVEstimatorIS(
                stat, costs,
                known_quantities={(1, "mean"): bkd.asarray([0.5])},
            )

    def test_mean_and_variance_partial_rejected(self, bkd) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        stat, costs, cov = _setup_mean_variance_stat(
            bkd, bench, model_idx, qoi_idx
        )
        with pytest.raises(ValueError, match="all-or-nothing"):
            GroupACVEstimatorIS(
                stat, costs,
                known_quantities={
                    (1, "mean"): bkd.asarray([0.5]),
                },
            )

    def test_mean_and_variance_partial_reversed_rejected(self, bkd) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        stat, costs, cov = _setup_mean_variance_stat(
            bkd, bench, model_idx, qoi_idx
        )
        var_vals = _compute_known_variance_values(bkd, cov, 1, 1)
        with pytest.raises(ValueError, match="all-or-nothing"):
            GroupACVEstimatorIS(
                stat, costs,
                known_quantities={
                    (1, "variance"): var_vals,
                },
            )


class TestHelperConvergence:
    """Verify _compute_known_*_values helpers via Monte Carlo convergence."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize(
        "stat_type,qoi_idx",
        STAT_CONFIGS,
    )
    @pytest.mark.slow_on("TorchBkd")
    def test_helper_converges(self, bkd, stat_type, qoi_idx) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        if stat_type == "variance":
            stat, costs, cov = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
        else:
            stat, costs, cov = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )

        models = bench.models_subproblem(model_idx, qoi_idx)
        nsamples = 200_000
        prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)
        samples = prior.rvs(nsamples)

        for m_idx in range(1, nmodels):
            vals = models[m_idx](samples)
            per_model_vals = vals[:nqoi, :]
            est = stat.sample_estimate(per_model_vals)

            if stat_type == "variance":
                expected = _compute_known_variance_values(
                    bkd, cov, m_idx, nqoi
                )
                bkd.assert_allclose(est, expected, rtol=0.05)
            else:
                mean_expected = _compute_known_mean_values(
                    bkd, bench, m_idx, qoi_idx
                )
                var_expected = _compute_known_variance_values(
                    bkd, cov, m_idx, nqoi
                )
                bkd.assert_allclose(est[:nqoi], mean_expected, rtol=0.05)
                bkd.assert_allclose(est[nqoi:], var_expected, rtol=0.05)


class TestEmptyKVariance:
    """Empty-K produces identical results for variance/mean+variance stats."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize("stat_type,qoi_idx", STAT_CONFIGS)
    @pytest.mark.parametrize(
        "EstimatorCls",
        [GroupACVEstimatorIS, GroupACVEstimatorNested],
        ids=["IS", "Nested"],
    )
    def test_empty_k_bit_identical(
        self, bkd, stat_type, qoi_idx, EstimatorCls
    ) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]

        if stat_type == "variance":
            stat_std, costs, _ = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            stat_kq, _, _ = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
        else:
            stat_std, costs, _ = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            stat_kq, _, _ = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )

        est_std = EstimatorCls(stat_std, costs)
        est_kq = EstimatorCls(stat_kq, costs, known_quantities=None)
        est_empty = EstimatorCls(stat_kq, costs, known_quantities={})

        nps = bkd.full((est_std.nsubsets(),), 50.0)

        cov_std = est_std._covariance_from_npartition_samples(nps)
        bkd.assert_allclose(
            est_kq._covariance_from_npartition_samples(nps), cov_std
        )
        bkd.assert_allclose(
            est_empty._covariance_from_npartition_samples(nps), cov_std
        )


class TestCVRecoveryVariance:
    """All-known LF recovers CVEstimator for variance/mean+variance stats."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize("stat_type,qoi_idx", STAT_CONFIGS)
    @pytest.mark.parametrize(
        "EstimatorCls",
        [GroupACVEstimatorIS, GroupACVEstimatorNested],
        ids=["IS", "Nested"],
    )
    def test_cv_recovery_estimator_value(
        self, bkd, stat_type, qoi_idx, EstimatorCls
    ) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        if stat_type == "variance":
            stat_gacv, costs, cov = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            stat_cv, _, _ = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_variance(
                bkd, cov, nqoi, list(range(1, nmodels))
            )
            lowfi_stats = _build_lowfi_stats_variance(
                bkd, stat_cv, cov, nqoi, nmodels
            )
        else:
            stat_gacv, costs, cov = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            stat_cv, _, _ = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_mean_variance(
                bkd, bench, cov, nqoi, qoi_idx, list(range(1, nmodels))
            )
            lowfi_stats = _build_lowfi_stats_mean_variance(
                bkd, bench, stat_cv, cov, nqoi, qoi_idx, nmodels
            )

        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        est_gacv = EstimatorCls(
            stat_gacv, costs,
            model_subsets=subsets,
            known_quantities=kq,
        )
        nsamples = 100
        nps = bkd.full((est_gacv.nsubsets(),), float(nsamples))
        alloc = _make_allocation(est_gacv, nps)
        fitted_gacv = FittedGroupACVEstimator(est_gacv, alloc)

        est_cv = CVEstimator(stat_cv, costs, lowfi_stats=lowfi_stats)
        fitted_cv = CVAllocator(est_cv).allocate(float(bkd.sum(costs)) * nsamples)

        np.random.seed(123)
        nsamples_int = int(nsamples)
        values_per_model = [
            bkd.asarray(np.random.randn(nqoi, nsamples_int))
            for _ in range(nmodels)
        ]

        est_gacv_val = fitted_gacv(values_per_model)
        est_cv_val = fitted_cv(values_per_model)
        bkd.assert_allclose(est_gacv_val, est_cv_val, rtol=1e-8)

    @pytest.mark.parametrize("stat_type,qoi_idx", STAT_CONFIGS)
    @pytest.mark.parametrize(
        "EstimatorCls",
        [GroupACVEstimatorIS, GroupACVEstimatorNested],
        ids=["IS", "Nested"],
    )
    def test_cv_recovery_covariance(
        self, bkd, stat_type, qoi_idx, EstimatorCls
    ) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        if stat_type == "variance":
            stat_gacv, costs, cov = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            stat_cv, _, _ = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_variance(
                bkd, cov, nqoi, list(range(1, nmodels))
            )
            lowfi_stats = _build_lowfi_stats_variance(
                bkd, stat_cv, cov, nqoi, nmodels
            )
        else:
            stat_gacv, costs, cov = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            stat_cv, _, _ = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_mean_variance(
                bkd, bench, cov, nqoi, qoi_idx, list(range(1, nmodels))
            )
            lowfi_stats = _build_lowfi_stats_mean_variance(
                bkd, bench, stat_cv, cov, nqoi, qoi_idx, nmodels
            )

        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        est_gacv = EstimatorCls(
            stat_gacv, costs,
            model_subsets=subsets,
            known_quantities=kq,
        )
        nsamples = 100
        nps = bkd.full((est_gacv.nsubsets(),), float(nsamples))

        est_cv = CVEstimator(stat_cv, costs, lowfi_stats=lowfi_stats)
        fitted_cv = CVAllocator(est_cv).allocate(float(bkd.sum(costs)) * nsamples)

        bkd.assert_allclose(
            est_gacv._covariance_from_npartition_samples(nps),
            fitted_cv.covariance(),
            rtol=1e-8,
        )


class TestVarianceMonotonicityVariance:
    """Variance decreases as |K| grows for variance/mean+variance stats."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.mark.parametrize("stat_type,qoi_idx", STAT_CONFIGS)
    @pytest.mark.parametrize(
        "EstimatorCls",
        [GroupACVEstimatorIS, GroupACVEstimatorNested],
        ids=["IS", "Nested"],
    )
    def test_monotonicity_sweep(
        self, bkd, stat_type, qoi_idx, EstimatorCls
    ) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        nqoi = len(qoi_idx)
        nmodels = len(model_idx)
        nsamples = 200.0

        prev_var = None
        for k_size in range(nmodels):
            if stat_type == "variance":
                stat, costs, cov = _setup_variance_stat(
                    bkd, bench, model_idx, qoi_idx
                )
                if k_size > 0:
                    kq = _build_known_quantities_variance(
                        bkd, cov, nqoi, list(range(1, 1 + k_size))
                    )
                else:
                    kq = None
            else:
                stat, costs, cov = _setup_mean_variance_stat(
                    bkd, bench, model_idx, qoi_idx
                )
                if k_size > 0:
                    kq = _build_known_quantities_mean_variance(
                        bkd, bench, cov, nqoi, qoi_idx,
                        list(range(1, 1 + k_size))
                    )
                else:
                    kq = None

            est = EstimatorCls(stat, costs, known_quantities=kq)
            nps = bkd.full((est.nsubsets(),), nsamples)
            var = float(
                est._covariance_from_npartition_samples(nps)[0, 0]
            )
            if prev_var is not None:
                assert var <= prev_var + 1e-6, (
                    f"|K|={k_size}: var={var:.2e} > prev={prev_var:.2e}"
                )
            prev_var = var


class TestEmpiricalUnbiasednessVariance:
    """Empirical unbiasedness for variance/mean+variance with known quantities."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)

    @pytest.mark.parametrize(
        "stat_type,qoi_idx",
        [("variance", [0]), ("mean_and_variance", [0])],
    )
    @pytest.mark.parametrize(
        "EstimatorCls",
        [GroupACVEstimatorIS, GroupACVEstimatorNested],
        ids=["IS", "Nested"],
    )
    @pytest.mark.slow_on("*")
    def test_empirical_bias(
        self, bkd, stat_type, qoi_idx, EstimatorCls
    ) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        if stat_type == "variance":
            stat, costs, cov = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_variance(
                bkd, cov, nqoi, [1]
            )
            true_stat_0 = float(
                _compute_known_variance_values(bkd, cov, 0, nqoi)[0]
            )
        else:
            stat, costs, cov = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_mean_variance(
                bkd, bench, cov, nqoi, qoi_idx, [1]
            )
            true_stat_0 = float(bench.ensemble_means()[0, qoi_idx[0]])

        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        est = EstimatorCls(
            stat, costs,
            model_subsets=subsets,
            known_quantities=kq,
        )
        nps = bkd.full((est.nsubsets(),), 50.0)
        alloc = _make_allocation(est, nps)
        fitted = FittedGroupACVEstimator(est, alloc)

        models = bench.models_subproblem(model_idx, qoi_idx)
        prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)

        nsamps_per = np.array(
            est._compute_nsamples_per_model(nps), dtype=int
        )
        nsamples_int = int(nsamps_per[0])
        assert np.all(nsamps_per == nsamples_int)

        ntrials = 5000
        all_samples = prior.rvs(ntrials * nsamples_int)
        all_vals = [m(all_samples)[:nqoi, :] for m in models]

        estimates = np.empty(ntrials)
        for trial in range(ntrials):
            sl = slice(trial * nsamples_int, (trial + 1) * nsamples_int)
            values_per_model = [v[:, sl] for v in all_vals]
            result = fitted(values_per_model)
            estimates[trial] = float(result[0])

        empirical_mean = np.mean(estimates)
        empirical_se = np.std(estimates) / np.sqrt(ntrials)
        bias = abs(empirical_mean - true_stat_0)
        assert bias / empirical_se < 3, (
            f"bias/SE={bias/empirical_se:.2f}, "
            f"empirical_mean={empirical_mean:.6f}, "
            f"true_stat_0={true_stat_0:.6f}"
        )


class TestAnalyticalVsEmpiricalVarianceVariance:
    """Analytical variance matches empirical variance for variance stats."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)

    @pytest.mark.parametrize(
        "stat_type,qoi_idx",
        [("variance", [0]), ("mean_and_variance", [0])],
    )
    @pytest.mark.parametrize(
        "EstimatorCls",
        [GroupACVEstimatorIS, GroupACVEstimatorNested],
        ids=["IS", "Nested"],
    )
    @pytest.mark.slow_on("*")
    def test_variance_match(
        self, bkd, stat_type, qoi_idx, EstimatorCls
    ) -> None:
        bench = _bench(bkd)
        model_idx = [0, 1, 2]
        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        if stat_type == "variance":
            stat, costs, cov = _setup_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_variance(
                bkd, cov, nqoi, [1]
            )
        else:
            stat, costs, cov = _setup_mean_variance_stat(
                bkd, bench, model_idx, qoi_idx
            )
            kq = _build_known_quantities_mean_variance(
                bkd, bench, cov, nqoi, qoi_idx, [1]
            )

        subsets = [bkd.asarray(list(range(nmodels)), dtype=int)]
        est = EstimatorCls(
            stat, costs,
            model_subsets=subsets,
            known_quantities=kq,
        )
        nps = bkd.full((est.nsubsets(),), 50.0)
        alloc = _make_allocation(est, nps)
        fitted = FittedGroupACVEstimator(est, alloc)

        analytical_var = float(fitted.covariance()[0, 0])

        models = bench.models_subproblem(model_idx, qoi_idx)
        prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)

        nsamps_per = np.array(
            est._compute_nsamples_per_model(nps), dtype=int
        )
        nsamples_int = int(nsamps_per[0])
        assert np.all(nsamps_per == nsamples_int)

        ntrials = 5000
        all_samples = prior.rvs(ntrials * nsamples_int)
        all_vals = [m(all_samples)[:nqoi, :] for m in models]

        estimates = np.empty(ntrials)
        for trial in range(ntrials):
            sl = slice(trial * nsamples_int, (trial + 1) * nsamples_int)
            values_per_model = [v[:, sl] for v in all_vals]
            result = fitted(values_per_model)
            estimates[trial] = float(result[0])

        empirical_var = np.var(estimates)
        rel_err = abs(empirical_var - analytical_var) / analytical_var
        assert rel_err < 0.15, (
            f"rel_err={rel_err:.3f}, "
            f"empirical={empirical_var:.2e}, analytical={analytical_var:.2e}"
        )
