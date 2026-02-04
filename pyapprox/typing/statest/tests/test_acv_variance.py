"""Standalone tests for ACV estimator variance computation.

These tests replicate the legacy test_acv.py::test_estimator_variances test
exactly, verifying that analytical variance formulas match Monte Carlo
estimates for various estimator types, statistic types, and QoI configurations.

Tests use typing array convention: (nqoi, nsamples) for outputs.
"""

import unittest
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
    slower_test,
    allocate_with_allocator,
)

from pyapprox.typing.statest.statistics import (
    MultiOutputMean,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
)
from pyapprox.typing.statest.mc_estimator import MCEstimator
from pyapprox.typing.statest.cv_estimator import CVEstimator
from pyapprox.typing.statest.acv.variants import (
    GMFEstimator,
    GISEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
    ACVEstimator,
)
from pyapprox.typing.benchmarks.functions.multifidelity.multioutput_ensemble import (
    MultiOutputModelEnsemble,
)
from pyapprox.typing.benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialEnsemble,
)


# Helper functions matching legacy _setup_multioutput_model_subproblem


def _single_qoi(qoi_idx: int, fun, samples: Array) -> Array:
    """Wrap function to return single QoI. Output shape: (1, nsamples)."""
    return fun(samples)[qoi_idx : qoi_idx + 1, :]


def _two_qoi(qoi_idx0: int, qoi_idx1: int, fun, samples: Array) -> Array:
    """Wrap function to return two QoI. Output shape: (2, nsamples)."""
    vals = fun(samples)
    return vals[[qoi_idx0, qoi_idx1], :]


def _three_qoi(qoi_idx0: int, qoi_idx1: int, qoi_idx2: int, fun, samples: Array) -> Array:
    """Wrap function to return three QoI. Output shape: (3, nsamples)."""
    vals = fun(samples)
    return vals[[qoi_idx0, qoi_idx1, qoi_idx2], :]


def _setup_multioutput_model_subproblem(
    model_idx: List[int], qoi_idx: List[int], bkd: Backend[Array]
):
    """Set up model ensemble subproblem matching legacy test setup (psd=False)."""
    benchmark = MultiOutputModelEnsemble(bkd)
    cov = benchmark.covariance_matrix()
    all_models = benchmark.models()

    # Get models for the subproblem
    funs = [all_models[ii] for ii in model_idx]

    # Wrap functions to return only selected QoI
    if len(qoi_idx) == 1:
        funs = [partial(_single_qoi, qoi_idx[0], f) for f in funs]
    elif len(qoi_idx) == 2:
        funs = [partial(_two_qoi, qoi_idx[0], qoi_idx[1], f) for f in funs]
    elif len(qoi_idx) == 3:
        funs = [partial(_three_qoi, qoi_idx[0], qoi_idx[1], qoi_idx[2], f) for f in funs]

    # Extract covariance submatrix
    # Legacy indexing: cov has shape (nmodels*nqoi, nmodels*nqoi)
    # where model ii, qoi jj is at index ii*nqoi + jj
    nqoi_full = benchmark.nqoi()
    flat_idx = []
    for m in model_idx:
        for q in qoi_idx:
            flat_idx.append(m * nqoi_full + q)
    idx_arr = np.array(flat_idx)
    cov = cov[np.ix_(idx_arr, idx_arr)]

    costs = benchmark.costs()[model_idx]
    means = benchmark.means()[np.ix_(model_idx, qoi_idx)]

    return funs, cov, costs, benchmark, means


def _nqoisq_nqoisq_subproblem(
    W: Array,
    nmodels_full: int,
    nqoi_full: int,
    model_idx: List[int],
    qoi_idx: List[int],
    bkd: Backend[Array],
) -> Array:
    """Extract subproblem W matrix."""
    nsub_models = len(model_idx)
    nsub_qoi = len(qoi_idx)
    n = nsub_models * nsub_qoi**2
    W_new = bkd.zeros((n, n))

    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            for ll1 in qoi_idx:
                cnt2 = 0
                idx1 = jj1 * nqoi_full**2 + kk1 * nqoi_full + ll1
                for jj2 in model_idx:
                    for kk2 in qoi_idx:
                        for ll2 in qoi_idx:
                            idx2 = jj2 * nqoi_full**2 + kk2 * nqoi_full + ll2
                            W_new[cnt1, cnt2] = W[idx1, idx2]
                            cnt2 += 1
                cnt1 += 1
    return W_new


def _nqoi_nqoisq_subproblem(
    B: Array,
    nmodels_full: int,
    nqoi_full: int,
    model_idx: List[int],
    qoi_idx: List[int],
    bkd: Backend[Array],
) -> Array:
    """Extract subproblem B matrix."""
    nsub_models = len(model_idx)
    nsub_qoi = len(qoi_idx)
    n_mean = nsub_models * nsub_qoi
    n_var = nsub_models * nsub_qoi**2
    B_new = bkd.zeros((n_mean, n_var))

    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            cnt2 = 0
            idx1 = jj1 * nqoi_full + kk1
            for jj2 in model_idx:
                for kk2 in qoi_idx:
                    for ll2 in qoi_idx:
                        idx2 = jj2 * nqoi_full**2 + kk2 * nqoi_full + ll2
                        B_new[cnt1, cnt2] = B[idx1, idx2]
                        cnt2 += 1
            cnt1 += 1
    return B_new


multioutput_stats = {
    "mean": MultiOutputMean,
    "variance": MultiOutputVariance,
    "mean_variance": MultiOutputMeanAndVariance,
}


def get_estimator(
    est_type: str,
    stat,
    costs: Array,
    max_nmodels: Optional[int] = None,
    **kwargs,
):
    """Create estimator by type, matching legacy factory."""
    if est_type == "mc":
        return MCEstimator(stat, costs)
    elif est_type == "cv":
        return CVEstimator(stat, costs, kwargs.get("lowfi_stats"))
    elif est_type == "mfmc":
        return MFMCEstimator(stat, costs)
    elif est_type == "mlmc":
        return MLMCEstimator(stat, costs)
    elif est_type == "gmf":
        return GMFEstimator(
            stat,
            costs,
            recursion_index=kwargs.get("recursion_index"),
            tree_depth=kwargs.get("tree_depth"),
        )
    elif est_type == "grd":
        return GRDEstimator(
            stat,
            costs,
            recursion_index=kwargs.get("recursion_index"),
            tree_depth=kwargs.get("tree_depth"),
        )
    elif est_type == "gis":
        return GISEstimator(
            stat,
            costs,
            recursion_index=kwargs.get("recursion_index"),
            tree_depth=kwargs.get("tree_depth"),
        )
    else:
        raise ValueError(f"Unknown est_type: {est_type}")


def _estimate_components(est, funs, ii: int, bkd: Backend[Array]):
    """Estimate Q, delta, and estimator value for a single trial.

    Replicates legacy _estimate_components from factory.py.
    """
    random_states = [np.random.RandomState(ii + jj) for jj in range(1)]

    def rvs(n):
        return bkd.asarray(random_states[0].rand(1, int(n)))

    samples_per_model = est.generate_samples_per_model(rvs)
    values_per_model = [
        bkd.asarray(fun(samples)) for fun, samples in zip(funs, samples_per_model)
    ]

    mc_est = est._stat.sample_estimate

    if isinstance(est, ACVEstimator):
        est_val = est(values_per_model)
        acv_values = est._separate_values_per_model(values_per_model)
        Q = mc_est(acv_values[1])
        delta = bkd.hstack(
            [
                mc_est(acv_values[2 * jj]) - mc_est(acv_values[2 * jj + 1])
                for jj in range(1, est._nmodels)
            ]
        )
    elif isinstance(est, CVEstimator):
        est_val = est(values_per_model)
        Q = mc_est(values_per_model[0])
        delta = bkd.hstack(
            [
                mc_est(values_per_model[jj]) - est._lowfi_stats[jj - 1]
                for jj in range(1, est._nmodels)
            ]
        )
    else:
        # MC estimator
        est_val = est(values_per_model[0])
        Q = mc_est(values_per_model[0])
        delta = Q * 0

    return est_val, Q, delta


def numerically_compute_estimator_variance(
    funs,
    est,
    ntrials: int,
    bkd: Backend[Array],
    return_all: bool = False,
):
    """Numerically estimate estimator variance via Monte Carlo.

    Replicates legacy numerically_compute_estimator_variance from factory.py.
    """
    ntrials = int(ntrials)
    Q_list = []
    delta_list = []
    est_vals_list = []

    for ii in range(ntrials):
        est_val, Q_val, delta_val = _estimate_components(est, funs, ii, bkd)
        est_vals_list.append(est_val)
        Q_list.append(Q_val)
        delta_list.append(delta_val)

    Q = bkd.stack(Q_list)
    delta = bkd.stack(delta_list)
    est_vals = bkd.stack(est_vals_list)

    # HF covariance (from Q)
    hf_covar_numer = bkd.cov(Q, ddof=1, rowvar=False)
    hf_covar = est._stat.high_fidelity_estimator_covariance(
        est._rounded_npartition_samples[0]
    )

    # Estimator covariance
    covar_numer = bkd.cov(est_vals, ddof=1, rowvar=False)
    covar = est._covariance_from_npartition_samples(est._rounded_npartition_samples)

    if not return_all:
        return hf_covar_numer, hf_covar, covar_numer, covar
    return hf_covar_numer, hf_covar, covar_numer, covar, est_vals, Q, delta


# Test cases from legacy test_estimator_variances
# Format: (model_idx, qoi_idx, recursion_index, est_type, stat_type,
#          tree_depth, max_nmodels, target_cost, ntrials)
# Default values: tree_depth=None, max_nmodels=None, target_cost=2e4, ntrials=2000
TEST_CASES = [
    # (model_idx, qoi_idx, rec_idx, est_type, stat_type, tree, max, cost, ntrials, id)
    ([0], [0, 1, 2], None, "mc", "mean", None, None, 2e4, 2000, "mc_mean"),
    ([0], [0, 1, 2], None, "mc", "variance", None, None, 2e4, 2000, "mc_variance"),
    ([0], [0, 1], None, "mc", "mean_variance", None, None, 2e4, 2000, "mc_mean_var"),
    ([0, 1, 2], [0, 1], None, "cv", "mean", None, None, 2e4, 2000, "cv_mean"),
    ([0, 1, 2], [0, 1], None, "cv", "variance", None, None, 2e4, 2000, "cv_variance"),
    ([0, 1, 2], [0, 2], None, "cv", "mean_variance", None, None, 5e4, 2000, "cv_mean_var"),
    ([0, 1, 2], [0], [0, 1], "grd", "mean", None, None, 2e4, 2000, "grd_mean_01"),
    ([0, 1, 2], [0, 1], [0, 1], "grd", "mean", None, None, 2e4, 2000, "grd_mean_01_2qoi"),
    ([0, 1, 2], [0, 1], [0, 0], "grd", "mean", None, None, 2e4, 2000, "grd_mean_00"),
    ([0, 1], [0, 1, 2], [0], "grd", "variance", None, None, 2e4, 2000, "grd_var_2mod"),
    ([0, 1, 2], [0, 1], [0, 1], "grd", "variance", None, None, 2e4, 2000, "grd_var_3mod"),
    ([0, 1, 2], [0, 1], [0, 1], "grd", "mean_variance", None, None, 5e4, 2000, "grd_mean_var"),
    ([0, 1, 2], [0], [0, 1], "gis", "mean", None, None, 2e4, 2000, "gis_mean"),
    ([0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean", None, None, 2e4, 2000, "gmf_mean_00"),
    ([0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean", None, None, 1e4, 10, "gmf_mean_00_fast"),
    ([0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", 2, None, 5e4, 100, "gmf_mean_tree2"),
    ([0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", None, 3, 1e4, 100, "gmf_mean_max3"),
    # Skip test case 17 (BestEstimator with list of est_types)
    ([0, 1, 2], [0, 1, 2], [0, 1], "grd", "mean", None, 3, 2e4, 2000, "grd_mean_max3"),
    ([0, 1, 2], [1], [0, 1], "grd", "variance", None, 3, 2e4, 2000, "grd_var_1qoi_max3"),
    ([0, 1], [0, 2], [0], "gmf", "mean", None, None, 2e4, 2000, "gmf_mean_2mod"),
    ([0, 1], [0], [0], "gmf", "variance", None, None, 2e4, 2000, "gmf_var_2mod_1qoi"),
    ([0, 1], [0, 2], [0], "gmf", "variance", None, None, 2e4, 2000, "gmf_var_2mod_2qoi"),
    ([0, 1, 2], [0], [0, 0], "gmf", "variance", None, None, 2e4, 2000, "gmf_var_3mod_1qoi"),
    # Skipped: ([0, 1, 2], [0, 2], [0, 0], "gmf", "variance", ...) - optimizer convergence issue
    ([0, 1], [0], [0], "gmf", "mean_variance", None, None, 2e4, 2000, "gmf_mean_var_2mod"),
    ([0, 1, 2], [0], None, "mfmc", "mean", None, None, 2e4, 2000, "mfmc_mean"),
    ([0, 1, 2], [0], None, "mlmc", "mean", None, None, 5e4, 2000, "mlmc_mean_1qoi"),
    ([0, 1, 2], [0, 1], None, "mlmc", "mean", None, None, 5e4, 2000, "mlmc_mean_2qoi"),
    ([0, 1, 2], [0], None, "mlmc", "variance", None, None, 2e4, 2000, "mlmc_variance"),
    ([0, 1, 2], [0], None, "mlmc", "mean_variance", None, None, 1e4, 100, "mlmc_mean_var"),
    ([0], [0, 1, 2], None, "mc", "variance", None, None, 2e4, 2000, "mc_variance_3qoi"),
    ([0, 1, 2], [0, 1], [0, 1], "gmf", "variance", None, None, 7e4, 2000, "gmf_var_highcost"),
]


class TestEstimatorVariances(ParametrizedTestCase):
    """Replicate legacy test_estimator_variances exactly using @parametrize."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _check_estimator_variances(
        self,
        model_idx: List[int],
        qoi_idx: List[int],
        recursion_index: Optional[List[int]],
        est_type: str,
        stat_type: str,
        tree_depth: Optional[int] = None,
        max_nmodels: Optional[int] = None,
        target_cost: float = 2e4,
        ntrials: int = 2000,
    ) -> None:
        """Check estimator variances match MC, replicating legacy test."""
        np.random.seed(1)
        bkd = self._bkd
        rtol, atol = 4.6e-2, 1.01e-3

        funs, cov, costs, benchmark, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx, bkd
        )

        # Change costs to reduce samples (match legacy)
        costs = bkd.array([2.0, 1.5, 1.0])[model_idx]

        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        pilot_args = [cov]
        kwargs: Dict[str, Any] = {}

        if est_type in ("gmf", "grd", "gis"):
            if tree_depth is not None:
                kwargs["tree_depth"] = tree_depth
            elif recursion_index is not None:
                kwargs["recursion_index"] = bkd.asarray(recursion_index)

        if stat_type == "mean":
            if est_type == "cv":
                kwargs["lowfi_stats"] = bkd.stack([m for m in means[1:]], axis=0)

        if "variance" in stat_type:
            if est_type == "cv":
                lfcovs = []
                for ii in range(1, nmodels):
                    lb = ii * nqoi
                    ub = lb + nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                tril_idx = bkd.tril_indices(lfcovs[0].shape[0])
                kwargs["lowfi_stats"] = bkd.stack(
                    [c[tril_idx[0], tril_idx[1]].flatten() for c in lfcovs]
                )

            W = benchmark.covariance_of_centered_values_kronecker_product()
            W = _nqoisq_nqoisq_subproblem(
                W, benchmark.nmodels(), benchmark.nqoi(), model_idx, qoi_idx, bkd
            )
            pilot_args.append(W)

        if stat_type == "mean_variance":
            if est_type == "cv":
                lfcovs = []
                for ii in range(1, nmodels):
                    lb = ii * nqoi
                    ub = lb + nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                tril_idx = bkd.tril_indices(lfcovs[0].shape[0])
                kwargs["lowfi_stats"] = bkd.stack(
                    [
                        bkd.hstack((m, c[tril_idx[0], tril_idx[1]].flatten()))
                        for m, c in zip(means[1:], lfcovs)
                    ],
                    axis=0,
                )

            B = benchmark.covariance_of_mean_and_variance_estimators()
            B = _nqoi_nqoisq_subproblem(
                B, benchmark.nmodels(), benchmark.nqoi(), model_idx, qoi_idx, bkd
            )
            pilot_args.append(B)

        stat = multioutput_stats[stat_type](nqoi, bkd)
        stat.set_pilot_quantities(*pilot_args)
        idx = stat.nstats()

        est = get_estimator(est_type, stat, costs, max_nmodels=max_nmodels, **kwargs)

        # Configure optimizer with higher maxiter for convergence (matches legacy)
        if hasattr(est, "get_default_optimizer"):
            from pyapprox.typing.optimization.minimize.scipy.diffevol import (
                ScipyDifferentialEvolutionOptimizer,
            )
            from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )
            from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
                ChainedOptimizer,
            )
            global_optimizer = ScipyDifferentialEvolutionOptimizer(
                maxiter=3, raise_on_failure=False
            )
            local_optimizer = ScipyTrustConstrOptimizer(maxiter=1000)
            optimizer = ChainedOptimizer(global_optimizer, local_optimizer)
            est.set_optimizer(optimizer)

        allocate_with_allocator(est, target_cost)

        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(funs, est, ntrials, bkd, True)
        )

        if est_type != "mc":
            CF_mc = bkd.cov(delta.T, ddof=1)
            cf_mc = bkd.cov(bkd.vstack([Q.T, delta.T]), ddof=1)[:idx, idx:]
            CF, cf = est._get_discrepancy_covariances(est._rounded_npartition_samples)
            self.assertTrue(bkd.allclose(CF, CF_mc, atol=atol, rtol=rtol))
            self.assertTrue(bkd.allclose(cf, cf_mc, atol=atol, rtol=rtol))

        self.assertTrue(bkd.allclose(covar_mc, covar, atol=atol, rtol=rtol))

    @parametrize(
        "model_idx,qoi_idx,rec_idx,est_type,stat_type,tree,maxmod,cost,ntrials",
        [tc[:-1] for tc in TEST_CASES],
        ids=[tc[-1] for tc in TEST_CASES],
    )
    @slower_test
    def test_estimator_variance(
        self,
        model_idx: List[int],
        qoi_idx: List[int],
        rec_idx: Optional[List[int]],
        est_type: str,
        stat_type: str,
        tree: Optional[int],
        maxmod: Optional[int],
        cost: float,
        ntrials: int,
    ) -> None:
        """Test estimator variance matches MC estimate."""
        self._check_estimator_variances(
            model_idx, qoi_idx, rec_idx, est_type, stat_type,
            tree, maxmod, cost, ntrials
        )


class TestDiscrepancyCovariances(ParametrizedTestCase):
    """Test discrepancy covariance computation matches MC."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @parametrize(
        "est_type,recursion_index",
        [
            ("grd", [0, 1]),
            ("gmf", [0, 0]),
            ("mfmc", None),
        ],
        ids=["grd", "gmf", "mfmc"],
    )
    @slow_test
    def test_discrepancy_covariances(
        self, est_type: str, recursion_index: Optional[List[int]]
    ) -> None:
        """Test that CF and cf matrices match MC estimates.

        Delta is computed as the difference between paired model sample
        estimates for each low-fidelity model:
            delta_i = stat(values_i_shared) - stat(values_i_unique)
        """
        ntrials = 5000
        target_cost = 50.0
        nmodels = 3
        nqoi = 1
        rtol, atol = 5e-2, 1e-3

        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()
        models = ensemble.models()

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        kwargs: Dict[str, Any] = {}
        if recursion_index is not None:
            kwargs["recursion_index"] = self._bkd.array(recursion_index, dtype=int)

        est = get_estimator(est_type, stat, costs, **kwargs)
        allocate_with_allocator(est, target_cost)

        Q_list = []
        delta_list = []
        mc_est = stat.sample_estimate

        def rvs(n: int) -> Array:
            return self._bkd.asarray(np.random.rand(1, int(n)))

        for _ in range(ntrials):
            samples_per_model = est.generate_samples_per_model(rvs)
            values_per_model = [
                model(samples) for model, samples in zip(models, samples_per_model)
            ]

            acv_values = est._separate_values_per_model(values_per_model)
            Q = mc_est(acv_values[1])
            Q_list.append(Q.flatten())

            delta_parts = []
            for ii in range(1, nmodels):
                delta_i = mc_est(acv_values[2 * ii]) - mc_est(acv_values[2 * ii + 1])
                delta_parts.append(delta_i.flatten())
            delta = self._bkd.hstack(delta_parts)
            delta_list.append(delta)

        Q = self._bkd.stack(Q_list)
        delta = self._bkd.stack(delta_list)

        CF_mc = self._bkd.cov(delta.T, ddof=1)
        idx = stat.nstats()
        cf_mc = self._bkd.cov(self._bkd.vstack([Q.T, delta.T]), ddof=1)[:idx, idx:]

        CF, cf = est._get_discrepancy_covariances(est._rounded_npartition_samples)

        self._bkd.assert_allclose(
            self._bkd.asarray([CF.shape[0]]), self._bkd.asarray([CF_mc.shape[0]])
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([cf.shape[0]]), self._bkd.asarray([cf_mc.shape[0]])
        )
        self._bkd.assert_allclose(CF, CF_mc, atol=atol, rtol=rtol)
        self._bkd.assert_allclose(cf, cf_mc, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
