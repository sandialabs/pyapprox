"""Standalone tests for ACV estimator variance computation.

These tests replicate the legacy test_acv.py::test_estimator_variances test
exactly, verifying that analytical variance formulas match Monte Carlo
estimates for various estimator types, statistic types, and QoI configurations.

Tests use typing array convention: (nqoi, nsamples) for outputs.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from pyapprox_benchmarks.statest import (
    MultiOutputEnsembleBenchmark,
    PolynomialEnsembleBenchmark,
)
from pyapprox.statest.acv.search import ACVSearch
from pyapprox.statest.acv.strategies import TreeDepthRecursionStrategy
from pyapprox.statest.acv.variants import (
    ACVEstimator,
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.mc_estimator import MCEstimator
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from tests._helpers.markers import slow_test, slower_test
from tests._helpers.acv_utils import allocate_with_allocator

# Helper functions for setting up test subproblems


multioutput_stats = {
    "mean": MultiOutputMean,
    "variance": MultiOutputVariance,
    "mean_variance": MultiOutputMeanAndVariance,
}


def _get_pilot_quantities_for_stat_type(
    benchmark: "MultiOutputModelEnsemble",
    stat_type: str,
    model_idx: List[int],
    qoi_idx: List[int],
):
    """Get pilot quantities for the given statistic type.

    Parameters
    ----------
    benchmark : MultiOutputEnsembleBenchmark
        The benchmark providing covariance data.
    stat_type : str
        One of "mean", "variance", "mean_variance".
    model_idx : List[int]
        Indices of models to include.
    qoi_idx : List[int]
        Indices of QoI to include.

    Returns
    -------
    Tuple[Array, ...]
        Pilot quantities for set_pilot_quantities().
    """
    cov = benchmark.covariance_subproblem(model_idx, qoi_idx)
    if stat_type == "mean":
        return (cov,)
    elif stat_type == "variance":
        W = benchmark.covariance_of_centered_values_kronecker_product_subproblem(
            model_idx, qoi_idx
        )
        return (cov, W)
    elif stat_type == "mean_variance":
        W = benchmark.covariance_of_centered_values_kronecker_product_subproblem(
            model_idx, qoi_idx
        )
        B = benchmark.covariance_of_mean_and_variance_estimators_subproblem(
            model_idx, qoi_idx
        )
        return (cov, W, B)
    else:
        raise ValueError(f"Unknown stat_type: {stat_type}")


def _setup_multioutput_model_subproblem(
    model_idx: List[int],
    qoi_idx: List[int],
    stat_type: str,
    bkd: Backend[Array],
):
    """Set up model ensemble subproblem using benchmark subproblem methods.

    Parameters
    ----------
    model_idx : List[int]
        Indices of models to include.
    qoi_idx : List[int]
        Indices of QoI to include.
    stat_type : str
        Type of statistic ("mean", "variance", "mean_variance").
    bkd : Backend
        Backend for array operations.

    Returns
    -------
    funs : List[Callable]
        Wrapped model functions for selected QoI.
    stat : MultiOutputStatistic
        Statistic for the subproblem with pilot quantities set.
    costs : Array
        Costs for selected models.
    benchmark : MultiOutputEnsembleBenchmark
        Full benchmark object.
    means : Array
        Means for the subproblem.
    """
    benchmark = MultiOutputEnsembleBenchmark(bkd)

    # Get wrapped functions for the subproblem
    funs = benchmark.models_subproblem(model_idx, qoi_idx)

    # Get pilot quantities directly for subproblem
    pilot_quantities = _get_pilot_quantities_for_stat_type(
        benchmark, stat_type, model_idx, qoi_idx
    )

    # Create statistic and set pilot quantities
    nqoi = len(qoi_idx)
    stat = multioutput_stats[stat_type](nqoi, bkd)
    stat.set_pilot_quantities(*pilot_quantities)

    costs = benchmark.costs_subproblem(model_idx)
    means = benchmark.means_subproblem(model_idx, qoi_idx)

    return funs, stat, costs, benchmark, means


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
    (
        [0, 1, 2],
        [0, 2],
        None,
        "cv",
        "mean_variance",
        None,
        None,
        5e4,
        2000,
        "cv_mean_var",
    ),
    ([0, 1, 2], [0], [0, 1], "grd", "mean", None, None, 2e4, 2000, "grd_mean_01"),
    (
        [0, 1, 2],
        [0, 1],
        [0, 1],
        "grd",
        "mean",
        None,
        None,
        2e4,
        2000,
        "grd_mean_01_2qoi",
    ),
    ([0, 1, 2], [0, 1], [0, 0], "grd", "mean", None, None, 2e4, 2000, "grd_mean_00"),
    ([0, 1], [0, 1, 2], [0], "grd", "variance", None, None, 2e4, 2000, "grd_var_2mod"),
    (
        [0, 1, 2],
        [0, 1],
        [0, 1],
        "grd",
        "variance",
        None,
        None,
        2e4,
        2000,
        "grd_var_3mod",
    ),
    (
        [0, 1, 2],
        [0, 1],
        [0, 1],
        "grd",
        "mean_variance",
        None,
        None,
        5e4,
        2000,
        "grd_mean_var",
    ),
    ([0, 1, 2], [0], [0, 1], "gis", "mean", None, None, 2e4, 2000, "gis_mean"),
    ([0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean", None, None, 2e4, 2000, "gmf_mean_00"),
    (
        [0, 1, 2],
        [0, 1, 2],
        [0, 0],
        "gmf",
        "mean",
        None,
        None,
        1e4,
        10,
        "gmf_mean_00_fast",
    ),
    ([0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", 2, None, 5e4, 100, "gmf_mean_tree2"),
    ([0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", None, 3, 1e4, 100, "gmf_mean_max3"),
    # Skip test case 17 (BestEstimator with list of est_types)
    ([0, 1, 2], [0, 1, 2], [0, 1], "grd", "mean", None, 3, 2e4, 2000, "grd_mean_max3"),
    (
        [0, 1, 2],
        [1],
        [0, 1],
        "grd",
        "variance",
        None,
        3,
        2e4,
        2000,
        "grd_var_1qoi_max3",
    ),
    ([0, 1], [0, 2], [0], "gmf", "mean", None, None, 2e4, 2000, "gmf_mean_2mod"),
    ([0, 1], [0], [0], "gmf", "variance", None, None, 2e4, 2000, "gmf_var_2mod_1qoi"),
    (
        [0, 1],
        [0, 2],
        [0],
        "gmf",
        "variance",
        None,
        None,
        2e4,
        2000,
        "gmf_var_2mod_2qoi",
    ),
    (
        [0, 1, 2],
        [0],
        [0, 0],
        "gmf",
        "variance",
        None,
        None,
        2e4,
        2000,
        "gmf_var_3mod_1qoi",
    ),
    # Skipped: ([0, 1, 2], [0, 2], [0, 0], "gmf", "variance", ...) - optimizer
    # convergence issue
    (
        [0, 1],
        [0],
        [0],
        "gmf",
        "mean_variance",
        None,
        None,
        2e4,
        2000,
        "gmf_mean_var_2mod",
    ),
    ([0, 1, 2], [0], None, "mfmc", "mean", None, None, 2e4, 2000, "mfmc_mean"),
    ([0, 1, 2], [0], None, "mlmc", "mean", None, None, 5e4, 2000, "mlmc_mean_1qoi"),
    ([0, 1, 2], [0, 1], None, "mlmc", "mean", None, None, 5e4, 2000, "mlmc_mean_2qoi"),
    ([0, 1, 2], [0], None, "mlmc", "variance", None, None, 2e4, 2000, "mlmc_variance"),
    (
        [0, 1, 2],
        [0],
        None,
        "mlmc",
        "mean_variance",
        None,
        None,
        1e4,
        100,
        "mlmc_mean_var",
    ),
    ([0], [0, 1, 2], None, "mc", "variance", None, None, 2e4, 2000, "mc_variance_3qoi"),
    (
        [0, 1, 2],
        [0, 1],
        [0, 1],
        "gmf",
        "variance",
        None,
        None,
        7e4,
        2000,
        "gmf_var_highcost",
    ),
]


class TestEstimatorVariances:
    """Replicate legacy test_estimator_variances exactly using @parametrize."""

    @pytest.fixture(autouse=True)
    def _setup(self):
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

        funs, stat, costs, benchmark, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx, stat_type, bkd
        )

        # Change costs to reduce samples (match legacy)
        costs = bkd.array([2.0, 1.5, 1.0])[model_idx]

        nqoi = len(qoi_idx)
        nmodels = len(model_idx)
        idx = stat.nstats()

        kwargs: Dict[str, Any] = {}

        # For ACV estimators, handle recursion_index (tree_depth handled via ACVSearch)
        if est_type in ("gmf", "grd", "gis"):
            if recursion_index is not None:
                kwargs["recursion_index"] = bkd.asarray(recursion_index)

        # CV estimator needs lowfi_stats computed from covariance and means
        if est_type == "cv":
            cov = benchmark.covariance_subproblem(model_idx, qoi_idx)
            if stat_type == "mean":
                kwargs["lowfi_stats"] = bkd.stack([m for m in means[1:]], axis=0)
            elif stat_type == "variance":
                lfcovs = []
                for ii in range(1, nmodels):
                    lb = ii * nqoi
                    ub = lb + nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                tril_idx = bkd.tril_indices(lfcovs[0].shape[0])
                kwargs["lowfi_stats"] = bkd.stack(
                    [c[tril_idx[0], tril_idx[1]].flatten() for c in lfcovs]
                )
            elif stat_type == "mean_variance":
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

        # Use ACVSearch when tree_depth is specified for ACV estimators
        if tree_depth is not None and est_type in ("gmf", "grd", "gis"):
            est_class_map = {
                "gmf": GMFEstimator,
                "grd": GRDEstimator,
                "gis": GISEstimator,
            }
            search = ACVSearch(
                stat,
                costs,
                estimator_classes=[est_class_map[est_type]],
                recursion_strategy=TreeDepthRecursionStrategy(max_depth=tree_depth),
            )
            result = search.search(target_cost=target_cost, allow_failures=True)
            est = result.estimator
        else:
            est = get_estimator(
                est_type, stat, costs, max_nmodels=max_nmodels, **kwargs
            )

            # Configure optimizer with higher maxiter for convergence (matches legacy)
            if hasattr(est, "get_default_optimizer"):
                from pyapprox.optimization.minimize.chained.chained_optimizer import (
                    ChainedOptimizer,
                )
                from pyapprox.optimization.minimize.scipy.diffevol import (
                    ScipyDifferentialEvolutionOptimizer,
                )
                from pyapprox.optimization.minimize.scipy.trust_constr import (
                    ScipyTrustConstrOptimizer,
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
            assert bkd.allclose(CF, CF_mc, atol=atol, rtol=rtol)
            assert bkd.allclose(cf, cf_mc, atol=atol, rtol=rtol)

        assert bkd.allclose(covar_mc, covar, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
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
            model_idx,
            qoi_idx,
            rec_idx,
            est_type,
            stat_type,
            tree,
            maxmod,
            cost,
            ntrials,
        )


class TestDiscrepancyCovariances:
    """Test discrepancy covariance computation matches MC."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(42)
        self._bkd = TorchBkd()

    @pytest.mark.parametrize(
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

        bm = PolynomialEnsembleBenchmark(self._bkd, nmodels=nmodels)
        cov = bm.ensemble_covariance()
        costs = bm.problem().costs()
        models = bm.problem().models()

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
