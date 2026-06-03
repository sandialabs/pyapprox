"""Compare GroupACV allocation optimizers across problem configs.

Sweeps target costs × optimizer/config combinations on the 5-model polynomial
ensemble (Mean statistic). Two sets: local-only (from default init guess) and
DE-initialized (differential evolution global → local refinement).

Produces:
  1. Scatter plot: estimator variance vs optimizer wall time (per budget panel)
  2. Bar chart: cost per iteration (wall time / nit) by optimizer

Usage:
    python groupacv_optimizer_comparison.py
    python groupacv_optimizer_comparison.py --optimizers slsqp,trustconstr
    python groupacv_optimizer_comparison.py --configs none/ineq,full/eq
    python groupacv_optimizer_comparison.py --budgets 50,500
    python groupacv_optimizer_comparison.py --sets local
    python groupacv_optimizer_comparison.py --sets de
    python groupacv_optimizer_comparison.py --no-spd
    python groupacv_optimizer_comparison.py --tol 1e-8 --maxiter 2000
    python groupacv_optimizer_comparison.py --no-plot

All flags accept comma-separated values. Defaults run everything.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pyapprox.optimization.minimize.chained.chained_optimizer import (
    ChainedOptimizer,
)
from pyapprox.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)
from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.statest.groupacv import (
    GroupACVAllocationOptimizer,
    GroupACVEstimatorIS,
    MeanGuidedSubsetFitter,
    MLBLUEEstimator,
    MLBLUESPDAllocationOptimizer,
    get_model_subsets,
)
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.statest.statistics import MultiOutputMean, MultiOutputVariance
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.statest import PolynomialEnsembleBenchmark


# ---------------------------------------------------------------------------
# Benchmark setup
# ---------------------------------------------------------------------------
def _setup_problem(bkd, stat_name: str = "mean", npilot: int = 10000):
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=5)
    costs = benchmark.problem().costs()
    nqoi = 1

    if stat_name == "mean":
        cov = benchmark.ensemble_covariance()
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
    elif stat_name == "variance":
        np.random.seed(0)
        variable = benchmark.problem().prior()
        models = benchmark.problem().models()
        pilot_samples = variable.rvs(npilot)
        pilot_values = [m(pilot_samples) for m in models]
        stat = MultiOutputVariance(nqoi, bkd)
        cov, W = stat.compute_pilot_quantities(pilot_values)
        stat.set_pilot_quantities(cov, W)
    else:
        raise ValueError(f"Unknown stat: {stat_name}")

    return costs, stat


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
    set_name: str  # "local", "de", or "spd"
    optimizer_name: str
    config_name: str
    target_cost: float
    objective_value: float
    estimator_variance: float
    success: bool
    wall_time_s: float
    actual_cost: float
    nit: Optional[int] = None
    nfev: Optional[int] = None
    message: str = ""

    def cost_per_iter(self) -> Optional[float]:
        if self.nit is not None and self.nit > 0:
            return self.wall_time_s / self.nit
        return None

    def label(self) -> str:
        return f"{self.optimizer_name}/{self.config_name}"

    def cache_key(self) -> str:
        return (
            f"{self.set_name}|{self.optimizer_name}|{self.config_name}"
            f"|{self.target_cost}"
        )


# ---------------------------------------------------------------------------
# Iteration capture via monkey-patching
# ---------------------------------------------------------------------------
def _extract_nit_nfev(result_obj) -> Tuple[Optional[int], Optional[int]]:
    """Extract nit/nfev from an optimizer result, if available."""
    if hasattr(result_obj, "get_raw_result"):
        raw = result_obj.get_raw_result()
        return getattr(raw, "nit", None), getattr(raw, "nfev", None)
    return None, None


def _wrap_minimize_for_capture(optimizer, captured: Dict):
    """Monkey-patch optimizer.minimize to capture the result object."""
    orig = optimizer.minimize

    def _wrapped(init_guess):
        r = orig(init_guess)
        captured["result"] = r
        return r

    optimizer.minimize = _wrapped


def _wrap_chained_for_capture(chained_opt, captured: Dict):
    """Capture iteration counts from both DE and local stages."""
    de_orig = chained_opt._global_optimizer.minimize
    local_orig = chained_opt._local_optimizer.minimize

    def _de_wrapped(init_guess):
        r = de_orig(init_guess)
        captured["de_result"] = r
        return r

    def _local_wrapped(init_guess):
        r = local_orig(init_guess)
        captured["local_result"] = r
        return r

    chained_opt._global_optimizer.minimize = _de_wrapped
    chained_opt._local_optimizer.minimize = _local_wrapped


# ---------------------------------------------------------------------------
# Optimizer factories (parameterized by tol/maxiter)
# ---------------------------------------------------------------------------
def _make_optimizer_factories(
    tol: float = 1e-6, maxiter: int = 1000, de_maxiter: int = 100,
) -> tuple:
    """Build optimizer factory functions with given tolerances.

    Returns (local_optimizers_dict, de_global_factory, spd_solver_kwargs).
    """

    def _de_global():
        return ScipyDifferentialEvolutionOptimizer(
            maxiter=de_maxiter, polish=False, seed=1, tol=tol,
            raise_on_failure=False,
        )

    def _local_slsqp():
        return ScipySLSQPOptimizer(maxiter=maxiter, ftol=tol)

    def _local_trust_constr():
        return ScipyTrustConstrOptimizer(maxiter=maxiter, gtol=tol)

    def _local_rol():
        from pyapprox.optimization.minimize.rol.rol_optimizer import (
            ROLOptimizer,
        )
        return ROLOptimizer(verbosity=0, fast_vector=True)

    local_optimizers: Dict[str, Callable] = {
        "slsqp": _local_slsqp,
        "trustconstr": _local_trust_constr,
    }

    try:
        _local_rol()
        local_optimizers["rol"] = _local_rol
    except (ImportError, RuntimeError):
        pass

    spd_solver_kwargs = {
        "tol_gap_abs": tol,
        "tol_gap_rel": tol,
        "tol_feas": tol,
    }

    return local_optimizers, _de_global, spd_solver_kwargs


# Check ROL availability at import time for reporting
try:
    from pyapprox.optimization.minimize.rol.rol_optimizer import ROLOptimizer
    ROLOptimizer(verbosity=0)
    _ROL_AVAILABLE = True
except (ImportError, RuntimeError):
    _ROL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config registry (all 6 combinations)
# ---------------------------------------------------------------------------
ALL_CONFIGS = {
    "none/ineq": AllocationProblemConfig(
        variable_scaling="none",
        budget_constraint_form="inequality",
    ),
    "none/eq": AllocationProblemConfig(
        variable_scaling="none",
        budget_constraint_form="equality",
    ),
    "constraint_only/ineq": AllocationProblemConfig(
        variable_scaling="constraint_only",
        budget_constraint_form="inequality",
    ),
    "constraint_only/eq": AllocationProblemConfig(
        variable_scaling="constraint_only",
        budget_constraint_form="equality",
    ),
    "full/ineq": AllocationProblemConfig(
        variable_scaling="full",
        budget_constraint_form="inequality",
    ),
    "full/eq": AllocationProblemConfig(
        variable_scaling="full",
        budget_constraint_form="equality",
    ),
    "log/ineq": AllocationProblemConfig(
        variable_scaling="log",
        budget_constraint_form="inequality",
    ),
    "log/eq": AllocationProblemConfig(
        variable_scaling="log",
        budget_constraint_form="equality",
    ),
}


# ---------------------------------------------------------------------------
# Estimator variance computation
# ---------------------------------------------------------------------------
def _compute_estimator_variance(stat, costs, bkd, npartition_samples):
    """Compute tr(estimator covariance) at a given allocation."""
    subsets = get_model_subsets(len(costs), bkd)
    est = GroupACVEstimatorIS(stat, costs, model_subsets=subsets)
    nps_float = bkd.asarray(npartition_samples, dtype=bkd.double_dtype())
    est_cov = est._covariance_from_npartition_samples(nps_float)
    return float(bkd.to_numpy(bkd.trace(est_cov)))


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------
def _run_local_trial(
    stat, costs, bkd, optimizer_factory, optimizer_name,
    config, config_name, target_cost,
) -> TrialResult:
    subsets = get_model_subsets(len(costs), bkd)
    est = GroupACVEstimatorIS(stat, costs, model_subsets=subsets)

    optimizer = optimizer_factory()
    captured: Dict = {}
    _wrap_minimize_for_capture(optimizer, captured)

    allocator = GroupACVAllocationOptimizer(
        est, optimizer=optimizer, problem_config=config
    )

    try:
        t0 = time.perf_counter()
        result = allocator.optimize(target_cost, round_nsamples=False)
        elapsed = time.perf_counter() - t0
    except (ValueError, RuntimeError) as e:
        return TrialResult(
            set_name="local",
            optimizer_name=optimizer_name,
            config_name=config_name,
            target_cost=target_cost,
            objective_value=float("inf"),
            estimator_variance=float("inf"),
            success=False,
            wall_time_s=0.0,
            actual_cost=0.0,
            message=str(e)[:80],
        )

    nit, nfev = None, None
    if "result" in captured:
        nit, nfev = _extract_nit_nfev(captured["result"])

    obj_val = float("inf")
    est_var = float("inf")
    if result.success:
        obj_val = float(bkd.to_numpy(result.objective_value)[0])
        est_var = _compute_estimator_variance(
            stat, costs, bkd, result.npartition_samples
        )

    return TrialResult(
        set_name="local",
        optimizer_name=optimizer_name,
        config_name=config_name,
        target_cost=target_cost,
        objective_value=obj_val,
        estimator_variance=est_var,
        success=result.success,
        wall_time_s=elapsed,
        actual_cost=result.actual_cost,
        nit=nit,
        nfev=nfev,
        message=result.message,
    )


def _run_de_trial(
    stat, costs, bkd, local_factory, de_factory, optimizer_name,
    config, config_name, target_cost,
) -> TrialResult:
    subsets = get_model_subsets(len(costs), bkd)
    est = GroupACVEstimatorIS(stat, costs, model_subsets=subsets)

    chained = ChainedOptimizer(de_factory(), local_factory())
    captured: Dict = {}
    _wrap_chained_for_capture(chained, captured)

    allocator = GroupACVAllocationOptimizer(
        est, optimizer=chained, problem_config=config
    )

    try:
        t0 = time.perf_counter()
        result = allocator.optimize(target_cost, round_nsamples=False)
        elapsed = time.perf_counter() - t0
    except (ValueError, RuntimeError) as e:
        return TrialResult(
            set_name="de",
            optimizer_name=optimizer_name,
            config_name=config_name,
            target_cost=target_cost,
            objective_value=float("inf"),
            estimator_variance=float("inf"),
            success=False,
            wall_time_s=0.0,
            actual_cost=0.0,
            message=str(e)[:80],
        )

    # Sum iterations from both stages
    nit_total, nfev_total = 0, 0
    for key in ["de_result", "local_result"]:
        if key in captured:
            nit_k, nfev_k = _extract_nit_nfev(captured[key])
            if nit_k is not None:
                nit_total += nit_k
            if nfev_k is not None:
                nfev_total += nfev_k

    obj_val = float("inf")
    est_var = float("inf")
    if result.success:
        obj_val = float(bkd.to_numpy(result.objective_value)[0])
        est_var = _compute_estimator_variance(
            stat, costs, bkd, result.npartition_samples
        )

    return TrialResult(
        set_name="de",
        optimizer_name=optimizer_name,
        config_name=config_name,
        target_cost=target_cost,
        objective_value=obj_val,
        estimator_variance=est_var,
        success=result.success,
        wall_time_s=elapsed,
        actual_cost=result.actual_cost,
        nit=nit_total if nit_total > 0 else None,
        nfev=nfev_total if nfev_total > 0 else None,
        message=result.message,
    )


def _run_spd_trial(
    stat, costs, bkd, target_cost, solver_kwargs,
) -> TrialResult:
    subsets = get_model_subsets(len(costs), bkd)
    est = MLBLUEEstimator(stat, costs, model_subsets=subsets)
    allocator = MLBLUESPDAllocationOptimizer(est, solver_kwargs=solver_kwargs)

    try:
        t0 = time.perf_counter()
        result = allocator.optimize(target_cost, round_nsamples=False)
        elapsed = time.perf_counter() - t0
    except (ValueError, RuntimeError) as e:
        return TrialResult(
            set_name="spd",
            optimizer_name="spd",
            config_name="convex_sdp",
            target_cost=target_cost,
            objective_value=float("inf"),
            estimator_variance=float("inf"),
            success=False,
            wall_time_s=0.0,
            actual_cost=0.0,
            message=str(e)[:80],
        )

    obj_val = float("inf")
    est_var = float("inf")
    if result.success:
        obj_val = float(bkd.to_numpy(result.objective_value)[0])
        est_var = _compute_estimator_variance(
            stat, costs, bkd, result.npartition_samples
        )

    return TrialResult(
        set_name="spd",
        optimizer_name="spd",
        config_name="convex_sdp",
        target_cost=target_cost,
        objective_value=obj_val,
        estimator_variance=est_var,
        success=result.success,
        wall_time_s=elapsed,
        actual_cost=result.actual_cost,
    )


def _run_mean_guided_trial(
    stat, costs, bkd, optimizer_factory, optimizer_name,
    config, config_name, target_cost,
) -> TrialResult:
    subsets = get_model_subsets(len(costs), bkd)

    fitter = MeanGuidedSubsetFitter(
        stat, costs, GroupACVEstimatorIS,
        candidate_subsets=subsets,
        optimizer=optimizer_factory(),
        problem_config=config,
    )

    try:
        t0 = time.perf_counter()
        result = fitter.fit(target_cost, min_nhf_samples=1)
        elapsed = time.perf_counter() - t0
    except (ValueError, RuntimeError) as e:
        return TrialResult(
            set_name="meanguided",
            optimizer_name=optimizer_name,
            config_name=config_name,
            target_cost=target_cost,
            objective_value=float("inf"),
            estimator_variance=float("inf"),
            success=False,
            wall_time_s=0.0,
            actual_cost=0.0,
            message=str(e)[:80],
        )

    alloc = result.best_allocation
    obj_val = float(bkd.to_numpy(alloc.objective_value)[0])

    # Compute estimator variance from the fitted reduced estimator
    fitted = result.best_estimator
    est_var = float(bkd.to_numpy(bkd.trace(fitted.covariance())))

    npruned = result.partitions_pruned()

    return TrialResult(
        set_name="meanguided",
        optimizer_name=optimizer_name,
        config_name=config_name,
        target_cost=target_cost,
        objective_value=obj_val,
        estimator_variance=est_var,
        success=True,
        wall_time_s=elapsed,
        actual_cost=alloc.actual_cost,
        message=f"pruned {npruned} partitions",
    )


# ---------------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------------
_DEFAULT_CACHE = Path(__file__).with_suffix(".json")


def _load_cache(path: Path) -> Dict[str, TrialResult]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    cache: Dict[str, TrialResult] = {}
    for d in data:
        tr = TrialResult(**d)
        cache[tr.cache_key()] = tr
    return cache


def _save_cache(cache: Dict[str, TrialResult], path: Path) -> None:
    data = [asdict(tr) for tr in cache.values()]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Cache saved: {path} ({len(data)} results)")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(
    target_costs: List[float],
    optimizer_names: List[str],
    config_names: List[str],
    set_names: List[str],
    stat_name: str = "mean",
    tol: float = 1e-6,
    maxiter: int = 1000,
    de_maxiter: int = 100,
    run_spd: bool = True,
    run_mean_guided: bool = False,
    cache_path: Optional[Path] = None,
    clear_cache: bool = False,
) -> List[TrialResult]:
    if cache_path is None:
        cache_path = _DEFAULT_CACHE

    cache = {} if clear_cache else _load_cache(cache_path)
    if cache:
        print(f"  Loaded {len(cache)} cached results from {cache_path}")

    bkd = NumpyBkd()
    costs, stat = _setup_problem(bkd, stat_name=stat_name)

    local_optimizers, de_global_factory, spd_solver_kwargs = (
        _make_optimizer_factories(tol=tol, maxiter=maxiter, de_maxiter=de_maxiter)
    )

    def _run_or_cached(key, runner):
        if key in cache:
            print(f"    [cached] ", end="")
            _print_trial(cache[key])
            return cache[key]
        trial = runner()
        _print_trial(trial)
        cache[trial.cache_key()] = trial
        _save_cache(cache, cache_path)
        return trial

    results: List[TrialResult] = []

    for target_cost in target_costs:
        print(f"\n{'='*80}")
        print(f" target_cost = {target_cost:.0f}")
        print(f"{'='*80}")

        if "local" in set_names:
            print(f"\n  [LOCAL] (from default init guess)")
            for opt_name in optimizer_names:
                if opt_name not in local_optimizers:
                    continue
                for cfg_name in config_names:
                    if cfg_name not in ALL_CONFIGS:
                        continue
                    key = f"local|{opt_name}|{cfg_name}|{target_cost}"
                    trial = _run_or_cached(key, lambda o=opt_name, c=cfg_name: (
                        _run_local_trial(
                            stat, costs, bkd,
                            local_optimizers[o],
                            o, ALL_CONFIGS[c],
                            c, target_cost,
                        )
                    ))
                    results.append(trial)

        if "de" in set_names:
            print(f"\n  [DE+LOCAL] (differential evolution -> local refinement)")
            for opt_name in optimizer_names:
                if opt_name not in local_optimizers:
                    continue
                for cfg_name in config_names:
                    if cfg_name not in ALL_CONFIGS:
                        continue
                    key = f"de|{opt_name}|{cfg_name}|{target_cost}"
                    trial = _run_or_cached(key, lambda o=opt_name, c=cfg_name: (
                        _run_de_trial(
                            stat, costs, bkd,
                            local_optimizers[o],
                            de_global_factory,
                            o, ALL_CONFIGS[c],
                            c, target_cost,
                        )
                    ))
                    results.append(trial)

        if run_spd:
            print(f"\n  [SPD] (convex SDP reference)")
            key = f"spd|spd|convex_sdp|{target_cost}"
            try:
                trial = _run_or_cached(key, lambda: (
                    _run_spd_trial(
                        stat, costs, bkd, target_cost, spd_solver_kwargs
                    )
                ))
                results.append(trial)
            except ImportError as e:
                print(f"    SPD skipped: {e}")

        if run_mean_guided and stat_name == "mean":
            print(f"\n  [MEAN-GUIDED] skipped (no-op for Mean stat; "
                  f"use --stat variance)")

        if run_mean_guided and stat_name != "mean":
            print(f"\n  [MEAN-GUIDED] (mean screening -> reduced target-stat solve)")
            for opt_name in optimizer_names:
                if opt_name not in local_optimizers:
                    continue
                for cfg_name in config_names:
                    if cfg_name not in ALL_CONFIGS:
                        continue
                    key = f"meanguided|{opt_name}|{cfg_name}|{target_cost}"
                    trial = _run_or_cached(key, lambda o=opt_name, c=cfg_name: (
                        _run_mean_guided_trial(
                            stat, costs, bkd,
                            local_optimizers[o],
                            o, ALL_CONFIGS[c],
                            c, target_cost,
                        )
                    ))
                    results.append(trial)

    return results


def _print_trial(trial: TrialResult):
    status = "OK" if trial.success else "FAIL"
    obj_str = f"{trial.objective_value:.6e}" if trial.success else "FAILED"
    var_str = f"{trial.estimator_variance:.6e}" if trial.success else ""
    nit_str = f"{trial.nit}" if trial.nit is not None else "n/a"
    cpi = trial.cost_per_iter()
    cpi_str = f"{cpi:.4f}" if cpi is not None else "n/a"
    print(
        f"    {trial.optimizer_name:<12} {trial.config_name:<20} "
        f"{obj_str:>13}  var={var_str:<13} "
        f"{trial.wall_time_s:>7.3f}s  nit={nit_str:<5} "
        f"cpi={cpi_str:<8} {status}"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _mc_variance(stat, costs, bkd, target_cost: float) -> float:
    """Single-fidelity MC estimator variance: tr(Cov(stat_hat)) at n=budget/c0."""
    nhf = target_cost / float(bkd.to_numpy(costs[0]))
    nhf_arr = bkd.array([nhf])
    hf_cov = stat.high_fidelity_estimator_covariance(nhf_arr)
    return float(bkd.to_numpy(bkd.trace(hf_cov)))


def plot_results(results: List[TrialResult], stat=None, costs=None, bkd=None):
    import matplotlib.pyplot as plt

    successful = [r for r in results if r.success]
    if not successful:
        print("No successful results to plot.")
        return

    target_costs = sorted(set(r.target_cost for r in successful))
    set_names = sorted(set(r.set_name for r in successful))

    # Marker by optimizer, color by config
    optimizer_markers = {
        "slsqp": "o", "trustconstr": "s", "rol": "D", "spd": "*",
    }
    config_colors = {
        "none/ineq": "C0", "none/eq": "C1",
        "constraint_only/ineq": "C2", "constraint_only/eq": "C3",
        "full/ineq": "C4", "full/eq": "C5",
        "log/ineq": "C6", "log/eq": "C8",
        "convex_sdp": "C7",
    }
    set_linestyles = {"local": "none", "de": "none", "spd": "none", "meanguided": "none"}
    set_edge = {"local": "black", "de": "red", "spd": "green", "meanguided": "orange"}

    # --- Figure 1: Variance vs wall time (one panel per budget) ---
    ncols = len(target_costs)
    fig1, axes1 = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    fig1.suptitle("Estimator Variance vs Optimizer Wall Time", fontsize=14)

    for col, tc in enumerate(target_costs):
        ax = axes1[0, col]
        ax.set_title(f"budget = {tc:.0f}")
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("Estimator variance")
        ax.set_yscale("log")
        ax.set_xscale("log")

        for r in successful:
            if r.target_cost != tc:
                continue
            marker = optimizer_markers.get(r.optimizer_name, "^")
            color = config_colors.get(r.config_name, "gray")
            edge = set_edge.get(r.set_name, "black")
            size = 120 if r.set_name == "spd" else 60
            ax.scatter(
                r.wall_time_s, r.estimator_variance,
                marker=marker, c=color, edgecolors=edge,
                s=size, linewidths=1.5, zorder=3,
            )

        # MC reference line
        if stat is not None and costs is not None and bkd is not None:
            mc_var = _mc_variance(stat, costs, bkd, tc)
            ax.axhline(mc_var, color="black", linestyle=":", linewidth=1,
                        alpha=0.6, label="MC (single-fidelity)")

        ax.grid(True, alpha=0.3)

    # Build legend from all unique optimizer/config/set combos
    from matplotlib.lines import Line2D
    legend_elements = []
    for opt_name, marker in optimizer_markers.items():
        if any(r.optimizer_name == opt_name for r in successful):
            legend_elements.append(
                Line2D([0], [0], marker=marker, color="gray",
                       linestyle="none", markersize=8,
                       label=f"opt: {opt_name}")
            )
    for cfg_name, color in config_colors.items():
        if any(r.config_name == cfg_name for r in successful):
            legend_elements.append(
                Line2D([0], [0], marker="o", color=color,
                       linestyle="none", markersize=8,
                       label=f"cfg: {cfg_name}")
            )
    for sn, edge in set_edge.items():
        if any(r.set_name == sn for r in successful):
            legend_elements.append(
                Line2D([0], [0], marker="o", color="lightgray",
                       markeredgecolor=edge, linestyle="none",
                       markersize=8, markeredgewidth=2,
                       label=f"set: {sn}")
            )
    if stat is not None:
        legend_elements.append(
            Line2D([0], [0], color="black", linestyle=":",
                   linewidth=1, label="MC (single-fidelity)")
        )
    fig1.legend(
        handles=legend_elements, loc="lower center",
        ncol=min(len(legend_elements), 6),
        bbox_to_anchor=(0.5, -0.02), fontsize=8,
    )
    fig1.tight_layout(rect=[0, 0.08, 1, 0.95])

    # --- Figure 2: Cost per iteration bar chart ---
    with_cpi = [r for r in successful if r.cost_per_iter() is not None]
    if with_cpi:
        labels = [f"{r.set_name}/{r.optimizer_name}\n{r.config_name}" for r in with_cpi]
        cpis = [r.cost_per_iter() for r in with_cpi]
        nits = [r.nit for r in with_cpi]
        budgets = [r.target_cost for r in with_cpi]

        # Group by budget
        fig2, axes2 = plt.subplots(
            2, ncols, figsize=(6 * ncols, 8), squeeze=False,
        )
        fig2.suptitle("Iteration Count and Cost per Iteration", fontsize=14)

        for col, tc in enumerate(target_costs):
            tc_trials = [r for r in with_cpi if r.target_cost == tc]
            if not tc_trials:
                continue

            tc_labels = [
                f"{r.set_name[:2]}/{r.optimizer_name[:4]}\n{r.config_name}"
                for r in tc_trials
            ]
            tc_nits = [r.nit for r in tc_trials]
            tc_cpis = [r.cost_per_iter() for r in tc_trials]
            tc_colors = [config_colors.get(r.config_name, "gray") for r in tc_trials]
            x = np.arange(len(tc_trials))

            # Top: number of iterations
            ax_top = axes2[0, col]
            ax_top.bar(x, tc_nits, color=tc_colors, edgecolor="black", linewidth=0.5)
            ax_top.set_title(f"budget = {tc:.0f}")
            ax_top.set_ylabel("Number of iterations")
            ax_top.set_xticks(x)
            ax_top.set_xticklabels(tc_labels, fontsize=6, rotation=45, ha="right")

            # Bottom: cost per iteration
            ax_bot = axes2[1, col]
            ax_bot.bar(x, tc_cpis, color=tc_colors, edgecolor="black", linewidth=0.5)
            ax_bot.set_ylabel("Cost per iter (s)")
            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels(tc_labels, fontsize=6, rotation=45, ha="right")

        fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Figure 3: Variance vs budget (decay rate check) ---
    if len(target_costs) > 1:
        from collections import defaultdict
        set_linestyle = {"local": "-", "de": "--", "spd": ":", "meanguided": "-."}
        set_marker_edge = {"local": None, "de": "white", "spd": None, "meanguided": "orange"}

        groups: Dict[str, List[TrialResult]] = defaultdict(list)
        for r in successful:
            key = f"{r.set_name}/{r.label()}"
            groups[key].append(r)

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.set_title("Estimator Variance vs Budget")
        ax3.set_xlabel("Target cost (budget)")
        ax3.set_ylabel("Estimator variance")
        ax3.set_xscale("log")
        ax3.set_yscale("log")

        for key, trials in sorted(groups.items()):
            trials.sort(key=lambda r: r.target_cost)
            tcs = [r.target_cost for r in trials]
            vs = [r.estimator_variance for r in trials]
            cfg = trials[0].config_name
            sn = trials[0].set_name
            color = config_colors.get(cfg, "gray")
            marker = optimizer_markers.get(trials[0].optimizer_name, "^")
            ls = set_linestyle.get(sn, "-")
            mec = set_marker_edge.get(sn)
            ax3.plot(tcs, vs, marker=marker, color=color, linestyle=ls,
                     markeredgecolor=mec, label=key,
                     linewidth=1.5, markersize=6)

        # MC (single-fidelity) reference line
        tc_arr = np.array(sorted(target_costs))
        if stat is not None and costs is not None and bkd is not None:
            mc_vars = [_mc_variance(stat, costs, bkd, tc) for tc in tc_arr]
            ax3.plot(tc_arr, mc_vars, "k:", linewidth=2, alpha=0.7,
                     label="MC (single-fidelity)")

        ax3.legend(fontsize=7, loc="best")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def print_summary(results: List[TrialResult]):
    print("\n\n")
    print("=" * 130)
    print("FULL RESULTS")
    print("=" * 130)
    print(
        f"{'Set':<6} {'Optimizer':<12} {'Config':<20} {'Budget':<8} "
        f"{'Objective':<14} {'EstVar':<14} {'Time(s)':<9} "
        f"{'nit':<6} {'cpi(s)':<9} {'OK?':<5} {'ActCost':<10}"
    )
    print("-" * 130)

    target_costs = sorted(set(r.target_cost for r in results))
    for tc in target_costs:
        tc_results = [r for r in results if r.target_cost == tc]
        tc_results.sort(key=lambda r: (r.estimator_variance, r.wall_time_s))
        for r in tc_results:
            obj_str = f"{r.objective_value:.6e}" if r.success else "FAILED"
            var_str = f"{r.estimator_variance:.6e}" if r.success else "FAILED"
            nit_str = f"{r.nit}" if r.nit is not None else "n/a"
            cpi = r.cost_per_iter()
            cpi_str = f"{cpi:.5f}" if cpi is not None else "n/a"
            print(
                f"{r.set_name:<6} {r.optimizer_name:<12} {r.config_name:<20} "
                f"{r.target_cost:<8.0f} {obj_str:<14} {var_str:<14} "
                f"{r.wall_time_s:<9.4f} {nit_str:<6} {cpi_str:<9} "
                f"{'Y' if r.success else 'N':<5} {r.actual_cost:<10.2f}"
            )
        print()

    # Best per target cost per set
    print("=" * 130)
    print("BEST per (target_cost, set) — ranked by estimator variance:")
    print("-" * 130)
    for tc in target_costs:
        for sn in ["local", "de", "spd"]:
            subset = [
                r for r in results
                if r.target_cost == tc and r.set_name == sn and r.success
            ]
            if not subset:
                continue
            best = min(subset, key=lambda r: r.estimator_variance)
            fastest = min(subset, key=lambda r: r.wall_time_s)
            print(
                f"  budget={tc:>7.0f} [{sn:<5}] "
                f"best: {best.optimizer_name}/{best.config_name} "
                f"var={best.estimator_variance:.6e} ({best.wall_time_s:.3f}s)"
            )
            if fastest is not best:
                print(
                    f"  {'':>15}{'':>7} "
                    f"fast: {fastest.optimizer_name}/{fastest.config_name} "
                    f"var={fastest.estimator_variance:.6e} "
                    f"({fastest.wall_time_s:.3f}s)"
                )

    # Failures
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\n{'='*130}")
        print(f"FAILURES ({len(failures)}):")
        print("-" * 130)
        for r in failures:
            print(
                f"  [{r.set_name}] {r.optimizer_name:<12} {r.config_name:<20} "
                f"budget={r.target_cost:<8.0f} {r.message}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GroupACV allocation optimizers."
    )
    parser.add_argument(
        "--stat", type=str, default="mean",
        help="Statistic type: mean or variance (default: mean)",
    )
    parser.add_argument(
        "--optimizers", type=str, default=None,
        help="Comma-separated optimizer names: slsqp,trustconstr,rol",
    )
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated config names: none/ineq,full/eq,...",
    )
    parser.add_argument(
        "--budgets", type=str, default=None,
        help="Comma-separated target costs: 50,500,5000",
    )
    parser.add_argument(
        "--sets", type=str, default=None,
        help="Comma-separated set names: local,de (default: both)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6,
        help="Optimizer tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--maxiter", type=int, default=1000,
        help="Max iterations for local optimizers (default: 1000)",
    )
    parser.add_argument(
        "--de-maxiter", type=int, default=100,
        help="Max iterations for DE global optimizer (default: 100)",
    )
    parser.add_argument(
        "--no-spd", action="store_true",
        help="Skip the SPD (CLARABEL) convex reference",
    )
    parser.add_argument(
        "--mean-guided", action="store_true",
        help="Run MeanGuidedSubsetFitter (two-stage: mean screening + target solve)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting (text output only)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Discard cached results and re-run everything",
    )
    parser.add_argument(
        "--cache-file", type=str, default=None,
        help=f"Path to cache file (default: {_DEFAULT_CACHE})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_optimizer_names = ["slsqp", "trustconstr"]
    if _ROL_AVAILABLE:
        all_optimizer_names.append("rol")

    optimizer_names = all_optimizer_names
    config_names = list(ALL_CONFIGS.keys())
    target_costs = [50.0, 500.0, 5000.0]
    set_names = ["local", "de"]

    if args.optimizers:
        optimizer_names = [s.strip() for s in args.optimizers.split(",")]
    if args.configs:
        config_names = [s.strip() for s in args.configs.split(",")]
    if args.budgets:
        target_costs = [float(s.strip()) for s in args.budgets.split(",")]
    if args.sets:
        set_names = [s.strip() for s in args.sets.split(",")]

    cache_path = Path(args.cache_file) if args.cache_file else _DEFAULT_CACHE

    stat_name = args.stat

    print("Configuration:")
    print(f"  Stat:        {stat_name}")
    print(f"  Optimizers:  {optimizer_names}")
    print(f"  Configs:     {config_names}")
    print(f"  Budgets:     {target_costs}")
    print(f"  Sets:        {set_names}")
    print(f"  tol:         {args.tol}")
    print(f"  maxiter:     {args.maxiter}")
    print(f"  de_maxiter:  {args.de_maxiter}")
    print(f"  SPD:         {not args.no_spd}")
    print(f"  MeanGuided:  {args.mean_guided}")
    print(f"  Plot:        {not args.no_plot}")
    print(f"  Cache:       {cache_path}")
    if _ROL_AVAILABLE:
        print(f"  ROL:         available")
    else:
        print(f"  ROL:         not installed (skipped)")

    bkd = NumpyBkd()
    costs, stat = _setup_problem(bkd, stat_name=stat_name)

    results = run_sweep(
        target_costs=target_costs,
        optimizer_names=optimizer_names,
        config_names=config_names,
        set_names=set_names,
        stat_name=stat_name,
        tol=args.tol,
        maxiter=args.maxiter,
        de_maxiter=args.de_maxiter,
        run_spd=not args.no_spd,
        run_mean_guided=args.mean_guided,
        cache_path=cache_path,
        clear_cache=args.clear_cache,
    )
    print_summary(results)

    if not args.no_plot:
        plot_results(results, stat=stat, costs=costs, bkd=bkd)


if __name__ == "__main__":
    main()
