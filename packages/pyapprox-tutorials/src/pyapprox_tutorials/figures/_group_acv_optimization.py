"""Plotting functions for the group ACV optimization tutorial.

Covers: group_acv_optimization.qmd

Demonstrates:
  - Why log-space variable scaling outperforms none/constraint_only/full
    across budgets, with full-scaling exhibiting non-monotonic
    convergence (more budget -> worse local minimum).
  - SPD reference for the convex MLBLUE-mean special case.
  - Variance estimation failure under direct SLSQP at low budgets
    where the n>=2 dead region forces infeasibility.
  - Mean-guided two-stage rescue: cheap Mean screening identifies a
    sparse active subset, then the reduced variance solve succeeds.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.statest.groupacv import (
    GroupACVAllocationOptimizer,
    GroupACVEstimatorIS,
    GroupACVLogDetObjective,
    GroupACVTraceObjective,
    MeanGuidedSubsetFitter,
    MLBLUEEstimator,
    MLBLUESPDAllocationOptimizer,
    get_model_subsets,
)
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
from pyapprox.statest.statistics import MultiOutputMean, MultiOutputVariance
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.statest import (
    MultiOutputEnsembleBenchmark,
    PolynomialEnsembleBenchmark,
)


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
_SCALING_CONFIGS = {
    "log/ineq": AllocationProblemConfig(
        variable_scaling="log", budget_constraint_form="inequality",
    ),
    "none/ineq": AllocationProblemConfig(
        variable_scaling="none", budget_constraint_form="inequality",
    ),
    "constraint_only/ineq": AllocationProblemConfig(
        variable_scaling="constraint_only",
        budget_constraint_form="inequality",
    ),
    "full/ineq": AllocationProblemConfig(
        variable_scaling="full", budget_constraint_form="inequality",
    ),
}

_SCALING_COLORS = {
    "log/ineq": "#2C7FB8",
    "none/ineq": "#27AE60",
    "constraint_only/ineq": "#8E44AD",
    "full/ineq": "#C0392B",
}

_SCALING_STYLES = {
    "log/ineq": "-",
    "none/ineq": "--",
    "constraint_only/ineq": "-.",
    "full/ineq": ":",
}

_SCALING_MARKERS = {
    "log/ineq": "o",
    "none/ineq": "s",
    "constraint_only/ineq": "^",
    "full/ineq": "D",
}


def _slsqp(maxiter: int = 1000, ftol: float = 1e-6) -> ScipySLSQPOptimizer:
    return ScipySLSQPOptimizer(maxiter=maxiter, ftol=ftol)


def _setup_mean_problem(nmodels: int = 5) -> Tuple:
    """Return (bkd, costs, stat, subsets) for the mean-estimation benchmark."""
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=nmodels)
    costs = benchmark.problem().costs()
    cov = benchmark.ensemble_covariance()
    nqoi = benchmark.problem().models()[0].nqoi()
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    subsets = get_model_subsets(nmodels, bkd)
    return bkd, costs, stat, subsets


def _setup_variance_problem(
    nmodels: int = 5,
) -> Tuple:
    """Return (bkd, costs, stat, subsets) for variance estimation.

    Variance pilot quantities (cov, W) are computed via exact Gauss
    quadrature on the known polynomial model functions.
    """
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=nmodels)
    costs = benchmark.problem().costs()
    models = benchmark.problem().models()
    nqoi = models[0].nqoi()

    cov = benchmark.covariance_matrix()
    W = benchmark.covariance_of_centered_values_kronecker_product()
    stat = MultiOutputVariance(nqoi, bkd)
    stat.set_pilot_quantities(cov, W)
    subsets = get_model_subsets(nmodels, bkd)
    return bkd, costs, stat, subsets


def _run_groupacv_slsqp(
    stat, costs, subsets, config: AllocationProblemConfig,
    target_cost: float,
) -> Optional[float]:
    """Run SLSQP on GroupACVEstimatorIS with given config; return est variance.

    Returns None on optimization failure.
    """
    template = GroupACVEstimatorIS(stat, costs, model_subsets=subsets)
    allocator = GroupACVAllocationOptimizer(
        template, optimizer=_slsqp(), problem_config=config,
    )
    try:
        result = allocator.optimize(target_cost, round_nsamples=False)
    except (ValueError, RuntimeError):
        return None
    if not result.success:
        return None
    bkd = stat.bkd()
    nps = bkd.asarray(result.npartition_samples, dtype=bkd.double_dtype())
    cov = template._covariance_from_npartition_samples(nps)
    return float(bkd.to_numpy(cov[0, 0]))


def _run_mlblue_spd(
    stat, costs, subsets, target_cost: float,
) -> Optional[float]:
    """Run the MLBLUE SPD allocator; return est variance, or None on failure."""
    est = MLBLUEEstimator(stat, costs, model_subsets=subsets)
    allocator = MLBLUESPDAllocationOptimizer(est)
    try:
        result = allocator.optimize(target_cost, round_nsamples=False)
    except (ValueError, RuntimeError, ImportError):
        return None
    if not result.success:
        return None
    bkd = stat.bkd()
    nps = bkd.asarray(result.npartition_samples, dtype=bkd.double_dtype())
    cov = est._covariance_from_npartition_samples(nps)
    return float(bkd.to_numpy(cov[0, 0]))


def _run_mean_guided(
    stat, costs, subsets, target_cost: float,
    config: Optional[AllocationProblemConfig] = None,
) -> Optional[Tuple[float, int]]:
    """Run MeanGuidedSubsetFitter; return (est_variance, partitions_pruned).

    Returns None on failure.
    """
    if config is None:
        config = _SCALING_CONFIGS["log/ineq"]
    fitter = MeanGuidedSubsetFitter(
        stat, costs, GroupACVEstimatorIS,
        candidate_subsets=subsets,
        optimizer=_slsqp(),
        problem_config=config,
    )
    try:
        guided = fitter.fit(target_cost, min_nhf_samples=1)
    except (ValueError, RuntimeError):
        return None
    bkd = stat.bkd()
    est_var = float(bkd.to_numpy(guided.best_estimator.covariance()[0, 0]))
    return est_var, guided.partitions_pruned()


def _mc_variance(stat, costs, target_cost: float) -> float:
    """Single-fidelity MC variance at the same budget."""
    bkd = stat.bkd()
    nhf = target_cost / float(bkd.to_numpy(costs[0]))
    hf_cov = stat.high_fidelity_estimator_covariance(bkd.array([nhf]))
    return float(bkd.to_numpy(hf_cov[0, 0]))


# ---------------------------------------------------------------------------
# group_acv_optimization.qmd -> fig-scaling-comparison
# ---------------------------------------------------------------------------
def plot_scaling_comparison(ax, budgets=(50.0, 100.0, 500.0, 1000.0, 5000.0)):
    """Compare four variable-scaling configurations across budgets.

    All four configs use SLSQP with inequality budget constraint on
    GroupACVEstimatorIS for mean estimation. Shows that log scaling is
    consistently best; full scaling exhibits non-monotonic convergence
    (more budget can produce a worse local minimum); none and
    constraint_only widen their gap to log as budget grows.

    Parameters
    ----------
    ax : matplotlib Axes
    budgets : iterable of float
        Target costs to sweep.
    """
    bkd, costs, stat, subsets = _setup_mean_problem()

    budgets = list(budgets)
    config_results = {}
    for name, config in _SCALING_CONFIGS.items():
        variances = []
        for tc in budgets:
            v = _run_groupacv_slsqp(stat, costs, subsets, config, tc)
            variances.append(v if v is not None else np.nan)
        config_results[name] = variances

    for name in ["log/ineq", "none/ineq", "constraint_only/ineq", "full/ineq"]:
        vs = config_results[name]
        ax.loglog(
            budgets, vs,
            color=_SCALING_COLORS[name],
            linestyle=_SCALING_STYLES[name],
            marker=_SCALING_MARKERS[name],
            lw=1.8, ms=7,
            label=name,
        )

    mc_vars = [_mc_variance(stat, costs, tc) for tc in budgets]
    ax.loglog(
        budgets, mc_vars, color="#666666", linestyle=(0, (1, 1)),
        lw=1.3, label="MC (single fidelity)",
    )

    ax.set_xlabel("Target cost $P$", fontsize=11)
    ax.set_ylabel(r"$\mathbb{V}[\hat{\mu}_0]$", fontsize=11)
    ax.set_title(
        "SLSQP estimator variance vs budget by scaling configuration",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# group_acv_optimization.qmd -> fig-spd-vs-slsqp
# ---------------------------------------------------------------------------
def plot_spd_vs_slsqp(ax, budgets=(50.0, 100.0, 500.0, 1000.0, 5000.0)):
    """Compare MLBLUE SPD against SLSQP+log/ineq across budgets.

    The MLBLUE-mean allocation problem is convex, so in theory the SDP
    (interior-point) solver returns the global optimum. In practice the
    Schur-complement encoding suffers a conditioning gap between the
    Psi block (which grows with budget) and the scalar variance variable
    t. Interior-point solvers (CLARABEL, SCS, MOSEK) drive the duality
    gap to zero on the equilibrated internal problem but cannot enforce
    the PSD constraint to the accuracy required when the block matrix's
    condition number exceeds ~1e8.

    The figure shows SPD failing increasingly badly as the budget grows:
    SLSQP+log/ineq decreases monotonically with budget while SPD stalls.
    By P=5000 the gap is order 50x.

    Parameters
    ----------
    ax : matplotlib Axes
    budgets : iterable of float
        Target costs to sweep.
    """
    bkd, costs, stat, subsets = _setup_mean_problem()

    budgets = list(budgets)
    spd_vars, slsqp_vars = [], []
    for tc in budgets:
        spd_vars.append(_run_mlblue_spd(stat, costs, subsets, tc) or np.nan)
        slsqp_vars.append(
            _run_groupacv_slsqp(
                stat, costs, subsets, _SCALING_CONFIGS["log/ineq"], tc
            ) or np.nan
        )

    spd_arr = np.array(spd_vars, dtype=float)
    slsqp_arr = np.array(slsqp_vars, dtype=float)

    ax.loglog(
        budgets, slsqp_arr, color="#2C7FB8", linestyle="-",
        marker="o", ms=7, lw=2.0,
        label="SLSQP + log/ineq (gradient solver)",
    )
    ax.loglog(
        budgets, spd_arr, color="#C0392B", linestyle="--",
        marker="*", ms=12, lw=1.8,
        label="SPD (cvxpy / CLARABEL)",
    )

    # Shade the gap between the two lines to make the suboptimality visible
    valid = np.isfinite(spd_arr) & np.isfinite(slsqp_arr)
    if np.any(valid):
        ax.fill_between(
            np.array(budgets, dtype=float)[valid],
            slsqp_arr[valid],
            spd_arr[valid],
            color="#C0392B",
            alpha=0.12,
            label="SPD suboptimality",
        )

    # Annotate the ratio at each budget
    for tc, vs, vsp in zip(budgets, slsqp_arr, spd_arr):
        if not (np.isfinite(vs) and np.isfinite(vsp)):
            continue
        ratio = vsp / vs
        if ratio < 1.2:
            continue  # don't clutter when they're close
        # place annotation midway between the lines on log scale
        y_mid = float(np.exp(0.5 * (np.log(vs) + np.log(vsp))))
        ax.annotate(
            rf"${ratio:.1f}\times$",
            xy=(tc, y_mid),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center",
            fontsize=9, color="#7B241C",
        )

    ax.set_xlabel("Target cost $P$", fontsize=11)
    ax.set_ylabel(r"$\mathbb{V}[\hat{\mu}_0]$", fontsize=11)
    ax.set_title(
        "MLBLUE mean: SPD allocator vs SLSQP+log/ineq",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# group_acv_optimization.qmd -> fig-variance-rescue
# ---------------------------------------------------------------------------
def plot_variance_rescue(
    axes, budgets=(10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0),
):
    """Two-panel demo of variance estimation under tight budgets.

    Panel (a): estimator variance vs budget for direct SLSQP+log/ineq vs
    MeanGuidedSubsetFitter. At small budgets, direct SLSQP either fails
    or finds a poor local minimum because the n>=2 dead region forces
    every partition to >=2 samples (infeasible / wasteful). The
    mean-guided fitter screens with a cheap Mean stat (dead threshold
    zero), identifies a sparse active subset, then solves the variance
    problem on the reduced estimator.

    Panel (b): number of active partitions retained by the fitter at
    each budget. Tight budgets prune aggressively; ample budgets keep
    most partitions.

    Parameters
    ----------
    axes : sequence of 2 matplotlib Axes
    budgets : iterable of float
        Target costs to sweep.
    """
    bkd, costs, stat, subsets = _setup_variance_problem()

    budgets = list(budgets)
    direct_vars: List[Optional[float]] = []
    guided_vars: List[Optional[float]] = []
    pruned_counts: List[Optional[int]] = []
    nsubsets_total = len(subsets)

    for tc in budgets:
        direct_vars.append(
            _run_groupacv_slsqp(
                stat, costs, subsets,
                _SCALING_CONFIGS["log/ineq"], tc,
            )
        )
        mg = _run_mean_guided(stat, costs, subsets, tc)
        if mg is None:
            guided_vars.append(None)
            pruned_counts.append(None)
        else:
            v, npruned = mg
            guided_vars.append(v)
            pruned_counts.append(npruned)

    # Panel (a): estimator variance vs budget
    ax_var = axes[0]
    direct_plot = [v if v is not None else np.nan for v in direct_vars]
    guided_plot = [v if v is not None else np.nan for v in guided_vars]

    ax_var.loglog(
        budgets, direct_plot, color="#C0392B", linestyle=":",
        marker="x", ms=9, mew=2.0, lw=1.6,
        label="Direct SLSQP + log/ineq",
    )
    ax_var.loglog(
        budgets, guided_plot, color="#27AE60", linestyle="-",
        marker="o", ms=7, lw=1.8,
        label="Mean-guided fitter",
    )

    # Mark direct failures on the x-axis
    failure_budgets = [
        budgets[i] for i, v in enumerate(direct_vars) if v is None
    ]
    if failure_budgets:
        ymin, ymax = ax_var.get_ylim()
        ax_var.scatter(
            failure_budgets,
            [ymin * 1.5] * len(failure_budgets),
            marker="X", s=140, color="#C0392B",
            edgecolor="black", lw=0.8,
            zorder=5, label="Direct SLSQP failure",
        )

    ax_var.set_xlabel("Target cost $P$", fontsize=11)
    ax_var.set_ylabel(r"$\mathbb{V}[\hat{\sigma}_0^2]$", fontsize=11)
    ax_var.set_title(
        "(a) Variance estimator: direct vs mean-guided", fontsize=11,
    )
    ax_var.legend(fontsize=9, loc="lower left")
    ax_var.grid(True, alpha=0.25, which="both")
    ax_var.set_axisbelow(True)

    # Panel (b): active partitions after pruning
    ax_act = axes[1]
    active_counts = [
        (nsubsets_total - p) if p is not None else 0
        for p in pruned_counts
    ]
    bar_x = np.arange(len(budgets))
    bars = ax_act.bar(
        bar_x, active_counts, color="#27AE60",
        edgecolor="black", lw=0.6, alpha=0.85,
    )
    ax_act.axhline(
        nsubsets_total, color="#666666", linestyle=":", lw=1.2,
        label=rf"Candidate subsets ($2^M-1 = {nsubsets_total}$)",
    )
    for b, count in zip(bars, active_counts):
        ax_act.text(
            b.get_x() + b.get_width() / 2, count + 0.5, str(count),
            ha="center", va="bottom", fontsize=9,
        )

    ax_act.set_xticks(bar_x)
    ax_act.set_xticklabels([f"{int(tc)}" for tc in budgets])
    ax_act.set_xlabel("Target cost $P$", fontsize=11)
    ax_act.set_ylabel("Active partitions", fontsize=11)
    ax_act.set_title(
        "(b) Mean-guided pruning by budget", fontsize=11,
    )
    ax_act.set_ylim(0, nsubsets_total + 4)
    ax_act.legend(fontsize=9, loc="upper left")
    ax_act.grid(True, alpha=0.2, axis="y")
    ax_act.set_axisbelow(True)


# ---------------------------------------------------------------------------
# group_acv_optimization.qmd -> fig-trace-vs-logdet
# ---------------------------------------------------------------------------
def plot_trace_vs_logdet(axes, target_cost: float = 100.0):
    """Compare trace-optimal vs log-det-optimal allocations on a multi-QoI problem.

    Uses MultiOutputEnsembleBenchmark (3 models, 3 QoIs). For each
    objective, computes the resulting estimator covariance and reports
    both metrics. Each objective wins on its own metric — the cross
    evaluation quantifies the suboptimality of the wrong choice.

    Parameters
    ----------
    axes : sequence of 2 matplotlib Axes
    target_cost : float
        Target budget used for both optimizations.
    """
    bkd = NumpyBkd()
    bench = MultiOutputEnsembleBenchmark(bkd)
    costs = bench.problem().costs()
    cov = bench.ensemble_covariance()
    nqoi = bench.problem().models()[0].nqoi()
    nmodels = bench.problem().nmodels()

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    subsets = get_model_subsets(nmodels, bkd)
    config = _SCALING_CONFIGS["log/ineq"]

    def _solve(objective_class):
        template = GroupACVEstimatorIS(
            stat, costs, model_subsets=subsets,
        )
        allocator = GroupACVAllocationOptimizer(
            template,
            optimizer=_slsqp(maxiter=2000, ftol=1e-8),
            problem_config=config,
            objective=objective_class(),
        )
        result = allocator.optimize(target_cost, round_nsamples=False)
        if not result.success:
            return None
        nps = bkd.asarray(
            result.npartition_samples, dtype=bkd.double_dtype()
        )
        est_cov = template._covariance_from_npartition_samples(nps)
        return np.asarray(bkd.to_numpy(est_cov))

    cov_T = _solve(GroupACVTraceObjective)
    cov_L = _solve(GroupACVLogDetObjective)
    if cov_T is None or cov_L is None:
        for ax in axes:
            ax.text(0.5, 0.5, "optimization failed",
                    ha="center", va="center", transform=ax.transAxes)
        return

    trace_T = float(np.trace(cov_T))
    trace_L = float(np.trace(cov_L))
    _, ld_T = np.linalg.slogdet(cov_T)
    _, ld_L = np.linalg.slogdet(cov_L)
    logdet_T = float(ld_T)
    logdet_L = float(ld_L)

    labels = ["Trace-opt", "LogDet-opt"]
    colors = ["#27AE60", "#2C7FB8"]

    # Panel (a): trace metric — trace-opt should win
    ax_t = axes[0]
    vals_t = [trace_T, trace_L]
    winner_t = int(np.argmin(vals_t))
    edgecolors_t = ["black"] * 2
    linewidths_t = [0.8] * 2
    edgecolors_t[winner_t] = "#222"
    linewidths_t[winner_t] = 2.4
    bars_t = ax_t.bar(
        labels, vals_t, color=colors,
        edgecolor=edgecolors_t, linewidth=linewidths_t,
    )
    for b, v in zip(bars_t, vals_t):
        ax_t.text(
            b.get_x() + b.get_width() / 2, v, f"{v:.3e}",
            ha="center", va="bottom", fontsize=9,
        )
    ratio_t = vals_t[1 - winner_t] / vals_t[winner_t]
    ax_t.set_title(
        rf"(a) $\mathrm{{tr}}(\mathrm{{Cov}})$ — wrong choice {ratio_t:.2f}$\times$ worse",
        fontsize=11,
    )
    ax_t.set_ylabel(r"$\mathrm{tr}(\mathrm{Cov})$", fontsize=11)
    ax_t.set_yscale("log")
    ax_t.grid(True, alpha=0.2, axis="y", which="both")
    ax_t.set_axisbelow(True)

    # Panel (b): logdet metric — logdet-opt should win
    ax_d = axes[1]
    vals_d = [logdet_T, logdet_L]
    winner_d = int(np.argmin(vals_d))
    edgecolors_d = ["black"] * 2
    linewidths_d = [0.8] * 2
    edgecolors_d[winner_d] = "#222"
    linewidths_d[winner_d] = 2.4
    bars_d = ax_d.bar(
        labels, vals_d, color=colors,
        edgecolor=edgecolors_d, linewidth=linewidths_d,
    )
    for b, v in zip(bars_d, vals_d):
        va = "bottom" if v >= 0 else "top"
        ax_d.text(
            b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
            ha="center", va=va, fontsize=9,
        )
    diff_d = vals_d[1 - winner_d] - vals_d[winner_d]
    ax_d.set_title(
        rf"(b) $\log\det(\mathrm{{Cov}})$ — wrong choice +{diff_d:.3f} larger",
        fontsize=11,
    )
    ax_d.set_ylabel(r"$\log\det(\mathrm{Cov})$", fontsize=11)
    ax_d.grid(True, alpha=0.2, axis="y")
    ax_d.set_axisbelow(True)

