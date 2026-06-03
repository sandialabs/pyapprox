"""Plotting functions for group ACV tutorials.

Covers: group_acv_concept.qmd, group_acv_analysis.qmd,
        group_acv_multistat_concept.qmd
"""

import matplotlib.patches as mpatches
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
from pyapprox.statest import (
    GMFEstimator,
    MCEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.acv import ACVAllocator, default_allocator_factory
from pyapprox.statest.acv.base import FittedACVEstimator
from pyapprox.statest.allocation import MCAllocator
from pyapprox.statest.groupacv import GroupACVEstimatorIS, MLBLUEEstimator
from pyapprox.statest.groupacv.allocation import GroupACVAllocationOptimizer
from pyapprox.statest.groupacv.base import FittedGroupACVEstimator
from pyapprox.statest.groupacv.utils import get_model_subsets
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.statest import PolynomialEnsembleBenchmark

_MODEL_COLORS = ["#2C7FB8", "#95a5a6", "#27AE60", "#8E44AD", "#C0392B"]
_MODEL_LABELS = ["$f_0$ (HF)", "$f_1$", "$f_2$", "$f_3$", "$f_4$"]


# ---------------------------------------------------------------------------
# group_acv_concept.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_group_structures(axes):
    """group_acv_concept.qmd -> fig-group-structures

    Five-panel side-by-side comparison of the group structure underlying
    MC, MLMC, MFMC, ACVMF, and MLBLUE on a four-model ensemble.

    Each panel renders a fixed (illustrative, not optimised) sample-count
    profile so the structural pattern reads clearly. Group sizes are not
    drawn to scale across panels.

    Parameters
    ----------
    axes : sequence of 5 matplotlib Axes
        Will be populated with the five panels in order
        (MC, MLMC, MFMC, ACVMF, MLBLUE).
    """
    nmodels = 4

    group_specs = {
        "MC": [
            ([0], 1.6),
        ],
        "MLMC": [
            ([0, 1], 0.9),
            ([1, 2], 1.1),
            ([2, 3], 1.4),
        ],
        "MFMC": [
            ([0],          0.5),
            ([0, 1],       0.7),
            ([0, 1, 2],    1.0),
            ([0, 1, 2, 3], 1.4),
        ],
        "ACVMF": [
            ([0],    0.5),
            ([0, 1], 0.7),
            ([0, 2], 0.9),
            ([0, 3], 1.2),
        ],
        "MLBLUE": [
            ([0, 1],       0.6),
            ([0, 2, 3],    0.7),
            ([1, 2],       0.9),
            ([2, 3],       1.0),
            ([3],          1.4),
        ],
    }

    subtitles = {
        "MC":     "single group; all budget on $f_0$",
        "MLMC":   "pairwise consecutive-level groups",
        "MFMC":   "nested chain of groups",
        "ACVMF":  "every LF model grouped with $f_0$",
        "MLBLUE": "arbitrary subsets allowed",
    }

    block_h = 0.55
    row_gap = 0.18
    panel_left_pad = 0.4

    max_total_width = 0.0
    for groups in group_specs.values():
        widths_sum = sum(w for _, w in groups) + 0.2 * (len(groups) - 1)
        max_total_width = max(max_total_width, widths_sum)

    for ax, (name, groups) in zip(axes, group_specs.items()):
        x_cursor = panel_left_pad
        n_groups = len(groups)

        for g_idx, (model_idx_list, width) in enumerate(groups):
            n_in_group = len(model_idx_list)
            block_w = width / n_in_group
            y_center = -(g_idx + 0.5) * (block_h + row_gap)

            for slot, m_idx in enumerate(model_idx_list):
                x_left = x_cursor + slot * block_w
                rect = mpatches.FancyBboxPatch(
                    (x_left, y_center - block_h / 2),
                    block_w, block_h,
                    boxstyle="round,pad=0.02",
                    facecolor=_MODEL_COLORS[m_idx],
                    edgecolor="k", lw=0.9, alpha=0.85,
                )
                ax.add_patch(rect)
                ax.text(
                    x_left + block_w / 2, y_center,
                    f"$f_{{{m_idx}}}$",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white",
                )

            ax.text(
                x_cursor + width + 0.12, y_center,
                rf"$\mathcal{{G}}^{{{g_idx + 1}}}$",
                ha="left", va="center",
                fontsize=9, color="#444",
            )

        ax.set_xlim(0, max_total_width + 0.9)
        ax.set_ylim(-(n_groups + 0.5) * (block_h + row_gap), 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=12, fontweight="bold", pad=4)
        ax.text(
            (max_total_width + 0.9) / 2, 0.25,
            subtitles[name],
            ha="center", va="bottom",
            fontsize=8.5, color="#555", style="italic",
        )

    handles = [
        mpatches.Patch(facecolor=_MODEL_COLORS[i], edgecolor="k",
                       label=_MODEL_LABELS[i])
        for i in range(nmodels)
    ]
    axes[-1].legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(-2.0, -0.18), ncol=nmodels,
        fontsize=9, frameon=False,
    )


# ---------------------------------------------------------------------------

def _cv_limits(cov_np):
    """Return [CV-1, CV-2, ..., CV-(L)] floors (variance / sigma_0^2)."""
    sigma2 = cov_np[0, 0]
    nmodels = cov_np.shape[0]
    cv_lims = []
    for k in range(1, nmodels):
        sub = cov_np[: k + 1, : k + 1]
        c0 = sub[0, 1:]
        Sll = sub[1:, 1:]
        cv_lims.append(1 - c0 @ np.linalg.solve(Sll, c0) / sigma2)
    return cv_lims


def _ceiling_panel(ax, bkd, benchmark):
    """Left panel of fig-gacv-convergence: structural ceiling at N_0=1."""
    models = benchmark.problem().models()
    costs = benchmark.problem().costs()
    nqoi = models[0].nqoi()
    nmodels = len(models)
    M = nmodels - 1
    cov = benchmark.ensemble_covariance()
    cov_np = bkd.to_numpy(cov)
    sigma2 = cov_np[0, 0]
    mc_var = sigma2
    cv_lims = _cv_limits(cov_np)

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    ri_zeros = bkd.zeros(M, dtype=int)

    nhf = 1
    base_ratio = np.array([2, 2, 2, 2])
    factors = np.arange(22)

    def sweep_acv(est_template):
        nparts = est_template._npartitions
        cs, rs = [], []
        for f in factors:
            npartition_samples = bkd.asarray(
                nhf * np.hstack(
                    (1, base_ratio[: nparts - 1] * 2**f)
                ),
            )
            ec = est_template._covariance_from_npartition_samples(
                npartition_samples
            )
            actual_cost = bkd.to_float(
                est_template._estimator_cost(npartition_samples)
            )
            cs.append(actual_cost)
            rs.append(bkd.to_float(ec[0, 0]) / mc_var)
        return cs, rs

    mlmc_c, mlmc_r = sweep_acv(MLMCEstimator(stat, costs))
    mfmc_c, mfmc_r = sweep_acv(MFMCEstimator(stat, costs))
    acvmf_c, acvmf_r = sweep_acv(
        GMFEstimator(stat, costs, recursion_index=ri_zeros)
    )

    mlblue_subsets = [
        bkd.asarray(list(range(nmodels)), dtype=int),
        bkd.asarray(list(range(1, nmodels)), dtype=int),
        bkd.asarray(list(range(2, nmodels)), dtype=int),
        bkd.asarray(list(range(3, nmodels)), dtype=int),
        bkd.asarray([nmodels - 1], dtype=int),
    ]
    mlblue_template = MLBLUEEstimator(
        stat, costs, model_subsets=mlblue_subsets
    )
    mlblue_c, mlblue_r = [], []
    for f in factors:
        npart = bkd.asarray(
            nhf * np.hstack((1, base_ratio * 2**f)), dtype=float,
        )
        actual_cost = bkd.to_float(
            mlblue_template._estimator_cost(npart)
        )
        ec = mlblue_template._covariance_from_npartition_samples(npart)
        mlblue_c.append(actual_cost)
        mlblue_r.append(bkd.to_float(ec[0, 0]) / mc_var)

    ax.loglog(mlmc_c, mlmc_r, "-", color="#2C7FB8", lw=2, label="MLMC")
    ax.loglog(mfmc_c, mfmc_r, "--", color="#E67E22", lw=2, label="MFMC")
    ax.loglog(acvmf_c, acvmf_r, "-.", color="#8E44AD", lw=2, label="ACVMF")
    ax.loglog(mlblue_c, mlblue_r, ":", color="#27AE60", lw=2.5,
              label="MLBLUE (GACV-IS)")

    cv_colors = ["#c0392b", "#d35400", "#f39c12", "#2c3e50"]
    cv_ls = ["--", "-.", ":", (0, (5, 1))]
    for i, (lim, col, ls) in enumerate(zip(cv_lims, cv_colors, cv_ls)):
        ax.axhline(lim, color=col, ls=ls, lw=1.1,
                   label=f"CV-{i + 1} limit")

    ax.axhline(1.0, color="k", lw=1, ls=":", alpha=0.4, label="MC")
    ax.set_xlabel("Total cost  (HF=1, LF samples grow)", fontsize=11)
    ax.set_ylabel(r"$\mathbb{V}[\hat{\mu}_0] / \mathbb{V}_{\mathrm{MC}}$",
                  fontsize=11)
    ax.set_title("Structural ceiling ($N_0$ fixed)", fontsize=11)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.2, which="both")


def _optimized_panel(ax, bkd, benchmark, target_costs):
    """Right panel of fig-gacv-convergence: optimised allocation across
    three target budgets.
    """
    models = benchmark.problem().models()
    costs = benchmark.problem().costs()
    nqoi = models[0].nqoi()
    nmodels = len(models)
    M = nmodels - 1
    cov = benchmark.ensemble_covariance()

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    ri_zeros = bkd.zeros(M, dtype=int)
    optimizer = ScipySLSQPOptimizer(maxiter=200)

    def variance_at(est_type, target_cost):
        tc = float(target_cost)
        if est_type == "MC":
            fitted = MCAllocator(MCEstimator(stat, costs)).allocate(tc)
            return bkd.to_float(fitted.covariance()[0, 0])
        elif est_type == "MLMC":
            template = MLMCEstimator(stat, costs)
            result = default_allocator_factory(template).allocate(tc)
            return bkd.to_float(FittedACVEstimator(template, result).covariance()[0, 0])
        elif est_type == "MFMC":
            template = MFMCEstimator(stat, costs)
            result = default_allocator_factory(template).allocate(tc)
            return bkd.to_float(FittedACVEstimator(template, result).covariance()[0, 0])
        elif est_type == "ACVMF":
            template = GMFEstimator(stat, costs, recursion_index=ri_zeros)
            result = ACVAllocator(template, optimizer=optimizer).allocate(tc)
            return bkd.to_float(FittedACVEstimator(template, result).covariance()[0, 0])
        elif est_type == "GACV-IS":
            return _gacv_is_variance(cov, costs, nqoi, nmodels, tc)
        else:
            raise ValueError(est_type)

    est_types = ["MC", "MLMC", "MFMC", "ACVMF", "GACV-IS"]
    colors = {
        "MC":      "#aaaaaa",
        "MLMC":    "#2C7FB8",
        "MFMC":    "#E67E22",
        "ACVMF":   "#8E44AD",
        "GACV-IS": "#27AE60",
    }
    markers = {
        "MC": "o", "MLMC": "s", "MFMC": "D",
        "ACVMF": "^", "GACV-IS": "v",
    }
    linestyles = {
        "MC": ":", "MLMC": "-", "MFMC": "--",
        "ACVMF": "-.", "GACV-IS": "-",
    }

    target_costs_np = np.array([float(c) for c in target_costs])
    for et in est_types:
        vars_at_cost = []
        for tc in target_costs:
            try:
                v = variance_at(et, tc)
            except Exception:
                v = np.nan
            vars_at_cost.append(v)
        ax.loglog(target_costs_np, vars_at_cost,
                  linestyles[et] + markers[et],
                  color=colors[et], lw=1.8, ms=8, label=et)

    ax.set_xlabel("Total cost $P$", fontsize=11)
    ax.set_ylabel(r"Optimised $\mathbb{V}[\hat{\mu}_0]$", fontsize=11)
    ax.set_title("Optimised allocation across budgets", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2, which="both")


def _multistat_optimizer():
    """Chained DE + trust-constr optimizer for multistat tutorials."""
    global_opt = ScipyDifferentialEvolutionOptimizer(
        maxiter=3, popsize=5, polish=False, seed=1, tol=1e-8,
        raise_on_failure=False,
    )
    local_opt = ScipyTrustConstrOptimizer(gtol=1e-6, maxiter=500)
    return ChainedOptimizer(global_opt, local_opt)


def _gacv_is_variance(cov, costs, nqoi, nmodels, target_cost):
    """Compute GACV-IS optimised variance using analytical derivatives."""
    bkd = NumpyBkd()
    cov = bkd.asarray(cov)
    costs = bkd.asarray(costs)
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    subsets = get_model_subsets(nmodels, bkd)
    template = GroupACVEstimatorIS(stat, costs, model_subsets=subsets)
    opt = _multistat_optimizer()
    result = GroupACVAllocationOptimizer(
        template, optimizer=opt
    ).optimize(target_cost)
    fitted = FittedGroupACVEstimator(template, result)
    return bkd.to_float(fitted.covariance()[0, 0])


def plot_gacv_convergence(axes, target_costs=(20.0, 100.0, 500.0)):
    """group_acv_concept.qmd -> fig-gacv-convergence

    Two-panel figure. Left: structural ceiling at fixed N_0 = 1 with growing
    LF samples. Right: optimised allocation across three target budgets.

    Both panels run on the polynomial 5-model benchmark.

    Parameters
    ----------
    axes : sequence of 2 matplotlib Axes
        Left receives the ceiling panel, right receives the optimised panel.
    target_costs : tuple of float, optional
        Three target costs for the optimised-allocation panel. Default
        (20, 100, 500) covers small/medium/large regimes on the polynomial
        benchmark.
    """
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=5)

    _ceiling_panel(axes[0], bkd, benchmark)
    _optimized_panel(axes[1], bkd, benchmark, target_costs)


# ---------------------------------------------------------------------------
# group_acv_multistat_concept.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_multistat_reduction(ax, target_costs=None):
    """group_acv_multistat_concept.qmd -> fig-multistat-reduction

    Variance reduction (estimator variance / MC variance) vs total budget for
    three target statistics on the polynomial 5-model benchmark:
      1. Mean — group ACV mean estimator (MultiOutputMean)
      2. Variance, standalone — group ACV variance estimator
         (MultiOutputVariance)
      3. Variance, from joint estimation — the variance component of the
         joint mean+variance estimator (MultiOutputMeanAndVariance)

    Parameters
    ----------
    ax : matplotlib Axes
        Single axes to draw on.
    target_costs : sequence of float, optional
        Budget values to sweep. Defaults to logspace(2, 3, 3).
    """
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=5)
    models = benchmark.problem().models()
    costs = benchmark.problem().costs()
    nqoi = models[0].nqoi()
    nmodels = len(costs)
    variable = benchmark.problem().prior()
    cov = benchmark.ensemble_covariance()

    if target_costs is None:
        target_costs = np.logspace(2, 3, 3)

    np.random.seed(42)
    npilot = 10000
    pilot_samples = variable.rvs(npilot)
    pilot_values = [m(pilot_samples) for m in models]

    subsets = get_model_subsets(nmodels, bkd)

    stat_mean = MultiOutputMean(nqoi, bkd)
    stat_mean.set_pilot_quantities(cov)

    stat_var = MultiOutputVariance(nqoi, bkd)
    cov_var, W = stat_var.compute_pilot_quantities(pilot_values)
    stat_var.set_pilot_quantities(cov_var, W)

    stat_joint = MultiOutputMeanAndVariance(nqoi, bkd)
    cov_joint, W_joint, B_joint = stat_joint.compute_pilot_quantities(
        pilot_values
    )
    stat_joint.set_pilot_quantities(cov_joint, W_joint, B_joint)

    mc_mean_var = bkd.to_float(cov[0, 0])
    mc_var_var = bkd.to_float(
        stat_var.high_fidelity_estimator_covariance(bkd.asarray(1.0))[0, 0]
    )

    configs = [
        ("Mean", stat_mean, mc_mean_var, 0, "#2C7FB8", "-o"),
        ("Variance (standalone)", stat_var, mc_var_var, 0,
         "#E67E22", "--s"),
        ("Variance (from joint)", stat_joint, mc_var_var, nqoi,
         "#27AE60", "-.^"),
    ]

    for label, stat, mc_ref, diag_idx, color, fmt in configs:
        reductions = []
        for tc in target_costs:
            template = GroupACVEstimatorIS(
                stat, costs, model_subsets=subsets
            )
            opt = _multistat_optimizer()
            result = GroupACVAllocationOptimizer(
                template, optimizer=opt
            ).optimize(float(tc))
            fitted = FittedGroupACVEstimator(template, result)
            est_var = bkd.to_float(
                fitted.covariance()[diag_idx, diag_idx]
            )
            reductions.append(est_var / mc_ref)
        ax.loglog(target_costs, reductions, fmt, color=color, lw=1.8,
                  ms=7, label=label)

    ax.set_xlabel("Total cost $P$", fontsize=11)
    ax.set_ylabel(
        r"$\mathbb{V}[\hat{\theta}] \;/\; \mathbb{V}_{\mathrm{MC}}$",
        fontsize=11,
    )
    ax.set_title("Variance reduction vs budget by statistic", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
