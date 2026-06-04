"""Plotting functions for group ACV tutorials.

Covers: group_acv_concept.qmd, group_acv_analysis.qmd,
        group_acv_multistat_concept.qmd
"""

import matplotlib.patches as mpatches
import numpy as np

from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.statest import (
    GMFEstimator,
    MCEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.acv import ACVAllocator, default_allocator_factory
from pyapprox.statest.acv.base import FittedACVEstimator
from pyapprox.statest.allocation import MCAllocator
from pyapprox.statest.groupacv import (
    GroupACVEstimatorIS,
    MeanGuidedSubsetFitter,
    MLBLUEEstimator,
)
from pyapprox.statest.groupacv.allocation import GroupACVAllocationOptimizer
from pyapprox.statest.groupacv.base import FittedGroupACVEstimator
from pyapprox.statest.groupacv.utils import get_model_subsets
from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
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

    Four-panel (2x2) comparison of the group structure underlying
    MLMC, MFMC, ACVMF, and MLBLUE on a four-model ensemble.

    Each panel renders a fixed illustrative group collection. Block
    widths are uniform per model; the MLBLUE panel uses vertical
    ellipses to indicate the full $2^M - 1$ subset collection.

    Parameters
    ----------
    axes : 2x2 array of matplotlib Axes
        Will be populated in order (MLMC, MFMC, ACVMF, MLBLUE),
        row-major.
    """
    nmodels = 4

    block_w_per_model = 0.35
    group_specs = {
        "MLMC": [
            [0, 1],
            [1, 2],
            [2, 3],
            [3],
        ],
        "MFMC": [
            [0],
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3],
        ],
        "ACVMF": [
            [0],
            [0, 1],
            [0, 2],
            [0, 3],
        ],
        "MLBLUE": [
            [2],
            [1, 3],
            "...",
            [0, 2, 3],
            [1, 2, 3],
            "...",
            [0, 1, 2, 3],
        ],
    }

    subtitles = {
        "MLMC":   "pairwise consecutive-level groups",
        "MFMC":   "nested chain of groups",
        "ACVMF":  "every LF model grouped with $f_0$",
        "MLBLUE": "all $2^M{-}1$ non-empty subsets",
    }

    block_h = 0.55
    row_gap = 0.18
    panel_left_pad = 0.4

    max_total_width = nmodels * block_w_per_model + 0.6

    flat_axes = axes.ravel() if hasattr(axes, "ravel") else axes
    for ax, (name, groups) in zip(flat_axes, group_specs.items()):
        x_cursor = panel_left_pad
        n_groups = len(groups)
        label_counter = 0

        for g_idx, entry in enumerate(groups):
            y_center = -(g_idx + 0.5) * (block_h + row_gap)

            if isinstance(entry, str):
                ax.text(
                    x_cursor + 0.3, y_center,
                    r"$\vdots$",
                    ha="center", va="center",
                    fontsize=12, color="#888",
                )
                continue

            label_counter += 1
            model_idx_list = entry
            width = len(model_idx_list) * block_w_per_model

            for slot, m_idx in enumerate(model_idx_list):
                x_left = x_cursor + slot * block_w_per_model
                rect = mpatches.FancyBboxPatch(
                    (x_left, y_center - block_h / 2),
                    block_w_per_model, block_h,
                    boxstyle="round,pad=0.02",
                    facecolor=_MODEL_COLORS[m_idx],
                    edgecolor="k", lw=0.9, alpha=0.85,
                )
                ax.add_patch(rect)
                ax.text(
                    x_left + block_w_per_model / 2, y_center,
                    f"$f_{{{m_idx}}}$",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white",
                )

            ax.text(
                x_cursor + width + 0.12, y_center,
                rf"$\mathcal{{G}}^{{{label_counter}}}$",
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
    flat_axes[-1].legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(-0.1, -0.25), ncol=nmodels,
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


_LOG_INEQ_CONFIG = AllocationProblemConfig(
    variable_scaling="log",
    budget_constraint_form="inequality",
)


def _slsqp():
    return ScipySLSQPOptimizer(maxiter=1000, ftol=1e-10)


def _gacv_is_variance(cov, costs, nqoi, nmodels, target_cost):
    """Compute GACV-IS optimised variance for Mean stat (SLSQP + log/ineq)."""
    bkd = NumpyBkd()
    cov = bkd.asarray(cov)
    costs = bkd.asarray(costs)
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    subsets = get_model_subsets(nmodels, bkd)
    template = GroupACVEstimatorIS(stat, costs, model_subsets=subsets)
    result = GroupACVAllocationOptimizer(
        template, optimizer=_slsqp(), problem_config=_LOG_INEQ_CONFIG,
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

def plot_multistat_reduction(axes, target_cost=500.0):
    """group_acv_multistat_concept.qmd -> fig-multistat-reduction

    Three-panel bar chart comparing standalone vs joint estimation at a single
    budget. Left: mean estimation variance reduction. Middle: variance
    estimation variance reduction. Right: determinant of the joint estimator
    covariance (the objective minimized by joint estimation).

    Parameters
    ----------
    axes : array of matplotlib Axes, shape (3,)
        Three panels: mean, variance, det.
    target_cost : float, optional
        Budget for the comparison. Default 500.
    """
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=5)
    models = benchmark.problem().models()
    costs = benchmark.problem().costs()
    nqoi = models[0].nqoi()
    nmodels = len(costs)
    hf_cost = bkd.to_float(costs[0])

    cov_full = benchmark.covariance_matrix()
    W = benchmark.covariance_of_centered_values_kronecker_product()
    B = benchmark.covariance_of_mean_and_variance_estimators()

    all_subsets = get_model_subsets(nmodels, bkd)
    mlmc_subsets = [
        bkd.asarray([k, k + 1], dtype=int)
        for k in range(nmodels - 1)
    ] + [bkd.asarray([nmodels - 1], dtype=int)]

    stat_mean = MultiOutputMean(nqoi, bkd)
    stat_mean.set_pilot_quantities(cov_full)

    stat_var = MultiOutputVariance(nqoi, bkd)
    stat_var.set_pilot_quantities(cov_full, W)

    stat_joint = MultiOutputMeanAndVariance(nqoi, bkd)
    stat_joint.set_pilot_quantities(cov_full, W, B)

    n_mc = float(target_cost) / hf_cost
    mc_mean_var = bkd.to_float(
        stat_mean.high_fidelity_estimator_covariance(
            bkd.asarray(n_mc)
        )[0, 0]
    )
    mc_var_var = bkd.to_float(
        stat_var.high_fidelity_estimator_covariance(
            bkd.asarray(n_mc)
        )[0, 0]
    )
    mc_det = mc_mean_var * mc_var_var

    def _solve_guided(stat, model_subsets):
        fitter = MeanGuidedSubsetFitter(
            stat, costs, GroupACVEstimatorIS,
            candidate_subsets=model_subsets,
            optimizer=_slsqp(),
            problem_config=_LOG_INEQ_CONFIG,
        )
        result = fitter.fit(float(target_cost))
        active = [
            model_subsets[i] for i in result.active_subset_indices
        ]
        nps = bkd.asarray(
            result.best_allocation.relaxed_npartition_samples,
            dtype=float,
        )
        return nps, active

    def _solve_direct(stat, model_subsets):
        template = GroupACVEstimatorIS(
            stat, costs, model_subsets=model_subsets
        )
        result = GroupACVAllocationOptimizer(
            template, optimizer=_slsqp(),
            problem_config=_LOG_INEQ_CONFIG,
        ).optimize(float(target_cost), round_nsamples=False)
        nps = bkd.asarray(result.npartition_samples, dtype=float)
        return nps, model_subsets

    def _eval_cov(stat, nps, subsets_used):
        template = GroupACVEstimatorIS(
            stat, costs, model_subsets=subsets_used
        )
        return template._covariance_from_npartition_samples(nps)

    # Solve each configuration
    # Standalone (all subsets): separate mean and variance allocations
    nps_mean_all, subs_mean_all = _solve_guided(stat_mean, all_subsets)
    nps_var_all, subs_var_all = _solve_guided(stat_var, all_subsets)
    # Joint (all subsets)
    nps_joint_all, subs_joint_all = _solve_guided(stat_joint, all_subsets)
    # Standalone (MLMC subsets)
    nps_mean_mlmc, subs_mean_mlmc = _solve_direct(stat_mean, mlmc_subsets)
    nps_var_mlmc, subs_var_mlmc = _solve_direct(stat_var, mlmc_subsets)
    # Joint (MLMC subsets)
    nps_joint_mlmc, subs_joint_mlmc = _solve_direct(stat_joint, mlmc_subsets)

    # Evaluate covariances for panels (a) and (b)
    cov_mean_all = _eval_cov(stat_mean, nps_mean_all, subs_mean_all)
    cov_var_all = _eval_cov(stat_var, nps_var_all, subs_var_all)
    cov_joint_all = _eval_cov(stat_joint, nps_joint_all, subs_joint_all)
    cov_mean_mlmc = _eval_cov(stat_mean, nps_mean_mlmc, subs_mean_mlmc)
    cov_var_mlmc = _eval_cov(stat_var, nps_var_mlmc, subs_var_mlmc)
    cov_joint_mlmc = _eval_cov(stat_joint, nps_joint_mlmc, subs_joint_mlmc)

    # Panel (c): det(Cov_joint) evaluated at each allocation separately.
    # Standalone mean and variance use different allocations, so we evaluate
    # the joint covariance at each one to show that joint allocation achieves
    # the lowest determinant.
    cov_joint_at_mean_all = _eval_cov(
        stat_joint, nps_mean_all, subs_mean_all
    )
    cov_joint_at_var_all = _eval_cov(
        stat_joint, nps_var_all, subs_var_all
    )
    cov_joint_at_mean_mlmc = _eval_cov(
        stat_joint, nps_mean_mlmc, subs_mean_mlmc
    )
    cov_joint_at_var_mlmc = _eval_cov(
        stat_joint, nps_var_mlmc, subs_var_mlmc
    )

    # Panels (a) and (b): 4 bars each
    bars_ab = [
        ("Standalone\n(all subsets)", "#2C7FB8",
         bkd.to_float(cov_mean_all[0, 0]) / mc_mean_var,
         bkd.to_float(cov_var_all[0, 0]) / mc_var_var),
        ("Joint\n(all subsets)", "#27AE60",
         bkd.to_float(cov_joint_all[0, 0]) / mc_mean_var,
         bkd.to_float(cov_joint_all[nqoi, nqoi]) / mc_var_var),
        ("Standalone\n(MLMC subsets)", "#8E44AD",
         bkd.to_float(cov_mean_mlmc[0, 0]) / mc_mean_var,
         bkd.to_float(cov_var_mlmc[0, 0]) / mc_var_var),
        ("Joint\n(MLMC subsets)", "#E67E22",
         bkd.to_float(cov_joint_mlmc[0, 0]) / mc_mean_var,
         bkd.to_float(cov_joint_mlmc[nqoi, nqoi]) / mc_var_var),
    ]

    # Panel (c): 6 bars — det(Cov_joint) at mean-only, var-only, and joint
    # allocations for each subset configuration
    bars_det = [
        ("Mean alloc\n(all)", "#2C7FB8",
         bkd.to_float(bkd.det(cov_joint_at_mean_all)) / mc_det),
        ("Var alloc\n(all)", "#C0392B",
         bkd.to_float(bkd.det(cov_joint_at_var_all)) / mc_det),
        ("Joint alloc\n(all)", "#27AE60",
         bkd.to_float(bkd.det(cov_joint_all)) / mc_det),
        ("Mean alloc\n(MLMC)", "#8E44AD",
         bkd.to_float(bkd.det(cov_joint_at_mean_mlmc)) / mc_det),
        ("Var alloc\n(MLMC)", "#E67E22",
         bkd.to_float(bkd.det(cov_joint_at_var_mlmc)) / mc_det),
        ("Joint alloc\n(MLMC)", "#2ECC71",
         bkd.to_float(bkd.det(cov_joint_mlmc)) / mc_det),
    ]

    # --- Plot panels (a) and (b) ---
    labels_ab = [b[0] for b in bars_ab]
    colors_ab = [b[1] for b in bars_ab]
    mean_ratios = [b[2] for b in bars_ab]
    var_ratios = [b[3] for b in bars_ab]

    x_ab = np.arange(len(bars_ab))
    width = 0.6

    ax_m, ax_v, ax_d = axes[0], axes[1], axes[2]

    ax_m.bar(x_ab, mean_ratios, width, color=colors_ab, edgecolor="k", lw=0.5)
    ax_m.set_yscale("log")
    ax_m.set_xticks(x_ab)
    ax_m.set_xticklabels(labels_ab, fontsize=8)
    ax_m.set_ylabel(
        r"$\mathbb{V}[\hat{\mu}_0]\;/\;\mathbb{V}_{\mathrm{MC}}$",
        fontsize=10,
    )
    ax_m.set_title(f"(a) Mean ($P={int(target_cost)}$)", fontsize=11)
    ax_m.grid(True, alpha=0.2, axis="y", which="both")
    ax_m.set_axisbelow(True)

    ax_v.bar(x_ab, var_ratios, width, color=colors_ab, edgecolor="k", lw=0.5)
    ax_v.set_yscale("log")
    ax_v.set_xticks(x_ab)
    ax_v.set_xticklabels(labels_ab, fontsize=8)
    ax_v.set_ylabel(
        r"$\mathbb{V}[\hat{\sigma}_0^2]\;/\;\mathbb{V}_{\mathrm{MC}}$",
        fontsize=10,
    )
    ax_v.set_title(f"(b) Variance ($P={int(target_cost)}$)", fontsize=11)
    ax_v.grid(True, alpha=0.2, axis="y", which="both")
    ax_v.set_axisbelow(True)

    # --- Plot panel (c): 6 bars ---
    labels_det = [b[0] for b in bars_det]
    colors_det = [b[1] for b in bars_det]
    det_ratios = [b[2] for b in bars_det]

    x_det = np.arange(len(bars_det))
    width_det = 0.5

    ax_d.bar(x_det, det_ratios, width_det, color=colors_det,
             edgecolor="k", lw=0.5)
    ax_d.set_yscale("log")
    ax_d.set_xticks(x_det)
    ax_d.set_xticklabels(labels_det, fontsize=7)
    ax_d.set_ylabel(
        r"$\det(\mathrm{Cov})\;/\;\det(\mathrm{Cov}_{\mathrm{MC}})$",
        fontsize=10,
    )
    ax_d.set_title(f"(c) Joint covariance determinant ($P={int(target_cost)}$)",
                   fontsize=11)
    ax_d.grid(True, alpha=0.2, axis="y", which="both")
    ax_d.set_axisbelow(True)


# ---------------------------------------------------------------------------
# group_acv_mixed_concept.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

_KNOWN_COLORS = ["#2C7FB8", "#E67E22", "#27AE60", "#8E44AD", "#C0392B"]
_KNOWN_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


_CHEAPEST_FIRST_ORDER = [4, 3, 2, 1]


def _make_known_quantities(means, nknown, nqoi):
    """Build known_quantities dict, adding cheapest LF models first."""
    if nknown == 0:
        return None
    kq = {}
    for m in _CHEAPEST_FIRST_ORDER[:nknown]:
        kq[(m, "mean")] = means[m, :nqoi]
    return kq


def _mixed_ceiling_panel(ax, bkd, benchmark):
    """Panel (a): variance ratio vs LF samples for each |K|.

    Uses the same 5-partition MLBLUE structure and sweep as
    ``_ceiling_panel`` so that all curves share the same x-axis
    (total cost). Known-mean partitions stay at nhf.
    """
    cov = benchmark.ensemble_covariance()
    costs = benchmark.problem().costs()
    means = benchmark.ensemble_means()
    nqoi = benchmark.problem().models()[0].nqoi()
    nmodels = len(costs)
    M = nmodels - 1
    cov_np = bkd.to_numpy(cov)
    mc_var = cov_np[0, 0]

    mlblue_subsets = [
        bkd.asarray(list(range(nmodels)), dtype=int),
        bkd.asarray(list(range(1, nmodels)), dtype=int),
        bkd.asarray(list(range(2, nmodels)), dtype=int),
        bkd.asarray(list(range(3, nmodels)), dtype=int),
        bkd.asarray([nmodels - 1], dtype=int),
    ]

    cv_lim = _cv_limits(cov_np)[-1]

    factors = np.arange(22)
    nhf = 1
    base_ratio = np.array([2, 2, 2, 2])

    stat_ref = MultiOutputMean(nqoi, bkd)
    stat_ref.set_pilot_quantities(cov)
    mfmc_template = MFMCEstimator(stat_ref, costs)
    mfmc_c, mfmc_r = [], []
    for f in factors:
        nps = bkd.asarray(nhf * np.hstack((1, base_ratio[:M] * 2**f)))
        mfmc_c.append(bkd.to_float(mfmc_template._estimator_cost(nps)))
        ec = mfmc_template._covariance_from_npartition_samples(nps)
        mfmc_r.append(bkd.to_float(ec[0, 0]) / mc_var)
    ax.loglog(mfmc_c, mfmc_r, "-", color="#aaaaaa", lw=1.5, alpha=0.7,
              label="MFMC")

    for nknown in range(M):
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        kq = _make_known_quantities(means, nknown, nqoi)
        est = GroupACVEstimatorIS(
            stat, costs, model_subsets=mlblue_subsets,
            known_quantities=kq,
        )
        nparts = est.npartitions()
        known_set = set(_CHEAPEST_FIRST_ORDER[:nknown])

        costs_list, ratios = [], []
        for f in factors:
            npart = nhf * np.hstack((1, base_ratio * 2**f))
            for p in range(1, nparts):
                models_in_p = set(
                    bkd.to_numpy(mlblue_subsets[p]).astype(int).tolist()
                )
                if models_in_p <= known_set:
                    npart[p] = nhf
            npart_arr = bkd.asarray(npart, dtype=float)
            actual_cost = bkd.to_float(est._estimator_cost(npart_arr))
            ec = est._covariance_from_npartition_samples(npart_arr)
            costs_list.append(actual_cost)
            ratios.append(bkd.to_float(ec[0, 0]) / mc_var)

        ax.loglog(
            costs_list, ratios,
            linestyle=_KNOWN_STYLES[nknown],
            color=_KNOWN_COLORS[nknown],
            lw=1.8,
            label=rf"$|\mathcal{{K}}| = {nknown}$",
        )

    cv_lims = _cv_limits(cov_np)
    ax.axhline(cv_lims[0], color="#E74C3C", ls="--", lw=1.5, zorder=5,
               label="CV-1 floor")
    ax.axhline(cv_lim, color="#2c3e50", ls="--", lw=1.5, zorder=5,
               label=f"CV-{M} floor")
    ax.set_xlabel("Total cost (HF fixed, LF grows)", fontsize=10)
    ax.set_ylabel(
        r"$\mathbb{V}[\hat{\mu}_0] / \mathbb{V}_{\mathrm{MC}}$",
        fontsize=10,
    )
    ax.set_title("(a) Convergence rate", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.15, which="both")


def _mixed_optimized_panel(ax, bkd, benchmark, target_costs):
    """Panel (b): optimized variance vs budget, one curve per |K|."""
    cov = benchmark.ensemble_covariance()
    costs = benchmark.problem().costs()
    means = benchmark.ensemble_means()
    nqoi = benchmark.problem().models()[0].nqoi()
    nmodels = len(costs)
    M = nmodels - 1
    mc_var = bkd.to_float(cov[0, 0])
    subsets = get_model_subsets(nmodels, bkd)

    tc_array = np.array([float(c) for c in target_costs])

    for nknown in range(M + 1):
        ratios = []
        for tc in target_costs:
            stat = MultiOutputMean(nqoi, bkd)
            stat.set_pilot_quantities(cov)
            kq = _make_known_quantities(means, nknown, nqoi)
            template = GroupACVEstimatorIS(
                stat, costs, model_subsets=subsets, known_quantities=kq,
            )
            result = GroupACVAllocationOptimizer(
                template, optimizer=_slsqp(),
                problem_config=_LOG_INEQ_CONFIG,
            ).optimize(float(tc))
            fitted = FittedGroupACVEstimator(template, result)
            est_var = bkd.to_float(fitted.covariance()[0, 0])
            ratios.append(est_var / mc_var)
        ax.loglog(
            tc_array, ratios,
            linestyle=_KNOWN_STYLES[nknown],
            marker="o",
            color=_KNOWN_COLORS[nknown],
            lw=1.8, ms=6,
            label=rf"$|\mathcal{{K}}| = {nknown}$",
        )

    ax.set_xlabel("Total cost $P$", fontsize=10)
    ax.set_ylabel(
        r"$\mathbb{V}[\hat{\mu}_0] / \mathbb{V}_{\mathrm{MC}}$",
        fontsize=10,
    )
    ax.set_title("(b) Optimized allocation across budgets", fontsize=11)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2, which="both")


def _mixed_summary_panel(ax, bkd, benchmark, fixed_budget=100.0):
    """Panel (c): variance vs |K| at one fixed budget."""
    cov = benchmark.ensemble_covariance()
    costs = benchmark.problem().costs()
    means = benchmark.ensemble_means()
    nqoi = benchmark.problem().models()[0].nqoi()
    nmodels = len(costs)
    M = nmodels - 1
    mc_var = bkd.to_float(cov[0, 0])
    subsets = get_model_subsets(nmodels, bkd)

    ratios = []
    for nknown in range(M + 1):
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        kq = _make_known_quantities(means, nknown, nqoi)
        template = GroupACVEstimatorIS(
            stat, costs, model_subsets=subsets, known_quantities=kq,
        )
        result = GroupACVAllocationOptimizer(
            template, optimizer=_slsqp(),
            problem_config=_LOG_INEQ_CONFIG,
        ).optimize(float(fixed_budget))
        fitted = FittedGroupACVEstimator(template, result)
        est_var = bkd.to_float(fitted.covariance()[0, 0])
        ratios.append(est_var / mc_var)

    ax.semilogy(
        range(M + 1), ratios, "-o",
        color="#2C7FB8", lw=2, ms=8,
    )
    ax.fill_between(
        range(M + 1), ratios, alpha=0.15, color="#2C7FB8",
    )

    ax.set_xlabel(r"$|\mathcal{K}|$ (number of known LF means)", fontsize=10)
    ax.set_ylabel(
        r"$\mathbb{V}[\hat{\mu}_0] / \mathbb{V}_{\mathrm{MC}}$",
        fontsize=10,
    )
    ax.set_title(f"(c) Summary at $P = {int(fixed_budget)}$", fontsize=11)
    ax.set_xticks(range(M + 1))
    ax.grid(True, alpha=0.2, which="both")


def plot_mixed_known_spectrum(axes, target_costs=(50.0, 200.0, 1000.0)):
    """group_acv_mixed_concept.qmd -> fig-mixed-known-spectrum

    Three-panel figure showing how known LF means affect group ACV.

    Panel (a): structural convergence rate — variance ratio vs total cost
    with N_0 fixed and LF samples growing, one curve per |K|.

    Panel (b): optimized-allocation variance ratio vs |K| at three budgets.

    Panel (c): variance ratio vs |K| at a single fixed budget (P=100),
    showing the monotonic improvement from constraint relaxation.

    Parameters
    ----------
    axes : sequence of 3 matplotlib Axes
    target_costs : tuple of float
        Three budgets for panel (b).
    """
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=5)

    _mixed_ceiling_panel(axes[0], bkd, benchmark)
    _mixed_optimized_panel(axes[1], bkd, benchmark, target_costs)
    _mixed_summary_panel(axes[2], bkd, benchmark, fixed_budget=100.0)


def plot_mixed_allocation_shift(ax, target_cost=100.0):
    """group_acv_mixed_concept.qmd -> fig-allocation-shift

    Grouped bar chart: samples per model for each |K| value.
    X-axis groups by |K|, bars within each group are models f0-f4.
    Log-scale y-axis.

    Parameters
    ----------
    ax : matplotlib Axes
    target_cost : float
    """
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=5)
    cov = benchmark.ensemble_covariance()
    costs = benchmark.problem().costs()
    means = benchmark.ensemble_means()
    nqoi = benchmark.problem().models()[0].nqoi()
    nmodels = len(costs)
    M = nmodels - 1
    subsets = get_model_subsets(nmodels, bkd)

    all_npm = []
    for nknown in range(M + 1):
        stat = MultiOutputMean(nqoi, bkd)
        stat.set_pilot_quantities(cov)
        kq = _make_known_quantities(means, nknown, nqoi)
        est = GroupACVEstimatorIS(
            stat, costs, model_subsets=subsets, known_quantities=kq,
        )
        result = GroupACVAllocationOptimizer(
            est, optimizer=_slsqp(), problem_config=_LOG_INEQ_CONFIG,
        ).optimize(float(target_cost))
        nps = bkd.to_numpy(result.npartition_samples).astype(float)
        npm = bkd.to_numpy(est._compute_nsamples_per_model(bkd.asarray(nps)))
        all_npm.append(npm)

    bar_width = 0.15
    x = np.arange(M + 1)

    for m in range(nmodels):
        offset = (m - nmodels / 2 + 0.5) * bar_width
        vals = [all_npm[k][m] for k in range(M + 1)]
        bars = ax.bar(
            x + offset, vals, bar_width,
            color=_MODEL_COLORS[m], edgecolor="k", lw=0.4,
            label=_MODEL_LABELS[m],
        )
        known_at_k = [set(_CHEAPEST_FIRST_ORDER[:k]) for k in range(M + 1)]
        for k in range(M + 1):
            if m in known_at_k[k]:
                bars[k].set_hatch("//")
                bars[k].set_edgecolor("#444")

    ax.set_yscale("log")
    ax.set_xlabel(r"$|\mathcal{K}|$ (number of known LF means)", fontsize=10)
    ax.set_ylabel("Samples per model", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(M + 1)])
    handles, labels = ax.get_legend_handles_labels()
    known_patch = mpatches.Patch(
        facecolor="#cccccc", edgecolor="#444", hatch="//",
        label="known mean",
    )
    handles.append(known_patch)
    labels.append("known mean")
    ax.legend(handles, labels, fontsize=8, loc="upper right", ncol=2)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.2, which="both", axis="y")


# ---------------------------------------------------------------------------
# group_acv_multistat_concept.qmd — allocation target mismatch
# ---------------------------------------------------------------------------

def plot_allocation_target_mismatch(axes, target_cost=500.0):
    """group_acv_multistat_concept.qmd -> fig-allocation-target-mismatch

    Two-panel bar chart showing the suboptimality of using an allocation
    optimized for one statistic to evaluate another, using all 2^M-1 subsets.

    Three allocation strategies (mean-target, variance-target,
    joint-target) are each evaluated under two report statistics
    (mean in panel a, variance in panel b).

    Parameters
    ----------
    axes : array of 2 matplotlib Axes
        Panel (a) = variance report, panel (b) = mean report.
    target_cost : float, optional
        Budget for the comparison. Default 500.
    """
    bkd = NumpyBkd()
    benchmark = PolynomialEnsembleBenchmark(bkd, nmodels=5)
    models = benchmark.problem().models()
    costs = benchmark.problem().costs()
    nqoi = models[0].nqoi()
    nmodels = len(costs)
    hf_cost = bkd.to_float(costs[0])

    all_subsets = get_model_subsets(nmodels, bkd)

    cov_full = benchmark.covariance_matrix()
    W = benchmark.covariance_of_centered_values_kronecker_product()
    B = benchmark.covariance_of_mean_and_variance_estimators()

    stat_mean = MultiOutputMean(nqoi, bkd)
    stat_mean.set_pilot_quantities(cov_full)

    stat_var = MultiOutputVariance(nqoi, bkd)
    stat_var.set_pilot_quantities(cov_full, W)

    stat_joint = MultiOutputMeanAndVariance(nqoi, bkd)
    stat_joint.set_pilot_quantities(cov_full, W, B)

    n_mc = float(target_cost) / hf_cost
    mc_mean_var = bkd.to_float(
        stat_mean.high_fidelity_estimator_covariance(
            bkd.asarray(n_mc)
        )[0, 0]
    )
    mc_var_var = bkd.to_float(
        stat_var.high_fidelity_estimator_covariance(
            bkd.asarray(n_mc)
        )[0, 0]
    )

    def _allocate_direct(stat_alloc):
        template = GroupACVEstimatorIS(
            stat_alloc, costs, model_subsets=all_subsets
        )
        result = GroupACVAllocationOptimizer(
            template, optimizer=_slsqp(), problem_config=_LOG_INEQ_CONFIG,
        ).optimize(float(target_cost), round_nsamples=False)
        return bkd.asarray(
            result.npartition_samples, dtype=bkd.double_dtype()
        )

    def _allocate_guided(stat_alloc):
        fitter = MeanGuidedSubsetFitter(
            stat_alloc, costs, GroupACVEstimatorIS,
            candidate_subsets=all_subsets,
            optimizer=_slsqp(),
            problem_config=_LOG_INEQ_CONFIG,
        )
        result = fitter.fit(float(target_cost))
        return (
            bkd.asarray(
                result.best_allocation.relaxed_npartition_samples,
                dtype=bkd.double_dtype(),
            ),
            [all_subsets[i] for i in result.active_subset_indices],
        )

    def _evaluate(stat_report, diag_idx, nps, subsets_used):
        template = GroupACVEstimatorIS(
            stat_report, costs, model_subsets=subsets_used
        )
        est_cov = template._covariance_from_npartition_samples(nps)
        return bkd.to_float(est_cov[diag_idx, diag_idx])

    # Mean: direct (no dead threshold issue)
    nps_mean = _allocate_direct(stat_mean)
    subs_mean = all_subsets

    # Variance: mean-guided screening
    nps_var, subs_var = _allocate_guided(stat_var)

    # Joint: mean-guided screening
    nps_joint, subs_joint = _allocate_guided(stat_joint)

    # Each alloc spec: (label, color, nps, subsets_used,
    #   mean_stat, mean_diag, var_stat, var_diag)
    alloc_specs = [
        ("Allocate\nfor variance", "#2C7FB8", nps_var, subs_var,
         stat_mean, 0, stat_var, 0),
        ("Allocate\njointly", "#27AE60", nps_joint, subs_joint,
         stat_joint, 0, stat_joint, nqoi),
        ("Allocate\nfor mean", "#E67E22", nps_mean, subs_mean,
         stat_mean, 0, stat_var, 0),
    ]

    panel_specs = [
        (mc_mean_var,
         r"$\mathbb{V}[\hat{\mu}_0]\;/\;\mathbb{V}_{\mathrm{MC}}$",
         f"(a) Mean report ($P={int(target_cost)}$)",
         4, 5),
        (mc_var_var,
         r"$\mathbb{V}[\hat{\sigma}_0^2]\;/\;\mathbb{V}_{\mathrm{MC}}$",
         f"(b) Variance report ($P={int(target_cost)}$)",
         6, 7),
    ]

    x = np.arange(len(alloc_specs))
    width = 0.6

    for panel_idx, (mc_ref, ylabel, title,
                    si, di) in enumerate(panel_specs):
        ax = axes[panel_idx]
        ratios = []
        labels = []
        colors = []
        for spec in alloc_specs:
            label, color, nps, subs = spec[0], spec[1], spec[2], spec[3]
            stat_report = spec[si]
            diag_idx = spec[di]
            v = _evaluate(stat_report, diag_idx, nps, subs)
            ratios.append(v / mc_ref)
            labels.append(label)
            colors.append(color)

        ax.bar(x, ratios, width, color=colors, edgecolor="k", lw=0.5)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y", which="both")
        ax.set_axisbelow(True)
