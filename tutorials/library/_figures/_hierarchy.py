"""Plotting functions for multi-level and multi-fidelity hierarchy tutorials.

Covers: mlmc_concept.qmd, mlmc_analysis.qmd,
        mfmc_concept.qmd, mfmc_analysis.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# mlmc_concept.qmd — echo:false figures → Convention A
# ---------------------------------------------------------------------------

def plot_level_variances(ax1, ax2):
    """mlmc_concept.qmd -> fig-level-variances

    Variance of level corrections vs cost per sample for the 5-model polynomial hierarchy.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks.instances.multifidelity.polynomial_ensemble import (
        polynomial_ensemble_5model,
    )
    from ._style import apply_style

    bkd = NumpyBkd()
    benchmark = polynomial_ensemble_5model(bkd)
    variable = benchmark.prior()
    models = benchmark.models()
    costs = bkd.to_numpy(benchmark.costs())
    M = len(models) - 1

    np.random.seed(0)
    N_est = 50000
    samples = variable.rvs(N_est)
    values = [bkd.to_numpy(m(samples)).ravel() for m in models]

    # Variance of each correction Y_ell = f_ell - f_{ell+1}  (f_{M+1} = 0)
    diff_vars = []
    for ell in range(M):
        diff_vars.append(np.var(values[ell] - values[ell + 1]))
    diff_vars.append(np.var(values[M]))

    level_labels = [rf"$Y_{{\ell={ell}}}$" for ell in range(M + 1)]
    x = np.arange(M + 1)

    ax1.bar(x, diff_vars, color="#2C7FB8", alpha=0.75, width=0.5,
            edgecolor="k", lw=0.5, label=r"$\mathrm{Var}[Y_\ell]$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(level_labels, fontsize=12)
    ax1.set_ylabel(r"Variance of correction $\mathrm{Var}[Y_\ell]$",
                   fontsize=11, color="#2C7FB8")
    ax1.tick_params(axis="y", labelcolor="#2C7FB8")
    ax1.set_yscale("log")
    ax1.set_xlabel("Level $\\ell$  (0 = HF, 4 = cheapest)", fontsize=12)

    ax2.plot(x, costs, "o-", color="#C0392B", lw=2, ms=8,
             label="Cost per sample $C_\\ell$")
    ax2.set_ylabel("Cost per sample $C_\\ell$", fontsize=11, color="#C0392B")
    ax2.tick_params(axis="y", labelcolor="#C0392B")
    ax2.set_yscale("log")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10,
               loc="upper right")
    ax1.set_title("Level correction variances vs cost per sample", fontsize=11)
    apply_style(ax1)


def plot_mlmc_vs_mc(axes):
    """mlmc_concept.qmd -> fig-mlmc-vs-mc

    Distribution of MC and MLMC mean estimates at equal total cost.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks.instances.multifidelity.polynomial_ensemble import (
        polynomial_ensemble_5model,
    )
    from pyapprox.statest.statistics import MultiOutputMean
    from pyapprox.statest import MLMCEstimator
    from pyapprox.statest.mc_estimator import MCEstimator
    from ._style import apply_style

    bkd = NumpyBkd()
    np.random.seed(0)

    benchmark = polynomial_ensemble_5model(bkd)
    variable = benchmark.prior()
    models = benchmark.models()
    costs = benchmark.costs()
    nqoi = models[0].nqoi()

    true_mean = bkd.to_float(benchmark.ensemble_means()[0, 0])

    # Pilot to estimate covariance
    N_pilot = 500
    s_pilot = variable.rvs(N_pilot)
    vals_pilot = [m(s_pilot) for m in models]
    stat = MultiOutputMean(nqoi, bkd)
    cov_pilot, = stat.compute_pilot_quantities(vals_pilot)
    stat.set_pilot_quantities(cov_pilot)

    target_cost = 50.0
    n_trials = 1000

    # MC
    mc_stat = MultiOutputMean(nqoi, bkd)
    mc_stat.set_pilot_quantities(cov_pilot[0:1, 0:1])
    mc_est = MCEstimator(mc_stat, costs[0:1])
    mc_est.allocate_samples(target_cost)
    mc_means = np.empty(n_trials)
    for i in range(n_trials):
        np.random.seed(i)
        s = variable.rvs(mc_est._rounded_nsamples_per_model[0])
        mc_means[i] = float(bkd.mean(models[0](s)[0]))

    # MLMC
    mlmc_est = MLMCEstimator(stat, costs)
    mlmc_est.allocate_samples(target_cost)
    mlmc_means = np.empty(n_trials)
    for i in range(n_trials):
        np.random.seed(i + 10000)
        samples_per_model = mlmc_est.generate_samples_per_model(
            lambda n: variable.rvs(int(n))
        )
        vals = [models[ell](samples_per_model[ell])
                for ell in range(len(models))]
        mlmc_means[i] = float(mlmc_est(vals))

    all_vals = np.concatenate([mc_means, mlmc_means])
    pad = 0.04 * (all_vals.max() - all_vals.min())
    xlim = (all_vals.min() - pad, all_vals.max() + pad)

    for ax, ests, color, label in zip(
        axes,
        [mc_means, mlmc_means],
        ["#2C7FB8", "#E67E22"],
        [f"MC\n(N = {mc_est._rounded_nsamples_per_model[0]})",
         "MLMC\n(optimal allocation)"],
    ):
        ax.hist(ests, bins=50, density=True, color=color, alpha=0.75,
                edgecolor="k", lw=0.2)
        ax.axvline(true_mean, color="#C0392B", ls="--", lw=2,
                   label="True mean")
        ax.set_xlabel(r"Estimate of $\mathbb{E}[f_0]$", fontsize=11)
        ax.set_title(f"{label}\nStd = {ests.std():.4f}", fontsize=10)
        ax.set_xlim(xlim)
        ax.legend(fontsize=9)
        apply_style(ax)

    axes[0].set_ylabel("Density", fontsize=11)

    return target_cost, n_trials


# ---------------------------------------------------------------------------
# mlmc_analysis.qmd — echo:true figures → Convention B
# ---------------------------------------------------------------------------

def plot_variance_vs_cost(target_costs_sweep, mc_vars_pred, mc_vars_emp,
                          mlmc_vars_pred, mlmc_vars_emp, n_trials, ax):
    """mlmc_analysis.qmd -> fig-variance-vs-cost

    MLMC and MC variance vs total cost: predicted lines and empirical markers.
    """
    from ._style import apply_style

    ax.loglog(target_costs_sweep, mc_vars_pred, "-", color="#2C7FB8", lw=2,
              label="MC (predicted)")
    ax.loglog(target_costs_sweep, mc_vars_emp, "o", color="#2C7FB8", ms=8,
              label="MC (empirical)")
    ax.loglog(target_costs_sweep, mlmc_vars_pred, "-", color="#E67E22", lw=2,
              label="MLMC (predicted)")
    ax.loglog(target_costs_sweep, mlmc_vars_emp, "s", color="#E67E22", ms=8,
              label="MLMC (empirical)")
    ax.set_xlabel("Total cost $P$", fontsize=12)
    ax.set_ylabel(r"Estimator variance", fontsize=12)
    ax.set_title(f"Predicted vs empirical variance  ({n_trials} trials)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")


def plot_mlmc_vs_opt(labels, vals, colors, ax):
    """mlmc_analysis.qmd -> fig-mlmc-vs-opt

    Bar chart comparing variance reduction of MLMC vs ACV-GRD vs MC.
    """
    from ._style import apply_style

    bars = ax.bar(labels, vals, color=colors, edgecolor="k", lw=0.5,
                  alpha=0.85)
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel("Variance reduction vs MC", fontsize=12)
    ax.set_title("Standard MLMC vs optimal-weight MLMC", fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.1f}x",
                ha="center", fontsize=10)
    apply_style(ax)


# ---------------------------------------------------------------------------
# mfmc_concept.qmd — echo:false figure → Convention A
# ---------------------------------------------------------------------------

def plot_sample_structure(axes):
    """mfmc_concept.qmd -> fig-sample-structure

    Sample allocation diagram for MLMC (independent) vs MFMC (nested).
    """
    import matplotlib.patches as mpatches
    from ._style import apply_style

    np.random.seed(42)

    mlmc_N = [3, 8, 28]
    mfmc_N = [3, 10, 40]
    n_models = 3
    mnames = [r"$f_0$  (HF)", r"$f_1$", r"$f_2$  (cheap)"]

    # -- MLMC --
    ax = axes[0]
    mlmc_colors = ["#C0392B", "#E67E22", "#2C7FB8"]
    offset = 0
    for m in range(n_models):
        N = mlmc_N[m]
        xs = np.arange(offset, offset + N)
        ax.barh([m] * N, [0.88] * N, left=xs, height=0.6,
                color=mlmc_colors[m], edgecolor="k", lw=0.4, alpha=0.85)
        offset += N + 1.5
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(mnames, fontsize=11)
    ax.set_xlabel("Sample index", fontsize=11)
    ax.set_title("MLMC: independent sets per level", fontsize=11, pad=8)
    ax.set_xlim(-0.5, offset - 0.5)
    ax.set_ylim(-0.5, n_models - 0.4)
    ax.legend(handles=[mpatches.Patch(color=c, label=f"Level {m} only")
                       for m, c in enumerate(mlmc_colors)], fontsize=9)
    ax.grid(alpha=0.12, axis="x")

    # -- MFMC --
    ax = axes[1]
    n_hf = mfmc_N[0]
    n_lf1_ex = mfmc_N[1] - mfmc_N[0]
    n_lf2_ex = mfmc_N[2] - mfmc_N[1]
    c_shared = "#C0392B"
    c_lf12 = "#E67E22"
    c_lf2 = "#2C7FB8"

    for m in range(n_models):
        ax.barh([m] * n_hf, [0.88] * n_hf, left=np.arange(n_hf),
                height=0.6, color=c_shared, edgecolor="k", lw=0.4,
                alpha=0.85)
    for m in range(1, n_models):
        ax.barh([m] * n_lf1_ex, [0.88] * n_lf1_ex,
                left=np.arange(n_hf, n_hf + n_lf1_ex),
                height=0.6, color=c_lf12, edgecolor="k", lw=0.4, alpha=0.85)
    ax.barh([2] * n_lf2_ex, [0.88] * n_lf2_ex,
            left=np.arange(n_hf + n_lf1_ex, mfmc_N[2]),
            height=0.6, color=c_lf2, edgecolor="k", lw=0.4, alpha=0.85)

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(mnames, fontsize=11)
    ax.set_xlabel("Sample index", fontsize=11)
    ax.set_title("MFMC: nested sets, HF samples always shared", fontsize=11,
                 pad=8)
    ax.set_xlim(-0.5, mfmc_N[-1] + 0.5)
    ax.set_ylim(-0.5, n_models - 0.4)
    ax.legend(handles=[
        mpatches.Patch(color=c_shared,
                       label=r"$\mathcal{Z}_0$: HF + all LF"),
        mpatches.Patch(color=c_lf12,
                       label=r"$\mathcal{Z}_1\!\setminus\!\mathcal{Z}_0$"
                             r": LF1 + LF2"),
        mpatches.Patch(color=c_lf2,
                       label=r"$\mathcal{Z}_2\!\setminus\!\mathcal{Z}_1$"
                             r": LF2 only"),
    ], fontsize=9)
    ax.grid(alpha=0.12, axis="x")


# ---------------------------------------------------------------------------
# mfmc_analysis.qmd — echo:true figure → Convention B
# ---------------------------------------------------------------------------

def plot_mfmc_variance_verify(target_costs_sweep, mc_vars_pred, mc_vars_emp,
                              mfmc_vars_pred, mfmc_vars_emp, n_trials, ax):
    """mfmc_analysis.qmd -> fig-variance-verify

    MFMC and MC variance vs total cost: predicted lines and empirical markers.
    """
    from ._style import apply_style

    ax.loglog(target_costs_sweep, mc_vars_pred, "-", color="#aaaaaa", lw=2,
              label="MC (predicted)")
    ax.loglog(target_costs_sweep, mc_vars_emp, "o", color="#aaaaaa", ms=8,
              label="MC (empirical)")
    ax.loglog(target_costs_sweep, mfmc_vars_pred, "-", color="#E67E22", lw=2,
              label="MFMC (predicted)")
    ax.loglog(target_costs_sweep, mfmc_vars_emp, "s", color="#E67E22", ms=8,
              label="MFMC (empirical)")
    ax.set_xlabel("Total cost $P$", fontsize=12)
    ax.set_ylabel(r"Estimator variance", fontsize=12)
    ax.set_title(f"Predicted vs empirical variance  ({n_trials} trials)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")
