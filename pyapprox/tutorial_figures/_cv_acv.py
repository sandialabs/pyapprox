"""Plotting functions for control variate and ACV tutorials.

Covers: control_variate_concept.qmd, control_variate_analysis.qmd,
        acv_concept.qmd, acv_many_models_concept.qmd,
        acv_many_models_analysis.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# control_variate_concept.qmd — all echo:false -> Convention A
# ---------------------------------------------------------------------------

def plot_model_surfaces(benchmark, bkd, ax_3d, ax_scatter):
    """control_variate_concept.qmd -> fig-model-surfaces

    Overlaid HF/LF response surfaces and output correlation scatter.
    """
    from ._style import apply_style

    hf_model = benchmark.models()[0]
    lf_model = benchmark.models()[1]
    variable = benchmark.prior()

    # --- Surface grids ---
    n_grid = 40
    x1d = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(x1d, x1d)
    grid_samples = bkd.array(np.column_stack([X.ravel(), Y.ravel()]).T)

    Z_hf = bkd.to_numpy(hf_model(grid_samples)).reshape(n_grid, n_grid)
    Z_lf = bkd.to_numpy(lf_model(grid_samples)).reshape(n_grid, n_grid)

    # --- Random samples for scatter ---
    N_scatter = 100
    samples = variable.rvs(N_scatter)
    vals_hf = bkd.to_numpy(hf_model(samples)).ravel()
    vals_lf = bkd.to_numpy(lf_model(samples)).ravel()

    cov = bkd.to_numpy(benchmark.ensemble_covariance()[:2, :2])
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # --- 3D surface ---
    ax_3d.plot_surface(X, Y, Z_hf, cmap="Blues", alpha=0.6, edgecolor="none",
                       label=r"$f_\alpha$")
    ax_3d.plot_surface(X, Y, Z_lf, cmap="Oranges", alpha=0.6,
                       edgecolor="none", label=r"$f_\kappa$")
    ax_3d.set_xlabel("$x_1$")
    ax_3d.set_ylabel("$x_2$")
    ax_3d.set_zlabel("$f$")
    ax_3d.set_title(r"$f_\alpha$ (blue) and $f_\kappa$ (orange)", fontsize=11)

    # --- Scatter ---
    ax_scatter.scatter(vals_hf, vals_lf, s=20, alpha=0.6, color="#2C7FB8",
                       edgecolor="k", lw=0.3)
    lo = min(vals_hf.min(), vals_lf.min())
    hi = max(vals_hf.max(), vals_lf.max())
    ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.3)
    ax_scatter.set_xlabel(r"$f_\alpha$", fontsize=11)
    ax_scatter.set_ylabel(r"$f_\kappa$", fontsize=11)
    ax_scatter.set_title(rf"Output correlation $\rho = {rho:.2f}$",
                         fontsize=11)
    ax_scatter.set_aspect("equal")
    apply_style(ax_scatter)


def plot_variance_reduction_vs_rho(ax):
    """control_variate_concept.qmd -> fig-variance-reduction-vs-rho

    CVMC variance reduction factor (1-rho^2) vs model correlation.
    """
    from ._style import apply_style

    rho = np.linspace(-1, 1, 500)
    gamma = 1 - rho**2

    ax.plot(rho, gamma, color="#2C7FB8", lw=2.5)
    ax.axhline(1, color="k", ls="--", lw=1, alpha=0.4, label="No reduction (MC)")
    ax.fill_between(rho, gamma, 1, alpha=0.12, color="#2C7FB8",
                    label="Variance saved")

    for rho_mark, label in [(-0.9, r"$\rho = -0.9$"), (0.9, r"$\rho = 0.9$")]:
        g = 1 - rho_mark**2
        ax.plot(rho_mark, g, "o", color="#C0392B", ms=8, zorder=4)
        ax.annotate(
            f"{label}\n$\\gamma = {g:.2f}$",
            xy=(rho_mark, g),
            xytext=(rho_mark + 0.05 * np.sign(rho_mark), g + 0.08),
            fontsize=9,
            ha="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )

    ax.set_xlabel(r"Correlation $\rho_{\alpha\kappa}$", fontsize=12)
    ax.set_ylabel(r"Variance reduction factor $1 - \rho^2$", fontsize=12)
    ax.set_title("CVMC variance reduction vs model correlation", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10)
    apply_style(ax)


def plot_cvmc_histograms(benchmark, bkd, axes):
    """control_variate_concept.qmd -> fig-cvmc-histograms

    Distribution of MC vs CVMC mean estimates over independent trials.
    """
    from ._style import apply_style

    hf_model = benchmark.models()[0]
    lf_model = benchmark.models()[1]
    variable = benchmark.prior()

    cov = bkd.to_numpy(benchmark.ensemble_covariance()[:2, :2])
    eta_opt = -cov[0, 1] / cov[1, 1]
    means = bkd.to_numpy(benchmark.ensemble_means())
    mu_kappa = means[1, 0].item()
    mu_alpha_true = means[0, 0].item()
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    N = 100
    n_trials = 1000
    mc_means = np.empty(n_trials)
    cv_means = np.empty(n_trials)

    for i in range(n_trials):
        np.random.seed(i)
        samples = variable.rvs(N)
        v_alpha = bkd.to_numpy(hf_model(samples)).ravel()
        v_kappa = bkd.to_numpy(lf_model(samples)).ravel()
        mc_means[i] = v_alpha.mean()
        cv_means[i] = v_alpha.mean() + eta_opt * (v_kappa.mean() - mu_kappa)

    actual_reduction = cv_means.var() / mc_means.var()

    xlim = (
        min(mc_means.min(), cv_means.min()),
        max(mc_means.max(), cv_means.max()),
    )
    pad = 0.05 * (xlim[1] - xlim[0])
    xlim = (xlim[0] - pad, xlim[1] + pad)

    for ax, vals, color, label, subtitle in zip(
        axes,
        [mc_means, cv_means],
        ["#2C7FB8", "#E67E22"],
        [r"MC: $\hat{\mu}_\alpha$",
         r"CVMC: $\hat{\mu}_\alpha^{\mathrm{CV}}$"],
        [
            f"Var = {mc_means.var():.2e}",
            f"Var = {cv_means.var():.2e}  ({actual_reduction:.0%} of MC)",
        ],
    ):
        ax.hist(vals, bins=50, density=True, color=color, alpha=0.7,
                edgecolor="k", lw=0.3)
        ax.axvline(mu_alpha_true, color="#C0392B", ls="--", lw=2,
                   label="True mean")
        ax.set_xlabel(r"Estimate of $\mu_\alpha$", fontsize=11)
        ax.set_title(f"{label}\n{subtitle}", fontsize=10)
        ax.set_xlim(xlim)
        ax.legend(fontsize=9)
        apply_style(ax)

    axes[0].set_ylabel("Density", fontsize=11)

    return N, rho, n_trials


# ---------------------------------------------------------------------------
# control_variate_analysis.qmd — echo:true -> Convention B
# ---------------------------------------------------------------------------

def plot_cv_verification(rho_vals, theoretical_reductions, empirical_reductions,
                         n_trials, N, ax):
    """control_variate_analysis.qmd -> fig-verification

    Empirical CVMC variance reduction dots vs theoretical 1-rho^2 line.
    """
    from ._style import apply_style

    ax.plot(rho_vals, theoretical_reductions, "--", color="#C0392B", lw=1.8,
            label=r"Theory: $1 - \rho^2$")
    ax.scatter(rho_vals, empirical_reductions, color="#2C7FB8", s=60, zorder=4,
               label=f"Empirical ({n_trials} trials, $N={N}$)")
    ax.set_xlabel(r"Correlation $\rho_{\alpha\kappa}$", fontsize=12)
    ax.set_ylabel("Variance reduction factor", fontsize=12)
    ax.set_title("Theoretical vs empirical CVMC variance reduction",
                 fontsize=11)
    ax.legend(fontsize=10)
    apply_style(ax)


# ---------------------------------------------------------------------------
# acv_concept.qmd — all echo:false -> Convention A
# ---------------------------------------------------------------------------

def plot_unknown_mean_problem(benchmark, bkd, axes):
    """acv_concept.qmd -> fig-unknown-mean-problem

    Histograms showing effect of unknown mu_kappa: CVMC, ACV r=2, ACV r=20.
    """
    from ._style import apply_style

    hf_model = benchmark.models()[0]
    lf_model = benchmark.models()[1]
    variable = benchmark.prior()

    cov_mat = bkd.to_numpy(benchmark.ensemble_covariance()[:2, :2])
    eta_opt = -cov_mat[0, 1] / cov_mat[1, 1]
    mu_kappa_true = bkd.to_float(benchmark.ensemble_means()[1, 0])
    mu_alpha_true = bkd.to_float(benchmark.ensemble_means()[0, 0])

    N = 20
    n_trials = 2000

    scenarios = [
        ("CVMC\n($\\mu_\\kappa$ known)", None),
        (r"ACV, $r = 2$" + "\n(few LF samples)", 2),
        (r"ACV, $r = 20$" + "\n(many LF samples)", 20),
    ]

    all_estimates = {}
    for label, r in scenarios:
        ests = np.empty(n_trials)
        for i in range(n_trials):
            rng = np.random.default_rng(i)
            samples_shared = variable.rvs(N)
            v_hf = bkd.to_numpy(hf_model(samples_shared)).ravel()
            v_lf_shared = bkd.to_numpy(lf_model(samples_shared)).ravel()
            mu_hf = v_hf.mean()
            mu_lf_shared = v_lf_shared.mean()

            if r is None:
                mu_lf_large = mu_kappa_true
            else:
                samples_lf = variable.rvs(r * N)
                v_lf_large = bkd.to_numpy(lf_model(samples_lf)).ravel()
                mu_lf_large = v_lf_large.mean()

            ests[i] = mu_hf + eta_opt * (mu_lf_shared - mu_lf_large)
        all_estimates[label] = ests

    # Shared x-axis
    all_vals = np.concatenate(list(all_estimates.values()))
    pad = 0.04 * (all_vals.max() - all_vals.min())
    xlim = (all_vals.min() - pad, all_vals.max() + pad)

    colors = ["#27AE60", "#E67E22", "#2C7FB8"]

    for ax, (label, r), color in zip(axes, scenarios, colors):
        ests = all_estimates[label]
        ax.hist(ests, bins=60, density=True, color=color, alpha=0.75,
                edgecolor="k", lw=0.2)
        ax.axvline(mu_alpha_true, color="#C0392B", ls="--", lw=2,
                   label="True mean")
        ax.set_xlabel(r"Estimate of $\mu_\alpha$", fontsize=11)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(xlim)
        apply_style(ax)
        ax.text(
            0.97, 0.95,
            f"Std = {ests.std():.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].legend(fontsize=9)

    return N, n_trials


def plot_acv_variance_reduction_vs_r(ax):
    """acv_concept.qmd -> fig-variance-reduction-vs-r

    ACV variance reduction factor vs LF-to-HF sample ratio r for
    several correlation values.
    """
    from ._style import apply_style

    r_vals = np.linspace(1.01, 30, 400)
    rho_vals = [0.7, 0.9, 0.99]
    colors = ["#2C7FB8", "#E67E22", "#27AE60"]

    for rho, color in zip(rho_vals, colors):
        gamma = 1 - (r_vals - 1) / r_vals * rho**2
        cvmc_limit = 1 - rho**2
        ax.plot(r_vals, gamma, color=color, lw=2,
                label=rf"$\rho = {rho}$")
        ax.axhline(cvmc_limit, color=color, lw=1.2, ls="--", alpha=0.6)

    ax.axhline(1.0, color="k", lw=1, ls=":", alpha=0.4,
               label="No reduction (MC)")
    ax.set_xlabel(r"Sample ratio $r = rN / N$", fontsize=12)
    ax.set_ylabel(r"Variance reduction factor $\gamma$", fontsize=12)
    ax.set_title("ACV variance reduction vs LF sample ratio", fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=10)
    ax.annotate(
        "CVMC limits\n(dashed)",
        xy=(28, 1 - 0.99**2 + 0.01),
        fontsize=8, color="gray", ha="right",
    )
    apply_style(ax)


# ---------------------------------------------------------------------------
# acv_many_models_concept.qmd — all echo:false -> Convention A
# ---------------------------------------------------------------------------

def plot_direct_vs_indirect(axes):
    """acv_many_models_concept.qmd -> fig-direct-vs-indirect

    Sample-set wiring diagrams for MFMC (indirect) and ACVMF (direct).
    """
    import matplotlib.patches as mpatches

    colors = ["#e74c3c", "#2C7FB8", "#27AE60", "#8E44AD"]
    labels = [r"$f_0$ (HF)", r"$f_1$ (LF)", r"$f_2$ (LF)", r"$f_3$ (LF)"]
    bw, bh = 0.9, 0.40

    def draw_box(ax, x, y, text, fc, ec, lw=1.2, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.05",
            fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                fontweight="bold" if bold else "normal", zorder=4)

    def draw_arrow(ax, x0, y0, x1, y1, color):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->, head_width=0.2",
                                    color=color, lw=1.8))

    # -- Left panel: MFMC (vertical chain) --
    ax = axes[0]
    chain_x_z, chain_x_p = 0.8, 2.0
    chain_ys = [3.0, 2.0, 1.0, 0.0]
    for i, (y, col, lbl) in enumerate(zip(chain_ys, colors, labels)):
        fc = colors[0] if i == 0 else "white"
        ec = colors[0] if i == 0 else col
        ztxt = (r"$\mathcal{Z}_0$" if i == 0
                else r"$\mathcal{Z}_{" + str(i) + r"}^*$")
        draw_box(ax, chain_x_z, y, ztxt, fc, ec, lw=2.0 if i == 0 else 1.2,
                 bold=(i == 0))
        if i > 0:
            draw_box(ax, chain_x_p, y,
                     r"$\mathcal{P}_{" + str(i) + r"}$",
                     col, col, lw=1.2)
        ax.text(chain_x_z - bw / 2 - 0.08, y, lbl, ha="right", va="center",
                fontsize=9, color=col, fontweight="bold")
    for i in range(1, 4):
        draw_arrow(ax, chain_x_z - 0.15, chain_ys[i] + bh / 2 + 0.04,
                   chain_x_z - 0.15, chain_ys[i - 1] - bh / 2 - 0.04,
                   colors[i])
    ax.set_xlim(-0.7, 2.8)
    ax.set_ylim(-0.6, 3.6)
    ax.set_title("MFMC: indirect (chained)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # -- Right panel: ACVMF (flat LF row) --
    ax = axes[1]
    hf_x, hf_y = 1.1, 2.8
    lf_xs = [0.0, 1.1, 2.2]
    lf_y_z, lf_y_p = 1.2, 0.2

    draw_box(ax, hf_x, hf_y, r"$\mathcal{Z}_0$", colors[0], colors[0],
             lw=2.0, bold=True)
    ax.text(hf_x, hf_y + bh / 2 + 0.15, labels[0], ha="center",
            va="bottom", fontsize=9, color=colors[0], fontweight="bold")

    for j, (lx, col, lbl) in enumerate(zip(lf_xs, colors[1:], labels[1:])):
        alpha = j + 1
        draw_box(ax, lx, lf_y_z,
                 r"$\mathcal{Z}_{" + str(alpha) + r"}^*$",
                 "white", col)
        draw_box(ax, lx, lf_y_p,
                 r"$\mathcal{P}_{" + str(alpha) + r"}$",
                 col, col)
        ax.text(lx, lf_y_z + bh / 2 + 0.12, lbl, ha="center", va="bottom",
                fontsize=9, color=col, fontweight="bold")
        draw_arrow(ax, lx, lf_y_z + bh / 2 + 0.04,
                   hf_x, hf_y - bh / 2 - 0.04, col)

    ax.set_xlim(-0.8, 3.0)
    ax.set_ylim(-0.4, 3.6)
    ax.set_title(r"ACVMF: direct (all $\rightarrow\,f_0$)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")


def plot_acv_ceiling(benchmark, bkd, ax):
    """acv_many_models_concept.qmd -> fig-acv-ceiling

    Variance/MC variance vs total cost for MLMC, MFMC, ACVMF with CV-k
    limit lines.
    """
    from pyapprox.statest.statistics import MultiOutputMean
    from pyapprox.statest import MLMCEstimator, MFMCEstimator, GMFEstimator

    models = benchmark.models()
    costs = benchmark.costs()
    nqoi = models[0].nqoi()
    cov = bkd.to_numpy(benchmark.ensemble_covariance())
    sigma2_hf = cov[0, 0]
    n_models = len(models)
    M = n_models - 1

    # -- Multi-model CV limits --
    cv_limits = []
    for k in range(1, n_models):
        sub = cov[:k + 1, :k + 1]
        c0l = sub[0, 1:]
        Sl = sub[1:, 1:]
        r2 = c0l @ np.linalg.solve(Sl, c0l) / sigma2_hf
        cv_limits.append(1 - r2)

    # -- Scan: fix nhf=1, grow LF partition sizes --
    nhf_samples = 1
    mc_var = sigma2_hf / nhf_samples
    partition_ratio_base = np.array([2, 2, 2, 2])
    factors = np.arange(22)

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(bkd.asarray(cov))
    ri_zeros = bkd.zeros(M, dtype=int)

    def compute_variance_vs_cost(estimator):
        nparts = estimator._npartitions
        est_costs, est_ratios = [], []
        for factor in factors:
            ratios = bkd.asarray(
                partition_ratio_base[:nparts - 1] * 2**factor, dtype=float
            )
            model_ratios = estimator._partition_ratios_to_model_ratios(ratios)
            target_cost = bkd.to_float(
                nhf_samples * (costs[0] + bkd.dot(model_ratios, costs[1:]))
            )
            est_cov = estimator.covariance_from_ratios(target_cost, ratios)
            est_var = bkd.to_float(est_cov[0, 0])
            est_costs.append(target_cost)
            est_ratios.append(est_var / mc_var)
        return est_costs, est_ratios

    mlmc_costs, mlmc_ratio = compute_variance_vs_cost(
        MLMCEstimator(stat, costs))
    mfmc_costs, mfmc_ratio = compute_variance_vs_cost(
        MFMCEstimator(stat, costs))
    acvmf_costs, acvmf_ratio = compute_variance_vs_cost(
        GMFEstimator(stat, costs, recursion_index=ri_zeros))

    # -- Plot --
    cv_colors = ["#e74c3c", "#8e44ad", "#16a085", "#2c3e50"]
    cv_ls = ["--", "-.", ":", (0, (5, 1))]

    ax.loglog(mlmc_costs, mlmc_ratio, "-", color="#2C7FB8", lw=2.5,
              label="MLMC")
    ax.loglog(mfmc_costs, mfmc_ratio, "--", color="#E67E22", lw=2.5,
              label="MFMC")
    ax.loglog(acvmf_costs, acvmf_ratio, "-.", color="#27AE60", lw=2.5,
              label="ACVMF (direct correction)")

    for k, (lim, col, ls) in enumerate(zip(cv_limits, cv_colors, cv_ls)):
        ax.axhline(lim, color=col, ls=ls, lw=1.5,
                   label=f"CV-{k + 1} limit "
                         f"({k + 1} LF model{'s' if k else ''})")

    ax.axhline(1.0, color="k", lw=1, ls=":", alpha=0.4,
               label="MC (no reduction)")
    ax.set_xlabel("Total cost", fontsize=12)
    ax.set_ylabel("Variance / MC variance", fontsize=12)
    ax.set_title("Direct correction breaks the CV-1 ceiling", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.2, which="both")


# ---------------------------------------------------------------------------
# acv_many_models_analysis.qmd — echo:true -> Convention B
# ---------------------------------------------------------------------------

def plot_acv_two_model_verification(r_values, empirical_reductions, rho,
                                    N, n_trials, ax):
    """acv_many_models_analysis.qmd -> fig-acv-verification

    ACV two-model variance reduction: theory line vs empirical dots.
    """
    from ._style import apply_style

    r_fine = np.linspace(1.01, 55, 300)
    theory_line = 1 - (r_fine - 1) / r_fine * rho**2

    ax.plot(r_fine, theory_line, "--", color="#C0392B", lw=1.8,
            label=r"Theory: $1 - \frac{r-1}{r}\rho^2$")
    ax.scatter(r_values, empirical_reductions, color="#2C7FB8", s=60, zorder=4,
               label=f"Empirical ($N={N}$, {n_trials} trials)")
    ax.axhline(1 - rho**2, color="#27AE60", ls=":", lw=1.5,
               label=r"CVMC limit: $1 - \rho^2$")
    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.3,
               label="MC (no reduction)")
    ax.set_xlabel(r"Sample ratio $r$", fontsize=12)
    ax.set_ylabel("Variance reduction factor $\\gamma$", fontsize=12)
    ax.set_title("ACV variance reduction: theory vs empirical", fontsize=11)
    ax.legend(fontsize=9)
    apply_style(ax)


def plot_allocation_matrix(ax, est, bkd, show_sizes=True):
    """acv_many_models_analysis.qmd -> fig-allocation-matrices (helper)

    Single allocation matrix heatmap for one estimator.
    """
    A = bkd.to_numpy(est._get_allocation_matrix())
    p = bkd.to_numpy(est.npartition_samples())
    nrows, ncols = A.shape
    ax.imshow(A, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    if show_sizes:
        for i in range(nrows):
            for j in range(ncols):
                if A[i, j] > 0:
                    ax.text(j, i, f"{int(p[i])}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([rf"$\mathcal{{P}}_{{{m}}}$" for m in range(nrows)],
                       fontsize=8)
    col_labels = []
    for alpha in range(ncols // 2):
        col_labels += [rf"$Z_{{{alpha}}}^*$", rf"$Z_{{{alpha}}}$"]
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")


def plot_allocation_matrices(estimators, bkd, axes):
    """acv_many_models_analysis.qmd -> fig-allocation-matrices

    Allocation matrix heatmaps for MLMC, MFMC, ACVMF, ACVIS side by side.
    """
    for ax, (name, est) in zip(axes, estimators.items()):
        plot_allocation_matrix(ax, est, bkd)
        ax.set_title(name, fontsize=12, pad=8)


def plot_variance_verification(target_costs_sweep, mc_vars_pred, mc_vars_emp,
                               mfmc_vars_pred, mfmc_vars_emp,
                               acvmf_vars_pred, acvmf_vars_emp,
                               n_trials, ax):
    """acv_many_models_analysis.qmd -> fig-variance-verify

    MC, MFMC, ACVMF variance vs cost: predicted lines and empirical markers.
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
    ax.loglog(target_costs_sweep, acvmf_vars_pred, "-", color="#8E44AD", lw=2,
              label="ACVMF (predicted)")
    ax.loglog(target_costs_sweep, acvmf_vars_emp, "D", color="#8E44AD", ms=8,
              label="ACVMF (empirical)")
    ax.set_xlabel("Total cost $P$", fontsize=12)
    ax.set_ylabel("Estimator variance", fontsize=12)
    ax.set_title(f"Predicted vs empirical variance  ({n_trials} trials)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
