"""Plotting functions for sparse grid tutorials.

Covers: multifidelity_sparse_grids.qmd, uq_with_sparse_grids.qmd,
        sparse_grid_to_pce_conversion.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# multifidelity_sparse_grids.qmd
# ---------------------------------------------------------------------------

def plot_model_hierarchy(
    z_plot, epsilons, bkd, result_f0, result_delta, s0, sd, z_eval,
    ax_models, ax_discrep, ax_recon,
):
    """multifidelity_sparse_grids.qmd -> fig-model-hierarchy

    Two-level model hierarchy: models, discrepancy, and reconstruction.
    """
    # Left: both models
    for alpha in [0, 1]:
        f_alpha = np.cos(np.pi * (z_plot + 1) / 2 + epsilons[alpha])
        ax_models.plot(
            z_plot, f_alpha, lw=2,
            label=f"$f_{alpha}$ ($\\epsilon={epsilons[alpha]}$)",
        )
    ax_models.set_xlabel("$z$", fontsize=12)
    ax_models.set_ylabel("$f_\\alpha(z)$", fontsize=12)
    ax_models.set_title("Model hierarchy", fontsize=12)
    ax_models.legend(fontsize=10)
    ax_models.grid(True, alpha=0.3)

    # Middle: discrepancy
    f0 = np.cos(np.pi * (z_plot + 1) / 2 + epsilons[0])
    f1 = np.cos(np.pi * (z_plot + 1) / 2 + epsilons[1])
    delta = f1 - f0
    ax_discrep.plot(z_plot, f1, "--", lw=2, color="#E74C3C",
                    label="$f_1$ (exact)")
    ax_discrep.plot(z_plot, delta, lw=2, color="#2C7FB8",
                    label="$\\delta = f_1 - f_0$")
    ax_discrep.set_xlabel("$z$", fontsize=12)
    ax_discrep.set_title("Discrepancy is smoother", fontsize=12)
    ax_discrep.legend(fontsize=10)
    ax_discrep.grid(True, alpha=0.3)

    # Right: reconstruction
    approx_f0 = bkd.to_numpy(result_f0.surrogate(z_eval)).ravel()
    approx_delta = bkd.to_numpy(result_delta.surrogate(z_eval)).ravel()
    approx_ml = approx_f0 + approx_delta

    ax_recon.plot(z_plot, f1, "--", lw=2, color="#E74C3C",
                  label="$f_1$ (exact)")
    ax_recon.plot(z_plot, approx_ml, lw=2, color="#27AE60",
                  label="$\\hat{f}_0 + \\hat{\\delta}$ (8 evals)")
    pts0 = bkd.to_numpy(s0[0])
    ptsd = bkd.to_numpy(sd[0])
    ax_recon.plot(pts0, np.cos(np.pi * (pts0 + 1) / 2 + epsilons[0]),
                  "ko", ms=7, label=f"$f_0$ evals ({len(pts0)})")
    ax_recon.plot(ptsd,
                  np.cos(np.pi * (ptsd + 1) / 2 + epsilons[1])
                  - np.cos(np.pi * (ptsd + 1) / 2 + epsilons[0]),
                  "bs", ms=7, label=f"$\\delta$ evals ({len(ptsd)})")
    ax_recon.set_xlabel("$z$", fontsize=12)
    ax_recon.set_title("Multi-level reconstruction", fontsize=12)
    ax_recon.legend(fontsize=9)
    ax_recon.grid(True, alpha=0.3)


def plot_config_vars(axes):
    """multifidelity_sparse_grids.qmd -> fig-config-vars

    Configuration variables as mesh refinement levels.
    """
    import matplotlib.pyplot as plt

    configs = [
        (3, 3, "$\\alpha_x=0,\\; \\alpha_y=0$\n(coarse $\\times$ coarse)"),
        (7, 3, "$\\alpha_x=1,\\; \\alpha_y=0$\n(fine $x$ $\\times$ coarse $y$)"),
        (7, 7, "$\\alpha_x=1,\\; \\alpha_y=1$\n(fine $\\times$ fine)"),
    ]
    for ax_i, (nx, ny, label) in zip(axes, configs):
        for i in range(nx):
            for j in range(ny):
                ax_i.add_patch(plt.Rectangle(
                    (i / nx, j / ny), 1 / nx, 1 / ny,
                    facecolor="white", edgecolor="steelblue", linewidth=1.2,
                ))
        ax_i.set_xlim(0, 1)
        ax_i.set_ylim(0, 1)
        ax_i.set_aspect("equal")
        ax_i.set_title(label, fontsize=11)
        ax_i.set_xlabel("$x$", fontsize=11)
        ax_i.set_ylabel("$y$", fontsize=11)


def plot_mf_indices_1d(ax, sel_np):
    """multifidelity_sparse_grids.qmd -> fig-mf-indices-1d

    Selected multi-indices for 1D physical + 1 config variable.
    """
    from pyapprox.surrogates.affine.indices import (
        plot_indices_2d, format_index_axes,
    )

    plot_indices_2d(
        ax, sel_np,
        colors=lambda idx: "#2C7FB8" if idx[-1] == 0 else "#E74C3C",
        labels=True, box_height=0.6, label_fontsize=9,
    )
    format_index_axes(
        ax, sel_np,
        axis_labels=["Physical level $\\beta$", "Config level $\\alpha$"],
    )
    ax.set_title("Selected multi-indices [physical, config]", fontsize=12)


def plot_mf_animation(fig, ax_idx, ax_fn, snapshots, f1_exact, z_plot_arr):
    """multifidelity_sparse_grids.qmd -> fig-mf-animation

    Animated multi-index refinement with selected/candidate indices.
    """
    import matplotlib.animation as animation
    from matplotlib.patches import Patch
    from IPython.display import HTML
    from pyapprox.surrogates.affine.indices import plot_index_sets

    _mf_color = lambda idx: "#2C7FB8" if idx[-1] == 0 else "#E74C3C"
    _mf_cand_color = lambda idx: "#93C5E8" if idx[-1] == 0 else "#F5A9A9"

    # Compute fixed axis limits from the union of all frames
    _all_idx = np.hstack(
        [s[0] for s in snapshots]
        + [s[1] for s in snapshots if s[1].shape[1] > 0]
    )
    _anim_max_indices = list(_all_idx.max(axis=1).astype(int))

    legend_handles = [
        Patch(facecolor="#2C7FB8", edgecolor="black", alpha=0.7,
              label="Selected ($\\alpha=0$)"),
        Patch(facecolor="#E74C3C", edgecolor="black", alpha=0.7,
              label="Selected ($\\alpha=1$)"),
        Patch(facecolor="#93C5E8", edgecolor="black", alpha=0.4,
              linestyle="--", label="Candidate ($\\alpha=0$)"),
        Patch(facecolor="#F5A9A9", edgecolor="black", alpha=0.4,
              linestyle="--", label="Candidate ($\\alpha=1$)"),
    ]

    def animate(ii):
        ax_idx.clear()
        ax_fn.clear()
        sel_idx, cand_idx, approx_vals, err = snapshots[ii]

        plot_index_sets(
            ax_idx, sel_idx, cand_idx,
            selected_colors=_mf_color, candidate_colors=_mf_cand_color,
            selected_labels=True, candidate_labels=True,
            box_height=0.6, selected_alpha=0.7, candidate_alpha=0.4,
            axis_labels=["Physical level $\\beta$",
                         "Config level $\\alpha$"],
            max_indices=_anim_max_indices,
        )
        ax_idx.legend(handles=legend_handles, fontsize=8, loc="upper right")
        ax_idx.set_title(
            f"Step {ii + 1}: {sel_idx.shape[1]} selected, "
            f"{cand_idx.shape[1]} candidates", fontsize=12,
        )

        ax_fn.plot(z_plot_arr, f1_exact, "--", lw=2,
                   color="#E74C3C", label="$f_1$ (exact)")
        ax_fn.plot(z_plot_arr, approx_vals, lw=2,
                   color="#27AE60", label="surrogate")
        ax_fn.set_xlabel("$z$", fontsize=12)
        ax_fn.set_ylabel("$f(z)$", fontsize=12)
        ax_fn.set_title(f"Error estimate: {err:.2e}", fontsize=12)
        ax_fn.legend(fontsize=10, loc="upper right")
        ax_fn.set_ylim(-1.3, 1.3)
        ax_fn.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(
        fig, animate, frames=len(snapshots), interval=800, repeat_delay=2000,
    )
    return HTML(ani.to_jshtml())


def plot_mf_manual_convergence(ax, steps, error_estimates, test_errors):
    """multifidelity_sparse_grids.qmd -> fig-mf-manual-convergence

    Multi-index refinement convergence: error estimate vs true error.
    """
    ax.semilogy(steps, error_estimates, "o-", lw=2, ms=6, color="steelblue",
                label="Error estimate")
    ax.semilogy(steps, test_errors, "s--", lw=2, ms=6, color="#E74C3C",
                label="True test error")
    ax.axhline(1e-5, color="gray", ls=":", lw=1, label="$10^{-5}$ target")
    ax.set_xlabel("Refinement step", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Multi-index refinement convergence", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def plot_cost_weighted_animation(
    fig, axes, snaps_uw, snaps_cw, n_frames,
):
    """multifidelity_sparse_grids.qmd -> fig-cost-weighted-animation

    Side-by-side comparison of unweighted vs cost-weighted refinement.
    """
    import matplotlib.animation as animation
    from matplotlib.patches import Patch
    from IPython.display import HTML
    from pyapprox.surrogates.affine.indices import plot_index_sets

    _mf_color = lambda idx: "#2C7FB8" if idx[-1] == 0 else "#E74C3C"
    _mf_cand_color = lambda idx: "#93C5E8" if idx[-1] == 0 else "#F5A9A9"

    # Compute fixed axis limits from both runs
    _cw_all = np.hstack(
        [s[0] for snaps in [snaps_uw, snaps_cw] for s in snaps]
        + [s[1] for snaps in [snaps_uw, snaps_cw]
           for s in snaps if s[1].shape[1] > 0]
    )
    _cw_max_indices = list(_cw_all.max(axis=1).astype(int))

    legend_handles_cw = [
        Patch(facecolor="#2C7FB8", edgecolor="black", alpha=0.7,
              label="Selected ($\\alpha=0$)"),
        Patch(facecolor="#E74C3C", edgecolor="black", alpha=0.7,
              label="Selected ($\\alpha=1$)"),
        Patch(facecolor="#93C5E8", edgecolor="black", alpha=0.4,
              linestyle="--", label="Candidate ($\\alpha=0$)"),
        Patch(facecolor="#F5A9A9", edgecolor="black", alpha=0.4,
              linestyle="--", label="Candidate ($\\alpha=1$)"),
    ]

    def animate_cw(ii):
        axes[0].clear()
        axes[1].clear()
        sel_uw, cand_uw, _, err_uw, cost_uw = snaps_uw[ii]
        sel_cw, cand_cw, _, err_cw, cost_cw = snaps_cw[ii]
        for ax_i, sel_i, cand_i, title_i in [
            (axes[0], sel_uw, cand_uw,
             f"Unweighted — step {ii+1}\n"
             f"cost={cost_uw:.1f}, error={err_uw:.1e}"),
            (axes[1], sel_cw, cand_cw,
             f"Cost-weighted (base=4) — step {ii+1}\n"
             f"cost={cost_cw:.1f}, error={err_cw:.1e}"),
        ]:
            plot_index_sets(
                ax_i, sel_i, cand_i,
                selected_colors=_mf_color,
                candidate_colors=_mf_cand_color,
                selected_labels=True, candidate_labels=True,
                box_height=0.6, selected_alpha=0.7, candidate_alpha=0.4,
                axis_labels=["Physical level $\\beta$",
                             "Config level $\\alpha$"],
                max_indices=_cw_max_indices,
            )
            ax_i.set_title(title_i, fontsize=11)
        axes[1].legend(handles=legend_handles_cw, fontsize=7,
                       loc="upper right")

    ani_cw = animation.FuncAnimation(
        fig, animate_cw, frames=n_frames, interval=800, repeat_delay=2000,
    )
    return HTML(ani_cw.to_jshtml())


def plot_mf_indices_2d(ax, sel_2d_np):
    """multifidelity_sparse_grids.qmd -> fig-mf-indices-2d

    Selected multi-indices for 2D physical + 1 config variable (3D voxels).
    """
    from pyapprox.surrogates.affine.indices import (
        plot_indices_3d, format_index_axes,
    )
    from matplotlib.patches import Patch

    plot_indices_3d(
        ax, sel_2d_np,
        colors=lambda idx: "#2C7FB8CC" if idx[-1] == 0 else "#E74C3CCC",
        labels=True,
    )
    format_index_axes(
        ax, sel_2d_np,
        axis_labels=[
            "$\\beta_1$ (physical dim 1)",
            "$\\beta_2$ (physical dim 2)",
            "$\\alpha$ (config level)",
        ],
    )

    legend_handles = [
        Patch(facecolor="#2C7FB8", edgecolor="black",
              label="$\\alpha=0$ (cheap)"),
        Patch(facecolor="#E74C3C", edgecolor="black",
              label="$\\alpha=1$ (expensive)"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper left")
    ax.set_title("Selected multi-indices $[\\beta_1, \\beta_2, \\alpha]$",
                 fontsize=12)


def plot_mf_2d_accuracy(
    axes, fig, plotter_true, plotter_surr, bkd, result_2d,
    true_fn, plot_limits,
):
    """multifidelity_sparse_grids.qmd -> fig-mf-2d-accuracy

    Multi-fidelity surrogate accuracy: true, surrogate, and error.
    """
    from pyapprox.interface.functions.plot.plot2d_rectangular import (
        meshgrid_samples,
    )

    plotter_true.plot_contours(axes[0], qoi=0, npts_1d=80, cmap="viridis")
    axes[0].set_title("True $f_1(z_1, z_2)$", fontsize=12)
    axes[0].set_xlabel("$z_1$", fontsize=11)
    axes[0].set_ylabel("$z_2$", fontsize=11)

    plotter_surr.plot_contours(axes[1], qoi=0, npts_1d=80, cmap="viridis")
    axes[1].set_title("Multi-fidelity surrogate", fontsize=12)
    axes[1].set_xlabel("$z_1$", fontsize=11)
    axes[1].set_ylabel("$z_2$", fontsize=11)

    X, Y, pts = meshgrid_samples(80, plot_limits, bkd)
    true_vals = true_fn(pts)
    surr_vals = result_2d.surrogate(pts)
    error = bkd.to_numpy(bkd.abs(true_vals - surr_vals)).reshape(
        bkd.to_numpy(X).shape
    )
    cs = axes[2].contourf(
        bkd.to_numpy(X), bkd.to_numpy(Y), error, levels=20, cmap="inferno",
    )
    fig.colorbar(cs, ax=axes[2])
    axes[2].set_title("Pointwise error $|f_1 - \\hat{f}|$", fontsize=12)
    axes[2].set_xlabel("$z_1$", fontsize=11)
    axes[2].set_ylabel("$z_2$", fontsize=11)


# ---------------------------------------------------------------------------
# uq_with_sparse_grids.qmd
# ---------------------------------------------------------------------------

def plot_sg_sobol(
    axes, main_effects, total_effects, bkd, qoi_labels,
):
    """uq_with_sparse_grids.qmd -> fig-sg-sobol

    Sobol indices from SG-to-PCE conversion: pie + bar chart.
    """
    from pyapprox.sensitivity.plots import (
        plot_main_effects,
        plot_total_effects,
    )

    plot_main_effects(main_effects, axes[0], bkd, rv=r"\xi", qoi=0)
    axes[0].set_title(f"Main Effects ({qoi_labels[0]})", fontsize=12)

    plot_total_effects(total_effects, axes[1], bkd, rv=r"\xi", qoi=0)
    axes[1].set_ylabel("Sobol Index", fontsize=11)
    axes[1].set_title(f"Total Effects ({qoi_labels[0]})", fontsize=12)


def plot_sg_marginals(
    axes, vals_mc, preds_sg, N_mc, N_sg, M_sg, nqoi, qoi_labels,
):
    """uq_with_sparse_grids.qmd -> fig-sg-marginals

    Marginal density estimates at equal wall-clock cost: MC vs SG.
    """
    for k, ax in enumerate(axes):
        ax.hist(vals_mc[k, :], bins=40, density=True, alpha=0.5,
                color="#2C7FB8", edgecolor="k", lw=0.2,
                label=f"MC ({N_mc} beam evals)")
        ax.hist(preds_sg[k, :], bins=80, density=True, alpha=0.4,
                color="#E67E22", edgecolor="k", lw=0.1,
                label=f"SG ({N_sg}+{M_sg:,} evals)")
        ax.set_xlabel(qoi_labels[k], fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)


def plot_sg_vs_mc(
    fig, ax1, ax2,
    sg_npts_list, mc_mean_errors, sg_mean_errors,
    mc_var_errors, sg_var_errors, sg_levels, qoi_labels, qoi_plot,
):
    """uq_with_sparse_grids.qmd -> fig-sg-vs-mc

    SG vs Monte Carlo efficiency for mean and variance estimation.
    """

    def _plot_convergence(ax, N_vals, mc_errs, sg_errs, ylabel, title,
                          level_labels):
        mc_med = [np.nanmedian(mc_errs[N]) for N in N_vals]
        mc_lo = [np.nanpercentile(mc_errs[N], 10) for N in N_vals]
        mc_hi = [np.nanpercentile(mc_errs[N], 90) for N in N_vals]
        sg_med = [np.nanmedian(sg_errs[N]) for N in N_vals]
        sg_lo = [np.nanpercentile(sg_errs[N], 10) for N in N_vals]
        sg_hi = [np.nanpercentile(sg_errs[N], 90) for N in N_vals]

        ax.fill_between(N_vals, mc_lo, mc_hi, alpha=0.15, color="#2C7FB8")
        ax.semilogy(N_vals, mc_med, "o-", ms=5, color="#2C7FB8", lw=1.5,
                    label="Monte Carlo")
        ax.fill_between(N_vals, sg_lo, sg_hi, alpha=0.15, color="#E67E22")
        ax.semilogy(N_vals, sg_med, "s-", ms=5, color="#E67E22", lw=1.5,
                    label="Sparse Grid")
        for N, lbl in zip(N_vals, level_labels):
            idx = N_vals.index(N)
            ax.annotate(f"l={lbl}", (N, sg_med[idx]),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=7, color="#E67E22")
        ax.set_xlabel("Number of model evaluations $N$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, which="both")

    level_labels = [str(lev) for lev in sg_levels]
    _plot_convergence(
        ax1, sg_npts_list, mc_mean_errors, sg_mean_errors,
        "Relative error in mean",
        f"Mean of {qoi_labels[qoi_plot]}",
        level_labels,
    )
    _plot_convergence(
        ax2, sg_npts_list, mc_var_errors, sg_var_errors,
        "Relative error in variance",
        f"Variance of {qoi_labels[qoi_plot]}",
        level_labels,
    )


# ---------------------------------------------------------------------------
# sparse_grid_to_pce_conversion.qmd
# ---------------------------------------------------------------------------

def plot_two_bases(ax_lagrange, ax_ortho, nodes, x_plot, ortho_vals):
    """sparse_grid_to_pce_conversion.qmd -> fig-two-bases

    Lagrange basis vs orthonormal Legendre polynomials side by side.
    """
    colors = ["#2C7FB8", "#E67E22", "#27AE60", "#8E44AD"]
    n = len(nodes)

    for j in range(n):
        L_j = np.ones_like(x_plot)
        for k in range(n):
            if k != j:
                L_j = L_j * (x_plot - nodes[k]) / (nodes[j] - nodes[k])
        ax_lagrange.plot(x_plot, L_j, color=colors[j], lw=2,
                         label=f"$L_{j}(x)$")
        ax_lagrange.plot(nodes[j], 1.0, "o", color=colors[j], ms=8)

    ax_lagrange.axhline(0, color="gray", lw=0.5, ls="--")
    ax_lagrange.set_xlabel("$x$", fontsize=11)
    ax_lagrange.set_ylabel("$L_j(x)$", fontsize=11)
    ax_lagrange.set_title(f"Lagrange basis ($n={n}$ nodes)", fontsize=12)
    ax_lagrange.legend(fontsize=9)
    ax_lagrange.grid(True, alpha=0.2)

    for k in range(n):
        ax_ortho.plot(x_plot, ortho_vals[:, k], color=colors[k], lw=2,
                      label=f"$P_{k}(x)$")

    ax_ortho.axhline(0, color="gray", lw=0.5, ls="--")
    ax_ortho.set_xlabel("$x$", fontsize=11)
    ax_ortho.set_ylabel("$P_k(x)$", fontsize=11)
    ax_ortho.set_title("Orthonormal Legendre polynomials", fontsize=12)
    ax_ortho.legend(fontsize=9)
    ax_ortho.grid(True, alpha=0.2)


def plot_projection_1d(
    ax_recon, ax_bar, x_plot, nodes, ortho_vals, proj_coefs_np, j_show,
):
    """sparse_grid_to_pce_conversion.qmd -> fig-projection-1d

    Single Lagrange basis function projected onto orthonormal basis.
    """
    colors = ["#2C7FB8", "#E67E22", "#27AE60", "#8E44AD"]
    n = len(nodes)

    # Compute L_j
    L_j_plot = np.ones_like(x_plot)
    for k in range(n):
        if k != j_show:
            L_j_plot *= (x_plot - nodes[k]) / (nodes[j_show] - nodes[k])

    # Reconstruction from orthonormal expansion
    reconstruction = np.zeros_like(x_plot)
    for k in range(n):
        reconstruction += proj_coefs_np[j_show, k] * ortho_vals[:, k]

    ax_recon.plot(x_plot, L_j_plot, color="#2C7FB8", lw=2.5,
                  label=f"$L_{j_show}(x)$ (Lagrange)")
    ax_recon.plot(x_plot, reconstruction, "--", color="#E67E22", lw=2,
                  label=(r"$\sum_k c_{" + str(j_show)
                         + r"k}\, P_k(x)$ (reconstruction)"))
    ax_recon.plot(nodes[j_show], 1.0, "o", color="#2C7FB8", ms=10, zorder=5)
    ax_recon.axhline(0, color="gray", lw=0.5, ls="--")
    ax_recon.set_xlabel("$x$", fontsize=11)
    ax_recon.set_title(
        f"$L_{j_show}(x)$ and its PCE reconstruction", fontsize=12,
    )
    ax_recon.legend(fontsize=9)
    ax_recon.grid(True, alpha=0.2)

    # Bar chart of projection coefficients
    bar_colors = colors[:n]
    ax_bar.bar(range(n), proj_coefs_np[j_show, :], color=bar_colors,
               edgecolor="k", lw=0.5)
    ax_bar.set_xticks(range(n))
    ax_bar.set_xticklabels([f"$P_{k}$" for k in range(n)])
    ax_bar.set_ylabel(f"$c_{{{j_show},k}}$", fontsize=12)
    ax_bar.set_title(
        f"Projection coefficients for $L_{j_show}$", fontsize=12,
    )
    ax_bar.axhline(0, color="gray", lw=0.5, ls="--")
    ax_bar.grid(True, alpha=0.2, axis="y")


def plot_2d_tp(
    ax_grid, ax_pce, fig, sg_samples, sg_values, pce_indices, pce_coefs,
    bkd,
):
    """sparse_grid_to_pce_conversion.qmd -> fig-2d-tp

    2D tensor product: grid values and PCE multi-indices with coefficients.
    """
    import matplotlib.pyplot as plt

    sg_np = bkd.to_numpy(sg_samples)
    vals_np = bkd.to_numpy(sg_values).flatten()
    nterms = pce_indices.shape[1]

    sc = ax_grid.scatter(sg_np[0], sg_np[1], c=vals_np, s=120,
                         cmap="coolwarm", edgecolor="k", lw=0.5, zorder=5)
    plt.colorbar(sc, ax=ax_grid, label="$f(x_1, x_2)$")
    ax_grid.set_xlabel("$x_1$", fontsize=11)
    ax_grid.set_ylabel("$x_2$", fontsize=11)
    ax_grid.set_title("Sparse grid sample values", fontsize=12)
    ax_grid.set_xlim(-1.2, 1.2)
    ax_grid.set_ylim(-1.2, 1.2)
    ax_grid.grid(True, alpha=0.3)

    coef_abs = np.abs(pce_coefs[:, 0])
    coef_max = coef_abs.max() if coef_abs.max() > 0 else 1.0
    sc2 = ax_pce.scatter(pce_indices[0], pce_indices[1], c=coef_abs, s=200,
                         cmap="YlOrRd", edgecolor="k", lw=0.5, zorder=5,
                         vmin=0, vmax=coef_max)
    plt.colorbar(sc2, ax=ax_pce, label="$|\\hat{c}_{k_1 k_2}|$")
    for t in range(nterms):
        ax_pce.annotate(f"{pce_coefs[t, 0]:.3f}",
                        (pce_indices[0, t], pce_indices[1, t]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=8)
    ax_pce.set_xlabel("$k_1$ (degree in $x_1$)", fontsize=11)
    ax_pce.set_ylabel("$k_2$ (degree in $x_2$)", fontsize=11)
    ax_pce.set_title("PCE multi-indices and coefficients", fontsize=12)
    ax_pce.set_xlim(-0.5, max(pce_indices[0]) + 0.8)
    ax_pce.set_ylim(-0.5, max(pce_indices[1]) + 0.8)
    ax_pce.grid(True, alpha=0.3)


def plot_smolyak_merge(
    fig, axes_flat, subspaces, smolyak_coefs, sub_converter, bkd, n_subs,
):
    """sparse_grid_to_pce_conversion.qmd -> fig-smolyak-merge

    Smolyak merging: subspace PCEs and accumulated global PCE.
    """
    merged = {}

    for idx, subspace in enumerate(subspaces):
        coef = smolyak_coefs[idx]
        indices, coefficients = sub_converter.convert_subspace(subspace)
        indices_np = bkd.to_numpy(indices)
        coefs_np = bkd.to_numpy(coefficients)

        for t in range(indices_np.shape[1]):
            key = (int(indices_np[0, t]), int(indices_np[1, t]))
            weighted = coef * coefs_np[0, t]
            merged[key] = merged.get(key, 0.0) + weighted

        ax = axes_flat[idx]
        for t in range(indices_np.shape[1]):
            val = coef * coefs_np[0, t]
            color = "#E67E22" if abs(val) > 1e-10 else "#BDC3C7"
            ax.scatter(indices_np[0, t], indices_np[1, t], c=color, s=150,
                       edgecolor="k", lw=0.5, zorder=5)
            if abs(val) > 1e-10:
                ax.annotate(f"{val:+.3f}",
                            (indices_np[0, t], indices_np[1, t]),
                            textcoords="offset points", xytext=(6, 5),
                            fontsize=7)

        sub_idx = bkd.to_numpy(subspace.get_index())
        ax.set_title(
            f"Subspace ({int(sub_idx[0])},{int(sub_idx[1])}), "
            f"$c = {coef:+.0f}$",
            fontsize=10,
        )
        ax.set_xlim(-0.5, 5)
        ax.set_ylim(-0.5, 5)
        ax.set_xlabel("$k_1$", fontsize=9)
        ax.set_ylabel("$k_2$", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Merged panel
    ax_merged = axes_flat[n_subs]
    for (k1, k2), val in merged.items():
        color = "#27AE60" if abs(val) > 1e-10 else "#BDC3C7"
        ax_merged.scatter(k1, k2, c=color, s=150, edgecolor="k", lw=0.5,
                          zorder=5)
        if abs(val) > 1e-10:
            ax_merged.annotate(f"{val:+.3f}", (k1, k2),
                               textcoords="offset points", xytext=(6, 5),
                               fontsize=7)
    ax_merged.set_title("Merged PCE (accumulated)", fontsize=10)
    ax_merged.set_xlim(-0.5, 5)
    ax_merged.set_ylim(-0.5, 5)
    ax_merged.set_xlabel("$k_1$", fontsize=9)
    ax_merged.set_ylabel("$k_2$", fontsize=9)
    ax_merged.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_subs + 1, len(axes_flat)):
        axes_flat[idx].axis("off")


def plot_sg_vs_pce(
    ax_sg, ax_err, fig, sg_vals, pce_vals, X_np, Y_np,
):
    """sparse_grid_to_pce_conversion.qmd -> fig-sg-vs-pce

    SG interpolant vs converted PCE: contours and pointwise error.
    """
    import matplotlib.pyplot as plt

    cs1 = ax_sg.contourf(X_np, Y_np,
                          sg_vals[0].reshape(X_np.shape),
                          levels=20, cmap="coolwarm")
    plt.colorbar(cs1, ax=ax_sg)
    ax_sg.set_xlabel("$x_1$", fontsize=11)
    ax_sg.set_ylabel("$x_2$", fontsize=11)
    ax_sg.set_title("SG interpolant", fontsize=12)

    error = np.abs(sg_vals[0] - pce_vals[0])
    cs2 = ax_err.contourf(X_np, Y_np,
                           error.reshape(X_np.shape),
                           levels=20, cmap="Reds")
    plt.colorbar(cs2, ax=ax_err)
    ax_err.set_xlabel("$x_1$", fontsize=11)
    ax_err.set_ylabel("$x_2$", fontsize=11)
    ax_err.set_title(f"|SG - PCE| (max = {error.max():.2e})", fontsize=12)
