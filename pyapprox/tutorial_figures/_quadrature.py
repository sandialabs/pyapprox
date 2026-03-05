"""Plotting functions for quadrature and sparse grid tutorials.

Covers: gauss_quadrature.qmd, piecewise_quadrature.qmd,
        isotropic_sparse_grids.qmd, adaptive_sparse_grids.qmd
"""

import numpy as np

from ._style import COLORS

# ---------------------------------------------------------------------------
# gauss_quadrature.qmd — all echo:true -> Convention B
# ---------------------------------------------------------------------------

def plot_gauss_nodes(ax, gauss_rule, bkd):
    """gauss_quadrature.qmd -> fig-gauss-nodes

    Gauss-Legendre nodes for M = 3, 5, 9 on [-1, 1].
    """
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    for ii, M in enumerate([3, 5, 9]):
        nd, _ = gauss_rule(M)
        nd_np = bkd.to_numpy(nd[0])
        ax.plot(
            nd_np,
            np.full_like(nd_np, ii),
            "o",
            ms=10,
            color=colors[ii],
            label=f"$M = {M}$",
        )
        ax.axhline(ii, color=colors[ii], lw=0.5, alpha=0.4)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["$M=3$", "$M=5$", "$M=9$"], fontsize=12)
    ax.set_xlabel("Node location $x$", fontsize=12)
    ax.set_title("Gauss-Legendre nodes on $[-1, 1]$", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)


def plot_quad_convergence_comparison(axs, Ms, gauss_err_analytic,
                                     trap_err_analytic, gauss_err_rough,
                                     trap_err_rough):
    """gauss_quadrature.qmd -> fig-convergence-comparison

    Gauss-Legendre vs. piecewise linear convergence on analytic and rough functions.
    """
    labels = [
        "$1/(1+16x^2)$ (analytic)",
        "$|x-0.3|^{1.5}$ (rough)",
    ]
    for ax, ge, te, title in zip(
        axs,
        [gauss_err_analytic, gauss_err_rough],
        [trap_err_analytic, trap_err_rough],
        labels,
    ):
        ax.semilogy(Ms, ge, "o-", color=COLORS["primary"],
                    label="Gauss-Legendre")
        ax.semilogy(Ms, te, "s-", color=COLORS["secondary"],
                    label="PiecewiseLinear (trapezoid)")
        ax.set_xlabel("Number of points $M$", fontsize=12)
        ax.set_ylabel("Absolute error", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)


def plot_gauss_hermite(ax, Ms, gh_errors):
    """gauss_quadrature.qmd -> fig-gauss-hermite

    Gauss-Hermite convergence for a smooth QoI with standard normal input.
    """
    ax.loglog(Ms, gh_errors, "o-", color=COLORS["accent"],
              label="Gauss-Hermite")
    ax.axhline(1e-15, color="gray", ls="--", lw=1, label="Machine precision floor")
    ax.set_xlabel("Number of quadrature points $M$", fontsize=12)
    ax.set_ylabel(r"Error in $\mathbb{E}[f(\theta)]$", fontsize=12)
    ax.set_title(
        r"Gauss-Hermite convergence for $\theta \sim \mathcal{N}(0,1)$",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


def plot_lobatto(ax, gauss_rule, lobatto_rule, bkd):
    """gauss_quadrature.qmd -> fig-lobatto

    Gauss vs. Gauss-Lobatto nodes side by side.
    """
    for ii, M in enumerate([5, 9]):
        nd_gauss, _ = gauss_rule(M)
        nd_lobatto, _ = lobatto_rule(M)
        nd_gauss_np = bkd.to_numpy(nd_gauss[0])
        nd_lobatto_np = bkd.to_numpy(nd_lobatto[0])
        offset_g = ii * 2
        offset_l = ii * 2 + 1
        ax.plot(
            nd_gauss_np, np.full(M, offset_g), "o", ms=9,
            color=COLORS["primary"],
            label=f"Gauss $M={M}$" if ii == 0 else None,
        )
        ax.plot(
            nd_lobatto_np, np.full(M, offset_l), "s", ms=9,
            color=COLORS["secondary"],
            label=f"Gauss-Lobatto $M={M}$" if ii == 0 else None,
        )
        ax.axhline(offset_g, color=COLORS["primary"], lw=0.4, alpha=0.5)
        ax.axhline(offset_l, color=COLORS["secondary"], lw=0.4, alpha=0.5)
        ax.text(1.05, offset_g, f"Gauss $M={M}$", va="center", fontsize=10)
        ax.text(1.05, offset_l, f"Lobatto $M={M}$", va="center", fontsize=10)

    ax.axvline(-1, color="k", lw=1, ls="--", alpha=0.4)
    ax.axvline(1, color="k", lw=1, ls="--", alpha=0.4)
    ax.set_xlim([-1.2, 1.6])
    ax.set_xlabel("Node location", fontsize=12)
    ax.set_yticks([])
    ax.set_title("Gauss vs. Gauss-Lobatto nodes", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)


# ---------------------------------------------------------------------------
# piecewise_quadrature.qmd — all echo:true -> Convention B
# ---------------------------------------------------------------------------

def plot_newton_cotes(axs, f_demo, a, b):
    """piecewise_quadrature.qmd -> fig-newton-cotes

    Newton-Cotes rules illustrated on one interval.
    """
    x_plot = np.linspace(a, b, 300)

    for ax, label, xs, color in zip(
        axs,
        ["Midpoint", "Trapezoid", "Simpson's"],
        [
            [(a + b) / 2],
            [a, b],
            [a, (a + b) / 2, b],
        ],
        [COLORS["primary"], COLORS["secondary"], COLORS["accent"]],
    ):
        ax.plot(x_plot, f_demo(x_plot), "k-", lw=2, label="$f(x)$")
        ys = [f_demo(xi) for xi in xs]
        ax.plot(xs, ys, "o", ms=8, color=color)

        if len(xs) == 1:          # midpoint: rectangle
            ax.fill_between([a, b], 0, ys[0], alpha=0.25, color=color,
                            label=label)
            ax.hlines(ys[0], a, b, color=color, lw=2)
        elif len(xs) == 2:        # trapezoid
            ax.fill_between([a, b], 0, ys, alpha=0.25, color=color,
                            label=label)
            ax.plot([a, b], ys, color=color, lw=2)
        else:                     # Simpson's: parabola through 3 points
            coeffs = np.polyfit(xs, ys, 2)
            p_vals = np.polyval(coeffs, x_plot)
            ax.fill_between(x_plot, 0, p_vals, alpha=0.25, color=color,
                            label=label)
            ax.plot(x_plot, p_vals, color=color, lw=2)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("$x$")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=10)

    axs[0].set_ylabel("$f(x)$")


def plot_pw_quad_convergence(ax, Ms, trap_errors, simp_errors):
    """piecewise_quadrature.qmd -> fig-convergence

    Convergence of composite trapezoid and Simpson's rules.
    """
    ax.loglog(Ms, trap_errors, "o-",
              label="PiecewiseLinear (trapezoid) $O(M^{-2})$")
    ax.loglog(Ms, simp_errors, "s-",
              label="PiecewiseQuadratic (Simpson) $O(M^{-4})$")

    # Reference slopes
    ax.loglog(Ms, 0.5 * Ms.astype(float)**-2, "k--", lw=1,
              label="$M^{-2}$ slope")
    ax.loglog(Ms, 0.5 * Ms.astype(float)**-4, "k:", lw=1,
              label="$M^{-4}$ slope")

    ax.set_xlabel("Number of nodes $M$", fontsize=12)
    ax.set_ylabel("Absolute error $|I_M[f] - I[f]|$", fontsize=12)
    ax.set_title("Piecewise polynomial quadrature convergence", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# isotropic_sparse_grids.qmd — all echo:true -> Convention B
# ---------------------------------------------------------------------------

def plot_smolyak_combo(fig, axs, max_level, iso_indices, iso_coefs,
                       eval_grid, X_plot, Y_plot, tp_interpolants):
    """isotropic_sparse_grids.qmd -> fig-smolyak-combo

    Tensor product sub-grids contributing to the 2D isotropic sparse grid.
    """
    text_props = dict(boxstyle="round", facecolor="white", alpha=0.6)

    for ii in range(max_level + 1):
        for jj in range(max_level + 1):
            ax = axs[max_level - jj][ii]
            if ii + jj > max_level:
                ax.axis("off")
                continue

            Z, sg = tp_interpolants[(ii, jj)]
            ax.contourf(X_plot, Y_plot, Z, levels=20, cmap="coolwarm")
            ax.plot(sg[0], sg[1], "ko", ms=3)

            # Find Smolyak coefficient
            for idx in range(iso_indices.shape[1]):
                if (int(iso_indices[0, idx]) == ii
                        and int(iso_indices[1, idx]) == jj):
                    coef = int(iso_coefs[idx])
                    break
            ax.text(
                0.05, 0.95, f"${coef:+d}$",
                transform=ax.transAxes, fontsize=16,
                verticalalignment="top", bbox=text_props,
            )
            ax.set_title(f"$\\beta = [{ii},{jj}]$", fontsize=10)

    fig.suptitle(
        f"Isotropic sparse grid components: $D=2$, $l={max_level}$",
        fontsize=13, y=1.01,
    )


def plot_sg_points(axes, levels, sample_arrays, nsamples_list):
    """isotropic_sparse_grids.qmd -> fig-sg-points

    Isotropic sparse grid points in 2D at increasing levels.
    """
    for i, (level, samples, nsamples) in enumerate(
        zip(levels, sample_arrays, nsamples_list)
    ):
        ax = axes[i]
        ax.scatter(samples[0], samples[1], s=40, c="steelblue", alpha=0.7)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"Level {level}: {nsamples} points")
        ax.set_xlabel("$z_1$")
        if i == 0:
            ax.set_ylabel("$z_2$")
        ax.grid(True, alpha=0.3)


def plot_point_counts(axs, dims_compare, levels_data, sg_counts_data,
                      tp_counts_data):
    """isotropic_sparse_grids.qmd -> fig-point-counts

    Sparse grid vs. tensor product point counts across dimensions.
    """
    for ax, d, levels_range, sg_counts, tp_counts in zip(
        axs, dims_compare, levels_data, sg_counts_data, tp_counts_data,
    ):
        ax.semilogy(
            levels_range, tp_counts, "-o", color=COLORS["secondary"],
            lw=2, ms=6, label="Tensor product",
        )
        ax.semilogy(
            levels_range, sg_counts, "--s", color=COLORS["primary"],
            lw=2, ms=6, label="Isotropic SG",
        )
        ax.set_title(f"$D = {d}$", fontsize=12)
        ax.set_xlabel("Level $l$", fontsize=12)
        ax.set_ylabel("Number of points", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)


def plot_4d_convergence(ax, sg_npts, sg_errors, tp_npts, tp_errors):
    """isotropic_sparse_grids.qmd -> fig-4d-convergence

    4D convergence of sparse grid vs. tensor product.
    """
    ax.loglog(
        sg_npts, sg_errors, "--s", color=COLORS["primary"], lw=2, ms=6,
        label="Isotropic sparse grid",
    )
    ax.loglog(
        tp_npts, tp_errors, "-o", color=COLORS["secondary"], lw=2, ms=6,
        label="Tensor product",
    )
    ax.set_xlabel("Number of points $N$", fontsize=12)
    ax.set_ylabel("RMS error", fontsize=12)
    ax.set_title("4D convergence: SG vs. TP (Leja nodes)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


def plot_growth_rules(ax, levels_gr, rules_data):
    """isotropic_sparse_grids.qmd -> fig-growth-rules

    Two standard univariate growth rules.
    """
    styles = [("o-", COLORS["primary"]), ("s-", COLORS["secondary"])]
    for (label, counts), (sty, col) in zip(rules_data, styles):
        ax.semilogy(levels_gr, counts, sty, color=col, lw=2, ms=6,
                    label=label)

    ax.set_xlabel("Level $\\ell$", fontsize=12)
    ax.set_ylabel("Nodes $M_\\ell$", fontsize=12)
    ax.set_title("Univariate growth rules", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# adaptive_sparse_grids.qmd — all echo:true -> Convention B
# ---------------------------------------------------------------------------

def plot_additive_function(ax, plotter):
    """adaptive_sparse_grids.qmd -> fig-additive-function

    Contour plot of the additive benchmark f(z1,z2) = cos(5z1) + 0.1z2.
    """
    plotter.plot_contours(ax, qoi=0, npts_1d=100, levels=21, cmap="coolwarm")
    ax.set_xlabel("$z_1$", fontsize=12)
    ax.set_ylabel("$z_2$", fontsize=12)
    ax.set_title(r"$f(z_1, z_2) = \cos(5z_1) + 0.1z_2$", fontsize=12)


def plot_index_set(ax, sel_np):
    """adaptive_sparse_grids.qmd -> fig-index-set

    Selected index set after adaptive refinement.
    """
    from pyapprox.surrogates.affine.indices import plot_index_sets

    plot_index_sets(
        ax, sel_np,
        selected_colors="steelblue", selected_labels=True,
        axis_labels=["$k_1$ (level in $z_1$)", "$k_2$ (level in $z_2$)"],
    )
    ax.set_title("Selected Index Set (Adaptive)", fontsize=12)


def plot_points_compare(axes, ada_selected, ada_candidate, ada_nsamples,
                        iso_samples, iso_nsamples, iso_level):
    """adaptive_sparse_grids.qmd -> fig-points-compare

    Point distribution for adaptive vs. isotropic sparse grids.
    """
    from pyapprox.surrogates.sparsegrids import plot_sparse_grid_points

    plot_sparse_grid_points(
        axes[0], ada_selected, ada_candidate,
        axis_labels=["$z_1$", "$z_2$"],
        title=f"Adaptive: {ada_nsamples} samples",
    )
    axes[0].legend(fontsize=9)

    plot_sparse_grid_points(
        axes[1], iso_samples,
        selected_color="coral",
        axis_labels=["$z_1$", "$z_2$"],
        title=f"Isotropic (L={iso_level}): {iso_nsamples} samples",
        selected_label="Isotropic",
    )


def plot_adaptive_vs_iso(ax, adapt_nsamples, adapt_errors,
                         iso_nsamples_list, iso_errors):
    """adaptive_sparse_grids.qmd -> fig-adaptive-vs-iso

    Adaptive vs. isotropic convergence on the additive benchmark.
    """
    ax.semilogy(
        adapt_nsamples, adapt_errors, "o-", color=COLORS["primary"],
        lw=2, ms=5, label="Adaptive SG",
    )
    ax.semilogy(
        iso_nsamples_list, iso_errors, "--s", color=COLORS["secondary"],
        lw=2, ms=5, label="Isotropic SG",
    )
    ax.set_xlabel("Number of function evaluations", fontsize=12)
    ax.set_ylabel("RMS error", fontsize=12)
    ax.set_title(
        "Adaptive vs. isotropic --- additive function ($D=2$)", fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


def plot_4d_adaptive(ax, adapt_nsamples, adapt_errors,
                     iso_nsamples_4d, iso_errors_4d):
    """adaptive_sparse_grids.qmd -> fig-4d-adaptive

    4D adaptive vs. isotropic convergence for an additive function.
    """
    ax.semilogy(
        adapt_nsamples, adapt_errors, "o-", color=COLORS["primary"],
        lw=2, ms=5, label="Adaptive SG",
    )
    ax.semilogy(
        iso_nsamples_4d, iso_errors_4d, "--s", color=COLORS["secondary"],
        lw=2, ms=5, label="Isotropic SG",
    )
    ax.set_xlabel("Number of function evaluations", fontsize=12)
    ax.set_ylabel("RMS error", fontsize=12)
    ax.set_title(
        "4D: adaptive vs. isotropic on additive function", fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
