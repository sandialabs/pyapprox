"""Plotting functions for interpolation tutorials.

Covers: lagrange_interpolation.qmd, tensor_product_interpolation.qmd,
        piecewise_interpolation.qmd, leja_sequences.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# lagrange_interpolation.qmd
# ---------------------------------------------------------------------------

def plot_lagrange_basis(bkd, cc_rule, ax):
    """lagrange_interpolation.qmd -> fig-lagrange-basis

    Lagrange basis polynomials for M=5 Clenshaw-Curtis nodes.
    """
    from pyapprox.surrogates.affine.univariate import LagrangeBasis1D
    from ._style import apply_style

    M = 5
    nodes_arr, _ = cc_rule(M)
    nodes = bkd.to_numpy(nodes_arr).ravel()

    basis = LagrangeBasis1D(bkd, cc_rule)
    basis.set_nterms(M)

    x_plot = np.linspace(-1, 1, 300)
    x_eval = bkd.array(x_plot.reshape(1, -1))
    phi = bkd.to_numpy(basis(x_eval))

    colors = plt_cm_tab10(M)
    for j in range(M):
        ax.plot(x_plot, phi[:, j], color=colors[j], lw=2,
                label=rf"$\phi_{j+1}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(nodes, np.zeros(M), "ko", ms=8, zorder=5)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel(r"$\phi_j(x)$", fontsize=12)
    ax.set_title(
        f"Lagrange basis functions — $M = {M}$ Clenshaw-Curtis nodes",
        fontsize=12,
    )
    ax.legend(fontsize=10, ncol=M, loc="lower center")
    apply_style(ax)


def plot_basis_comparison(x_plot, phi_cc, nodes_cc, phi_gauss, nodes_gauss,
                          M, axs):
    """lagrange_interpolation.qmd -> fig-basis-comparison

    Side-by-side CC vs Gauss-Legendre Lagrange basis functions.
    """
    from ._style import apply_style

    for ax, phi, nd, title in zip(
        axs,
        [phi_cc, phi_gauss],
        [nodes_cc, nodes_gauss],
        ["Clenshaw-Curtis nodes", "Gauss-Legendre nodes"],
    ):
        for j in range(M):
            ax.plot(x_plot, phi[:, j], lw=1.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.plot(nd, np.zeros(M), "ko", ms=7, zorder=5)
        ax.set_title(f"{title}, $M = {M}$", fontsize=11)
        ax.set_xlabel("$x$", fontsize=11)
        apply_style(ax)

    axs[0].set_ylabel(r"$\phi_j(x)$", fontsize=11)


def plot_runge(x_plot, true_vals, nodes_equi, interp_equi, M_equi,
               nodes_cc, interp_cc, M_cc, axs):
    """lagrange_interpolation.qmd -> fig-runge

    Runge phenomenon: equidistant vs Clenshaw-Curtis interpolation.
    """
    from ._style import apply_style

    def runge(x):
        return 1.0 / (1.0 + 25.0 * x**2)

    for ax, nodes, interp, M_label, title, color in zip(
        axs,
        [nodes_equi, nodes_cc],
        [interp_equi, interp_cc],
        [M_equi, M_cc],
        ["Equidistant nodes — Runge phenomenon",
         "Clenshaw-Curtis nodes — no oscillation"],
        ["#E67E22", "#27AE60"],
    ):
        ax.plot(x_plot, true_vals, "k-", lw=2,
                label=r"$f(x) = 1/(1+25x^2)$")
        ax.plot(x_plot, interp, "--", color=color, lw=2,
                label=f"Interpolant $M={M_label}$")
        ax.plot(nodes, runge(nodes), "o", color=color, ms=7)
        ax.set_ylim(-0.5, 1.4)
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        apply_style(ax)


def plot_nested_cc(cc_rule, bkd, levels, ax):
    """lagrange_interpolation.qmd -> fig-nested-cc

    Nested Clenshaw-Curtis node hierarchies.
    """
    from ._style import apply_style

    colors = ["#2C7FB8", "#E67E22", "#27AE60"]
    y_pos = list(range(len(levels)))

    for lvl, yy, col in zip(levels, y_pos, colors):
        nd, _ = cc_rule(lvl)
        nd_np = bkd.to_numpy(nd).ravel()
        ax.plot(nd_np, np.full_like(nd_np, yy), "o", ms=10, color=col,
                label=f"$M = {lvl}$")
        ax.axhline(yy, lw=0.4, color=col, alpha=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"$M={l}$" for l in levels], fontsize=12)
    ax.set_xlabel("Node location", fontsize=12)
    ax.set_title("Nested Clenshaw-Curtis hierarchies", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)


def plot_interp_convergence(Ms, errors_by_func, labels, ax):
    """lagrange_interpolation.qmd -> fig-convergence

    Interpolation error convergence for multiple test functions.
    """
    from ._style import apply_style

    colors = ["#2C7FB8", "#E67E22", "#27AE60"]
    for (errors, label), col in zip(zip(errors_by_func, labels), colors):
        ax.semilogy(Ms, errors, "o-", color=col, lw=2, ms=7, label=label)

    ax.set_xlabel("Number of nodes $M$", fontsize=12)
    ax.set_ylabel(r"$\max_{x} |f(x) - f_M(x)|$", fontsize=12)
    ax.set_title(
        "Lagrange interpolation convergence — Clenshaw-Curtis nodes",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# tensor_product_interpolation.qmd
# ---------------------------------------------------------------------------

def plot_tp_2d(Z_true, Z_interp, grid_np, M, nfine, axs):
    """tensor_product_interpolation.qmd -> fig-tp-2d

    True function, TP interpolant, and pointwise error for 2D test.
    """
    import matplotlib.pyplot as plt

    Z_err = np.abs(Z_true - Z_interp)
    kw = dict(cmap="coolwarm", extent=[-1, 1, -1, 1], origin="lower",
              aspect="equal", interpolation="bilinear")

    im0 = axs[0].imshow(Z_true, vmin=Z_true.min(), vmax=Z_true.max(), **kw)
    im1 = axs[1].imshow(Z_interp, vmin=Z_true.min(), vmax=Z_true.max(), **kw)
    im2 = axs[2].imshow(Z_err, vmin=0, vmax=max(Z_err.max(), 1e-16),
                        cmap="Reds", extent=[-1, 1, -1, 1],
                        origin="lower", aspect="equal",
                        interpolation="bilinear")

    axs[1].plot(grid_np[0], grid_np[1], "k.", ms=4, alpha=0.6)

    for ax, title in zip(
        axs, ["True $f$", f"TP Interpolant $M={M}$", "Pointwise error"]
    ):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("$z_1$", fontsize=11)

    axs[0].set_ylabel("$z_2$", fontsize=11)
    plt.colorbar(im0, ax=axs[0])
    plt.colorbar(im1, ax=axs[1])
    plt.colorbar(im2, ax=axs[2])


def plot_curse_of_dimensionality(levels, dims, ax):
    """tensor_product_interpolation.qmd -> fig-curse

    TP point counts vs refinement level for several dimensions.
    """
    from ._style import apply_style

    def double_plus_one(level):
        if level == 0:
            return 1
        return 2**level + 1

    colors = plt_cm_plasma(len(dims))
    for d, col in zip(dims, colors):
        counts = [double_plus_one(ell)**d for ell in levels]
        ax.semilogy(levels, counts, "o-", color=col, lw=2, ms=6,
                    label=f"$D = {d}$")

    ax.set_xlabel("Refinement level $\\ell$", fontsize=12)
    ax.set_ylabel("Number of grid points $M^D$", fontsize=12)
    ax.set_title(
        "Curse of dimensionality: TP point counts grow as $M^D$", fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


def plot_2d_convergence(Ms_cc, err_cc_smooth, err_cc_nonsmooth,
                        Ms_pw, err_pw_smooth, err_pw_nonsmooth,
                        func_labels, axs):
    """tensor_product_interpolation.qmd -> fig-2d-convergence

    2D TP convergence: CC/Lagrange vs piecewise linear, smooth vs non-smooth.
    """
    from ._style import apply_style

    data = [
        (err_cc_smooth, err_pw_smooth),
        (err_cc_nonsmooth, err_pw_nonsmooth),
    ]
    for ax, (err_cc, err_pw), label in zip(axs, data, func_labels):
        ax.semilogy(Ms_cc, err_cc, "o-", color="#2C7FB8", lw=2, ms=6,
                    label="CC Lagrange (nested)")
        ax.semilogy(Ms_pw, err_pw, "s-", color="#E67E22", lw=2, ms=6,
                    label="Piecewise linear")
        ax.set_xlabel("Nodes per dimension $M$", fontsize=12)
        ax.set_ylabel("Max error", fontsize=12)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# piecewise_interpolation.qmd
# ---------------------------------------------------------------------------

def plot_gibbs(x_plot, true_vals, interp_lagrange, interp_pw_vals,
               M_cc, M_pw, axs):
    """piecewise_interpolation.qmd -> fig-gibbs

    Step function: global Lagrange vs piecewise linear interpolation.
    """
    from ._style import apply_style

    titles = ["Global Lagrange — Gibbs oscillations",
              "Piecewise linear — localized error"]
    interps = [interp_lagrange, interp_pw_vals]
    Ms_labels = [M_cc, M_pw]

    for ax, interp, title, M_label in zip(axs, interps, titles, Ms_labels):
        ax.plot(x_plot, true_vals, "k-", lw=2, label="True $f(x)$")
        ax.plot(x_plot, interp, "--", color="#2C7FB8", lw=2,
                label=f"Interpolant $M={M_label}$")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$", fontsize=12)
        ax.legend(fontsize=10)
        apply_style(ax)


def plot_piecewise_basis(x_plot, phi_hat, nodes_np, M,
                         phi_lagrange_mid, j_mid, axs):
    """piecewise_interpolation.qmd -> fig-piecewise-basis

    Piecewise linear hat functions and local vs global support comparison.
    """
    from ._style import apply_style

    colors = plt_cm_tab10(M)

    ax = axs[0]
    for j in range(M):
        ax.plot(x_plot, phi_hat[:, j], color=colors[j], lw=2,
                label=f"$\\phi_{j+1}$")
    ax.plot(nodes_np, np.zeros(M), "ko", ms=7, zorder=5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Piecewise linear (hat) basis, $M = {M}$", fontsize=11)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel(r"$\phi_j(x)$", fontsize=12)
    ax.legend(fontsize=8, ncol=M, loc="upper center")
    apply_style(ax)

    ax = axs[1]
    phi_hat_mid = phi_hat[:, j_mid]
    ax.plot(x_plot, phi_lagrange_mid, "-", color="#E67E22", lw=2,
            label=f"Lagrange $\\phi_{j_mid+1}$ (global support)")
    ax.plot(x_plot, phi_hat_mid, "--", color="#2C7FB8", lw=2,
            label=f"Piecewise linear $\\phi_{j_mid+1}$ (local support)")
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(nodes_np[j_mid], 0, "ko", ms=8)
    ax.set_title("Global vs. local support — middle basis function",
                 fontsize=11)
    ax.set_xlabel("$x$", fontsize=12)
    ax.legend(fontsize=10)
    apply_style(ax)


def plot_pw_convergence(hs_linear, errors_linear, hs_cubic, errors_cubic, ax):
    """piecewise_interpolation.qmd -> fig-pw-convergence

    Piecewise linear and cubic convergence with reference slopes.
    """
    from ._style import apply_style

    ax.loglog(hs_linear, errors_linear, "o-", color="#2C7FB8", lw=2, ms=6,
              label="Linear ($p=1$)")
    ax.loglog(hs_cubic, errors_cubic, "^-", color="#E67E22", lw=2, ms=6,
              label="Cubic ($p=3$)")
    ax.loglog(hs_linear, 0.5 * hs_linear**2, "k--", lw=1,
              label="$h^2$ slope")
    ax.loglog(hs_cubic, 2.0 * hs_cubic**4, "k:", lw=1,
              label="$h^4$ slope")
    ax.set_xlabel("Cell width $h$", fontsize=12)
    ax.set_ylabel("Max error", fontsize=12)
    ax.set_title(r"Piecewise convergence on $e^{\sin(2\pi x)}$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def plot_discontinuous(Ms_cc, err_lag, Ms_pw, err_pw, ax):
    """piecewise_interpolation.qmd -> fig-discontinuous

    Kink function error: piecewise linear vs global Lagrange.
    """
    from ._style import apply_style

    ax.loglog(Ms_cc, err_lag, "o-", color="#E67E22", lw=2, ms=6,
              label="Global Lagrange (CC nodes)")
    ax.loglog(Ms_pw, err_pw, "s-", color="#2C7FB8", lw=2, ms=6,
              label="Piecewise linear")
    ax.loglog(Ms_pw, 1.0 / Ms_pw, "k--", lw=1, label="$M^{-1}$ slope")
    ax.set_xlabel("Number of nodes $M$", fontsize=12)
    ax.set_ylabel("Max error", fontsize=12)
    ax.set_title(
        "Kink function $|x - 1/3|$: piecewise linear converges; "
        "Lagrange stalls",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# leja_sequences.qmd
# ---------------------------------------------------------------------------

def plot_leja_growth(bkd, leja_seq, Ms, ax):
    """leja_sequences.qmd -> fig-leja-growth

    Leja sequence growth showing nestedness at multiple sizes.
    """
    from ._style import apply_style

    nodes_full, _ = leja_seq.quadrature_rule(max(Ms))
    nodes_full_np = bkd.to_numpy(nodes_full).ravel()

    colors = plt_cm_blues(len(Ms))

    for ii, M in enumerate(Ms):
        y = ii
        pts = nodes_full_np[:M]
        ax.plot(pts, np.full(M, y), "o", ms=9, color=colors[ii],
                label=f"$M = {M}$")
        ax.axhline(y, lw=0.4, color=colors[ii], alpha=0.5)

    ax.set_yticks(range(len(Ms)))
    ax.set_yticklabels([f"$M = {M}$" for M in Ms], fontsize=12)
    ax.set_xlabel("Node location", fontsize=12)
    ax.set_title("Leja sequence growth — each level adds one point",
                 fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim([-1.15, 1.15])


def plot_node_distribution(leja_np, M_leja, cc_np, M_cc, gauss_np, M_gauss,
                           axs):
    """leja_sequences.qmd -> fig-node-distribution

    Side-by-side node distributions for Leja, CC, and Gauss-Legendre.
    """
    from ._style import apply_style

    labels = ["Leja (Christoffel)", "Clenshaw-Curtis", "Gauss-Legendre"]
    colors_pts = ["#2C7FB8", "#E67E22", "#27AE60"]
    sizes = [M_leja, M_cc, M_gauss]

    for ax, nodes, lbl, col, sz in zip(
        axs,
        [leja_np, cc_np, gauss_np],
        labels,
        colors_pts,
        sizes,
    ):
        ax.plot(nodes, np.zeros_like(nodes), "o", ms=9, color=col)
        ax.set_yticks([])
        ax.set_ylabel(f"{lbl}\n($M={sz}$)", fontsize=10, rotation=0,
                      ha="right", va="center")
        ax.axhline(0, lw=0.4, color="k", alpha=0.3)
        ax.grid(True, axis="x", alpha=0.3)

    axs[-1].set_xlabel("Node location", fontsize=12)


def plot_lagrange_on_leja(x_plot, phi_leja, nodes_np, M, ax):
    """leja_sequences.qmd -> fig-lagrange-on-leja

    Lagrange basis functions evaluated on Leja points.
    """
    from ._style import apply_style

    for j in range(M):
        ax.plot(x_plot, phi_leja[:, j], lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(nodes_np, np.zeros(M), "ko", ms=7, zorder=5)
    ax.set_title(f"Lagrange basis on {M} Leja points", fontsize=12)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel(r"$\phi_j(x)$", fontsize=12)
    apply_style(ax)


def plot_leja_convergence(Ms, err_leja, cc_Ms, err_cc, err_gauss, ax):
    """leja_sequences.qmd -> fig-leja-convergence

    Interpolation error comparison: Leja vs CC vs Gauss-Legendre.
    """
    from ._style import apply_style

    ax.semilogy(Ms, err_leja, "o-", color="#2C7FB8", lw=2, ms=5,
                label="Leja (nested, any M)")
    ax.semilogy(cc_Ms, err_cc, "s-", color="#E67E22", lw=2, ms=6,
                label="Clenshaw-Curtis (nested, $2^k+1$)")
    ax.semilogy(Ms, err_gauss, "^-", color="#27AE60", lw=2, ms=5,
                label="Gauss-Legendre (non-nested)")
    ax.set_xlabel("Number of nodes $M$", fontsize=12)
    ax.set_ylabel(r"$\max_x |f(x) - f_M(x)|$", fontsize=12)
    ax.set_title(r"Convergence for $f(x) = e^{\cos(\pi x)}$", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


def plot_two_point_quadrature(Ms_quad, err_1pt, err_2pt, ax):
    """leja_sequences.qmd -> fig-two-point-quadrature

    Quadrature error: one-point vs two-point optimized Leja.
    """
    from ._style import apply_style

    ax.semilogy(Ms_quad, err_1pt, "o-", color="#2C7FB8", lw=2, ms=5,
                label="One-point Leja")
    ax.semilogy(Ms_quad, err_2pt, "s-", color="#E67E22", lw=2, ms=6,
                label="Two-point Leja")
    ax.set_xlabel("Number of quadrature points $M$", fontsize=12)
    ax.set_ylabel("Absolute quadrature error", fontsize=12)
    ax.set_title("Quadrature accuracy: one-point vs. two-point Leja",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


def plot_leja_beta(nodes_uni_01, nodes_beta_01, M, x_plot, rv_beta, axs):
    """leja_sequences.qmd -> fig-leja-beta

    Uniform vs Beta-matched Leja node distributions with density overlay.
    """
    titles = [
        "Uniform (Legendre) — nodes spread across $[0,1]$",
        r"Beta(3,5)-matched Jacobi — nodes concentrate near mode",
    ]

    for ax, seq, title in zip(axs, [nodes_uni_01, nodes_beta_01], titles):
        ax2 = ax.twinx()
        ax2.fill_between(x_plot, rv_beta.pdf(x_plot), alpha=0.15,
                         color="#E67E22", label=r"$p(\theta)$")
        ax2.set_ylabel(r"$p(\theta)$", fontsize=11, color="#E67E22")
        ax2.tick_params(axis="y", labelcolor="#E67E22")

        ax.plot(seq, np.zeros(M), "o", ms=9, color="#2C7FB8", zorder=5,
                label=f"$M={M}$ Leja nodes")
        ax.axhline(0, lw=0.5, color="k")
        ax.set_ylim(-0.1, 0.1)
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)

    axs[-1].set_xlabel(r"$\theta$", fontsize=12)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def plt_cm_tab10(n):
    """Return n evenly-spaced tab10 colors."""
    import matplotlib.pyplot as plt
    return plt.cm.tab10(np.linspace(0, 1, n))


def plt_cm_plasma(n):
    """Return n evenly-spaced plasma colors."""
    import matplotlib.pyplot as plt
    return plt.cm.plasma(np.linspace(0.1, 0.9, n))


def plt_cm_blues(n):
    """Return n evenly-spaced Blues colors."""
    import matplotlib.pyplot as plt
    return plt.cm.Blues(np.linspace(0.35, 0.9, n))
