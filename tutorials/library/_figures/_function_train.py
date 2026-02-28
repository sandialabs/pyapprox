"""Plotting functions for function train tutorials.

Covers: function_train_surrogates.qmd, functiontrain_sensitivity.qmd,
        uq_with_functiontrain.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# function_train_surrogates.qmd — echo:false figures → Convention A
# ---------------------------------------------------------------------------

def plot_ft_rank1_contours(axes):
    """function_train_surrogates.qmd → fig-rank1-contours

    Rank-1 function: 1D Gaussian factors and their product contour.
    """
    x = np.linspace(-3, 3, 200)
    X1, X2 = np.meshgrid(x, x)

    g1 = np.exp(-X1**2 / 2)
    g2 = np.exp(-X2**2 / 2)
    f_rank1 = g1 * g2

    axes[0].plot(x, np.exp(-x**2 / 2), color="#2563a8", lw=2.5)
    axes[0].set_title(r"$g_1(x_1) = e^{-x_1^2/2}$", fontsize=11)
    axes[0].set_xlabel(r"$x_1$")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, np.exp(-x**2 / 2), color="#1a7a4a", lw=2.5)
    axes[1].set_title(r"$g_2(x_2) = e^{-x_2^2/2}$", fontsize=11)
    axes[1].set_xlabel(r"$x_2$")
    axes[1].grid(True, alpha=0.25)

    cf = axes[2].contourf(X1, X2, f_rank1, levels=20, cmap="Blues")
    axes[2].set_title(
        r"$f(x_1,x_2) = g_1(x_1)\cdot g_2(x_2)$", fontsize=11
    )
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")

    return cf


def plot_ft_highrank_contours(axes, fig=None):
    """function_train_surrogates.qmd → fig-highrank-contours

    Rank-2 function: two rank-1 Gaussian terms and their sum.
    """
    x = np.linspace(-3, 3, 200)
    X1, X2 = np.meshgrid(x, x)

    g1_a = np.exp(-2 * (X1 + 1)**2)
    g2_a = np.exp(-2 * (X2 + 1)**2)
    g1_b = np.exp(-0.5 * (X1 - 1)**2)
    g2_b = np.exp(-0.5 * (X2 - 1)**2)
    f_rank2 = g1_a * g2_a + g1_b * g2_b

    cf1 = axes[0].contourf(X1, X2, g1_a * g2_a, levels=15, cmap="Blues")
    axes[0].set_title("Term 1:  narrow, at $(-1,-1)$", fontsize=11)
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    if fig is not None:
        fig.colorbar(cf1, ax=axes[0])

    cf2 = axes[1].contourf(X1, X2, g1_b * g2_b, levels=15, cmap="Greens")
    axes[1].set_title("Term 2:  broad, at $(+1,+1)$", fontsize=11)
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    if fig is not None:
        fig.colorbar(cf2, ax=axes[1])

    cf3 = axes[2].contourf(X1, X2, f_rank2, levels=20, cmap="Oranges")
    axes[2].set_title(r"Sum = rank-2 function", fontsize=11)
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")
    if fig is not None:
        fig.colorbar(cf3, ax=axes[2])


def plot_core_diagram(ax):
    """function_train_surrogates.qmd → fig-core-diagram

    Rank-2 function as a matrix product of function-valued vectors.
    """
    import matplotlib.pyplot as plt

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 3.2)
    ax.axis("off")

    BLUE = "#2563a8"
    GREEN = "#1a7a4a"
    LIGHT_B = "#dbeafe"
    LIGHT_G = "#d1fae5"
    BORDER = "#374151"

    bw, bh = 1.5, 0.9

    def _box(a, x, y, w, h, color, label, fontsize=10):
        rect = plt.Rectangle(
            (x, y), w, h, facecolor=color,
            edgecolor=BORDER, lw=1.5, zorder=3,
        )
        a.add_patch(rect)
        a.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, zorder=4,
        )

    # Core 1: 1x2 row vector
    c1_y = 1.0
    _box(ax, 0.2, c1_y, bw, bh, LIGHT_B, r"$g_1^{(1)}(x_1)$")
    _box(ax, 0.2 + bw + 0.1, c1_y, bw, bh, LIGHT_B, r"$g_1^{(2)}(x_1)$")
    ax.text(
        0.2 + bw + 0.05, c1_y + bh + 0.35,
        r"$\mathbf{G}_1(x_1)$", fontsize=11, color=BLUE,
        fontweight="bold", ha="center",
    )
    ax.text(
        0.2 + bw + 0.05, c1_y + bh + 0.08,
        r"$1{\times}2$", fontsize=9, color="gray", ha="center",
    )

    # Multiply sign
    mid1_x = 0.2 + 2 * bw + 0.1 + 0.3
    ax.text(
        mid1_x, c1_y + bh / 2, r"$\cdot$",
        fontsize=28, ha="center", va="center", color=BORDER,
    )

    # Core 2: 2x1 column vector
    c2_x = mid1_x + 0.4
    c2_top = c1_y + bh / 2 + 0.05
    c2_bot = c1_y - bh / 2 - 0.05
    _box(ax, c2_x, c2_top, bw, bh, LIGHT_G, r"$g_2^{(1)}(x_2)$")
    _box(ax, c2_x, c2_bot, bw, bh, LIGHT_G, r"$g_2^{(2)}(x_2)$")
    ax.text(
        c2_x + bw / 2, c2_top + bh + 0.35,
        r"$\mathbf{G}_2(x_2)$", fontsize=11, color=GREEN,
        fontweight="bold", ha="center",
    )
    ax.text(
        c2_x + bw / 2, c2_top + bh + 0.08,
        r"$2{\times}1$", fontsize=9, color="gray", ha="center",
    )

    # Rank annotation
    rank_x = (mid1_x + c2_x) / 2
    ax.text(rank_x, c1_y - 0.25, "rank = 2", fontsize=8, color="gray",
            ha="center")

    # Equals sign and result
    eq_x = c2_x + bw + 0.5
    ax.text(
        eq_x, c1_y + bh / 2, r"$=$",
        fontsize=22, ha="center", va="center", color=BORDER,
    )
    res_x = eq_x + 0.5
    ax.text(res_x, c1_y + bh / 2 + 0.4,
            r"$g_1^{(1)}(x_1)\,g_2^{(1)}(x_2)$", fontsize=10)
    ax.text(res_x, c1_y + bh / 2 - 0.1,
            r"$+\; g_1^{(2)}(x_1)\,g_2^{(2)}(x_2)$", fontsize=10)
    ax.text(res_x, c1_y + bh / 2 - 0.65,
            r"$= f(x_1, x_2)$", fontsize=11, color="black",
            fontstyle="italic")

    ax.set_title(
        "A rank-2 function as a matrix product of function-valued vectors",
        fontsize=11, pad=8,
    )


def plot_ft_3var(ax):
    """function_train_surrogates.qmd → fig-ft-3var

    Rank-2 FT for three variables: core diagram with 1x2, 2x2, 2x1 blocks.
    """
    import matplotlib.pyplot as plt

    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 4.2)
    ax.axis("off")

    BLUE = "#2563a8"
    LIGHT_B = "#dbeafe"
    GREEN = "#1a7a4a"
    LIGHT_G = "#d1fae5"
    PURPLE = "#6b3fa0"
    LIGHT_P = "#ede9fe"
    BORDER = "#374151"

    bw, bh, gap = 1.3, 0.85, 0.1

    def _box(a, x, y, w, h, color, label, fontsize=9):
        rect = plt.Rectangle(
            (x, y), w, h, facecolor=color,
            edgecolor=BORDER, lw=1.4, zorder=3,
        )
        a.add_patch(rect)
        a.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, zorder=4,
        )

    cy = 1.6

    # Core 1: 1x2
    c1_x = 0.2
    c1_y = cy - bh / 2
    _box(ax, c1_x, c1_y, bw, bh, LIGHT_B, r"$g_{11}(x_1)$")
    _box(ax, c1_x + bw + gap, c1_y, bw, bh, LIGHT_B, r"$g_{12}(x_1)$")
    c1_mid = c1_x + bw + gap / 2
    ax.text(c1_mid, c1_y + bh + 0.55,
            r"$\mathbf{G}_1(x_1)$", fontsize=11, color=BLUE,
            fontweight="bold", ha="center")
    ax.text(c1_mid, c1_y + bh + 0.2,
            r"$1 \times 2$", fontsize=9, color="gray", ha="center")

    # Dot 1
    dot1_x = c1_x + 2 * bw + gap + 0.35
    ax.text(dot1_x, cy, r"$\cdot$", fontsize=28, ha="center", va="center",
            color=BORDER)

    # Core 2: 2x2
    c2_x = dot1_x + 0.35
    c2_top = cy + gap / 2
    c2_bot = cy - bh - gap / 2
    _box(ax, c2_x, c2_top, bw, bh, LIGHT_G, r"$g_{21}(x_2)$")
    _box(ax, c2_x + bw + gap, c2_top, bw, bh, LIGHT_G, r"$g_{22}(x_2)$")
    _box(ax, c2_x, c2_bot, bw, bh, LIGHT_G, r"$g_{23}(x_2)$")
    _box(ax, c2_x + bw + gap, c2_bot, bw, bh, LIGHT_G, r"$g_{24}(x_2)$")
    c2_mid = c2_x + bw + gap / 2
    ax.text(c2_mid, c2_top + bh + 0.55,
            r"$\mathbf{G}_2(x_2)$", fontsize=11, color=GREEN,
            fontweight="bold", ha="center")
    ax.text(c2_mid, c2_top + bh + 0.2,
            r"$2 \times 2$", fontsize=9, color="gray", ha="center")

    # Dot 2
    dot2_x = c2_x + 2 * bw + gap + 0.35
    ax.text(dot2_x, cy, r"$\cdot$", fontsize=28, ha="center", va="center",
            color=BORDER)

    # Core 3: 2x1
    c3_x = dot2_x + 0.35
    _box(ax, c3_x, c2_top, bw, bh, LIGHT_P, r"$g_{31}(x_3)$")
    _box(ax, c3_x, c2_bot, bw, bh, LIGHT_P, r"$g_{32}(x_3)$")
    ax.text(c3_x + bw / 2, c2_top + bh + 0.55,
            r"$\mathbf{G}_3(x_3)$", fontsize=11, color=PURPLE,
            fontweight="bold", ha="center")
    ax.text(c3_x + bw / 2, c2_top + bh + 0.2,
            r"$2 \times 1$", fontsize=9, color="gray", ha="center")

    # Equals and result
    eq_x = c3_x + bw + 0.45
    ax.text(eq_x, cy, r"$=$", fontsize=22, ha="center", va="center",
            color=BORDER)
    ax.text(eq_x + 0.4, cy + 0.35,
            r"$f(x_1, x_2, x_3)$", fontsize=12, fontstyle="italic")
    ax.text(eq_x + 0.5, cy - 0.15, "(a scalar)", fontsize=9.5, color="gray")

    # Annotations
    ax.annotate(
        "univariate\nfunction", xy=(c1_x + bw / 2, c1_y),
        xytext=(c1_x + bw / 2, -0.35),
        fontsize=8, color="gray", ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.9),
    )
    ax.annotate(
        "core = matrix\nof functions", xy=(c2_mid, c2_bot),
        xytext=(c2_mid, -0.35),
        fontsize=8, color="gray", ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.9),
    )

    ax.set_title(
        r"Function Train for $d=3$, rank $r=2$: "
        r"$f = \mathbf{G}_1(x_1)\cdot\mathbf{G}_2(x_2)\cdot\mathbf{G}_3(x_3)$",
        fontsize=11, pad=8,
    )


def plot_ft_fitted(true_func, ft_fitted, plot_limits, bkd, axes, fig=None):
    """function_train_surrogates.qmd → fig-ft-fitted

    True rank-2 target vs fitted FT surrogate with pointwise error.
    """
    from pyapprox.interface.functions.plot.plot2d_rectangular import (
        Plotter2DRectangularDomain,
        meshgrid_samples,
    )

    npts = 100
    true_plotter = Plotter2DRectangularDomain(true_func, plot_limits)
    ft_plotter = Plotter2DRectangularDomain(ft_fitted, plot_limits)

    X, Y, eval_pts = meshgrid_samples(npts, plot_limits, bkd)
    f_true_vals = bkd.to_numpy(true_func(eval_pts)[0]).reshape(X.shape)
    f_pred_vals = bkd.to_numpy(ft_fitted(eval_pts)[0]).reshape(X.shape)
    err = np.abs(f_true_vals - f_pred_vals)
    rel_err = np.linalg.norm(f_true_vals - f_pred_vals) / np.linalg.norm(
        f_true_vals
    )

    kw = dict(levels=25, cmap="Oranges")
    true_plotter.plot_contours(axes[0], qoi=0, npts_1d=npts, **kw)
    axes[0].set_title("True rank-2 function", fontsize=11)
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")

    ft_plotter.plot_contours(axes[1], qoi=0, npts_1d=npts, **kw)
    axes[1].set_title("FT surrogate (rank 2, degree 12)", fontsize=11)
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")

    cf2 = axes[2].contourf(
        bkd.to_numpy(X), bkd.to_numpy(Y), err, levels=25, cmap="Reds",
    )
    axes[2].set_title(
        f"Pointwise error  (rel. $L_2$ = {rel_err:.2e})", fontsize=11,
    )
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")
    if fig is not None:
        fig.colorbar(cf2, ax=axes[2])


# ---------------------------------------------------------------------------
# function_train_surrogates.qmd — echo:true figure → Convention B
# ---------------------------------------------------------------------------

def plot_ft_rank_comparison(
    true_func, marginals_2d, s_fit, y_fit, eval_pts, X, f_true_vals,
    plot_limits, bkd, axes,
):
    """function_train_surrogates.qmd → fig-ft-rank-comparison

    FT approximation quality vs rank for a rank-2 target.
    """
    from pyapprox.surrogates.functiontrain import (
        create_pce_functiontrain,
        ALSFitter,
    )
    from pyapprox.surrogates.functiontrain.fitters import MSEFitter
    from pyapprox.interface.functions.plot.plot2d_rectangular import (
        Plotter2DRectangularDomain,
    )

    npts = 100
    for ax, rank in zip(axes, [1, 2, 4]):
        ranks_list = [rank] * (2 - 1)
        ft_r = create_pce_functiontrain(
            marginals_2d, max_level=12, ranks=ranks_list, bkd=bkd,
            init_scale=0.1,
        )
        als_r = ALSFitter(bkd, max_sweeps=50, tol=1e-14).fit(
            ft_r, s_fit, y_fit,
        )
        mse_r = MSEFitter(bkd).fit(als_r.surrogate(), s_fit, y_fit)
        ft_r_fit = mse_r.surrogate()

        f_r_vals = bkd.to_numpy(ft_r_fit(eval_pts)[0]).reshape(X.shape)
        rel_r = np.linalg.norm(f_true_vals - f_r_vals) / np.linalg.norm(
            f_true_vals
        )

        plotter_r = Plotter2DRectangularDomain(ft_r_fit, plot_limits)
        plotter_r.plot_contours(
            ax, qoi=0, npts_1d=npts, levels=25, cmap="Oranges",
        )
        ax.set_title(f"rank = {rank},  rel. $L_2$ = {rel_r:.1e}",
                     fontsize=11)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")


# ---------------------------------------------------------------------------
# functiontrain_sensitivity.qmd — echo:true figures → Convention B
# ---------------------------------------------------------------------------

def plot_ft_sobol(S_exact, S_ft, T_exact, T_ft, nvars, axes):
    """functiontrain_sensitivity.qmd → fig-ft-sobol

    FT vs analytical Sobol indices: main effects, total effects, scatter.
    """
    x = np.arange(nvars)
    width = 0.35

    # Main effect comparison
    ax = axes[0]
    ax.bar(x - width / 2, S_exact, width, label='Exact', alpha=0.8)
    ax.bar(x + width / 2, S_ft, width, label='FT', alpha=0.8)
    ax.set_xlabel('Variable')
    ax.set_ylabel('$S_k$')
    ax.set_title('Main Effect Sobol Indices')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$x_{k}$' for k in range(nvars)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total effect comparison
    ax = axes[1]
    ax.bar(x - width / 2, T_exact, width, label='Exact', alpha=0.8)
    ax.bar(x + width / 2, T_ft, width, label='FT', alpha=0.8)
    ax.set_xlabel('Variable')
    ax.set_ylabel('$S_k^T$')
    ax.set_title('Total Effect Sobol Indices')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$x_{k}$' for k in range(nvars)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # First vs total effect scatter
    ax = axes[2]
    ax.scatter(S_ft, T_ft, s=100, alpha=0.8)
    lim = max(max(T_ft), max(T_exact)) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', label='$S_k = S_k^T$')
    for k in range(nvars):
        ax.annotate(
            f'$x_{k}$', (S_ft[k], T_ft[k]),
            textcoords='offset points', xytext=(5, 5), fontsize=10,
        )
    ax.set_xlabel('Main effect $S_k$')
    ax.set_ylabel('Total effect $S_k^T$')
    ax.set_title('Main vs Total Effects')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_ft_decomposition(all_S, ax):
    """functiontrain_sensitivity.qmd → fig-ft-decomposition

    Full Sobol index decomposition bar chart colored by interaction order.
    """
    from matplotlib.patches import Patch

    labels = [
        str(u) for u in sorted(all_S.keys(), key=lambda x: (len(x), x))
    ]
    values = [
        all_S[u] for u in sorted(all_S.keys(), key=lambda x: (len(x), x))
    ]

    colors = [
        'tab:blue' if len(eval(l)) == 1
        else 'tab:orange' if len(eval(l)) == 2
        else 'tab:green'
        for l in labels
    ]

    ax.bar(range(len(values)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f'$S_{{{l[1:-1]}}}$' for l in labels])
    ax.set_ylabel('Sobol Index')
    ax.set_title('Sobol Index Decomposition (Ishigami)')

    legend_elements = [
        Patch(facecolor='tab:blue', alpha=0.8, label='First-order'),
        Patch(facecolor='tab:orange', alpha=0.8, label='Second-order'),
        Patch(facecolor='tab:green', alpha=0.8, label='Third-order'),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# uq_with_functiontrain.qmd — echo:false figure → Convention A
# ---------------------------------------------------------------------------

def plot_ft_vs_mc(model, prior, marginals, bkd, axes):
    """uq_with_functiontrain.qmd → fig-ft-vs-mc

    FT vs MC efficiency: relative error in mean and variance vs budget.
    """
    from pyapprox.surrogates.functiontrain import (
        ALSFitter,
        PCEFunctionTrain,
        create_pce_functiontrain,
    )
    from pyapprox.surrogates.functiontrain.statistics.moments import (
        FunctionTrainMoments,
    )
    from pyapprox.surrogates.quadrature import (
        TensorProductQuadratureRule,
        gauss_quadrature_rule,
    )

    nvars = len(marginals)

    # True moments via tensor-product Gauss quadrature
    nquad = 20
    quad_rules = [
        lambda n, m=m: gauss_quadrature_rule(m, n, bkd) for m in marginals
    ]
    quad_rule = TensorProductQuadratureRule(
        bkd, quad_rules, [nquad] * nvars,
    )
    quad_pts, quad_wts = quad_rule()
    qoi_quad = bkd.to_numpy(model(quad_pts))[0]
    quad_wts_np = bkd.to_numpy(quad_wts)
    true_mean = float(quad_wts_np @ qoi_quad)
    true_var = float(quad_wts_np @ (qoi_quad - true_mean) ** 2)

    # (degree, budget) pairs at oversampling ratio 3
    rank = 3
    degrees = [2, 3, 4, 5]
    budget_degree_pairs = []
    for deg in degrees:
        ft_probe = create_pce_functiontrain(
            marginals, max_level=deg, ranks=[rank] * (nvars - 1), bkd=bkd,
        )
        N_bud = 3 * ft_probe.nparams()
        budget_degree_pairs.append((N_bud, deg))

    N_budget_values = [pair[0] for pair in budget_degree_pairs]
    n_reps = 10

    mc_mean_errors = {N: [] for N in N_budget_values}
    mc_var_errors = {N: [] for N in N_budget_values}
    ft_mean_errors = {N: [] for N in N_budget_values}
    ft_var_errors = {N: [] for N in N_budget_values}

    for N_bud, deg in budget_degree_pairs:
        for rep in range(n_reps):
            np.random.seed(1000 * N_bud + rep)
            s = prior.rvs(N_bud)
            v = bkd.to_numpy(model(s))[0:1, :]

            # MC estimates
            mc_mu = np.mean(v[0, :])
            mc_va = np.var(v[0, :], ddof=1)
            mc_mean_errors[N_bud].append(
                abs(mc_mu - true_mean) / abs(true_mean)
            )
            mc_var_errors[N_bud].append(
                abs(mc_va - true_var) / abs(true_var)
            )

            # FT: ALS fit, then extract analytical moments
            ft_rep = create_pce_functiontrain(
                marginals, max_level=deg,
                ranks=[rank] * (nvars - 1), bkd=bkd,
            )
            try:
                ft_fitted = ALSFitter(
                    bkd, max_sweeps=50, tol=1e-14,
                ).fit(ft_rep, s, v).surrogate()
                pce_ft_rep = PCEFunctionTrain(ft_fitted)
                mom_rep = FunctionTrainMoments(pce_ft_rep)
                ft_mu = float(bkd.to_numpy(mom_rep.mean()))
                ft_va = float(bkd.to_numpy(mom_rep.variance()))
                ft_mean_errors[N_bud].append(
                    abs(ft_mu - true_mean) / abs(true_mean)
                )
                ft_var_errors[N_bud].append(
                    abs(ft_va - true_var) / abs(true_var)
                )
            except Exception:
                ft_mean_errors[N_bud].append(np.nan)
                ft_var_errors[N_bud].append(np.nan)

    deg_labels = [str(d) for _, d in budget_degree_pairs]

    def _plot_convergence(ax, N_vals, mc_errs, ft_errs, ylabel, title,
                          d_labels):
        mc_med = [np.nanmedian(mc_errs[N]) for N in N_vals]
        mc_lo = [np.nanpercentile(mc_errs[N], 10) for N in N_vals]
        mc_hi = [np.nanpercentile(mc_errs[N], 90) for N in N_vals]
        ft_med = [np.nanmedian(ft_errs[N]) for N in N_vals]
        ft_lo = [np.nanpercentile(ft_errs[N], 10) for N in N_vals]
        ft_hi = [np.nanpercentile(ft_errs[N], 90) for N in N_vals]

        ax.fill_between(N_vals, mc_lo, mc_hi, alpha=0.15, color="#2C7FB8")
        ax.semilogy(
            N_vals, mc_med, "o-", ms=5, color="#2C7FB8", lw=1.5,
            label="Monte Carlo",
        )
        ax.fill_between(N_vals, ft_lo, ft_hi, alpha=0.15, color="#E67E22")
        ax.semilogy(
            N_vals, ft_med, "s-", ms=5, color="#E67E22", lw=1.5,
            label="Function Train",
        )
        for N, dlbl in zip(N_vals, d_labels):
            idx = N_vals.index(N)
            ax.annotate(
                f"p={dlbl}", (N, ft_med[idx]),
                textcoords="offset points", xytext=(6, -12),
                fontsize=7, color="#E67E22",
            )
        ax.set_xlabel("Number of model evaluations $N$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, which="both")

    qoi_label = r"$\delta_{\mathrm{tip}}$"
    _plot_convergence(
        axes[0], N_budget_values, mc_mean_errors, ft_mean_errors,
        "Relative error in mean", f"Mean of {qoi_label}", deg_labels,
    )
    _plot_convergence(
        axes[1], N_budget_values, mc_var_errors, ft_var_errors,
        "Relative error in variance", f"Variance of {qoi_label}",
        deg_labels,
    )
