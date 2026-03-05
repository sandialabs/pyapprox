"""Shared plotting patterns reused across multiple tutorial categories."""

import numpy as np


def plot_estimator_histograms(
    mc_values, cv_values, true_value, mc_label, cv_label, ax_mc, ax_cv,
    mc_color="#2C7FB8", cv_color="#E67E22", bins=50,
):
    """Plot side-by-side histograms of MC vs variance-reduced estimator.

    Used by: cv_concept, acv_concept, mlmc_concept, mlblue_analysis.
    """
    from ._style import apply_style

    xlim = (
        min(mc_values.min(), cv_values.min()),
        max(mc_values.max(), cv_values.max()),
    )
    pad = 0.05 * (xlim[1] - xlim[0])
    xlim = (xlim[0] - pad, xlim[1] + pad)

    for ax, vals, color, label in [
        (ax_mc, mc_values, mc_color, mc_label),
        (ax_cv, cv_values, cv_color, cv_label),
    ]:
        ax.hist(vals, bins=bins, density=True, color=color, alpha=0.7,
                edgecolor="k", lw=0.3)
        ax.axvline(true_value, color="#C0392B", ls="--", lw=2,
                   label="True mean")
        ax.set_xlabel(r"Estimate of $\mu_\alpha$", fontsize=11)
        var_val = vals.var()
        ratio = cv_values.var() / mc_values.var() if ax is ax_cv else None
        subtitle = f"Var = {var_val:.2e}"
        if ratio is not None:
            subtitle += f"  ({ratio:.0%} of MC)"
        ax.set_title(f"{label}\n{subtitle}", fontsize=10)
        ax.set_xlim(xlim)
        ax.legend(fontsize=9)
        apply_style(ax)

    ax_mc.set_ylabel("Density", fontsize=11)


def plot_loglog_convergence(
    ns, errors, labels, colors, ax, xlabel="N", ylabel="Error",
    reference_slopes=None,
):
    """Plot log-log convergence of errors vs number of points.

    Used by: gauss_quadrature, lagrange_interp, pce, sparse_grids.
    """
    from ._style import apply_style

    for n, err, label, color in zip(ns, errors, labels, colors):
        ax.loglog(n, err, "o-", color=color, label=label, ms=5)

    if reference_slopes is not None:
        for slope, sl_label, sl_color in reference_slopes:
            x = np.array([ns[0][0], ns[0][-1]])
            y0 = errors[0][0]
            y = y0 * (x / x[0]) ** slope
            ax.loglog(x, y, "--", color=sl_color, alpha=0.5, label=sl_label)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    apply_style(ax)
