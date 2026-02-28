"""Plotting functions for forward UQ tutorials.

Covers: forward_uq.qmd, random_variables_distributions.qmd,
        monte_carlo_sampling.qmd, estimator_accuracy_mse.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# forward_uq.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_forward_uq_schematic(ax_in, ax_mid, ax_out):
    """forward_uq.qmd → fig-forward-uq-schematic

    Input distribution → model box → push-forward distribution.
    """
    from scipy.stats import norm
    from matplotlib.patches import FancyBboxPatch
    from ._style import apply_style

    # Left: input distribution (2D contour)
    t1 = np.linspace(-3, 3, 200)
    t2 = np.linspace(-3, 3, 200)
    T1, T2 = np.meshgrid(t1, t2)
    rho = 0.3
    Z = (T1**2 - 2 * rho * T1 * T2 + T2**2) / (2 * (1 - rho**2))
    pdf_2d = np.exp(-Z) / (2 * np.pi * np.sqrt(1 - rho**2))
    ax_in.contourf(T1, T2, pdf_2d, levels=8, cmap="Blues", alpha=0.8)
    ax_in.contour(T1, T2, pdf_2d, levels=6, colors="steelblue", linewidths=0.5)
    ax_in.set_xlabel(r"$\theta_1$", fontsize=12)
    ax_in.set_ylabel(r"$\theta_2$", fontsize=12)
    ax_in.set_title(r"Input distribution $p(\boldsymbol{\theta})$", fontsize=11)
    ax_in.set_aspect("equal")
    ax_in.set_xlim(-3, 3)
    ax_in.set_ylim(-3, 3)

    # Center: model box with arrow
    ax_mid.axis("off")
    box = FancyBboxPatch(
        (0.15, 0.35), 0.7, 0.3, boxstyle="round,pad=0.05",
        facecolor="#F0F0F0", edgecolor="k", linewidth=1.5,
    )
    ax_mid.add_patch(box)
    ax_mid.text(
        0.5, 0.5, r"$f(\boldsymbol{\theta})$",
        ha="center", va="center", fontsize=14, transform=ax_mid.transAxes,
    )
    ax_mid.annotate(
        "", xy=(0.12, 0.5), xytext=(-0.15, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2, color="#2C7FB8"),
    )
    ax_mid.annotate(
        "", xy=(1.15, 0.5), xytext=(0.88, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2, color="#2C7FB8"),
    )

    # Right: output distribution (1D)
    q_vals = np.linspace(-1, 5, 300)
    pdf_out = 0.7 * norm.pdf(q_vals, 1.5, 0.8) + 0.3 * norm.pdf(q_vals, 3.0, 0.5)
    ax_out.fill_between(q_vals, pdf_out, alpha=0.3, color="#E67E22")
    ax_out.plot(q_vals, pdf_out, color="#E67E22", lw=2)
    ax_out.set_xlabel(r"$q = f(\boldsymbol{\theta})$", fontsize=12)
    ax_out.set_ylabel("Density", fontsize=12)
    ax_out.set_title("Push-forward distribution", fontsize=11)
    ax_out.set_yticks([])
    ax_out.set_xlim(-1, 5)


def plot_linear_propagation(ax_in, ax_mid, ax_out):
    """forward_uq.qmd → fig-linear-propagation

    Linear propagation: Gaussian in → Gaussian out.
    """
    from scipy.stats import norm
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.pde.galerkin.physics.euler_bernoulli import (
        EulerBernoulliBeamFEM,
    )
    from ._style import apply_style

    bkd = NumpyBkd()
    L, H, q0_nom = 100.0, 30.0, 10.0
    A_skin = 2 * 5.0 * 1.0
    A_core = 20.0 * 1.0
    E_eff = (A_skin * 2e4 + A_core * 5e3) / (A_skin + A_core)
    I_rect = H**3 / 12.0

    beam_ref = EulerBernoulliBeamFEM(
        nx=40, length=L, EI=E_eff * I_rect,
        load_func=lambda x: 1.0 * x / L, bkd=bkd,
    )
    c_linear = float(beam_ref.tip_deflection())

    rng = np.random.default_rng(42)
    nsamples = 5000
    q0_gaussian = rng.normal(loc=q0_nom, scale=2.0, size=nsamples)
    delta_gaussian = c_linear * q0_gaussian

    ax_in.hist(
        q0_gaussian, bins=50, density=True, color="#2C7FB8",
        alpha=0.7, edgecolor="k", lw=0.3,
    )
    q_grid = np.linspace(q0_gaussian.min(), q0_gaussian.max(), 300)
    ax_in.plot(q_grid, norm.pdf(q_grid, q0_nom, 2.0), "k-", lw=1.5)
    ax_in.set_xlabel(r"$q_0$", fontsize=12)
    ax_in.set_ylabel("Density", fontsize=12)
    ax_in.set_title(r"Input: $q_0 \sim \mathcal{N}(10,\, 2^2)$", fontsize=10)
    apply_style(ax_in)

    ax_mid.axis("off")
    ax_mid.annotate(
        "", xy=(1.0, 0.5), xytext=(0.0, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#2C7FB8"),
    )
    ax_mid.text(
        0.5, 0.62, r"$\delta_{\mathrm{tip}} = c \cdot q_0$",
        ha="center", va="center", fontsize=11, color="#666",
        transform=ax_mid.transAxes,
    )

    ax_out.hist(
        delta_gaussian, bins=50, density=True, color="#E67E22",
        alpha=0.7, edgecolor="k", lw=0.3,
    )
    d_grid = np.linspace(delta_gaussian.min(), delta_gaussian.max(), 300)
    ax_out.plot(
        d_grid, norm.pdf(d_grid, c_linear * q0_nom, abs(c_linear) * 2.0),
        "k-", lw=1.5,
    )
    ax_out.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=12)
    ax_out.set_ylabel("Density", fontsize=12)
    ax_out.set_title("Push-forward: Gaussian shape preserved", fontsize=10)
    apply_style(ax_out)


def plot_nonlinear_propagation(ax_in, ax_mid, ax_out):
    """forward_uq.qmd → fig-nonlinear-propagation

    Nonlinear (1/E) propagation: Gaussian in → skewed out.
    """
    from scipy.stats import norm
    from ._style import apply_style

    L, H, q0_nom = 100.0, 30.0, 10.0
    A_skin = 2 * 5.0 * 1.0
    A_core = 20.0 * 1.0
    E_eff = (A_skin * 2e4 + A_core * 5e3) / (A_skin + A_core)
    I_rect = H**3 / 12.0
    E_mean = E_eff
    E_std = 0.15 * E_eff

    rng = np.random.default_rng(42)
    nsamples = 5000
    E_samples = rng.normal(loc=E_mean, scale=E_std, size=nsamples)
    E_samples = E_samples[E_samples > 0]
    delta_nonlinear = (11 * q0_nom * L**4) / (120 * E_samples * I_rect)

    ax_in.hist(
        E_samples, bins=60, density=True, color="#2C7FB8",
        alpha=0.7, edgecolor="k", lw=0.3,
    )
    e_grid = np.linspace(E_samples.min(), E_samples.max(), 300)
    ax_in.plot(e_grid, norm.pdf(e_grid, E_mean, E_std), "k-", lw=1.5)
    ax_in.set_xlabel(r"$E$", fontsize=12)
    ax_in.set_ylabel("Density", fontsize=12)
    ax_in.set_title(
        r"Input: $E \sim \mathcal{N}(\bar{E},\, (0.15\bar{E})^2)$",
        fontsize=10,
    )
    apply_style(ax_in)

    ax_mid.axis("off")
    ax_mid.annotate(
        "", xy=(1.0, 0.5), xytext=(0.0, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#2C7FB8"),
    )
    ax_mid.text(
        0.5, 0.62, r"$f \propto 1/E$", ha="center", va="center",
        fontsize=11, color="#666", transform=ax_mid.transAxes,
    )

    ax_out.hist(
        delta_nonlinear, bins=60, density=True, color="#E67E22",
        alpha=0.7, edgecolor="k", lw=0.3,
    )
    d_mean = np.mean(delta_nonlinear)
    d_std = np.std(delta_nonlinear)
    d_grid = np.linspace(delta_nonlinear.min(), delta_nonlinear.max(), 300)
    ax_out.plot(
        d_grid, norm.pdf(d_grid, d_mean, d_std), "r--", lw=1.5,
        label="Gaussian (same mean, var)",
    )
    ax_out.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=12)
    ax_out.set_ylabel("Density", fontsize=12)
    ax_out.set_title("Push-forward: skewed (non-Gaussian)", fontsize=10)
    ax_out.legend(fontsize=9)
    apply_style(ax_out)


def plot_different_priors(ax1, ax2):
    """forward_uq.qmd → fig-different-priors

    Same model, two different input distributions on E.
    """
    from ._style import apply_style

    L, H, q0_nom = 100.0, 30.0, 10.0
    A_skin = 2 * 5.0 * 1.0
    A_core = 20.0 * 1.0
    E_mean = (A_skin * 2e4 + A_core * 5e3) / (A_skin + A_core)
    I_rect = H**3 / 12.0

    rng = np.random.default_rng(42)
    nsamples = 5000
    E_narrow = rng.normal(loc=E_mean, scale=0.10 * E_mean, size=nsamples)
    E_wide = rng.normal(loc=E_mean, scale=0.25 * E_mean, size=nsamples)
    E_narrow = E_narrow[E_narrow > 0]
    E_wide = E_wide[E_wide > 0]

    delta_narrow = (11 * q0_nom * L**4) / (120 * E_narrow * I_rect)
    delta_wide = (11 * q0_nom * L**4) / (120 * E_wide * I_rect)

    ax1.hist(
        E_narrow, bins=60, density=True, color="#2C7FB8", alpha=0.4,
        edgecolor="none", label="Narrow (10% CV)",
    )
    ax1.hist(
        E_wide, bins=60, density=True, color="#C0392B", alpha=0.4,
        edgecolor="none", label="Wide (25% CV)",
    )
    ax1.set_xlabel(r"$E$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Input distributions", fontsize=11)
    ax1.legend(fontsize=9)
    apply_style(ax1)

    ax2.hist(
        delta_narrow, bins=60, density=True, color="#2C7FB8", alpha=0.4,
        edgecolor="none", label="From narrow input",
    )
    ax2.hist(
        delta_wide, bins=60, density=True, color="#C0392B", alpha=0.4,
        edgecolor="none", label="From wide input",
    )
    ax2.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Push-forward distributions", fontsize=11)
    ax2.legend(fontsize=9)
    apply_style(ax2)


def plot_different_models(ax1, ax2):
    """forward_uq.qmd → fig-different-models

    Same input, two models with different beam lengths.
    """
    from ._style import apply_style

    L, H, q0_nom = 100.0, 30.0, 10.0
    A_skin = 2 * 5.0 * 1.0
    A_core = 20.0 * 1.0
    E_mean = (A_skin * 2e4 + A_core * 5e3) / (A_skin + A_core)
    I_rect = H**3 / 12.0

    rng = np.random.default_rng(42)
    nsamples = 5000
    E_common = rng.normal(loc=E_mean, scale=0.15 * E_mean, size=nsamples)
    E_common = E_common[E_common > 0]

    L1, L2 = 100.0, 110.0
    delta_L1 = (11 * q0_nom * L1**4) / (120 * E_common * I_rect)
    delta_L2 = (11 * q0_nom * L2**4) / (120 * E_common * I_rect)

    ax1.hist(
        E_common, bins=60, density=True, color="#2C7FB8", alpha=0.7,
        edgecolor="k", lw=0.3,
    )
    ax1.set_xlabel(r"$E$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Shared input distribution", fontsize=11)
    apply_style(ax1)

    ax2.hist(
        delta_L1, bins=60, density=True, color="#2C7FB8", alpha=0.4,
        edgecolor="none", label=r"$L_1 = 100$",
    )
    ax2.hist(
        delta_L2, bins=60, density=True, color="#27AE60", alpha=0.4,
        edgecolor="none", label=r"$L_2 = 110$",
    )
    ax2.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Push-forward distributions", fontsize=11)
    ax2.legend(fontsize=9)
    apply_style(ax2)


def plot_summary_statistics(delta_nonlinear, ax):
    """forward_uq.qmd → fig-summary-statistics

    Push-forward PDF with mean, median, CI, failure region.
    Accepts pre-computed nonlinear delta samples.
    """
    from scipy.stats import gaussian_kde
    from ._style import apply_style

    kde = gaussian_kde(delta_nonlinear)
    d_grid = np.linspace(
        np.percentile(delta_nonlinear, 0.5),
        np.percentile(delta_nonlinear, 99.5),
        500,
    )
    pdf_vals = kde(d_grid)

    d_mean = np.mean(delta_nonlinear)
    d_median = np.median(delta_nonlinear)
    d_std = np.std(delta_nonlinear)
    q05 = np.percentile(delta_nonlinear, 5)
    q95 = np.percentile(delta_nonlinear, 95)
    threshold = d_mean + 2.0 * d_std

    ax.fill_between(d_grid, pdf_vals, alpha=0.15, color="#E67E22")
    ax.plot(d_grid, pdf_vals, color="#E67E22", lw=2, label="Push-forward PDF")

    mask_ci = (d_grid >= q05) & (d_grid <= q95)
    ax.fill_between(
        d_grid[mask_ci], pdf_vals[mask_ci], alpha=0.3, color="#2C7FB8",
        label=f"90% interval [{q05:.3f}, {q95:.3f}]",
    )

    mask_fail = d_grid >= threshold
    ax.fill_between(
        d_grid[mask_fail], pdf_vals[mask_fail], alpha=0.5, color="#C0392B",
        label=(
            f"$P(\\delta_{{\\mathrm{{tip}}}} > {threshold:.3f})$"
            f" = {np.mean(delta_nonlinear > threshold):.1%}"
        ),
    )

    ax.axvline(d_mean, color="#27AE60", ls="-", lw=2,
               label=f"Mean = {d_mean:.4f}")
    ax.axvline(d_median, color="#8E44AD", ls="--", lw=1.5,
               label=f"Median = {d_median:.4f}")
    ax.axvline(threshold, color="#C0392B", ls="-", lw=1.5)
    ax.text(
        threshold + 0.0002, max(pdf_vals) * 0.85,
        r"$\delta_{\mathrm{crit}}$",
        fontsize=12, color="#C0392B", ha="left",
    )

    bracket_y = max(pdf_vals) * 0.45
    ax.annotate(
        "", xy=(d_mean - d_std, bracket_y),
        xytext=(d_mean + d_std, bracket_y),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="#27AE60"),
    )
    ax.text(
        d_mean, bracket_y + max(pdf_vals) * 0.04,
        rf"$\pm\sigma$ = $\pm${d_std:.4f}",
        ha="center", fontsize=9, color="#27AE60",
    )

    ax.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Push-forward distribution with summary statistics",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(d_grid[0], d_grid[-1])
    apply_style(ax)


# ---------------------------------------------------------------------------
# random_variables_distributions.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_indep_beta(variable_indep, beta_E1, beta_E2, samples_indep, bkd,
                    ax_main, ax_top, ax_right):
    """random_variables_distributions.qmd → fig-indep-beta

    Joint scatter with marginal histograms for independent Beta prior.
    """
    domain = variable_indep.domain()
    E1_lo, E1_hi = float(domain[0, 0]), float(domain[0, 1])
    E2_lo, E2_hi = float(domain[1, 0]), float(domain[1, 1])

    plotter = variable_indep.plotter()
    plotter.plot_contours(ax_main, npts_1d=80, levels=8, cmap="Blues",
                          alpha=0.6)

    samples_np = bkd.to_numpy(samples_indep)
    ax_main.scatter(
        samples_np[0], samples_np[1], alpha=0.1, s=3, color="#2C7FB8",
        zorder=0,
    )
    ax_main.set_xlabel(r"$E_1$", fontsize=12)
    ax_main.set_ylabel(r"$E_2$", fontsize=12)
    ax_main.set_xlim(E1_lo - 200, E1_hi + 200)
    ax_main.set_ylim(E2_lo - 50, E2_hi + 50)

    ax_top.hist(
        samples_np[0], bins=40, density=True, color="#2C7FB8", alpha=0.5,
        edgecolor="none",
    )
    e1_grid = bkd.reshape(bkd.linspace(E1_lo, E1_hi, 200), (1, -1))
    ax_top.plot(
        bkd.to_numpy(e1_grid[0]),
        bkd.to_numpy(beta_E1.pdf(e1_grid)[0]), "k-", lw=1.2,
    )
    ax_top.set_xlim(E1_lo - 200, E1_hi + 200)
    ax_top.set_yticks([])
    ax_top.set_xticks([])

    ax_right.hist(
        samples_np[1], bins=40, density=True, color="#2C7FB8", alpha=0.5,
        edgecolor="none", orientation="horizontal",
    )
    e2_grid = bkd.reshape(bkd.linspace(E2_lo, E2_hi, 200), (1, -1))
    ax_right.plot(
        bkd.to_numpy(beta_E2.pdf(e2_grid)[0]),
        bkd.to_numpy(e2_grid[0]), "k-", lw=1.2,
    )
    ax_right.set_ylim(E2_lo - 50, E2_hi + 50)
    ax_right.set_xticks([])
    ax_right.set_yticks([])


def plot_copula_beta(variable_copula, beta_E1, beta_E2, samples_copula,
                     bkd, ax_main, ax_top, ax_right):
    """random_variables_distributions.qmd → fig-copula-beta

    Joint scatter with marginal histograms for copula Beta prior.
    """
    domain = variable_copula.domain()
    E1_lo, E1_hi = float(domain[0, 0]), float(domain[0, 1])
    E2_lo, E2_hi = float(domain[1, 0]), float(domain[1, 1])

    plotter_cop = variable_copula.plotter()
    plotter_cop.plot_contours(ax_main, npts_1d=80, levels=8, cmap="Greens",
                              alpha=0.6)

    samples_np = bkd.to_numpy(samples_copula)
    ax_main.scatter(
        samples_np[0], samples_np[1], alpha=0.1, s=3, color="#27AE60",
        zorder=0,
    )
    ax_main.set_xlabel(r"$E_1$", fontsize=12)
    ax_main.set_ylabel(r"$E_2$", fontsize=12)
    ax_main.set_xlim(E1_lo - 200, E1_hi + 200)
    ax_main.set_ylim(E2_lo - 50, E2_hi + 50)

    ax_top.hist(
        samples_np[0], bins=40, density=True, color="#27AE60", alpha=0.5,
        edgecolor="none",
    )
    e1_grid = bkd.reshape(bkd.linspace(E1_lo, E1_hi, 200), (1, -1))
    ax_top.plot(
        bkd.to_numpy(e1_grid[0]),
        bkd.to_numpy(beta_E1.pdf(e1_grid)[0]), "k-", lw=1.2,
    )
    ax_top.set_xlim(E1_lo - 200, E1_hi + 200)
    ax_top.set_yticks([])
    ax_top.set_xticks([])

    ax_right.hist(
        samples_np[1], bins=40, density=True, color="#27AE60", alpha=0.5,
        edgecolor="none", orientation="horizontal",
    )
    e2_grid = bkd.reshape(bkd.linspace(E2_lo, E2_hi, 200), (1, -1))
    ax_right.plot(
        bkd.to_numpy(beta_E2.pdf(e2_grid)[0]),
        bkd.to_numpy(e2_grid[0]), "k-", lw=1.2,
    )
    ax_right.set_ylim(E2_lo - 50, E2_hi + 50)
    ax_right.set_xticks([])
    ax_right.set_yticks([])


def plot_two_priors(plotter_indep, samples_indep_np, plotter_copula,
                    samples_copula_np, domain, ax1, ax2):
    """random_variables_distributions.qmd → fig-two-priors

    Side-by-side comparison of independent vs copula priors.
    """
    E1_lo, E1_hi = float(domain[0, 0]), float(domain[0, 1])
    E2_lo, E2_hi = float(domain[1, 0]), float(domain[1, 1])

    plotter_indep.plot_contours(ax1, npts_1d=80, levels=8, cmap="Blues",
                                alpha=0.5)
    ax1.scatter(
        samples_indep_np[0], samples_indep_np[1], alpha=0.08, s=3,
        color="#2C7FB8", zorder=0,
    )
    ax1.set_xlabel(r"$E_1$", fontsize=12)
    ax1.set_ylabel(r"$E_2$", fontsize=12)
    ax1.set_title("Independent Beta marginals", fontsize=11)
    ax1.set_xlim(E1_lo - 200, E1_hi + 200)
    ax1.set_ylim(E2_lo - 50, E2_hi + 50)

    plotter_copula.plot_contours(ax2, npts_1d=80, levels=8, cmap="Greens",
                                 alpha=0.5)
    ax2.scatter(
        samples_copula_np[0], samples_copula_np[1], alpha=0.08, s=3,
        color="#27AE60", zorder=0,
    )
    ax2.set_xlabel(r"$E_1$", fontsize=12)
    ax2.set_ylabel(r"$E_2$", fontsize=12)
    ax2.set_title(r"Gaussian copula ($\rho = 0.95$)", fontsize=11)
    ax2.set_xlim(E1_lo - 200, E1_hi + 200)
    ax2.set_ylim(E2_lo - 50, E2_hi + 50)


# ---------------------------------------------------------------------------
# monte_carlo_sampling.qmd — mixed conventions
# ---------------------------------------------------------------------------

def plot_beam_deflections(benchmark, bkd, ax1, ax2, cbar_fig=None):
    """monte_carlo_sampling.qmd → fig-beam-deflections (Convention A)

    Deflected beam shapes for two KLE parameter values, colored by EI.
    """
    from pyapprox.pde.galerkin.physics.euler_bernoulli import (
        EulerBernoulliBeamFEM,
    )
    from pyapprox.pde.field_maps.kle_factory import (
        create_lognormal_kle_field_map,
    )
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    nx, length, EI_mean, q0 = 40, 100.0, 1e6, 1.0
    sigma, corr_len, num_kle = 0.3, 0.3, 2

    midpoints = np.linspace(
        length / (2 * nx), length - length / (2 * nx), nx,
    )
    mesh_coords_norm = bkd.asarray(midpoints[np.newaxis, :] / length)
    mean_log = bkd.full((nx,), float(np.log(EI_mean)))
    elem_lengths = bkd.full((nx,), 1.0 / nx)
    field_map = create_lognormal_kle_field_map(
        mesh_coords=mesh_coords_norm, mean_log_field=mean_log, bkd=bkd,
        num_kle_terms=num_kle, sigma=sigma, correlation_length=corr_len,
        quad_weights=elem_lengths,
    )

    beam = EulerBernoulliBeamFEM(
        nx=nx, length=length, EI=EI_mean,
        load_func=lambda x: q0 * x / length, bkd=bkd,
    )
    x_nodes = beam.node_coordinates()

    xi_vals = [np.array([-1.5, 1.0]), np.array([1.5, -1.0])]
    xi_labels = [
        r"$\boldsymbol{\xi} = (-1.5,\; 1.0)$",
        r"$\boldsymbol{\xi} = (1.5,\; -1.0)$",
    ]

    EI_fields, deflections = [], []
    for xi in xi_vals:
        EI_field = bkd.to_numpy(field_map(bkd.asarray(xi)))
        beam.set_EI(EI_field)
        EI_fields.append(EI_field)
        deflections.append(bkd.to_numpy(beam.deflection_at_nodes()))

    all_EI = np.concatenate(EI_fields)
    norm_obj = Normalize(vmin=all_EI.min(), vmax=all_EI.max())

    all_w = np.concatenate(deflections)
    w_max = max(abs(all_w.min()), abs(all_w.max()))
    y_lim = (-w_max * 1.3, w_max * 0.3)

    for ax, EI_field, w, label in zip(
        [ax1, ax2], EI_fields, deflections, xi_labels,
    ):
        EI_seg = 0.5 * (EI_field[:-1] + EI_field[1:])
        EI_seg = np.concatenate([[EI_field[0]], EI_seg, [EI_field[-1]]])

        points = np.column_stack([x_nodes, -w])
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        lc = LineCollection(segments, cmap="magma", norm=norm_obj, lw=3.5)
        lc.set_array(EI_seg)
        ax.add_collection(lc)

        ax.plot(x_nodes, np.zeros_like(x_nodes), "k--", lw=0.8, alpha=0.4)
        ax.plot([0, 0], [y_lim[0] * 0.8, y_lim[1] * 2], "k-", lw=3)

        ax.set_xlim(-5, length + 5)
        ax.set_ylim(y_lim)
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("Deflection $w(x)$", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.2)

    if cbar_fig is not None:
        sm = plt.cm.ScalarMappable(cmap="magma", norm=norm_obj)
        cbar = cbar_fig.colorbar(
            sm, ax=[ax1, ax2], shrink=0.8, pad=0.04,
        )
        cbar.set_label("$EI(x)$", fontsize=11)


def plot_samples_and_surface(model, samples, qoi_values, bkd, ax1, ax2,
                             fig=None):
    """monte_carlo_sampling.qmd → fig-samples-and-surface (Convention A)

    Samples in input space and response surface with samples overlaid.
    """
    from pyapprox.interface.functions.plot.plot2d_rectangular import (
        Plotter2DRectangularDomain,
    )
    from matplotlib.patches import FancyArrowPatch
    from ._style import apply_style

    samples_np = bkd.to_numpy(samples)
    qoi_np = bkd.to_numpy(qoi_values)
    plot_limits = [-3.5, 3.5, -3.5, 3.5]

    ax1.scatter(
        samples_np[0], samples_np[1], c=qoi_np, cmap="magma",
        s=30, edgecolors="w", lw=0.4, zorder=3,
    )
    ax1.set_xlabel(r"$\xi_1$", fontsize=12)
    ax1.set_ylabel(r"$\xi_2$", fontsize=12)
    ax1.set_title("Samples in input space", fontsize=11)
    if fig is not None:
        cbar1 = fig.colorbar(ax1.collections[0], ax=ax1, shrink=0.8)
        cbar1.set_label(r"$\delta_{\mathrm{tip}}$")
    apply_style(ax1)

    plotter = Plotter2DRectangularDomain(model, plot_limits)
    cs = plotter.plot_contours(
        ax2, qoi=0, npts_1d=40, levels=20, cmap="magma", alpha=0.8,
    )
    ax2.scatter(
        samples_np[0], samples_np[1], c=qoi_np, cmap="magma",
        s=30, edgecolors="w", lw=0.5, zorder=3,
    )
    ax2.set_xlabel(r"$\xi_1$", fontsize=12)
    ax2.set_ylabel(r"$\xi_2$", fontsize=12)
    ax2.set_title("Response surface with samples", fontsize=11)
    if fig is not None:
        cbar2 = fig.colorbar(cs, ax=ax2, shrink=0.8)
        cbar2.set_label(r"$\delta_{\mathrm{tip}}$")

        arrow = FancyArrowPatch(
            (0.44, 0.5), (0.52, 0.5),
            transform=fig.transFigure,
            arrowstyle="->,head_length=8,head_width=5",
            lw=2, color="0.3", clip_on=False,
        )
        fig.patches.append(arrow)
        fig.text(
            0.48, 0.56, r"$f(\boldsymbol{\xi})$",
            ha="center", va="bottom", fontsize=14, color="0.3",
            transform=fig.transFigure,
        )


def plot_mc_variability_histogram(means, true_mean, n_reps, M, ax):
    """monte_carlo_sampling.qmd → fig-mc-variability-small (Convention B)

    Histogram of MC mean estimates from repeated experiments.
    """
    from ._style import apply_style

    ax.hist(
        means, bins=30, density=True, color="#2C7FB8", alpha=0.7,
        edgecolor="k", lw=0.3,
    )
    ax.axvline(
        true_mean, color="#C0392B", ls="--", lw=2,
        label=f"True mean = {true_mean:.6f}",
    )
    ax.axvline(
        np.mean(means), color="#27AE60", ls="-", lw=2,
        label=f"Mean of estimates = {np.mean(means):.6f}",
    )
    ax.set_xlabel(rf"$\hat{{\mu}}_{{{M}}}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Distribution of MC mean estimate ($M = {M}$, {n_reps} repetitions)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    apply_style(ax)


def plot_mc_variability_comparison(means_small, means_large, true_mean,
                                   M_small, M_large, ax1, ax2):
    """monte_carlo_sampling.qmd → fig-mc-variability-large (Convention B)

    Side-by-side histograms comparing M=10 vs M=100.
    """
    from ._style import apply_style

    xlim_shared = (
        min(means_small.min(), means_large.min()),
        max(means_small.max(), means_large.max()),
    )
    pad = 0.05 * (xlim_shared[1] - xlim_shared[0])
    xlim_shared = (xlim_shared[0] - pad, xlim_shared[1] + pad)

    ax1.hist(
        means_small, bins=30, density=True, color="#2C7FB8", alpha=0.7,
        edgecolor="k", lw=0.3,
    )
    ax1.axvline(true_mean, color="#C0392B", ls="--", lw=2)
    ax1.axvline(np.mean(means_small), color="#27AE60", ls="-", lw=1.5)
    ax1.set_xlabel(r"$\hat{\mu}_M$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(f"$M = {M_small}$", fontsize=11)
    ax1.set_xlim(xlim_shared)
    apply_style(ax1)

    ax2.hist(
        means_large, bins=30, density=True, color="#E67E22", alpha=0.7,
        edgecolor="k", lw=0.3,
    )
    ax2.axvline(
        true_mean, color="#C0392B", ls="--", lw=2, label="True mean",
    )
    ax2.axvline(
        np.mean(means_large), color="#27AE60", ls="-", lw=1.5,
        label="Mean of estimates",
    )
    ax2.set_xlabel(r"$\hat{\mu}_M$", fontsize=12)
    ax2.set_title(f"$M = {M_large}$", fontsize=11)
    ax2.set_xlim(xlim_shared)
    ax2.legend(fontsize=9)
    apply_style(ax2)


def plot_multi_qoi_scatter(qoi_multi_np, qoi_labels, axes):
    """monte_carlo_sampling.qmd → fig-multi-qoi-scatter (Convention B)

    Pairwise scatter of three QoIs.
    """
    from ._style import apply_style

    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(
            qoi_multi_np[i], qoi_multi_np[j],
            c="#2C7FB8", s=20, edgecolors="k", lw=0.3, alpha=0.7,
        )
        ax.set_xlabel(qoi_labels[i], fontsize=10)
        ax.set_ylabel(qoi_labels[j], fontsize=10)
        corr = np.corrcoef(qoi_multi_np[i], qoi_multi_np[j])[0, 1]
        ax.set_title(f"$\\rho = {corr:.2f}$", fontsize=11)
        apply_style(ax)


# ---------------------------------------------------------------------------
# estimator_accuracy_mse.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_mean_mse_convergence(model, variable, true_var, bkd, ax):
    """estimator_accuracy_mse.qmd → fig-mean-mse-convergence

    Log-log convergence of MC mean estimator variance.
    """
    from ._style import apply_style

    n_reps = 500
    M_values = [10, 20, 50, 100, 200, 500, 1000]

    empirical_var_of_mean = np.empty(len(M_values))
    for j, M in enumerate(M_values):
        means = np.empty(n_reps)
        for i in range(n_reps):
            np.random.seed(i + j * n_reps)
            s = variable.rvs(M)
            means[i] = np.mean(bkd.to_numpy(model(s))[0])
        empirical_var_of_mean[j] = np.var(means)

    ax.loglog(
        M_values, empirical_var_of_mean, "o", ms=7, color="#2C7FB8",
        label="Empirical Var[$\\hat{\\mu}_M$]", zorder=3,
    )
    M_theory = np.array([5, 2000])
    ax.loglog(
        M_theory, true_var / M_theory, "--", color="#C0392B", lw=1.5,
        label=r"$\sigma^2 / M$",
    )
    ax.set_xlabel("$M$ (number of samples)", fontsize=12)
    ax.set_ylabel("Var[$\\hat{\\mu}_M$]", fontsize=12)
    ax.set_title("Convergence of the MC mean estimator", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")

    return empirical_var_of_mean, M_values


def plot_mean_vs_var_convergence(model, variable, true_var, true_kappa4,
                                 empirical_var_of_mean, M_values, bkd, ax):
    """estimator_accuracy_mse.qmd → fig-mean-vs-var-convergence

    Convergence comparison: mean vs variance estimator.
    """
    n_reps = 500
    empirical_var_of_var = np.empty(len(M_values))
    for j, M in enumerate(M_values):
        variances = np.empty(n_reps)
        for i in range(n_reps):
            np.random.seed(i + j * n_reps + 100000)
            s = variable.rvs(M)
            q = bkd.to_numpy(model(s))[0]
            variances[i] = np.var(q, ddof=1)
        empirical_var_of_var[j] = np.var(variances)

    M_theory = np.array([5, 2000])
    ax.loglog(
        M_values, empirical_var_of_mean, "o", ms=7, color="#2C7FB8",
        label="Var[$\\hat{\\mu}_M$] (empirical)", zorder=3,
    )
    ax.loglog(
        M_theory, true_var / M_theory, "--", color="#2C7FB8", lw=1.5,
        label=r"$\sigma^2 / M$",
    )
    ax.loglog(
        M_values, empirical_var_of_var, "s", ms=7, color="#E67E22",
        label="Var[$\\hat{\\sigma}^2_M$] (empirical)", zorder=3,
    )
    ax.loglog(
        M_theory, (true_kappa4 - true_var**2) / M_theory, "--",
        color="#E67E22", lw=1.5,
        label=r"$(\kappa_4 - \sigma^4) / M$",
    )
    ax.set_xlabel("$M$ (number of samples)", fontsize=12)
    ax.set_ylabel("Estimator variance", fontsize=12)
    ax.set_title("Mean vs. variance estimator convergence", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")


def plot_estimator_covariance(empirical_est_cov, theoretical_est_cov,
                               labels, ax1, ax2, fig=None):
    """estimator_accuracy_mse.qmd → fig-estimator-covariance-empirical

    Side-by-side heatmaps of empirical vs theoretical estimator covariance.
    """
    n_stats = len(labels)
    vmin = min(empirical_est_cov.min(), theoretical_est_cov.min())
    vmax = max(empirical_est_cov.max(), theoretical_est_cov.max())

    im1 = ax1.imshow(
        empirical_est_cov, cmap="RdBu_r", aspect="equal",
        vmin=vmin, vmax=vmax,
    )
    ax1.set_xticks(range(n_stats))
    ax1.set_xticklabels(labels, fontsize=7, rotation=45)
    ax1.set_yticks(range(n_stats))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_title("Empirical", fontsize=11)
    if fig is not None:
        fig.colorbar(im1, ax=ax1, shrink=0.7)

    im2 = ax2.imshow(
        theoretical_est_cov, cmap="RdBu_r", aspect="equal",
        vmin=vmin, vmax=vmax,
    )
    ax2.set_xticks(range(n_stats))
    ax2.set_xticklabels(labels, fontsize=7, rotation=45)
    ax2.set_yticks(range(n_stats))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_title("Theoretical", fontsize=11)
    if fig is not None:
        fig.colorbar(im2, ax=ax2, shrink=0.7)


def plot_diagonal_variances(empirical_est_cov, theoretical_est_cov,
                             labels, ax):
    """estimator_accuracy_mse.qmd → fig-diagonal-variances

    Bar chart of individual estimator variances (diagonal entries).
    """
    from ._style import apply_style

    n_stats = len(labels)
    empirical_diag = np.diag(empirical_est_cov)
    theoretical_diag = np.diag(theoretical_est_cov)

    x = np.arange(n_stats)
    width = 0.35
    ax.bar(
        x - width / 2, empirical_diag, width, color="#2C7FB8", alpha=0.7,
        label="Empirical", edgecolor="k", lw=0.3,
    )
    ax.bar(
        x + width / 2, theoretical_diag, width, color="#E67E22", alpha=0.7,
        label="Theoretical", edgecolor="k", lw=0.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Estimator variance", fontsize=12)
    ax.set_title(
        "Individual estimator variances "
        "(diagonal of Cov[$\\hat{\\mathbf{Q}}_M$])",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
