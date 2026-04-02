"""Plotting functions for Bayesian OED tutorials.

Covers: boed_kl_concept.qmd, boed_kl_usage.qmd, boed_kl_estimator.qmd,
        boed_kl_gradients.qmd, boed_kl_nonlinear_usage.qmd,
        boed_kl_qmc.qmd, boed_kl_design_stability.qmd,
        boed_pred_concept.qmd, boed_pred_usage.qmd,
        boed_pred_nonlinear_usage.qmd, boed_data_workflow_usage.qmd
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from pyapprox.util.backends.protocols import Array, Backend

NDArrayFloat = npt.NDArray[np.floating[Any]]

# ---------------------------------------------------------------------------
# boed_kl_concept.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_kl_intuition(axes: Sequence[Axes]) -> None:
    """boed_kl_concept.qmd → fig-kl-intuition

    KL divergence from prior to posterior for weak and strong designs.
    """
    from scipy import stats

    prior_mu, prior_std = 0.0, 0.5
    weak_mu, weak_std = 0.05, 0.44
    strong_mu, strong_std = 0.32, 0.14

    def kl_gaussian(mu1: float, s1: float, mu2: float, s2: float) -> float:
        return float(
            np.log(s2 / s1) + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2) - 0.5
        )

    kl_weak = kl_gaussian(weak_mu, weak_std, prior_mu, prior_std)
    kl_strong = kl_gaussian(strong_mu, strong_std, prior_mu, prior_std)

    theta = np.linspace(-1.6, 1.6, 500)
    prior_pdf = stats.norm.pdf(theta, prior_mu, prior_std)
    weak_pdf = stats.norm.pdf(theta, weak_mu, weak_std)
    strong_pdf = stats.norm.pdf(theta, strong_mu, strong_std)

    GREY = "#888888"
    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"

    for ax, post_pdf, post_mu, post_std, post_lbl, post_col, kl_val in [
        (axes[0], weak_pdf, weak_mu, weak_std,
         "Posterior\n(weak design)", BLUE, kl_weak),
        (axes[1], strong_pdf, strong_mu, strong_std,
         "Posterior\n(strong design)", ORANGE, kl_strong),
    ]:
        ax.fill_between(theta, prior_pdf, alpha=0.18, color=GREY)
        ax.plot(theta, prior_pdf, color=GREY, lw=2,
                label="Prior  $p(\\theta)$", zorder=3)
        ax.fill_between(theta, post_pdf, alpha=0.22, color=post_col)
        ax.plot(theta, post_pdf, color=post_col, lw=2.5,
                label=f"{post_lbl}", zorder=4)
        ax.text(0.97, 0.94,
                f"KL = {kl_val:.2f} nats",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold", color=post_col,
                bbox=dict(fc="white", ec=post_col, lw=1.2, pad=3, alpha=0.85))
        ax.set_xlabel(r"Parameter $\theta$", fontsize=11)
        ax.set_xlim(theta[0], theta[-1])
        ax.set_ylim(0)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Probability density", fontsize=11)
    axes[0].set_title("Small KL  →  little was learned", fontsize=10)
    axes[1].set_title("Large KL  →  data was very informative", fontsize=10)


def plot_eig_as_average(axes: Sequence[Axes]) -> None:
    """boed_kl_concept.qmd → fig-eig-as-average

    EIG as average KL divergence over sampled observations.
    """
    from scipy import stats

    rng = np.random.default_rng(42)

    prior_std = 0.5
    noise_std = 0.5
    N_samples = 10

    marg_std = np.sqrt(prior_std**2 + noise_std**2)
    y_samples = rng.normal(0, marg_std, size=N_samples)

    var_post = 1.0 / (1.0 / prior_std**2 + 1.0 / noise_std**2)
    std_post = np.sqrt(var_post)
    mu_posts = var_post * y_samples / noise_std**2

    def kl_gauss_1d(mu1: float, s1: float, mu2: float, s2: float) -> float:
        return float(
            np.log(s2 / s1) + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2) - 0.5
        )

    kl_vals = np.array([kl_gauss_1d(mu, std_post, 0.0, prior_std)
                        for mu in mu_posts])
    eig_mc = kl_vals.mean()
    eig_exact = 0.5 * np.log(prior_std**2 / var_post)

    theta = np.linspace(-2.0, 2.0, 500)
    prior_pdf = stats.norm.pdf(theta, 0, prior_std)

    GREY = "#888888"
    BLUE = "#2C7FB8"
    cmap = __import__("matplotlib").pyplot.cm.cool

    # Left: posterior curves for each sampled observation
    ax = axes[0]
    ax.plot(theta, prior_pdf, color=GREY, lw=2.5, zorder=5,
            label="Prior  $p(\\theta)$", ls="--")
    for i, (mu, kl) in enumerate(zip(mu_posts, kl_vals)):
        c = cmap(i / N_samples)
        post_pdf = stats.norm.pdf(theta, mu, std_post)
        ax.plot(theta, post_pdf, color=c, lw=1.2, alpha=0.7)

    ax.plot([], [], color=cmap(0.5), lw=1.5, alpha=0.8,
            label=f"Posteriors (n={N_samples} draws)")
    ax.set_xlabel(r"Parameter $\theta$", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("Each observation produces\na different posterior", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_xlim(theta[0], theta[-1])
    ax.set_ylim(0)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: KL for each posterior, and their mean (= EIG)
    ax = axes[1]
    bar_cols = [cmap(i / N_samples) for i in range(N_samples)]
    ax.bar(range(N_samples), kl_vals, color=bar_cols, alpha=0.8,
           edgecolor="white", linewidth=0.8)
    ax.axhline(eig_mc, color=BLUE, lw=2.2, ls="--",
               label=f"MC average  = {eig_mc:.3f} nats")
    ax.axhline(eig_exact, color="black", lw=1.5, ls=":",
               label=f"Exact EIG   = {eig_exact:.3f} nats")
    ax.set_xlabel("Sampled observation index", fontsize=11)
    ax.set_ylabel("KL divergence (nats)", fontsize=11)
    ax.set_title("KL per outcome; average → EIG", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_xticks(range(N_samples))
    ax.set_ylim(0)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)


def plot_eig_vs_nobs(ax: Axes) -> None:
    """boed_kl_concept.qmd → fig-eig-vs-nobs

    EIG vs number of observations for a linear Gaussian model.
    """
    from pyapprox.benchmarks.instances.oed.linear_gaussian import (
        build_linear_gaussian_kl_benchmark,
    )
    from pyapprox.util.backends.numpy import NumpyBkd

    bkd = NumpyBkd()
    np.random.seed(0)

    nobs_list = [1, 2, 3, 5, 8, 10]
    noise_std, prior_std, degree = 0.5, 0.5, 2
    eigs = []
    for nobs in nobs_list:
        bench = build_linear_gaussian_kl_benchmark(
            nobs, degree, noise_std, prior_std, bkd)
        weights = bkd.ones((nobs, 1))
        eigs.append(bench.exact_eig(weights))

    ax.plot(nobs_list, eigs, "o-", color="#2C7FB8", lw=2)
    ax.set_xlabel("Number of observations $K$", fontsize=12)
    ax.set_ylabel("EIG (nats)", fontsize=12)
    ax.set_title("EIG grows with number of observations (unit weights)",
                 fontsize=11)
    ax.grid(True, alpha=0.25)


def plot_posterior_shrinkage(axes: Sequence[Axes]) -> None:
    """boed_kl_concept.qmd → fig-posterior-shrinkage

    Posterior shrinkage depends on sensor placement, not just count.
    """
    from scipy import stats

    np.random.seed(0)

    noise_std = 0.5
    prior_std = 0.5
    degree = 2
    d = degree + 1

    def poly_design_matrix(x_locs: NDArrayFloat) -> NDArrayFloat:
        return np.column_stack([x_locs**k for k in range(d)])

    def gaussian_posterior_cov(
        A: NDArrayFloat, noise_std_: float, prior_std_: float,
    ) -> NDArrayFloat:
        Sigma_prior_inv = np.eye(d) / prior_std_**2
        return np.linalg.inv(A.T @ A / noise_std_**2 + Sigma_prior_inv)

    def closed_form_eig(
        A: NDArrayFloat, noise_std_: float, prior_std_: float,
    ) -> float:
        Sigma_prior = np.eye(d) * prior_std_**2
        Sigma_post = gaussian_posterior_cov(A, noise_std_, prior_std_)
        _, ld_prior = np.linalg.slogdet(Sigma_prior)
        _, ld_post = np.linalg.slogdet(Sigma_post)
        return float(0.5 * (ld_prior - ld_post))

    def make_locations(n: int, design: str = "spread") -> NDArrayFloat:
        if design == "clustered":
            center = 0.5
            half = 0.08
            return np.linspace(center - half, center + half, n)
        else:
            return np.linspace(0.0, 1.0, n)

    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"

    nobs_sweep = list(range(1, 11))
    std_theta2_A, std_theta2_B = [], []
    eig_A, eig_B = [], []

    for n in nobs_sweep:
        x_A = make_locations(n, "clustered")
        x_B = make_locations(n, "spread")
        A_mat = poly_design_matrix(x_A)
        B_mat = poly_design_matrix(x_B)
        cov_A = gaussian_posterior_cov(A_mat, noise_std, prior_std)
        cov_B = gaussian_posterior_cov(B_mat, noise_std, prior_std)
        std_theta2_A.append(np.sqrt(cov_A[2, 2]))
        std_theta2_B.append(np.sqrt(cov_B[2, 2]))
        eig_A.append(closed_form_eig(A_mat, noise_std, prior_std))
        eig_B.append(closed_form_eig(B_mat, noise_std, prior_std))

    prior_std_theta2 = prior_std

    K_show = 5
    cov_A5 = gaussian_posterior_cov(
        poly_design_matrix(make_locations(K_show, "clustered")),
        noise_std, prior_std)
    cov_B5 = gaussian_posterior_cov(
        poly_design_matrix(make_locations(K_show, "spread")),
        noise_std, prior_std)

    theta_grid = np.linspace(-1.8, 1.8, 500)
    prior_pdf = stats.norm.pdf(theta_grid, 0, prior_std)
    post_A5 = stats.norm.pdf(theta_grid, 0, np.sqrt(cov_A5[2, 2]))
    post_B5 = stats.norm.pdf(theta_grid, 0, np.sqrt(cov_B5[2, 2]))

    # Panel 1 -- posterior std vs. N
    ax = axes[0]
    ax.axhline(prior_std_theta2, color="grey", lw=1.5, ls=":",
               label="Prior std  (prior_std=0.5)")
    ax.plot(nobs_sweep, std_theta2_A, "o-", color=ORANGE, lw=2, ms=6,
            label="Design A  (clustered)")
    ax.plot(nobs_sweep, std_theta2_B, "s-", color=BLUE, lw=2, ms=6,
            label="Design B  (spread)")
    ax.set_xlabel("Number of sensors $K$", fontsize=11)
    ax.set_ylabel(r"Posterior std of $\theta_2$", fontsize=11)
    ax.set_title("Posterior uncertainty vs. $K$", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2 -- posterior PDFs at K=5
    ax = axes[1]
    ax.fill_between(theta_grid, prior_pdf, alpha=0.12, color="grey")
    ax.plot(theta_grid, prior_pdf, color="grey", lw=2, ls="--", label="Prior")
    ax.fill_between(theta_grid, post_A5, alpha=0.18, color=ORANGE)
    ax.plot(theta_grid, post_A5, color=ORANGE, lw=2.2,
            label=f"Design A  ($K={K_show}$,  std={np.sqrt(cov_A5[2, 2]):.2f})")
    ax.fill_between(theta_grid, post_B5, alpha=0.18, color=BLUE)
    ax.plot(theta_grid, post_B5, color=BLUE, lw=2.2,
            label=f"Design B  ($K={K_show}$,  std={np.sqrt(cov_B5[2, 2]):.2f})")
    ax.set_xlabel(r"Quadratic coefficient $\theta_2$", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(f"Posterior PDFs at $K={K_show}$ sensors", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(theta_grid[0], theta_grid[-1])
    ax.set_ylim(0)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3 -- EIG vs. N
    ax = axes[2]
    ax.plot(nobs_sweep, eig_A, "o-", color=ORANGE, lw=2, ms=6,
            label="Design A  (clustered)")
    ax.plot(nobs_sweep, eig_B, "s-", color=BLUE, lw=2, ms=6,
            label="Design B  (spread)")
    ax.set_xlabel("Number of sensors $K$", fontsize=11)
    ax.set_ylabel("EIG (nats)", fontsize=12)
    ax.set_title("EIG vs. $K$  (closed form)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# boed_kl_usage.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_boed_convergence_panels(
    axes: Sequence[Axes],
    bkd: Backend[Array],
    values_mc: Dict[str, List[Array]],
    values_inner: Dict[str, List[Array]],
    outer_counts: List[int],
    inner_counts: List[int],
    M_fixed: int,
    inner_sweep: List[int],
) -> None:
    """boed_kl_usage.qmd → fig-convergence-panels

    MSE vs outer samples and bias/variance vs inner samples.
    """
    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"

    mse_array = bkd.to_numpy(bkd.vstack(values_mc["mse"]))
    bias2_arr = bkd.to_numpy(bkd.vstack(values_inner["sqbias"]))
    var_arr = bkd.to_numpy(bkd.vstack(values_inner["variance"]))

    # Left: MSE vs M
    ax = axes[0]
    colors = [BLUE, ORANGE]
    for ii, ninner in enumerate(inner_counts):
        ax.loglog(outer_counts, mse_array[ii, :], "o-",
                  color=colors[ii], lw=2,
                  label=f"$N_{{\\mathrm{{in}}}}={ninner}$")

    ref_x = np.array([outer_counts[0], outer_counts[-1]])
    ref_y = mse_array[0, 0] * ref_x[0] / ref_x
    ax.loglog(ref_x, ref_y, "k--", lw=1.5, alpha=0.55,
              label="$O(1/M)$ reference")
    ax.set_xlabel("Outer samples $M$", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("MSE vs. outer samples  (variance-dominated)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: bias^2 and variance vs N_in
    ax = axes[1]
    ax.loglog(inner_sweep, bias2_arr[:, 0], "s-", color=ORANGE, lw=2,
              label="bias²  (inner bias)")
    ax.loglog(inner_sweep, var_arr[:, 0], "o-", color=BLUE, lw=2,
              label=f"variance  ($M={M_fixed}$, fixed)")
    ax.set_xlabel("Inner samples $N_{\\mathrm{in}}$", fontsize=11)
    ax.set_ylabel("Error component", fontsize=11)
    ax.set_title("Bias² and variance vs. inner samples", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# boed_kl_estimator.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_double_loop(ax: Axes) -> None:
    """boed_kl_estimator.qmd → fig-double-loop

    Double-loop estimator flow diagram.
    """
    from matplotlib.patches import FancyBboxPatch

    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"
    GREEN = "#31A354"
    GREY = "#636363"
    LTBLUE = "#BDD7E7"
    LTORANGE = "#FDDBC7"
    LTGREEN = "#A1D99B"

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    def box(
        a: Axes, x: float, y: float, w: float, h: float,
        text: str, fc: str, ec: str, fontsize: float = 9.5, bold: bool = False,
    ) -> None:
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        a.add_patch(p)
        a.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=fontsize,
               fontweight="bold" if bold else "normal",
               zorder=4, multialignment="center")

    def arrow(
        a: Axes, x0: float, y0: float, x1: float, y1: float,
        color: str = GREY, lw: float = 1.6, ls: str = "solid",
    ) -> None:
        a.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=lw, linestyle=ls, mutation_scale=11),
            zorder=2)

    ax.text(1.85, 7.7, "OUTER LOOP  ($M$ samples)",
            ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(6.5, 7.7,
            "SHARED  (reuses outer $\\mathbf{y}$)",
            ha="center", fontsize=11, fontweight="bold", color=GREEN)
    ax.text(11.2, 7.7,
            "INNER LOOP  ($N_{\\mathrm{in}}$ samples)",
            ha="center", fontsize=11, fontweight="bold", color=ORANGE)

    box(ax, 0.15, 6.3, 2.1, 0.95,
        r"$\boldsymbol{\theta}^{(m)} \sim p(\boldsymbol{\theta})$",
        LTBLUE, BLUE, 9)
    box(ax, 0.15, 4.95, 2.1, 0.95,
        "$\\boldsymbol{\\varepsilon}^{(m)}"
        "\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I}_K)$",
        LTBLUE, BLUE, 9)
    box(ax, 0.15, 3.55, 2.1, 0.90,
        r"$\mathbf{f}(\boldsymbol{\theta}^{(m)})$",
        LTBLUE, BLUE, 9)

    box(ax, 4.5, 6.2, 4.0, 1.05,
        "$\\mathbf{y}^{(m)}=\\mathbf{f}(\\boldsymbol{\\theta}^{(m)})"
        "+\\mathbf{W}^{-1/2}\\boldsymbol{\\Gamma}^{1/2}"
        "\\boldsymbol{\\varepsilon}^{(m)}$",
        LTGREEN, GREEN, 8.8)
    box(ax, 4.5, 4.85, 4.0, 1.0,
        r"$\ell_m = \log p(\mathbf{y}^{(m)} "
        r"\mid \boldsymbol{\theta}^{(m)}, \mathbf{w})$"
        "\n(outer log-likelihood)",
        LTGREEN, GREEN, 8.5)
    box(ax, 4.5, 3.2, 4.0, 1.0,
        r"$\hat{p}(\mathbf{y}^{(m)}\!\mid\!\mathbf{w}) = "
        r"\frac{1}{N_{\mathrm{in}}}\!\sum_n "
        r"p(\mathbf{y}^{(m)}\!\mid\!\boldsymbol{\theta}^{(n)},\mathbf{w})$"
        "\n(inner evidence average)",
        LTGREEN, GREEN, 8.0)
    box(ax, 4.5, 1.4, 4.0, 1.2,
        r"$\widehat{\mathrm{EIG}}=\frac{1}{M}\sum_m"
        r"\left[\ell_m-\log\hat{p}(\mathbf{y}^{(m)}\mid\mathbf{w})\right]$",
        "#EDF8FB", GREEN, 9.2, bold=True)

    box(ax, 11.0, 6.2, 1.85, 1.05,
        "$\\boldsymbol{\\theta}^{(n)}\\sim p(\\boldsymbol{\\theta})$\n"
        "$n=1,\\ldots,N_{\\mathrm{in}}$",
        LTORANGE, ORANGE, 8.5)
    box(ax, 11.0, 4.85, 1.85, 0.95,
        r"$\mathbf{f}(\boldsymbol{\theta}^{(n)})$",
        LTORANGE, ORANGE, 9)
    box(ax, 11.0, 3.2, 1.85, 1.0,
        "$\\log p(\\mathbf{y}^{(m)}\\!\\mid\\!"
        "\\boldsymbol{\\theta}^{(n)},\\mathbf{w})$\nfor each $(m,n)$",
        LTORANGE, ORANGE, 7.8)

    arrow(ax, 1.2, 6.3, 1.2, 5.9, BLUE)
    arrow(ax, 1.2, 4.95, 1.2, 4.45, BLUE)
    arrow(ax, 2.25, 6.77, 4.5, 6.72, BLUE)
    arrow(ax, 2.25, 5.42, 4.5, 6.60, BLUE)
    arrow(ax, 2.25, 4.00, 4.5, 6.50, BLUE)
    arrow(ax, 6.5, 6.2, 6.5, 5.85, GREEN)
    arrow(ax, 6.5, 4.85, 6.5, 4.2, GREEN)
    arrow(ax, 6.5, 3.2, 6.5, 2.6, GREEN)
    arrow(ax, 11.93, 6.2, 11.93, 5.8, ORANGE)
    arrow(ax, 11.93, 4.85, 11.93, 4.2, ORANGE)
    arrow(ax, 8.5, 6.72, 11.0, 3.72, ORANGE)
    arrow(ax, 11.0, 3.7, 8.5, 3.7, ORANGE)

    ax.text(9.85, 5.5,
            "outer $\\mathbf{y}^{(m)}$\npassed to inner loop",
            ha="center", va="center", fontsize=8, color=ORANGE,
            fontstyle="italic",
            bbox=dict(fc="white", ec=ORANGE, lw=0.8, pad=2, alpha=0.85))


# ---------------------------------------------------------------------------
# boed_kl_gradients.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_eig_landscape(
    axes: Sequence[Axes],
    check_pts: NDArrayFloat,
    analytic_grads: NDArrayFloat,
    correct_grads: NDArrayFloat,
    wrong_grads: NDArrayFloat,
) -> None:
    """boed_kl_gradients.qmd → fig-eig-landscape

    EIG landscape and C1+C2+C3 vs incomplete C1+C2 gradient comparison.
    """
    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"
    GREEN = "#31A354"
    GREY = "#888888"

    def eig_exact(
        w1: float | NDArrayFloat,
    ) -> np.floating[Any] | NDArrayFloat:
        result = 0.5 * np.log(6 + 9 * w1 - 9 * w1**2)
        return result

    def eig_grad_exact(w1: float) -> float:
        det = 6 + 9 * w1 - 9 * w1**2
        return float((9 - 18 * w1) / (2 * det))

    w_fine = np.linspace(0.01, 0.99, 300)
    eig_vals = eig_exact(w_fine)

    ax = axes[0]
    ax.plot(w_fine, eig_vals, color=BLUE, lw=2.5,
            label="EIG$(w_1)$  (closed form)")
    ax.axvline(0.5, color=GREY, lw=1.3, ls="--",
               label="Optimum  $w_1^*=0.5$")
    ax.scatter([0.5], [eig_exact(0.5)], color=GREEN, s=90, zorder=5,
               label=f"Maximum: {eig_exact(0.5):.3f} nats")
    for w1 in [0.12, 0.28, 0.50, 0.72, 0.88]:
        g = eig_grad_exact(w1)
        eig_w1 = float(eig_exact(w1))
        ax.annotate(
            "", xy=(w1 + np.sign(g) * 0.10, eig_w1),
            xytext=(w1, eig_w1),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8,
                            mutation_scale=11))
    ax.set_xlabel("$w_1$   ($w_2 = 1 - w_1$)", fontsize=11)
    ax.set_ylabel("EIG (nats)", fontsize=11)
    ax.set_title("EIG$(w_1)$ and gradient direction", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(len(check_pts))
    width = 0.28
    ax.bar(x - width, analytic_grads, width,
           label="Analytic  $d\\,$EIG/$dw_1$", color=BLUE, alpha=0.85,
           edgecolor="white")
    ax.bar(x, correct_grads, width,
           label="MC  C1+C2+C3  (correct)", color=GREEN, alpha=0.85,
           edgecolor="white")
    ax.bar(x + width, wrong_grads, width,
           label="MC  C1+C2  (wrong)", color=ORANGE, alpha=0.85,
           edgecolor="white")
    ax.axhline(0, color=GREY, lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$w_1={w:.2f}$" for w in check_pts], fontsize=9)
    ax.set_ylabel("Gradient value", fontsize=11)
    ax.set_title("C1+C2+C3 vs incomplete C1+C2 gradient", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    if np.sign(wrong_grads[0]) != np.sign(analytic_grads[0]):
        ax.annotate("Wrong sign\n← diverges from optimum",
                    xy=(x[0] + width, wrong_grads[0]),
                    xytext=(x[0] + width + 0.4, wrong_grads[0] - 0.12),
                    fontsize=7.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))


# ---------------------------------------------------------------------------
# boed_kl_nonlinear_usage.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_lv_trajectories(
    axes: Sequence[Axes],
    benchmark: Any,
    obs_model: Any,
    prior: Any,
    sample: NDArrayFloat,
    observations: NDArrayFloat,
) -> None:
    """boed_kl_nonlinear_usage.qmd → fig-lv-trajectories

    Lotka-Volterra trajectories for prior ensemble plus nominal trajectory.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    bkd = NumpyBkd()

    titles = ["State 1 (prey)", "State 2", "State 3 (predator)"]

    for samp in prior.rvs(30).T:
        sol = obs_model.solve_trajectory(samp[:, None])[0]
        for ii in range(3):
            axes[ii].plot(
                benchmark.solution_times(), sol[ii],
                color="#AAAAAA", lw=0.8, alpha=0.5)

    sol_nom = obs_model.solve_trajectory(sample)[0]
    for ii in range(3):
        axes[ii].plot(benchmark.solution_times(), sol_nom[ii],
                      color="#2C7FB8", lw=2)
        axes[ii].set_title(titles[ii], fontsize=11)
        axes[ii].set_xlabel("Time (s)", fontsize=10)
        axes[ii].grid(True, alpha=0.2)

    obs_times = bkd.to_numpy(benchmark.observation_times())
    for ii, state_idx in enumerate([0, 2]):
        axes[state_idx].plot(
            obs_times,
            observations[ii],
            "o", ms=7, color="#D95F02", zorder=5, label="Candidate obs.")
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel("Population", fontsize=10)


def plot_lv_design(
    axes: Sequence[Axes],
    benchmark: Any,
    design_weights: Array,
    bkd: Backend[Array],
) -> None:
    """boed_kl_nonlinear_usage.qmd → fig-lv-design

    Optimal design weights for the Lotka-Volterra EIG problem.
    """
    weights_by_state = bkd.to_numpy(design_weights).reshape((2, -1))
    obs_times = bkd.to_numpy(benchmark.observation_times())

    labels = ["State 1 (prey)", "State 3 (predator)"]
    colors = ["#2C7FB8", "#D95F02"]

    for ii, ax in enumerate(axes):
        ax.bar(obs_times, weights_by_state[ii], color=colors[ii], alpha=0.8)
        ax.set_xlabel("Observation time (s)", fontsize=11)
        ax.set_title(labels[ii], fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Design weight", fontsize=11)


# ---------------------------------------------------------------------------
# boed_kl_qmc.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_points_comparison(axes: Sequence[Axes]) -> None:
    """boed_kl_qmc.qmd → fig-points-comparison

    200 samples in 2-D: standard MC vs randomly-shifted Halton RQMC.
    """
    from pyapprox.expdesign.quadrature.halton import HaltonSampler
    from pyapprox.probability.joint import IndependentJoint
    from pyapprox.probability.univariate import GaussianMarginal
    from pyapprox.util.backends.numpy import NumpyBkd

    bkd = NumpyBkd()

    rng_mc = np.random.default_rng(0)
    pts_mc = rng_mc.standard_normal((200, 2))

    std_normal_2d = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(2)], bkd,
    )
    halton_2d = HaltonSampler(2, bkd, distribution=std_normal_2d, seed=0)
    pts_qmc, _ = halton_2d.sample(200)
    pts_qmc = bkd.to_numpy(pts_qmc).T

    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"

    for ax, pts, title, color in [
        (axes[0], pts_mc, "Standard MC  (i.i.d.)", BLUE),
        (axes[1], pts_qmc, "RQMC  (randomly-shifted Halton)", ORANGE),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.55, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Dimension 1", fontsize=10)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.grid(True, alpha=0.15)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Dimension 2", fontsize=10)


def plot_mc_vs_qmc(
    ax: Axes,
    outer_counts: List[int],
    mse_mc: List[float],
    mse_qmc: List[float],
) -> None:
    """boed_kl_qmc.qmd → fig-mc-vs-qmc

    MSE vs outer sample count for MC and randomly-shifted Halton RQMC.
    """
    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"

    slope_mc = np.polyfit(np.log(outer_counts), np.log(mse_mc), 1)[0]
    slope_qmc = np.polyfit(np.log(outer_counts), np.log(mse_qmc), 1)[0]

    ax.loglog(outer_counts, mse_mc, "o-", color=BLUE, lw=2.2, ms=7,
              label=f"MC   (slope = {slope_mc:.2f})")
    ax.loglog(outer_counts, mse_qmc, "s-", color=ORANGE, lw=2.2, ms=7,
              label=f"RQMC (slope = {slope_qmc:.2f})")

    ref_x = np.array([outer_counts[0], outer_counts[-1]], dtype=float)
    ref_1 = mse_mc[0] * (ref_x / outer_counts[0])**(-1.0)
    ax.loglog(ref_x, ref_1, "k--", lw=1.3, alpha=0.50,
              label="$O(M^{-1})$ reference")

    ax.set_xlabel("Outer samples $M$", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title(
        "MC vs. RQMC convergence for the double-loop EIG estimator",
        fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# boed_kl_design_stability.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_design_variability(
    axes: Sequence[Axes],
    obs_times: NDArrayFloat,
    designs_fixed: List[NDArrayFloat],
    M_fixed: int,
) -> None:
    """boed_kl_design_stability.qmd → fig-variability

    Optimal design weights from independent MC realizations at fixed budget.
    """
    state_labels = ["State 1 (prey)", "State 3 (predator)"]
    seed_colors = ["#2C7FB8", "#D95F02", "#31A354"]

    for col in range(2):
        ax = axes[col]
        for trial, w_2d in enumerate(designs_fixed):
            ax.plot(obs_times, w_2d[col], "o-", ms=4, lw=1.5, alpha=0.7,
                    color=seed_colors[trial], label=f"seed {trial}")
        ax.set_title(f"{state_labels[col]}  ($M = {M_fixed:,}$)", fontsize=11)
        ax.set_xlabel("Observation time (s)", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Design weight", fontsize=10)
    axes[0].legend(fontsize=9)


def plot_design_convergence(
    axes: Any,
    obs_times: NDArrayFloat,
    budgets: List[int],
    all_designs: Dict[int, List[NDArrayFloat]],
) -> None:
    """boed_kl_design_stability.qmd → fig-design-convergence

    Design weights from independent MC realizations at increasing budget.
    """
    state_labels = ["State 1 (prey)", "State 3 (predator)"]
    seed_colors = ["#2C7FB8", "#D95F02", "#31A354"]

    for row, M in enumerate(budgets):
        for col in range(2):
            ax = axes[row, col]
            for trial, w_2d in enumerate(all_designs[M]):
                ax.plot(obs_times, w_2d[col], "o-", ms=3, lw=1.2, alpha=0.6,
                        color=seed_colors[trial],
                        label=(f"seed {trial}"
                               if row == 0 and col == 0 else None))
            ax.set_title(f"$M={M:,}$  —  {state_labels[col]}", fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Design weight", fontsize=10)
            if row == len(budgets) - 1:
                ax.set_xlabel("Observation time (s)", fontsize=10)

    axes[0, 0].legend(fontsize=8, ncol=3, loc="upper right")


# ---------------------------------------------------------------------------
# boed_pred_concept.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def _pred_concept_helpers() -> Tuple[
    float, float, float, NDArrayFloat, NDArrayFloat, NDArrayFloat,
    Any, Any, Any, Any,
]:
    """Return shared helpers for boed_pred_concept figures."""
    sigma1, sigma2, sigma_n = 2.0, 0.5, 0.5
    Sigma0 = np.diag([sigma1**2, sigma2**2])
    Sigma0_inv = np.diag([1 / sigma1**2, 1 / sigma2**2])
    c = np.array([0., 1.])

    def sensor_row(x: float) -> NDArrayFloat:
        return np.array([np.cos(np.pi * x / 2), np.sin(np.pi * x / 2)])

    def posterior_cov(x_locs: List[float]) -> NDArrayFloat:
        rows = np.vstack([sensor_row(x) for x in x_locs])
        return np.linalg.inv(rows.T @ rows / sigma_n**2 + Sigma0_inv)

    def pf_std(x_locs: List[float]) -> float:
        return float(np.sqrt(c @ posterior_cov(x_locs) @ c))

    def eig_val(x_locs: List[float]) -> float:
        Sp = posterior_cov(x_locs)
        _, ld0 = np.linalg.slogdet(Sigma0)
        _, ld1 = np.linalg.slogdet(Sp)
        return float(0.5 * (ld0 - ld1))

    return (sigma1, sigma2, sigma_n, Sigma0, Sigma0_inv, c,
            sensor_row, posterior_cov, pf_std, eig_val)


def plot_pushforward_intuition(
    ax_A2d: Axes, ax_B2d: Axes, ax_Apf: Axes, ax_Bpf: Axes,
) -> None:
    """boed_pred_concept.qmd → fig-pushforward-intuition

    2D posterior ellipses and 1D push-forward distributions.
    """
    from matplotlib.patches import Ellipse
    from scipy import stats

    (sigma1, sigma2, sigma_n, Sigma0, Sigma0_inv, c,
     sensor_row, posterior_cov, pf_std, eig_val) = _pred_concept_helpers()

    np.random.seed(42)

    def cov_ellipse(
        ax: Axes, cov: NDArrayFloat, center: Tuple[float, float],
        nstd: int, color: str, lw: float, ls: str = "-",
        label: str | None = None,
    ) -> None:
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        e = Ellipse(xy=center,
                    width=2 * nstd * np.sqrt(vals[0]),
                    height=2 * nstd * np.sqrt(vals[1]),
                    angle=angle, edgecolor=color, fc="none",
                    lw=lw, ls=ls, label=label)
        ax.add_patch(e)

    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"
    GREEN = "#31A354"
    GREY = "#888888"

    Spost_A = posterior_cov([0.0])
    Spost_B = posterior_cov([1.0])
    eig_A = eig_val([0.0])
    eig_B = eig_val([1.0])

    theta2_g = np.linspace(-1.5, 1.5, 400)

    for ax_2d, ax_pf, Spost, title, subtitle, color in [
        (ax_A2d, ax_Apf, Spost_A,
         "Design A  (EIG-optimal, $x=0$)",
         f"EIG = {eig_A:.2f} nats", ORANGE),
        (ax_B2d, ax_Bpf, Spost_B,
         "Design B  (Pred-OED-optimal, $x=1$)",
         f"EIG = {eig_B:.2f} nats", BLUE),
    ]:
        # 2D ellipses
        ax_2d.set_xlim(-6.5, 6.5)
        ax_2d.set_ylim(-1.55, 1.55)
        ax_2d.axhline(0, color="lightgrey", lw=0.8, zorder=0)
        ax_2d.axvline(0, color="lightgrey", lw=0.8, zorder=0)

        cov_ellipse(ax_2d, Sigma0, (0, 0), nstd=2, color=GREY, lw=2.0,
                    ls="--", label="Prior (2$\\sigma$)")
        cov_ellipse(ax_2d, Sigma0, (0, 0), nstd=1, color=GREY, lw=1.2,
                    ls="--")
        cov_ellipse(ax_2d, Spost, (0, 0), nstd=2, color=color, lw=2.5,
                    label="Posterior (2$\\sigma$)")
        cov_ellipse(ax_2d, Spost, (0, 0), nstd=1, color=color, lw=1.5)

        ax_2d.annotate("", xy=(5.8, 0.85), xytext=(5.8, -0.85),
                       arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.2))
        ax_2d.text(5.35, 0.0, "$q=\\theta_2$",
                   color=GREEN, fontsize=9, ha="right", va="center",
                   fontweight="bold")

        ax_2d.set_xlabel("$\\theta_1$  (trend)", fontsize=10)
        ax_2d.set_ylabel("$\\theta_2$  (QoI)", fontsize=10)
        ax_2d.set_title(f"{title}\n{subtitle}",
                        fontsize=10, fontweight="bold", color=color)
        ax_2d.legend(fontsize=8.5, loc="upper left")
        ax_2d.spines[["top", "right"]].set_visible(False)

        # 1D push-forward
        prior_pf = stats.norm.pdf(theta2_g, 0, sigma2)
        post_std = np.sqrt(c @ Spost @ c)
        post_pf = stats.norm.pdf(theta2_g, 0, post_std)

        ax_pf.fill_between(theta2_g, prior_pf, alpha=0.15, color=GREY)
        ax_pf.plot(theta2_g, prior_pf, color=GREY, lw=2.0, ls="--",
                   label=f"Prior $q$  (std = {sigma2:.2f})")
        ax_pf.fill_between(theta2_g, post_pf, alpha=0.25, color=color)
        ax_pf.plot(theta2_g, post_pf, color=color, lw=2.5,
                   label=f"Posterior $q$  (std = {post_std:.2f})")

        ax_pf.set_xlabel(
            "QoI  $q(\\boldsymbol{\\theta}) = \\theta_2$", fontsize=10)
        ax_pf.set_ylabel("Density", fontsize=10)
        ax_pf.set_xlim(-1.5, 1.5)
        ax_pf.set_ylim(0)
        ax_pf.legend(fontsize=8.5)
        ax_pf.grid(True, alpha=0.18)
        ax_pf.spines[["top", "right"]].set_visible(False)
        ax_pf.text(0.97, 0.93, f"Push-forward std = {post_std:.3f}",
                   transform=ax_pf.transAxes, ha="right", va="top",
                   fontsize=9.5, fontweight="bold", color=color,
                   bbox=dict(fc="white", ec=color, lw=1, pad=2, alpha=0.9))

    ax_Apf.set_title("Push-forward: prior $\\approx$ posterior",
                     fontsize=9, color=GREY, pad=4)
    ax_Bpf.set_title("Push-forward: posterior is narrower",
                     fontsize=9, color=BLUE, pad=4)


def plot_utility_as_average(axes: Sequence[Axes]) -> None:
    """boed_pred_concept.qmd → fig-utility-as-average

    Push-forward posteriors and EIG vs push-forward std comparison.
    """
    from scipy import stats

    rng = np.random.default_rng(42)

    (sigma1, sigma2, sigma_n, Sigma0, Sigma0_inv, c,
     sensor_row, posterior_cov, pf_std_fn, eig_fn) = _pred_concept_helpers()

    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"
    GREY = "#888888"

    # Left panel: multiple push-forward posteriors
    x_demo = [0.3, 0.6, 0.9]
    Spost = posterior_cov(x_demo)
    A_demo = np.vstack([sensor_row(x) for x in x_demo])
    pf_std_demo = np.sqrt(c @ Spost @ c)

    M = 12
    theta_true = rng.multivariate_normal([0, 0], Sigma0, size=M)
    eps = rng.normal(0, sigma_n, (M, 3))
    y_obs = theta_true @ A_demo.T + eps

    post_means = np.array(
        [(Spost @ (A_demo.T @ y_obs[m] / sigma_n**2))[1] for m in range(M)])

    theta2_g = np.linspace(-1.8, 1.8, 400)

    ax = axes[0]
    prior_pf = stats.norm.pdf(theta2_g, 0, sigma2)
    ax.fill_between(theta2_g, prior_pf, alpha=0.12, color=GREY)
    ax.plot(theta2_g, prior_pf, color=GREY, lw=2.5, ls="--",
            label=f"Prior push-forward  (std = {sigma2:.2f})")
    for m in range(M):
        ax.plot(theta2_g,
                stats.norm.pdf(theta2_g, post_means[m], pf_std_demo),
                color=BLUE, lw=1.0, alpha=0.45)
    ax.plot([], [], color=BLUE, lw=1.8, alpha=0.7,
            label=f"Posterior push-forwards  (std = {pf_std_demo:.2f} each)")
    ax.set_xlabel("QoI  $q(\\boldsymbol{\\theta}) = \\theta_2$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"$M={M}$ outer samples $\\to$ $M$ push-forward posteriors\n"
        "(same width, different centres — Gaussian case)",
        fontsize=9.5)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(0)
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.97, 0.93,
            "Width = push-forward std\n"
            "$U(\\mathbf{w})$ = expected (variance of) width",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(fc="white", ec=BLUE, lw=1, pad=2, alpha=0.9),
            color=BLUE)

    # Right panel: EIG(x) vs push-forward std(x)
    xs = np.linspace(0.01, 0.99, 200)
    eigs = [eig_fn([x]) for x in xs]
    pfstds = [pf_std_fn([x]) for x in xs]

    ax = axes[1]
    ax2 = ax.twinx()
    ln1, = ax.plot(xs, eigs, color=ORANGE, lw=2.5,
                   label="EIG  (max at $x=0$)")
    ln2, = ax2.plot(xs, pfstds, color=BLUE, lw=2.5, ls="--",
                    label="Push-forward std  (min at $x=1$)")
    ax.axvline(xs[np.argmax(eigs)], color=ORANGE, lw=1.2, ls=":", alpha=0.7)
    ax2.axvline(xs[np.argmin(pfstds)], color=BLUE, lw=1.2, ls=":", alpha=0.7)
    ax.set_xlabel("Sensor location $x$", fontsize=11)
    ax.set_ylabel("EIG (nats)", fontsize=11, color=ORANGE)
    ax2.set_ylabel("Push-forward std of $q$", fontsize=11, color=BLUE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax.set_title(
        "EIG and prediction utility disagree on the optimal sensor",
        fontsize=9.5)
    ax.grid(True, alpha=0.18)
    ax.legend([ln1, ln2], [str(ln1.get_label()), str(ln2.get_label())],
              fontsize=9.5, loc="center right")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)


def plot_pushforward_shrinkage(axes: Sequence[Axes]) -> None:
    """boed_pred_concept.qmd → fig-pushforward-shrinkage

    Push-forward shrinkage: Pred-OED-optimal vs EIG-optimal sensors.
    """
    from scipy import stats

    (sigma1, sigma2, sigma_n, Sigma0, Sigma0_inv, c,
     sensor_row, posterior_cov, pf_std, eig_val) = _pred_concept_helpers()

    BLUE = "#2C7FB8"
    ORANGE = "#E6550D"
    GREY = "#888888"

    nobs_sweep = list(range(1, 11))
    pf_A = [pf_std([0.0] * K) for K in nobs_sweep]
    pf_B = [pf_std([1.0] * K) for K in nobs_sweep]

    K_show = 5
    Spost_A5 = posterior_cov([0.0] * K_show)
    Spost_B5 = posterior_cov([1.0] * K_show)
    pf_A5 = np.sqrt(c @ Spost_A5 @ c)
    pf_B5 = np.sqrt(c @ Spost_B5 @ c)

    theta2_g = np.linspace(-1.5, 1.5, 400)
    prior_pdf = stats.norm.pdf(theta2_g, 0, sigma2)
    pf_A5_pdf = stats.norm.pdf(theta2_g, 0, pf_A5)
    pf_B5_pdf = stats.norm.pdf(theta2_g, 0, pf_B5)

    xs = np.linspace(0.01, 0.99, 200)
    pf_stds_x = [pf_std([x]) for x in xs]

    # Panel 1: push-forward std vs K
    ax = axes[0]
    ax.axhline(sigma2, color=GREY, lw=1.5, ls=":",
               label=f"Prior std = {sigma2}")
    ax.plot(nobs_sweep, pf_A, "o-", color=ORANGE, lw=2, ms=6,
            label="Design A  ($x=0$, EIG-optimal)")
    ax.plot(nobs_sweep, pf_B, "s-", color=BLUE, lw=2, ms=6,
            label="Design B  ($x=1$, Pred-OED-optimal)")
    ax.set_xlabel("Number of sensors $K$", fontsize=11)
    ax.set_ylabel("Push-forward std of $q$", fontsize=11)
    ax.set_title("Prediction uncertainty vs. $K$", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: PDFs at K=5
    ax = axes[1]
    ax.fill_between(theta2_g, prior_pdf, alpha=0.12, color=GREY)
    ax.plot(theta2_g, prior_pdf, color=GREY, lw=2, ls="--",
            label=f"Prior  (std = {sigma2:.2f})")
    ax.fill_between(theta2_g, pf_A5_pdf, alpha=0.20, color=ORANGE)
    ax.plot(theta2_g, pf_A5_pdf, color=ORANGE, lw=2.2,
            label=f"Design A  ($K={K_show}$,  std = {pf_A5:.2f})")
    ax.fill_between(theta2_g, pf_B5_pdf, alpha=0.20, color=BLUE)
    ax.plot(theta2_g, pf_B5_pdf, color=BLUE, lw=2.2,
            label=f"Design B  ($K={K_show}$,  std = {pf_B5:.2f})")
    ax.set_xlabel("QoI  $q(\\boldsymbol{\\theta}) = \\theta_2$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Push-forward PDFs at $K = {K_show}$ sensors", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: utility landscape
    ax = axes[2]
    ax.plot(xs, pf_stds_x, color=BLUE, lw=2.5)
    ax.axvline(xs[np.argmin(pf_stds_x)], color=BLUE, lw=1.5, ls="--",
               alpha=0.7, label="Optimum at $x=1$")
    ax.axhline(sigma2, color=GREY, lw=1.5, ls=":",
               label=f"Prior std = {sigma2}")
    ax.fill_between(xs, pf_stds_x, sigma2,
                    where=[p < sigma2 for p in pf_stds_x],
                    alpha=0.12, color=BLUE, label="Reduction vs. prior")
    ax.set_xlabel("Sensor location $x$", fontsize=11)
    ax.set_ylabel(
        "Push-forward std of $q$  $= \\sqrt{U(\\mathbf{w})}$", fontsize=10)
    ax.set_title("Utility landscape  (1 sensor)", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# boed_pred_usage.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_pred_mse_mc(
    axes: Sequence[Axes],
    bkd: Backend[Array],
    values_mc: Dict[str, List[Array]],
    outer_counts: List[int],
    inner_counts: List[int],
) -> None:
    """boed_pred_usage.qmd → fig-pred-mse-mc

    MSE of goal-oriented utility estimator vs inner samples for MC.
    """
    mse_array = bkd.to_numpy(bkd.vstack(values_mc["mse"]))

    colors = ["#2C7FB8", "#D95F02"]
    for ax_idx, nouter in enumerate(outer_counts):
        ax = axes[ax_idx]
        ax.loglog(inner_counts, mse_array[:, ax_idx], "o-",
                  color=colors[ax_idx], lw=2, label=f"$M = {nouter}$")

        ref_x = np.array([inner_counts[0], inner_counts[-1]])
        ref_y = mse_array[0, ax_idx] * ref_x[0] / ref_x
        ax.loglog(ref_x, ref_y, "k--", lw=1.5, alpha=0.6, label="$O(1/N)$")

        ax.set_title(f"$M = {nouter}$ outer samples", fontsize=11)
        ax.set_xlabel("Inner samples $N_{in}$", fontsize=11)
        ax.set_ylabel("MSE" if ax_idx == 0 else "", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, which="both")


# ---------------------------------------------------------------------------
# boed_pred_nonlinear_usage.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_lv_pred_target(
    ax: Axes,
    benchmark: Any,
    obs_model: Any,
    sample: Array,
    observations: Array,
    predictions: Array,
    bkd: Backend[Array],
) -> None:
    """boed_pred_nonlinear_usage.qmd → fig-lv-pred-target

    Observation times and prediction targets for Lotka-Volterra.
    """
    sol = obs_model.solve_trajectory(sample)[0]
    ax.plot(benchmark.solution_times(), bkd.to_numpy(sol).T, lw=2)

    obs_times = bkd.to_numpy(benchmark.observation_times())
    for ii, state_idx in enumerate([0, 2]):
        ax.plot(
            obs_times,
            bkd.to_numpy(observations[ii]),
            "o", ms=8, label=f"Obs. state {state_idx + 1}", alpha=0.8)

    pred_times = bkd.to_numpy(benchmark.prediction_times()).ravel()
    pred_vals = bkd.to_numpy(predictions).ravel()
    ax.plot(pred_times, pred_vals, "rx", ms=12, mew=2.5,
            label="Prediction targets", zorder=5)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Population", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Observation vs. prediction times", fontsize=11)
    ax.grid(True, alpha=0.2)


def plot_pred_design_weights(
    axes: Any,
    benchmark: Any,
    weights_std: Array,
    weights_ent: Array,
    bkd: Backend[Array],
) -> None:
    """boed_pred_nonlinear_usage.qmd → fig-lv-design-comparison

    Design weights for standard-deviation and entropic-deviation objectives.
    """
    weights_std_np = bkd.to_numpy(weights_std).reshape((2, -1))
    weights_ent_np = bkd.to_numpy(weights_ent).reshape((2, -1))
    obs_times = bkd.to_numpy(benchmark.observation_times())

    titles = ["State 1 (prey)", "State 3 (predator)"]
    row_labels = ["Std dev. objective", "Entropic deviation"]
    colors = ["#2C7FB8", "#D95F02"]

    for row, (w, rl) in enumerate(
        zip([weights_std_np, weights_ent_np], row_labels)
    ):
        for col in range(2):
            ax = axes[row][col]
            ax.bar(obs_times, w[col], color=colors[row], alpha=0.8)
            ax.set_xlabel("Time (s)", fontsize=10)
            if col == 0:
                ax.set_ylabel("Design weight", fontsize=10)
                ax.set_title(f"{rl} — {titles[col]}", fontsize=10)
            else:
                ax.set_title(titles[col], fontsize=10)
            ax.grid(True, axis="y", alpha=0.25)


# ---------------------------------------------------------------------------
# boed_data_workflow_usage.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_advec_diff_design(
    ax: Axes, obs_locs: NDArrayFloat, design_weights: NDArrayFloat,
) -> None:
    """boed_data_workflow_usage.qmd → fig-advec-diff-design

    Optimal sensor weights for the advection-diffusion domain.
    """
    import matplotlib.pyplot as plt

    w = design_weights
    w_norm = w / w.max()

    sc = ax.scatter(obs_locs[0], obs_locs[1], c=w, s=w_norm * 60 + 5,
                    cmap="YlOrRd", edgecolors="k", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Design weight")
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("Optimal sensor placement: advection-diffusion", fontsize=11)
