"""Plotting functions for GP tutorials.

Covers: gp_surrogate.qmd, gp_sensitivity.qmd, multioutput_gp.qmd,
        adaptive_gp_sampling.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# gp_surrogate.qmd — Convention A (echo:false) unless noted
# ---------------------------------------------------------------------------

def plot_prior_posterior_samples(
    axes, xi_grid, prior_std, prior_paths, mu_post, std_post,
    post_paths, X_obs, y_obs, n_samples,
):
    """gp_surrogate.qmd -> fig-prior-posterior-samples

    Prior and posterior GP sample paths on a 1D slice.
    """
    from ._style import apply_style

    colors = __import__("matplotlib").cm.tab10(
        np.linspace(0, 0.5, n_samples)
    )

    # Prior panel
    ax = axes[0]
    ax.fill_between(
        xi_grid, -2 * prior_std, 2 * prior_std,
        alpha=0.15, color="#2C7FB8", label=r"$\pm 2\sigma$ prior",
    )
    for i in range(n_samples):
        ax.plot(xi_grid, prior_paths[:, i], lw=1.2, alpha=0.8, color=colors[i])
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$\xi_1$", fontsize=11)
    ax.set_ylabel("Output", fontsize=11)
    ax.set_title("GP prior: 5 sample paths", fontsize=10)
    ax.legend(fontsize=8)
    apply_style(ax)

    # Posterior panel
    ax = axes[1]
    ax.fill_between(
        xi_grid, mu_post - 2 * std_post, mu_post + 2 * std_post,
        alpha=0.20, color="#117A65", label=r"$\pm 2\sigma^*$ posterior",
    )
    ax.plot(xi_grid, mu_post, color="#117A65", lw=2, label=r"$\mu^*$")
    for i in range(n_samples):
        ax.plot(xi_grid, post_paths[:, i], lw=1.2, alpha=0.8, color=colors[i])
    ax.scatter(X_obs[0], y_obs[0], s=50, color="k", zorder=5,
               label="Training data")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$\xi_1$", fontsize=11)
    ax.set_title("GP posterior: paths conditioned on data", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    apply_style(ax)


def plot_kernel_comparison(axes, bkd_preview, n_samples=5):
    """gp_surrogate.qmd -> fig-kernel-comparison

    Prior sample paths for SE, Matern 5/2, and Matern 3/2 kernels.
    """
    from pyapprox.surrogates.kernels.matern import (
        SquaredExponentialKernel, Matern52Kernel, Matern32Kernel,
    )
    from ._style import apply_style

    _xi = np.linspace(-3, 3, 300)
    _Xg = _xi.reshape(1, -1)
    _rng = np.random.default_rng(42)
    _ells = [0.8]
    _kernels_cmp = [
        ("Squared\nExponential",
         SquaredExponentialKernel(_ells, (0.1, 10.), 1, bkd_preview)),
        ("Matern 5/2",
         Matern52Kernel(_ells, (0.1, 10.), 1, bkd_preview)),
        ("Matern 3/2",
         Matern32Kernel(_ells, (0.1, 10.), 1, bkd_preview)),
    ]
    _palette = ["#2980B9", "#117A65", "#7D3C98"]

    for ax, (label, kern), color in zip(axes, _kernels_cmp, _palette):
        _K = bkd_preview.to_numpy(kern(_Xg, _Xg)) + 1e-8 * np.eye(len(_xi))
        _L = np.linalg.cholesky(_K)
        _s = np.sqrt(np.diag(_K))
        _Z = _rng.standard_normal((len(_xi), n_samples))
        _paths = _L @ _Z
        ax.fill_between(_xi, -2 * _s, 2 * _s, alpha=0.12, color=color)
        for i in range(n_samples):
            ax.plot(_xi, _paths[:, i], lw=1.3, alpha=0.85, color=color)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r"$\xi$", fontsize=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.3)
        apply_style(ax)
    axes[0].set_ylabel("Output", fontsize=10)


def plot_nlml_landscape(ax, bkd, nvars, samples_train, values_train_tip):
    """gp_surrogate.qmd -> fig-nlml-landscape

    NLML contour as a function of the first two length scales.
    """
    import matplotlib.pyplot as plt
    from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
    from pyapprox.surrogates.kernels.matern import Matern52Kernel

    n_grid = 25
    ell_vals = np.logspace(-1, 1, n_grid)
    nlml_grid = np.zeros((n_grid, n_grid))

    for ii, l1 in enumerate(ell_vals):
        for jj, l2 in enumerate(ell_vals):
            _ls = [l1, l2] + [1.0] * (nvars - 2)
            _k_tmp = Matern52Kernel(_ls, (0.1, 10.0), nvars, bkd)
            _gp_tmp = ExactGaussianProcess(
                _k_tmp, nvars=nvars, bkd=bkd, nugget=1e-6,
            )
            _gp_tmp.hyp_list().set_all_inactive()
            _gp_tmp.fit(samples_train, values_train_tip)
            nlml_grid[jj, ii] = bkd.to_float(
                bkd.asarray([_gp_tmp.neg_log_marginal_likelihood()])[0]
            )

    best_ii, best_jj = np.unravel_index(
        np.argmin(nlml_grid), nlml_grid.shape,
    )

    cf = ax.contourf(
        ell_vals, ell_vals, nlml_grid, levels=30, cmap="viridis_r",
    )
    ax.contour(
        ell_vals, ell_vals, nlml_grid, levels=15, colors="k",
        linewidths=0.4, alpha=0.3,
    )
    plt.colorbar(cf, ax=ax, label="NLML")
    ax.scatter(
        [ell_vals[best_ii]], [ell_vals[best_jj]],
        marker="*", s=200, color="red", zorder=5, label="Minimum",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Length scale $\ell_1$", fontsize=10)
    ax.set_ylabel(r"Length scale $\ell_2$", fontsize=10)
    ax.set_title(r"NLML landscape ($\ell_3 = 1.0$ fixed)", fontsize=10)
    ax.legend(fontsize=9)


def plot_length_scales(ax, gp, bkd):
    """gp_surrogate.qmd -> fig-length-scales

    Fitted length scales as horizontal bar chart.
    """
    _ls_log = [
        h for h in gp.hyp_list().hyperparameters()
        if "lenscale" in h._name.lower()
    ]
    _ls_vals = [
        v.item() for v in bkd.to_numpy(bkd.exp(_ls_log[0].get_values()))
    ]
    _ls_names = [
        rf"$\ell_{{{i+1}}}$ ($\xi_{{{i+1}}}$)"
        for i in range(len(_ls_vals))
    ]

    bars = ax.barh(
        _ls_names, _ls_vals, color="#2C7FB8", edgecolor="k",
        lw=0.7, height=0.5,
    )
    ax.axvline(1.0, color="red", lw=1.5, ls="--", label="Initial value (1.0)")
    for bar, v in zip(bars, _ls_vals):
        ax.text(
            v + 0.03, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}", va="center", fontsize=9,
        )
    ax.set_xlabel("Fitted length scale", fontsize=10)
    ax.set_title("Hyperparameter fitting result", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)


def plot_gp_predictions(axes, vals_np, mean_np, std_np, l2_err):
    """gp_surrogate.qmd -> fig-gp-predictions

    Predicted vs true scatter and standardised residual histogram.
    """
    import matplotlib.pyplot as plt
    from ._style import apply_style

    # Left: prediction vs truth coloured by std
    sc = axes[0].scatter(
        vals_np, mean_np, s=12, alpha=0.7,
        c=std_np, cmap="YlOrRd", edgecolors="none",
    )
    lims = [
        min(vals_np.min(), mean_np.min()),
        max(vals_np.max(), mean_np.max()),
    ]
    pad = 0.1 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]
    axes[0].plot(lims, lims, "k--", lw=1.5)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"True $\delta_{\mathrm{tip}}$", fontsize=11)
    axes[0].set_ylabel(r"GP mean $\hat{\delta}_{\mathrm{tip}}$", fontsize=11)
    axes[0].set_title(f"Rel. $L_2$ error = {l2_err:.2e}", fontsize=10)
    apply_style(axes[0])
    plt.colorbar(sc, ax=axes[0], label=r"$\sigma^*$")

    # Right: standardised residuals
    std_errors = (vals_np - mean_np) / np.maximum(std_np, 1e-12)
    axes[1].hist(
        std_errors, bins=35, density=True,
        color="#2C7FB8", edgecolor="k", lw=0.2, alpha=0.7,
    )
    x_ref = np.linspace(-4, 4, 200)
    axes[1].plot(
        x_ref, np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi),
        "r--", lw=1.5, label=r"$\mathcal{N}(0,1)$",
    )
    axes[1].set_xlabel(r"$(f - \mu^*) / \sigma^*$", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Standardised residuals", fontsize=10)
    axes[1].legend(fontsize=9)
    apply_style(axes[1])


def plot_calibration(ax, vals_np, mean_np, std_np):
    """gp_surrogate.qmd -> fig-calibration

    Reliability diagram comparing nominal vs empirical coverage.
    """
    from scipy import stats as sp_stats
    from ._style import apply_style

    nominal_levels = np.linspace(0.01, 0.99, 60)
    empirical_coverage = []
    for p in nominal_levels:
        z = sp_stats.norm.ppf(0.5 + p / 2)
        inside = np.abs(vals_np - mean_np) < z * std_np
        empirical_coverage.append(np.mean(inside))

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(
        nominal_levels, empirical_coverage, "o-", color="#2C7FB8",
        ms=4, lw=1.5, alpha=0.85, label="GP surrogate",
    )
    ax.fill_between(
        nominal_levels, nominal_levels, empirical_coverage,
        where=np.array(empirical_coverage) > nominal_levels,
        alpha=0.12, color="#117A65", label="Under-confident (safe)",
    )
    ax.fill_between(
        nominal_levels, nominal_levels, empirical_coverage,
        where=np.array(empirical_coverage) < nominal_levels,
        alpha=0.12, color="#E74C3C", label="Over-confident (risky)",
    )
    ax.set_xlabel("Nominal coverage probability", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Calibration diagram", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    apply_style(ax)


def plot_gp_convergence_n(
    ax, bkd, nvars, prior, model, samples_test, vals_np,
):
    """gp_surrogate.qmd -> fig-convergence-n

    Relative L2 error vs training set size for Matern 5/2 and SE kernels.
    """
    from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
    from pyapprox.surrogates.gaussianprocess.fitters import (
        GPMaximumLikelihoodFitter,
    )
    from pyapprox.surrogates.kernels.matern import (
        Matern52Kernel, SquaredExponentialKernel,
    )
    from ._style import apply_style

    budgets_n = [10, 20, 40, 60, 80, 100]

    np.random.seed(42)
    X_all_pool = prior.rvs(max(budgets_n))
    y_all_pool = model(X_all_pool)[:1, :]

    errors_matern = []
    errors_se = []
    for n in budgets_n:
        _k_m = Matern52Kernel([1.0] * nvars, (0.1, 10.0), nvars, bkd)
        _gp_m = ExactGaussianProcess(_k_m, nvars=nvars, bkd=bkd, nugget=1e-6)
        _res_m = GPMaximumLikelihoodFitter(bkd).fit(
            _gp_m, X_all_pool[:, :n], y_all_pool[:, :n],
        )
        _pred_m = bkd.to_numpy(_res_m.surrogate().predict(samples_test))[0, :]
        errors_matern.append(
            np.linalg.norm(vals_np - _pred_m) / np.linalg.norm(vals_np),
        )

        _k_s = SquaredExponentialKernel(
            [1.0] * nvars, (0.1, 10.0), nvars, bkd,
        )
        _gp_s = ExactGaussianProcess(_k_s, nvars=nvars, bkd=bkd, nugget=1e-6)
        _res_s = GPMaximumLikelihoodFitter(bkd).fit(
            _gp_s, X_all_pool[:, :n], y_all_pool[:, :n],
        )
        _pred_s = bkd.to_numpy(_res_s.surrogate().predict(samples_test))[0, :]
        errors_se.append(
            np.linalg.norm(vals_np - _pred_s) / np.linalg.norm(vals_np),
        )

    ax.loglog(
        budgets_n, errors_matern, "o-", color="#2C7FB8",
        lw=2, ms=7, label="Matern 5/2",
    )
    ax.loglog(
        budgets_n, errors_se, "s-", color="#E67E22",
        lw=2, ms=7, label="Squared Exponential",
    )
    ref = errors_matern[0] * (np.array(budgets_n) / budgets_n[0]) ** (-1.0)
    ax.loglog(
        budgets_n, ref, "k--", lw=1.2, alpha=0.6, label=r"$N^{-1}$ reference",
    )
    ax.set_xlabel("Training set size $N$", fontsize=11)
    ax.set_ylabel(r"Relative $L_2$ error", fontsize=11)
    ax.set_title("Convergence with training data", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)


def plot_uncertainty_map(ax, gp, bkd, samples_train, N_train):
    """gp_surrogate.qmd -> fig-uncertainty-map

    2D posterior std map with training point overlay.
    """
    import matplotlib.pyplot as plt

    n2d = 60
    xi1 = np.linspace(-3, 3, n2d)
    xi2 = np.linspace(-3, 3, n2d)
    Xi1, Xi2 = np.meshgrid(xi1, xi2)
    Xi3 = np.zeros_like(Xi1)

    X_2d = bkd.asarray(
        np.row_stack([Xi1.ravel(), Xi2.ravel(), Xi3.ravel()])
    )
    std_2d = bkd.to_numpy(gp.predict_std(X_2d))[0, :].reshape(n2d, n2d)

    X_tr_np = bkd.to_numpy(samples_train)

    cf = ax.contourf(xi1, xi2, std_2d, levels=25, cmap="YlOrBr")
    plt.colorbar(
        cf, ax=ax,
        label=r"$\sigma^*(\xi_1, \xi_2,\, \xi_3{=}0)$",
    )
    ax.scatter(
        X_tr_np[0, :], X_tr_np[1, :],
        s=22, color="k", marker="+", linewidths=1.2,
        label=f"Training points ($N={N_train}$)", zorder=5,
    )
    ax.set_xlabel(r"$\xi_1$", fontsize=11)
    ax.set_ylabel(r"$\xi_2$", fontsize=11)
    ax.set_title(
        r"Posterior uncertainty map: $\sigma^*(\xi_1, \xi_2)$", fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)


# ---------------------------------------------------------------------------
# gp_sensitivity.qmd — Convention A (echo:false)
# ---------------------------------------------------------------------------

def plot_gp_sobol(
    axes, nvars, S1_gp, ST_gp, S1_ff, ST_ff, gt_main, gt_total,
):
    """gp_sensitivity.qmd -> fig-gp-sobol

    GP kernel-integral vs fix-and-freeze Sobol indices with ground truth.
    """
    from ._style import apply_style

    x = np.arange(nvars)
    width = 0.35
    hw = width / 2 + 0.08
    input_labels = [f"$x_{i+1}$" for i in range(nvars)]

    ax1, ax2 = axes

    # First-order
    ax1.bar(
        x, S1_gp, width, label="GP kernel integrals",
        color="#2C7FB8", edgecolor="k", lw=0.5,
    )
    ax1.scatter(
        x, S1_ff, s=50, color="white", edgecolors="#084594",
        linewidths=1.5, zorder=5, label="Fix & freeze",
    )
    for i in range(nvars):
        ax1.plot(
            [i - hw, i + hw], [gt_main[i], gt_main[i]],
            color="k", linestyle="--", alpha=0.6, lw=1.2,
            label="Ground truth" if i == 0 else None,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(input_labels, fontsize=11)
    ax1.set_ylabel("Sensitivity index", fontsize=10)
    ax1.set_title("First-order $S_i$", fontsize=11)
    ax1.set_ylim(0, 0.65)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25, axis="y")

    # Total effect
    ax2.bar(
        x, ST_gp, width, label="GP kernel integrals",
        color="#FD8D3C", edgecolor="k", lw=0.5,
    )
    ax2.scatter(
        x, ST_ff, s=50, color="white", edgecolors="#A63603",
        linewidths=1.5, zorder=5, label="Fix & freeze",
    )
    for i in range(nvars):
        ax2.plot(
            [i - hw, i + hw], [gt_total[i], gt_total[i]],
            color="k", linestyle="--", alpha=0.6, lw=1.2,
            label="Ground truth" if i == 0 else None,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(input_labels, fontsize=11)
    ax2.set_title("Total effect $S_i^T$", fontsize=11)
    ax2.set_ylim(0, 0.7)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25, axis="y")


def plot_sobol_distribution(axes, nvars, S_dist, gt_main, bkd):
    """gp_sensitivity.qmd -> fig-sobol-distribution

    Histograms of first-order Sobol indices across GP posterior realizations.
    """
    from ._style import apply_style

    colors = ["#2C7FB8", "#E67E22", "#27AE60"]

    for d in range(nvars):
        ax = axes[d]
        S_d = bkd.to_numpy(S_dist[d])
        ax.hist(
            S_d, bins=30, density=True, alpha=0.7, color=colors[d],
            edgecolor="k", lw=0.3,
        )
        ax.axvline(
            gt_main[d], color="k", lw=1.5, ls="--", alpha=0.7,
            label=f"Truth: {gt_main[d]:.3f}",
        )
        ax.axvline(
            S_d.mean(), color=colors[d], lw=1.5, ls="-",
            label=f"Mean: {S_d.mean():.3f}",
        )
        ax.set_xlabel(f"$S_{d+1}$", fontsize=11)
        ax.set_title(f"$x_{d+1}$", fontsize=11)
        ax.legend(fontsize=8)
        apply_style(ax)

    axes[0].set_ylabel("Density", fontsize=10)


# ---------------------------------------------------------------------------
# multioutput_gp.qmd — Convention A (echo:false)
# ---------------------------------------------------------------------------

def plot_forrester_functions(ax, xr, y_hf_true, y_lf_true):
    """multioutput_gp.qmd -> fig-forrester-functions

    High- and low-fidelity Forrester benchmark curves.
    """
    from ._style import apply_style

    ax.plot(xr, y_hf_true, "k-", lw=2.5, label="High fidelity $f_h(x)$")
    ax.plot(
        xr, y_lf_true, "--", color="#E67E22", lw=2,
        label="Low fidelity $f_l(x)$",
    )
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$f(x)$", fontsize=12)
    ax.legend(fontsize=11)
    apply_style(ax)


def plot_predictions_uncertainty(
    axes, xr, y_test_hf,
    mf_pred, mf_std, l2_mf, n_lf, n_hf, total_cost,
    hf_pred, hf_std, l2_hf, n_hf_equiv,
    hf_few_pred, hf_few_std, l2_hf_few,
    xs_hf, ys_hf, xs_hf_only, ys_hf_only, costs_per_eval,
):
    """multioutput_gp.qmd -> fig-predictions-uncertainty

    Three-panel comparison of MF GP, HF-only equal budget, HF-only 4 samples.
    """
    from ._style import apply_style

    # Top: Multi-fidelity GP
    ax = axes[0]
    ax.plot(xr, y_test_hf, "k-", lw=2, label="HF truth", zorder=4)
    ax.plot(
        xr, mf_pred, "-", color="#1F77B4", lw=2,
        label=f"MF GP (L2={l2_mf:.2e})",
    )
    ax.fill_between(
        xr, mf_pred - 2 * mf_std, mf_pred + 2 * mf_std,
        color="#1F77B4", alpha=0.2, label="$\\pm 2\\sigma$",
    )
    ax.scatter(
        xs_hf, ys_hf, s=50, color="#D62728", edgecolor="k", lw=0.5,
        zorder=5, label=f"HF data ($N={n_hf}$)",
    )
    ax.set_ylabel("$f(x)$", fontsize=12)
    ax.set_title(
        f"Multi-fidelity GP ({n_lf} LF + {n_hf} HF, "
        f"cost = {total_cost:.0f} s)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    apply_style(ax)

    # Middle: HF-only GP at equal budget
    ax = axes[1]
    ax.plot(xr, y_test_hf, "k-", lw=2, label="HF truth", zorder=4)
    ax.plot(
        xr, hf_pred, "-", color="#E67E22", lw=2,
        label=f"HF-only GP (L2={l2_hf:.2e})",
    )
    ax.fill_between(
        xr, hf_pred - 2 * hf_std, hf_pred + 2 * hf_std,
        color="#E67E22", alpha=0.2, label="$\\pm 2\\sigma$",
    )
    ax.scatter(
        xs_hf_only, ys_hf_only, s=50, color="#D62728", edgecolor="k",
        lw=0.5, zorder=5, label=f"HF data ($N={n_hf_equiv}$)",
    )
    ax.set_ylabel("$f(x)$", fontsize=12)
    ax.set_title(
        f"HF-only GP ({n_hf_equiv} HF, cost = {total_cost:.0f} s)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    apply_style(ax)

    # Bottom: HF-only GP with same 4 HF samples
    ax = axes[2]
    ax.plot(xr, y_test_hf, "k-", lw=2, label="HF truth", zorder=4)
    ax.plot(
        xr, hf_few_pred, "-", color="#2CA02C", lw=2,
        label=f"HF-only GP (L2={l2_hf_few:.2e})",
    )
    ax.fill_between(
        xr, hf_few_pred - 2 * hf_few_std, hf_few_pred + 2 * hf_few_std,
        color="#2CA02C", alpha=0.2, label="$\\pm 2\\sigma$",
    )
    ax.scatter(
        xs_hf, ys_hf, s=50, color="#D62728", edgecolor="k", lw=0.5,
        zorder=5, label=f"HF data ($N={n_hf}$)",
    )
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$f(x)$", fontsize=12)
    ax.set_title(
        f"HF-only GP ({n_hf} HF, "
        f"cost = {n_hf * costs_per_eval[-1]:.0f} s)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    apply_style(ax)


def plot_kernel_matrix(ax, K_np, n_cov, level_names, nmodels):
    """multioutput_gp.qmd -> fig-kernel-matrix

    Block kernel matrix heatmap for multi-level GP.
    """
    import matplotlib.pyplot as plt

    im = ax.imshow(K_np, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Kernel value", fraction=0.046)
    block = n_cov
    for k, name in enumerate(level_names):
        mid = k * block + block // 2
        ax.text(mid, -1.5, name, ha="center", va="top", fontsize=9)
        ax.text(
            -1.5, mid, name, ha="right", va="center",
            fontsize=9, rotation=90,
        )
        if k < nmodels - 1:
            ax.axvline((k + 1) * block - 0.5, color="k", lw=0.8)
            ax.axhline((k + 1) * block - 0.5, color="k", lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Multi-level kernel matrix", fontsize=11)


# ---------------------------------------------------------------------------
# adaptive_gp_sampling.qmd — Convention B (echo:true)
# ---------------------------------------------------------------------------

def plot_sample_designs(axes, strategies):
    """adaptive_gp_sampling.qmd -> fig-sample-designs

    2D projections of Cholesky, IVAR, and Sobol sample designs.
    """
    from ._style import apply_style

    for ax, (pts, color, label) in zip(axes, strategies):
        ax.scatter(
            pts[0, :], pts[1, :], s=22, color=color, alpha=0.8,
            edgecolors="k", linewidths=0.3,
        )
        ax.set_xlabel(r"$\xi_1$", fontsize=11)
        ax.set_ylabel(r"$\xi_2$", fontsize=11)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        apply_style(ax)


def plot_gp_sampling_convergence(
    ax, budgets, chol_errors, ivar_errors, sobol_errors,
):
    """adaptive_gp_sampling.qmd -> fig-convergence

    Relative L2 error vs training samples for three sampling strategies.
    """
    from ._style import apply_style

    ax.semilogy(
        budgets, chol_errors, "o-", color="#1A5276",
        lw=2, ms=6, label="Cholesky-greedy",
    )
    ax.semilogy(
        budgets, ivar_errors, "s-", color="#117A65",
        lw=2, ms=6, label="IVAR",
    )
    ax.semilogy(
        budgets, sobol_errors, "^--", color="#7D3C98",
        lw=2, ms=6, label="Sobol (space-filling)", alpha=0.8,
    )
    ax.set_xlabel("Number of training samples $N$", fontsize=11)
    ax.set_ylabel(r"Relative $L_2$ error", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("GP convergence by sampling strategy", fontsize=10)
