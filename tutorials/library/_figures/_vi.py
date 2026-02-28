"""Plotting functions for variational inference tutorials.

Covers: vi_intro.qmd, vi_families.qmd, vi_objective.qmd,
        vi_optimization.qmd, vi_amortized.qmd

All figures are echo:false -> Convention A (figure generators that accept
PyApprox objects and handle all compute + plot internally).
"""

import math

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _grid_integrate(values, grid):
    """Integrate values on a 1D grid using piecewise-linear quadrature."""
    from pyapprox.surrogates.affine.univariate.piecewisepoly import (
        PiecewiseLinear,
    )
    from pyapprox.util.backends.numpy import NumpyBkd

    _bkd = NumpyBkd()
    pw = PiecewiseLinear(_bkd.asarray(grid), _bkd)
    _, weights = pw.quadrature_rule()
    return float(_bkd.sum(_bkd.asarray(values) * weights))


def _beam_exact_posterior(bkd, tip_model, mu_prior, sigma_prior,
                          y_obs, sigma_noise):
    """Compute exact posterior moments and grid PDF for the 1D beam."""
    from pyapprox.surrogates.affine.univariate import (
        HermitePolynomial1D,
        GaussQuadratureRule,
    )

    hermite_poly = HermitePolynomial1D(bkd)
    hermite_rule = GaussQuadratureRule(hermite_poly)
    nquad = 100
    quad_pts, quad_wts = hermite_rule(nquad)
    zq = bkd.to_numpy(quad_pts[0])
    wq = bkd.to_numpy(quad_wts[:, 0])
    Eq = mu_prior + sigma_prior * zq

    tip_preds_q = bkd.to_numpy(
        tip_model(bkd.asarray(Eq[np.newaxis, :]))
    )[0, :]
    lik_vals_q = norm.pdf(y_obs, tip_preds_q, sigma_noise)

    Z = wq @ lik_vals_q
    exact_mean = (wq @ (Eq * lik_vals_q)) / Z
    exact_var = (wq @ ((Eq - exact_mean)**2 * lik_vals_q)) / Z
    exact_std = np.sqrt(exact_var)

    E_grid = np.linspace(4_000, 20_000, 500)
    prior_vals = norm.pdf(E_grid, mu_prior, sigma_prior)
    tip_preds = bkd.to_numpy(
        tip_model(bkd.asarray(E_grid[np.newaxis, :]))
    )[0, :]
    lik_vals = norm.pdf(y_obs, tip_preds, sigma_noise)
    post_exact = lik_vals * prior_vals
    post_exact = post_exact / _grid_integrate(post_exact, E_grid)

    return exact_mean, exact_std, E_grid, post_exact


def _make_degree0_expansion(bkd, coeff=0.0):
    """Degree-0 expansion: a single tunable constant."""
    from pyapprox.probability.univariate import UniformMarginal
    from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
    from pyapprox.surrogates.affine.expansions import BasisExpansion
    from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
    from pyapprox.surrogates.affine.univariate import create_bases_1d

    marginals = [UniformMarginal(-1.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, 0, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    exp.set_coefficients(bkd.asarray([[coeff]]))
    return exp


def _run_vi_with_maxiter(bkd, tip_model, std_tip_model, noise_variances,
                         y_obs, mu_prior, sigma_prior, maxiter, seed=42):
    """Run VI in standardized space, return (mu_E, sigma_E, neg_elbo)."""
    from pyapprox.probability.conditional.gaussian import ConditionalGaussian
    from pyapprox.probability.conditional.joint import (
        ConditionalIndependentJoint,
    )
    from pyapprox.probability.joint.independent import IndependentJoint
    from pyapprox.probability.univariate.gaussian import GaussianMarginal
    from pyapprox.probability.likelihood import DiagonalGaussianLogLikelihood
    from pyapprox.probability.likelihood.gaussian import (
        MultiExperimentLogLikelihood,
    )
    from pyapprox.inverse.variational.elbo import make_single_problem_elbo
    from pyapprox.inverse.variational.fitter import VariationalFitter
    from pyapprox.optimization.minimize.scipy.trust_constr import (
        ScipyTrustConstrOptimizer,
    )

    mf = _make_degree0_expansion(bkd, 0.0)
    lsf = _make_degree0_expansion(bkd, 0.0)
    cond = ConditionalGaussian(mf, lsf, bkd)
    vd = ConditionalIndependentJoint([cond], bkd)
    vp = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

    base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
    multi_lik = MultiExperimentLogLikelihood(
        base_lik, bkd.asarray([[y_obs]]), bkd,
    )

    def log_lik_fn(z):
        return multi_lik.logpdf(std_tip_model(z))

    np.random.seed(seed)
    bs = bkd.asarray(np.random.normal(0, 1, (1, 1000)))
    wt = bkd.full((1, 1000), 1.0 / 1000)
    elbo = make_single_problem_elbo(vd, log_lik_fn, vp, bs, wt, bkd)

    opt = ScipyTrustConstrOptimizer(maxiter=maxiter, gtol=1e-8, verbosity=0)
    fitter = VariationalFitter(bkd, optimizer=opt)
    result = fitter.fit(elbo)

    dummy = bkd.zeros((cond.nvars(), 1))
    z_mu = float(cond._mean_func(dummy)[0, 0])
    z_sig = math.exp(float(cond._log_stdev_func(dummy)[0, 0]))
    return (mu_prior + sigma_prior * z_mu,
            sigma_prior * z_sig,
            result.neg_elbo())


# ---------------------------------------------------------------------------
# vi_intro.qmd
# ---------------------------------------------------------------------------

def plot_two_strategies(ax1, ax2):
    """vi_intro.qmd -> fig-two-strategies

    MCMC samples vs VI Gaussian fit on a mildly skewed posterior.
    """
    from ._style import apply_style

    rng = np.random.default_rng(42)

    theta_grid = np.linspace(-4, 6, 500)
    true_post = (0.6 * norm.pdf(theta_grid, 1.0, 0.8)
                 + 0.4 * norm.pdf(theta_grid, 2.5, 0.6))
    true_post /= _grid_integrate(true_post, theta_grid)

    # --- Left: MCMC ---
    chain = np.empty(300)
    chain[0] = 0.0
    for t in range(1, 300):
        prop = chain[t - 1] + rng.normal(0, 0.6)
        p_curr = np.interp(chain[t - 1], theta_grid, true_post)
        p_prop = np.interp(prop, theta_grid, true_post)
        if rng.uniform() < min(1, p_prop / max(p_curr, 1e-30)):
            chain[t] = prop
        else:
            chain[t] = chain[t - 1]

    ax1.fill_between(theta_grid, true_post, alpha=0.15, color="#E67E22")
    ax1.plot(theta_grid, true_post, color="#E67E22", lw=2,
             label="True posterior")
    ax1.hist(chain[50:], bins=30, density=True, color="#2C7FB8", alpha=0.5,
             edgecolor="k", lw=0.3, label="MCMC samples")
    ax1.set_xlabel(r"$\theta$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("MCMC: sample the posterior", fontsize=11)
    ax1.legend(fontsize=9)
    apply_style(ax1)

    # --- Right: VI ---
    candidates = [(0.5, 1.2, "#CCCCCC"), (2.5, 0.5, "#CCCCCC"),
                  (0.8, 0.6, "#CCCCCC"), (1.5, 0.9, "#2C7FB8")]
    ax2.fill_between(theta_grid, true_post, alpha=0.15, color="#E67E22")
    ax2.plot(theta_grid, true_post, color="#E67E22", lw=2,
             label="True posterior")
    for ii, (mu, sig, c) in enumerate(candidates[:-1]):
        q = norm.pdf(theta_grid, mu, sig)
        ax2.plot(theta_grid, q, color=c, lw=1.2, ls="--", alpha=0.6,
                 label="Other candidates" if ii == 0 else None)
    mu_best, sig_best, c_best = candidates[-1]
    q_best = norm.pdf(theta_grid, mu_best, sig_best)
    ax2.plot(theta_grid, q_best, color=c_best, lw=2.5,
             label="Best fit (VI result)")
    ax2.fill_between(theta_grid, q_best, alpha=0.1, color=c_best)
    ax2.set_xlabel(r"$\theta$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("VI: fit the best simple distribution", fontsize=11)
    ax2.legend(fontsize=9)
    apply_style(ax2)


def plot_candidate_gallery(E_grid, post_exact, exact_mean, exact_std,
                           E_true, ax):
    """vi_intro.qmd -> fig-candidate-gallery

    Five Gaussian candidates overlaid on exact beam posterior.
    """
    from ._style import apply_style

    candidates = [
        (8_000, 2_000, "Too far left"),
        (10_500, 500, "Too narrow"),
        (10_000, 4_000, "Too wide"),
        (14_000, 1_500, "Shifted right"),
        (exact_mean, exact_std, "Best fit"),
    ]
    gray = "#BBBBBB"
    colors = [gray, gray, gray, gray, "#2C7FB8"]
    lws = [1.2, 1.2, 1.2, 1.2, 2.5]
    lss = ["--", "--", "--", "--", "-"]

    ax.plot(E_grid, post_exact, color="#E67E22", lw=2.5,
            label="Exact posterior", zorder=10)
    ax.fill_between(E_grid, post_exact, alpha=0.12, color="#E67E22")

    for (mu, sig, label), c, lw, ls in zip(candidates, colors, lws, lss):
        q = norm.pdf(E_grid, mu, sig)
        is_best = label == "Best fit"
        ax.plot(E_grid, q, color=c, lw=lw, ls=ls,
                label=f"$\\mu={mu:.0f},\\; \\sigma={sig:.0f}$ — {label}",
                zorder=11 if is_best else 5,
                alpha=1.0 if is_best else 0.7)

    ax.axvline(E_true, color="#27AE60", ls="--", lw=1.5, alpha=0.6,
               label=f"True $E$ = {E_true}")
    ax.set_xlabel(r"$E$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Which Gaussian best approximates the posterior?",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    apply_style(ax)


def plot_optimization_snapshots(bkd, tip_model, std_tip_model,
                                noise_variances, y_obs,
                                mu_prior, sigma_prior,
                                E_grid, post_exact, axes):
    """vi_intro.qmd -> fig-optimization-snapshots

    Four snapshots of VI convergence from prior to posterior.
    """
    snapshots = [
        (0, float(mu_prior), float(sigma_prior), None),
    ]
    for mi in [2, 4, 200]:
        mu_i, sig_i, ne_i = _run_vi_with_maxiter(
            bkd, tip_model, std_tip_model, noise_variances,
            y_obs, mu_prior, sigma_prior, mi,
        )
        snapshots.append((mi, mu_i, sig_i, ne_i))

    for ax, (mi, mu_i, sig_i, ne_i) in zip(axes, snapshots):
        ax.plot(E_grid, post_exact, color="#E67E22", lw=2)
        ax.fill_between(E_grid, post_exact, alpha=0.12, color="#E67E22")
        q_i = norm.pdf(E_grid, mu_i, sig_i)
        ax.plot(E_grid, q_i, color="#2C7FB8", lw=2.5, ls="--")
        ax.fill_between(E_grid, q_i, alpha=0.08, color="#2C7FB8")

        if mi == 0:
            ax.set_title("Initial (prior)", fontsize=10, fontweight="bold")
        else:
            ax.set_title(f"After {mi} iterations", fontsize=10,
                         fontweight="bold")

        ax.text(0.97, 0.95,
                f"$\\mu = {mu_i:.0f}$\n$\\sigma = {sig_i:.0f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(fc="white", ec="gray", lw=0.5, alpha=0.85))
        ax.set_xlabel(r"$E$", fontsize=10)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel("Density", fontsize=11)


def plot_vi_result(E_grid, post_exact, vi_mean, vi_stdev, E_true, ax):
    """vi_intro.qmd -> fig-vi-result

    VI Gaussian approximation overlaid on exact grid posterior.
    """
    from ._style import apply_style

    vi_pdf = norm.pdf(E_grid, vi_mean, vi_stdev)

    ax.plot(E_grid, post_exact, color="#E67E22", lw=2.5,
            label="Exact posterior (grid)")
    ax.fill_between(E_grid, post_exact, alpha=0.15, color="#E67E22")
    ax.plot(E_grid, vi_pdf, color="#2C7FB8", lw=2.5, ls="--",
            label=(f"VI: $\\mathcal{{N}}({vi_mean:.0f},"
                   f"\\, {vi_stdev:.0f}^2)$"))
    ax.fill_between(E_grid, vi_pdf, alpha=0.1, color="#2C7FB8")
    ax.axvline(E_true, color="#27AE60", ls="--", lw=1.5, alpha=0.8,
               label=f"True $E$ = {E_true}")
    ax.set_xlabel(r"$E$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Variational inference vs. exact posterior", fontsize=11)
    ax.legend(fontsize=10)
    apply_style(ax)


def plot_vi_vs_mcmc(bkd, tip_model, noise_likelihood, prior,
                    E_grid, post_exact, vi_mean, vi_stdev, E_true,
                    mu_prior, ax):
    """vi_intro.qmd -> fig-vi-vs-mcmc

    Exact posterior, MCMC histogram, and VI curve on same axes.
    """
    from pyapprox.inverse.posterior.log_unnormalized import (
        LogUnNormalizedPosterior,
    )
    from pyapprox.inverse.sampling.metropolis import (
        MetropolisHastingsSampler,
    )
    from ._style import apply_style

    posterior_fn = LogUnNormalizedPosterior(
        model_fn=tip_model, likelihood=noise_likelihood,
        prior=prior, bkd=bkd,
    )

    n_steps = 2000
    proposal_cov = bkd.asarray([[800.0**2]])
    sampler = MetropolisHastingsSampler(
        log_posterior_fn=posterior_fn, nvars=1, bkd=bkd,
        proposal_cov=proposal_cov,
    )

    np.random.seed(0)
    mcmc_result = sampler.sample(
        nsamples=n_steps, burn=0,
        initial_state=bkd.asarray([[float(mu_prior)]]),
        bounds=bkd.asarray([[4_000.0, 20_000.0]]),
    )
    chain = bkd.to_numpy(mcmc_result.samples)[0, :]
    burnin = 200
    chain_post = chain[burnin:]

    vi_pdf = norm.pdf(E_grid, vi_mean, vi_stdev)

    # Exact
    ax.plot(E_grid, post_exact, color="#E67E22", lw=2.5,
            label="Exact (grid)", zorder=10)
    ax.fill_between(E_grid, post_exact, alpha=0.08, color="#E67E22")

    # MCMC
    ax.hist(chain_post, bins=50, density=True, color="#2C7FB8", alpha=0.35,
            edgecolor="k", lw=0.3,
            label=f"MCMC ({n_steps - burnin} samples)", zorder=5)

    # VI
    ax.plot(E_grid, vi_pdf, color="#2C7FB8", lw=2.5, ls="--",
            label=(f"VI: $\\mathcal{{N}}({vi_mean:.0f},"
                   f"\\, {vi_stdev:.0f}^2)$"),
            zorder=11)

    ax.axvline(E_true, color="#27AE60", ls="--", lw=1.5, alpha=0.6)
    ax.set_xlabel(r"$E$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Exact vs. MCMC vs. VI", fontsize=11)
    ax.legend(fontsize=9)
    apply_style(ax)


def plot_bimodal(ax1, ax2):
    """vi_intro.qmd -> fig-bimodal

    Bimodal posterior with Gaussian moment-matched and mode-collapsed fits.
    """
    from ._style import apply_style

    bimodal_grid = np.linspace(-4, 6, 500)
    bimodal = (0.5 * norm.pdf(bimodal_grid, -1.5, 0.4)
               + 0.5 * norm.pdf(bimodal_grid, 1.5, 0.4))
    bimodal /= _grid_integrate(bimodal, bimodal_grid)

    bm_mean = _grid_integrate(bimodal_grid * bimodal, bimodal_grid)
    bm_var = _grid_integrate(
        (bimodal_grid - bm_mean)**2 * bimodal, bimodal_grid,
    )
    bm_std = np.sqrt(bm_var)
    gauss_approx = norm.pdf(bimodal_grid, bm_mean, bm_std)

    mode_collapsed = norm.pdf(bimodal_grid, 1.5, 0.4)

    # Left: moment-matched
    ax1.plot(bimodal_grid, bimodal, color="#E67E22", lw=2.5,
             label="True posterior")
    ax1.fill_between(bimodal_grid, bimodal, alpha=0.12, color="#E67E22")
    ax1.plot(bimodal_grid, gauss_approx, color="#2C7FB8", lw=2.5, ls="--",
             label="Best Gaussian (covers both)")
    ax1.fill_between(bimodal_grid, gauss_approx, alpha=0.08, color="#2C7FB8")
    ax1.set_xlabel(r"$\theta$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Option A: spread across both modes", fontsize=10)
    ax1.legend(fontsize=9)
    apply_style(ax1)

    # Right: mode-collapsed
    ax2.plot(bimodal_grid, bimodal, color="#E67E22", lw=2.5,
             label="True posterior")
    ax2.fill_between(bimodal_grid, bimodal, alpha=0.12, color="#E67E22")
    ax2.plot(bimodal_grid, mode_collapsed, color="#8E44AD", lw=2.5, ls="--",
             label="Alternative: lock onto one mode")
    ax2.fill_between(bimodal_grid, mode_collapsed, alpha=0.08,
                     color="#8E44AD")
    ax2.set_xlabel(r"$\theta$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Option B: collapse onto one mode", fontsize=10)
    ax2.legend(fontsize=9)
    apply_style(ax2)


# ---------------------------------------------------------------------------
# vi_families.qmd
# ---------------------------------------------------------------------------

def plot_family_comparison(E1_m, E2_m, p_grid, E1_true, E2_true,
                           conds_diag, var_dist_full,
                           mu_prior_2d, sigma_prior_2d,
                           result_diag, result_full, bkd, ax1, ax2):
    """vi_families.qmd -> fig-family-comparison

    Diagonal vs full-covariance Gaussian VI on 2D composite beam posterior.
    """
    from matplotlib.patches import Ellipse
    from ._style import apply_style

    # Extract diagonal params
    diag_means = []
    diag_stds = []
    for ii, cd in enumerate(conds_diag):
        dummy = bkd.zeros((cd.nvars(), 1))
        z_mu = float(cd._mean_func(dummy)[0, 0])
        z_sig = math.exp(float(cd._log_stdev_func(dummy)[0, 0]))
        diag_means.append(mu_prior_2d[ii] + sigma_prior_2d[ii] * z_mu)
        diag_stds.append(sigma_prior_2d[ii] * z_sig)

    # Extract full-covariance params
    dummy_full = bkd.zeros((var_dist_full.nvars(), 1))
    full_z_mean = bkd.to_numpy(
        var_dist_full.reparameterize(dummy_full, bkd.zeros((2, 1)))
    ).flatten()
    full_z_cov = bkd.to_numpy(var_dist_full.covariance(dummy_full)[0])
    full_mean = mu_prior_2d + sigma_prior_2d * full_z_mean
    S = np.diag(sigma_prior_2d)
    full_cov = S @ full_z_cov @ S

    for ax, title, neg_elbo in [
        (ax1,
         f"Diagonal Gaussian (neg ELBO = {result_diag.neg_elbo():.2f})",
         result_diag.neg_elbo()),
        (ax2,
         f"Full-covariance Gaussian (neg ELBO = {result_full.neg_elbo():.2f})",
         result_full.neg_elbo()),
    ]:
        ax.contourf(E1_m, E2_m, p_grid, levels=10, cmap="Oranges",
                     alpha=0.4)
        ax.contour(E1_m, E2_m, p_grid, levels=8, colors="#E67E22",
                    linewidths=0.5, alpha=0.5)
        ax.scatter(E1_true, E2_true, marker="D", s=80, color="#27AE60",
                   edgecolors="k", lw=1, zorder=5)
        ax.set_xlabel(r"$E_1$ (skin)", fontsize=12)
        ax.set_ylabel(r"$E_2$ (core)", fontsize=12)
        ax.set_title(title, fontsize=10, fontweight="bold")
        apply_style(ax)

    # Diagonal ellipses
    for n_std in [1, 2]:
        ell = Ellipse(
            xy=(diag_means[0], diag_means[1]),
            width=2 * n_std * diag_stds[0],
            height=2 * n_std * diag_stds[1],
            angle=0,
            facecolor="none", edgecolor="#2C7FB8",
            lw=2 if n_std == 1 else 1.5,
            ls="-" if n_std == 1 else "--",
        )
        ax1.add_patch(ell)
    ax1.scatter(diag_means[0], diag_means[1], marker="+", s=150,
                color="#2C7FB8", lw=3, zorder=5)

    # Full-covariance ellipses
    eigenvalues, eigenvectors = np.linalg.eigh(full_cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for n_std in [1, 2]:
        ell = Ellipse(
            xy=(full_mean[0], full_mean[1]),
            width=2 * n_std * np.sqrt(eigenvalues[0]),
            height=2 * n_std * np.sqrt(eigenvalues[1]),
            angle=angle,
            facecolor="none", edgecolor="#2C7FB8",
            lw=2 if n_std == 1 else 1.5,
            ls="-" if n_std == 1 else "--",
        )
        ax2.add_patch(ell)
    ax2.scatter(full_mean[0], full_mean[1], marker="+", s=150,
                color="#2C7FB8", lw=3, zorder=5)


def plot_beta_vi(prior_alpha, prior_beta, exact_posterior_alpha,
                 exact_posterior_beta, rec_alpha, rec_beta, ax):
    """vi_families.qmd -> fig-beta-vi

    Beta VI approximation vs exact conjugate posterior for coin-flip.
    """
    from scipy.stats import beta as beta_dist
    from ._style import apply_style

    p_grid = np.linspace(0.001, 0.999, 300)
    exact_pdf = beta_dist.pdf(p_grid, exact_posterior_alpha,
                              exact_posterior_beta)
    vi_pdf = beta_dist.pdf(p_grid, rec_alpha, rec_beta)
    prior_pdf = beta_dist.pdf(p_grid, prior_alpha, prior_beta)

    ax.plot(p_grid, prior_pdf, color="#2C7FB8", lw=1.5, ls=":", alpha=0.5,
            label=f"Prior: Beta({prior_alpha}, {prior_beta})")
    ax.plot(p_grid, exact_pdf, color="#E67E22", lw=2.5,
            label=f"Exact: Beta({exact_posterior_alpha:.0f}, "
                  f"{exact_posterior_beta:.0f})")
    ax.fill_between(p_grid, exact_pdf, alpha=0.12, color="#E67E22")
    ax.plot(p_grid, vi_pdf, color="#2C7FB8", lw=2.5, ls="--",
            label=f"VI: Beta({rec_alpha:.1f}, {rec_beta:.1f})")
    ax.set_xlabel("$p$ (probability of heads)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Beta VI recovers the conjugate posterior", fontsize=11)
    ax.legend(fontsize=10)
    apply_style(ax)


# ---------------------------------------------------------------------------
# vi_objective.qmd
# ---------------------------------------------------------------------------

def plot_overlap_intuition(E_grid, post_exact, exact_mean, exact_std, axes):
    """vi_objective.qmd -> fig-overlap-intuition

    Three Gaussian candidates with mismatch shading against exact posterior.
    """
    from ._style import apply_style

    candidates = [
        (8_000, 1_000, "Shifted left"),
        (exact_mean, 3_500, "Too wide"),
        (exact_mean, exact_std, "Well-fit"),
    ]

    for ax, (mu, sig, title) in zip(axes, candidates):
        q = norm.pdf(E_grid, mu, sig)

        ax.plot(E_grid, post_exact, color="#E67E22", lw=2.5,
                label="Exact posterior")
        ax.plot(E_grid, q, color="#2C7FB8", lw=2, ls="--",
                label=f"$q$: $\\mu={mu:.0f},\\; \\sigma={sig:.0f}$")

        mismatch = np.abs(q - post_exact)
        ax.fill_between(E_grid, 0, mismatch, alpha=0.25, color="#C0392B",
                        label="Mismatch")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$E$", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        apply_style(ax)

    axes[0].set_ylabel("Density", fontsize=11)


def plot_kl_landscape(E_grid, post_exact, ax, fig):
    """vi_objective.qmd -> fig-kl-landscape

    KL divergence surface over (mu, sigma) with candidates marked.
    """
    from ._style import apply_style

    mu_range = np.linspace(6_000, 16_000, 120)
    sig_range = np.linspace(300, 4_000, 100)
    mu_mesh, sig_mesh = np.meshgrid(mu_range, sig_range)
    kl_surface = np.full_like(mu_mesh, np.nan)

    for i in range(len(sig_range)):
        for j in range(len(mu_range)):
            q_vals = norm.pdf(E_grid, mu_mesh[i, j], sig_mesh[i, j])
            q_safe = np.clip(q_vals, 1e-30, None)
            p_safe = np.clip(post_exact, 1e-30, None)
            integrand = q_safe * (np.log(q_safe) - np.log(p_safe))
            kl_surface[i, j] = _grid_integrate(integrand, E_grid)

    kl_surface = np.clip(kl_surface, 0, 8)

    cf = ax.contourf(mu_mesh, sig_mesh, kl_surface, levels=25,
                     cmap="YlOrRd")
    ax.contour(mu_mesh, sig_mesh, kl_surface, levels=12, colors="k",
               linewidths=0.3, alpha=0.3)
    fig.colorbar(cf, ax=ax, shrink=0.85, label="KL divergence")

    min_idx = np.unravel_index(np.nanargmin(kl_surface), kl_surface.shape)
    ax.plot(mu_mesh[min_idx], sig_mesh[min_idx], "*", ms=18,
            color="white", markeredgecolor="k", markeredgewidth=1.5,
            zorder=10, label="Optimum")

    cand_pts = [
        (8_000, 2_000, "Too far left"),
        (10_500, 500, "Too narrow"),
        (10_000, 4_000, "Too wide"),
        (14_000, 1_500, "Shifted right"),
    ]
    for mu_c, sig_c, lbl in cand_pts:
        ax.plot(mu_c, sig_c, "o", ms=8, color="#2C7FB8",
                markeredgecolor="k", markeredgewidth=1, zorder=9)
        ax.annotate(lbl, (mu_c, sig_c), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color="#2C7FB8",
                    fontweight="bold")

    ax.set_xlabel(r"$\mu$", fontsize=12)
    ax.set_ylabel(r"$\sigma$", fontsize=12)
    ax.set_title("KL divergence landscape: which Gaussian is closest?",
                 fontsize=11)
    ax.legend(fontsize=10, loc="upper right")


def plot_elbo_two_terms(bkd, tip_model, y_obs, sigma_noise,
                        mu_prior, sigma_prior, exact_std,
                        ax1, ax2, ax3):
    """vi_objective.qmd -> fig-elbo-two-terms

    Expected log-likelihood, KL from prior, and negative ELBO vs mu.
    """
    from ._style import apply_style

    sigma_z_fixed = exact_std / sigma_prior
    mu_E_scan = np.linspace(6_000, 16_000, 200)
    mu_z_scan = (mu_E_scan - mu_prior) / sigma_prior

    np.random.seed(42)
    n_mc = 5000
    eps_mc = np.random.normal(0, 1, n_mc)

    ell_term = np.empty_like(mu_z_scan)
    kl_prior_term = np.empty_like(mu_z_scan)

    for i, mu_z_i in enumerate(mu_z_scan):
        z_samples = mu_z_i + sigma_z_fixed * eps_mc
        E_samples = mu_prior + sigma_prior * z_samples
        tip_at_samples = bkd.to_numpy(
            tip_model(bkd.asarray(E_samples[np.newaxis, :]))
        )[0, :]
        log_lik_samples = -0.5 * ((y_obs - tip_at_samples) / sigma_noise)**2
        ell_term[i] = np.mean(log_lik_samples)

        kl_prior_term[i] = 0.5 * (sigma_z_fixed**2 + mu_z_i**2 - 1.0
                                   - 2.0 * np.log(sigma_z_fixed))

    neg_elbo_scan = -ell_term + kl_prior_term

    mu_mle_E = mu_E_scan[np.argmax(ell_term)]
    mu_elbo_opt_E = mu_E_scan[np.argmin(neg_elbo_scan)]

    # Expected log-likelihood
    ax1.plot(mu_E_scan, ell_term, color="#C0392B", lw=2.5)
    ax1.axvline(mu_mle_E, color="#C0392B", ls=":", lw=1.5, alpha=0.6)
    ax1.set_xlabel(r"$\mu$", fontsize=12)
    ax1.set_ylabel("Expected log-likelihood", fontsize=11)
    ax1.set_title(
        r"$\mathbb{E}_q[\log \mathcal{L}(\theta)]$ — fit the data",
        fontsize=10)
    apply_style(ax1)

    # KL from prior
    ax2.plot(mu_E_scan, kl_prior_term, color="#2C7FB8", lw=2.5)
    ax2.axvline(mu_prior, color="#2C7FB8", ls=":", lw=1.5, alpha=0.6,
                label=f"Prior mean ({mu_prior:,})")
    ax2.set_xlabel(r"$\mu$", fontsize=12)
    ax2.set_ylabel("KL from prior", fontsize=11)
    ax2.set_title(
        r"$\mathrm{KL}(q \,\|\, p(\theta))$ — stay close to prior",
        fontsize=10)
    ax2.legend(fontsize=9)
    apply_style(ax2)

    # Negative ELBO
    ax3.plot(mu_E_scan, neg_elbo_scan, color="#8E44AD", lw=2.5)
    ax3.axvline(mu_prior, color="#2C7FB8", ls=":", lw=1, alpha=0.4,
                label=f"Prior mean ({mu_prior:,})")
    ax3.axvline(mu_mle_E, color="#C0392B", ls=":", lw=1, alpha=0.4,
                label="MLE")
    ax3.axvline(mu_elbo_opt_E, color="#8E44AD", ls="--", lw=2, alpha=0.8,
                label="ELBO optimum")
    ax3.scatter(mu_elbo_opt_E, neg_elbo_scan[np.argmin(neg_elbo_scan)],
                s=100, color="#8E44AD", edgecolors="k", lw=1.5, zorder=5)
    ax3.set_xlabel(r"$\mu$", fontsize=12)
    ax3.set_ylabel("Negative ELBO", fontsize=11)
    ax3.set_title("$-$ELBO — the objective VI minimizes", fontsize=10)
    ax3.legend(fontsize=9)
    apply_style(ax3)


# ---------------------------------------------------------------------------
# vi_optimization.qmd
# ---------------------------------------------------------------------------

def plot_reparam_diagram(ax1, ax2):
    """vi_optimization.qmd -> fig-reparam-diagram

    Standard sampling vs reparameterized path for gradient flow.
    """
    from matplotlib.patches import FancyBboxPatch

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

    # --- Left: non-differentiable path ---
    ax1.set_title("Standard sampling (not differentiable)", fontsize=11,
                  fontweight="bold", color="#C0392B")

    box1 = FancyBboxPatch((0.5, 3.5), 2.5, 1.5,
                           boxstyle="round,pad=0.15",
                           facecolor="#D6EAF8", edgecolor="#2C7FB8", lw=2)
    ax1.add_patch(box1)
    ax1.text(1.75, 4.25, r"$(\mu, \sigma)$", ha="center", va="center",
             fontsize=14, color="#2C7FB8")

    box2 = FancyBboxPatch((4.5, 3.5), 2.5, 1.5,
                           boxstyle="round,pad=0.15",
                           facecolor="#FADBD8", edgecolor="#C0392B", lw=2)
    ax1.add_patch(box2)
    ax1.text(5.75, 4.25, r"$\theta \sim q$", ha="center", va="center",
             fontsize=14, color="#C0392B")

    box3 = FancyBboxPatch((4.5, 0.8), 2.5, 1.5,
                           boxstyle="round,pad=0.15",
                           facecolor="#F9E79F", edgecolor="#B7950B", lw=2)
    ax1.add_patch(box3)
    ax1.text(5.75, 1.55, "ELBO", ha="center", va="center",
             fontsize=14, color="#B7950B")

    ax1.annotate("", xy=(4.4, 4.25), xytext=(3.1, 4.25),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#C0392B"))
    ax1.text(3.75, 4.7, "sample", ha="center", fontsize=9, color="#C0392B")
    ax1.annotate("", xy=(5.75, 2.4), xytext=(5.75, 3.4),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#B7950B"))
    ax1.text(3.75, 3.1, "X  no gradient", ha="center", fontsize=10,
             color="#C0392B", fontweight="bold")

    # --- Right: reparameterized path ---
    ax2.set_title("Reparameterized (differentiable)", fontsize=11,
                  fontweight="bold", color="#27AE60")

    box4 = FancyBboxPatch((0.3, 3.5), 2.2, 1.5,
                           boxstyle="round,pad=0.15",
                           facecolor="#D6EAF8", edgecolor="#2C7FB8", lw=2)
    ax2.add_patch(box4)
    ax2.text(1.4, 4.25, r"$(\mu, \sigma)$", ha="center", va="center",
             fontsize=14, color="#2C7FB8")

    box5 = FancyBboxPatch((0.3, 0.8), 2.2, 1.5,
                           boxstyle="round,pad=0.15",
                           facecolor="#E8DAEF", edgecolor="#8E44AD", lw=2)
    ax2.add_patch(box5)
    ax2.text(1.4, 1.55, r"$\varepsilon \sim \mathcal{N}(0,1)$",
             ha="center", va="center", fontsize=12, color="#8E44AD")

    box6 = FancyBboxPatch((4.0, 2.2), 2.8, 1.5,
                           boxstyle="round,pad=0.15",
                           facecolor="#D5F5E3", edgecolor="#27AE60", lw=2)
    ax2.add_patch(box6)
    ax2.text(5.4, 2.95, r"$\theta = \mu + \sigma\varepsilon$",
             ha="center", va="center", fontsize=12, color="#27AE60")

    box7 = FancyBboxPatch((4.0, 0.3), 2.8, 1.2,
                           boxstyle="round,pad=0.15",
                           facecolor="#F9E79F", edgecolor="#B7950B", lw=2)
    ax2.add_patch(box7)
    ax2.text(5.4, 0.9, "ELBO", ha="center", va="center",
             fontsize=14, color="#B7950B")

    ax2.annotate("", xy=(3.9, 3.4), xytext=(2.6, 4.0),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#27AE60"))
    ax2.annotate("", xy=(3.9, 2.6), xytext=(2.6, 1.8),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#8E44AD"))
    ax2.annotate("", xy=(5.4, 1.6), xytext=(5.4, 2.1),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#B7950B"))

    ax2.text(8.0, 3.0, "gradient\n   flows", ha="center", fontsize=10,
             color="#27AE60", fontweight="bold")


def plot_reparam_visual(E_grid, post_exact, mu_prior, sigma_prior,
                        exact_mean, exact_std, axes):
    """vi_optimization.qmd -> fig-reparam-visual

    Base samples and their transformations at initial and optimal params.
    """
    exact_z_mean = (exact_mean - mu_prior) / sigma_prior
    exact_z_std = exact_std / sigma_prior

    np.random.seed(7)
    n_base = 15
    eps = np.sort(np.random.normal(0, 1, n_base))

    mu_z_init, sig_z_init = 0.0, 1.0
    mu_z_opt, sig_z_opt = exact_z_mean, exact_z_std

    E_init = mu_prior + sigma_prior * (mu_z_init + sig_z_init * eps)
    E_opt = mu_prior + sigma_prior * (mu_z_opt + sig_z_opt * eps)

    # Top: base samples
    ax0 = axes[0]
    ax0.scatter(eps, np.zeros_like(eps), s=60, c="#8E44AD",
                edgecolors="k", lw=0.8, zorder=5)
    x_ref = np.linspace(-4, 4, 200)
    ax0_t = ax0.twinx()
    ax0_t.plot(x_ref, norm.pdf(x_ref, 0, 1), color="#8E44AD", lw=1.5,
               alpha=0.3)
    ax0_t.set_yticks([])
    ax0_t.set_ylim(-0.05, 0.5)
    ax0.set_xlim(-4, 4)
    ax0.set_yticks([])
    ax0.set_xlabel(r"$\varepsilon$", fontsize=12)
    ax0.set_title(
        r"Base samples $\varepsilon_s \sim \mathcal{N}(0, 1)$"
        " — drawn once, fixed",
        fontsize=11)
    ax0.grid(True, alpha=0.15)

    # Middle: initial transformation (prior)
    ax1 = axes[1]
    ax1.scatter(E_init, np.zeros_like(E_init), s=60, c="#2C7FB8",
                edgecolors="k", lw=0.8, zorder=5)
    ax1_t = ax1.twinx()
    ax1_t.fill_between(E_grid, post_exact, alpha=0.08, color="#E67E22")
    ax1_t.plot(E_grid, post_exact, color="#E67E22", lw=1.5, alpha=0.3)
    ax1_t.set_yticks([])
    ax1_t.set_ylim(-0.05 * max(post_exact), max(post_exact) * 1.5)
    ax1.set_xlim(E_grid[0], E_grid[-1])
    ax1.set_yticks([])
    ax1.set_xlabel(r"$E$", fontsize=12)
    ax1.set_title(
        r"Initial: $E_s = \mu + \sigma \varepsilon_s$"
        f"  ($\\mu={mu_prior:,},\\, \\sigma={sigma_prior:,}$)"
        " — spread over the prior",
        fontsize=10)
    ax1.grid(True, alpha=0.15)

    # Bottom: optimized transformation
    ax2 = axes[2]
    ax2.scatter(E_opt, np.zeros_like(E_opt), s=60, c="#27AE60",
                edgecolors="k", lw=0.8, zorder=5)
    ax2_t = ax2.twinx()
    ax2_t.fill_between(E_grid, post_exact, alpha=0.15, color="#E67E22")
    ax2_t.plot(E_grid, post_exact, color="#E67E22", lw=1.5, alpha=0.4)
    ax2_t.set_yticks([])
    ax2_t.set_ylim(-0.05 * max(post_exact), max(post_exact) * 1.5)
    ax2.set_xlim(E_grid[0], E_grid[-1])
    ax2.set_yticks([])
    ax2.set_xlabel(r"$E$", fontsize=12)
    ax2.set_title(
        f"Optimized: $E_s = {exact_mean:.0f} + {exact_std:.0f}"
        r" \cdot \varepsilon_s$ — clustered on the posterior",
        fontsize=10)
    ax2.grid(True, alpha=0.15)


def plot_elbo_convergence(maxiters, neg_elbos, means_E, stdevs_E,
                          exact_mean, exact_std, ax1, ax2, ax3):
    """vi_optimization.qmd -> fig-elbo-convergence

    Negative ELBO and parameter trajectories across iterations.
    """
    from ._style import apply_style

    ax1.plot(maxiters, neg_elbos, "o-", color="#8E44AD", lw=2, ms=6)
    ax1.set_xlabel("Max iterations", fontsize=11)
    ax1.set_ylabel("Negative ELBO", fontsize=11)
    ax1.set_title("Objective convergence", fontsize=11)
    apply_style(ax1)

    ax2.plot(maxiters, means_E, "o-", color="#E67E22", lw=2, ms=6)
    ax2.axhline(exact_mean, color="gray", ls="--", lw=1.5,
                label=f"Exact mean = {exact_mean:.0f}")
    ax2.set_xlabel("Max iterations", fontsize=11)
    ax2.set_ylabel(r"$\mu_E$", fontsize=11)
    ax2.set_title("Variational mean", fontsize=11)
    ax2.legend(fontsize=9)
    apply_style(ax2)

    ax3.plot(maxiters, stdevs_E, "o-", color="#2C7FB8", lw=2, ms=6)
    ax3.axhline(exact_std, color="gray", ls="--", lw=1.5,
                label=f"Exact std = {exact_std:.0f}")
    ax3.set_xlabel("Max iterations", fontsize=11)
    ax3.set_ylabel(r"$\sigma_E$", fontsize=11)
    ax3.set_title("Variational std dev", fontsize=11)
    ax3.legend(fontsize=9)
    apply_style(ax3)


def plot_base_samples(bkd, tip_model, std_tip_model,
                      noise_variances, y_obs,
                      mu_prior, sigma_prior,
                      E_grid, post_exact,
                      sample_counts, axes, fig):
    """vi_optimization.qmd -> fig-base-samples

    Effect of number of base samples on VI accuracy.
    """
    from pyapprox.probability.conditional.gaussian import ConditionalGaussian
    from pyapprox.probability.conditional.joint import (
        ConditionalIndependentJoint,
    )
    from pyapprox.probability.joint.independent import IndependentJoint
    from pyapprox.probability.univariate.gaussian import GaussianMarginal
    from pyapprox.probability.likelihood import DiagonalGaussianLogLikelihood
    from pyapprox.probability.likelihood.gaussian import (
        MultiExperimentLogLikelihood,
    )
    from pyapprox.inverse.variational.elbo import make_single_problem_elbo
    from pyapprox.inverse.variational.fitter import VariationalFitter
    from pyapprox.optimization.minimize.scipy.trust_constr import (
        ScipyTrustConstrOptimizer,
    )

    for ax, ns in zip(axes, sample_counts):
        mf = _make_degree0_expansion(bkd, 0.0)
        lsf = _make_degree0_expansion(bkd, 0.0)
        cond = ConditionalGaussian(mf, lsf, bkd)
        vd = ConditionalIndependentJoint([cond], bkd)
        vp = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        multi_lik = MultiExperimentLogLikelihood(
            base_lik, bkd.asarray([[y_obs]]), bkd,
        )

        def log_lik_fn(z, _ml=multi_lik):
            return _ml.logpdf(std_tip_model(z))

        np.random.seed(42)
        bs = bkd.asarray(np.random.normal(0, 1, (1, ns)))
        wt = bkd.full((1, ns), 1.0 / ns)
        elbo_i = make_single_problem_elbo(vd, log_lik_fn, vp, bs, wt, bkd)

        opt = ScipyTrustConstrOptimizer(maxiter=50, gtol=1e-8, verbosity=0)
        fitter = VariationalFitter(bkd, optimizer=opt)
        fitter.fit(elbo_i)

        dummy_x = bkd.zeros((cond.nvars(), 1))
        z_mu = float(cond._mean_func(dummy_x)[0, 0])
        z_sig = math.exp(float(cond._log_stdev_func(dummy_x)[0, 0]))
        vi_mu_E = mu_prior + sigma_prior * z_mu
        vi_sig_E = sigma_prior * z_sig
        vi_pdf = norm.pdf(E_grid, vi_mu_E, vi_sig_E)

        ax.plot(E_grid, post_exact, color="#E67E22", lw=2)
        ax.fill_between(E_grid, post_exact, alpha=0.12, color="#E67E22")
        ax.plot(E_grid, vi_pdf, color="#2C7FB8", lw=2, ls="--")
        ax.set_title(f"$S = {ns}$", fontsize=11)
        ax.set_xlabel(r"$E$", fontsize=10)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel("Density", fontsize=11)
    fig.legend(["Exact posterior", "", "VI approximation"], fontsize=9,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))


def plot_vi_2d(bkd, tip_model_2d, noise_lik_2d, prior_2d,
               mu_prior_2d, sigma_prior_2d,
               E1_true, E2_true,
               vi_means_2d, vi_stdevs_2d,
               var_dist_full, elbo_2d, elbo_full,
               ax1, ax2, ax3):
    """vi_optimization.qmd -> fig-vi-2d

    Exact 2D posterior vs diagonal and full-covariance Gaussian VI.
    """
    from pyapprox.inverse.posterior.log_unnormalized import (
        LogUnNormalizedPosterior,
    )
    from matplotlib.patches import Ellipse
    from ._style import apply_style

    # Grid posterior in physical (E1, E2) space
    posterior_2d_fn = LogUnNormalizedPosterior(
        model_fn=tip_model_2d, likelihood=noise_lik_2d,
        prior=prior_2d, bkd=bkd,
    )

    n_g = 80
    E1_g = np.linspace(4_000, 20_000, n_g)
    E2_g = np.linspace(2_000, 18_000, n_g)
    E1_m, E2_m = np.meshgrid(E1_g, E2_g)
    grid_pts = bkd.asarray(np.vstack([E1_m.ravel(), E2_m.ravel()]))
    lp_flat = bkd.to_numpy(posterior_2d_fn(grid_pts))
    lp_grid = lp_flat.reshape(n_g, n_g)
    p_grid = np.exp(lp_grid - np.max(lp_grid))

    # Extract full-covariance params
    dummy_full = bkd.zeros((var_dist_full.nvars(), 1))
    full_z_mean = bkd.to_numpy(
        var_dist_full.reparameterize(dummy_full, bkd.zeros((2, 1)))
    ).flatten()
    full_z_cov = bkd.to_numpy(var_dist_full.covariance(dummy_full)[0])
    full_mean_E = mu_prior_2d + sigma_prior_2d * full_z_mean
    S = np.diag(sigma_prior_2d)
    full_cov_E = S @ full_z_cov @ S

    for ax in [ax1, ax2, ax3]:
        ax.contourf(E1_m, E2_m, p_grid, levels=10, cmap="Oranges",
                     alpha=0.5)
        ax.contour(E1_m, E2_m, p_grid, levels=8, colors="#E67E22",
                    linewidths=0.5, alpha=0.5)
        ax.scatter(E1_true, E2_true,
                   marker="D", s=80, color="#27AE60", edgecolors="k",
                   lw=1, zorder=5, label="True parameters")
        ax.set_xlabel(r"$E_1$ (skin)", fontsize=12)
        ax.set_ylabel(r"$E_2$ (core)", fontsize=12)
        ax.grid(True, alpha=0.15)

    ax1.set_title("Exact posterior (grid)", fontsize=11)
    ax1.legend(fontsize=9)

    # Diagonal VI ellipses
    for n_std in [1, 2]:
        ell = Ellipse(
            xy=(vi_means_2d[0], vi_means_2d[1]),
            width=2 * n_std * vi_stdevs_2d[0],
            height=2 * n_std * vi_stdevs_2d[1],
            angle=0,
            facecolor="none", edgecolor="#2C7FB8",
            lw=2 if n_std == 1 else 1.5,
            ls="-" if n_std == 1 else "--",
        )
        ax2.add_patch(ell)
    ax2.scatter(vi_means_2d[0], vi_means_2d[1], marker="+", s=150,
                color="#2C7FB8", lw=3, zorder=5, label="VI mean")
    ax2.set_title(f"Diagonal ({elbo_2d.nvars()} params)", fontsize=11)
    ax2.legend(fontsize=9)

    # Full-covariance VI ellipses
    eigenvalues, eigenvectors = np.linalg.eigh(full_cov_E)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for n_std in [1, 2]:
        ell = Ellipse(
            xy=(full_mean_E[0], full_mean_E[1]),
            width=2 * n_std * np.sqrt(eigenvalues[0]),
            height=2 * n_std * np.sqrt(eigenvalues[1]),
            angle=angle,
            facecolor="none", edgecolor="#2C7FB8",
            lw=2 if n_std == 1 else 1.5,
            ls="-" if n_std == 1 else "--",
        )
        ax3.add_patch(ell)
    ax3.scatter(full_mean_E[0], full_mean_E[1], marker="+", s=150,
                color="#2C7FB8", lw=3, zorder=5, label="VI mean")
    ax3.set_title(f"Full-covariance ({elbo_full.nvars()} params)",
                  fontsize=11)
    ax3.legend(fontsize=9)


# ---------------------------------------------------------------------------
# vi_amortized.qmd
# ---------------------------------------------------------------------------

def plot_amortization_concept(ax1, ax2):
    """vi_amortized.qmd -> fig-amortization-concept

    Single-problem VI vs amortized VI schematic diagram.
    """
    from matplotlib.patches import FancyBboxPatch

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 3.5)
        ax.axis("off")

    # --- Top: Single-problem VI ---
    ax1.set_title("Single-problem VI: one optimization per dataset",
                  fontsize=11, fontweight="bold", loc="left", pad=10)

    colors = ["#2C7FB8", "#E67E22", "#27AE60"]
    for i, (xpos, label, c) in enumerate([
        (1.5, "$y_1$", colors[0]),
        (6.0, "$y_2$", colors[1]),
        (10.5, "$y_3$", colors[2]),
    ]):
        box = FancyBboxPatch((xpos - 0.8, 2.2), 1.6, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor="#F0F0F0", edgecolor=c, lw=2)
        ax1.add_patch(box)
        ax1.text(xpos, 2.6, label, ha="center", va="center",
                 fontsize=13, color=c, fontweight="bold")

        ax1.annotate("", xy=(xpos, 1.7), xytext=(xpos, 2.1),
                     arrowprops=dict(arrowstyle="-|>", lw=1.5, color=c))
        ax1.text(xpos, 1.9, "optimize", ha="center", fontsize=8,
                 color="gray")

        box2 = FancyBboxPatch((xpos - 1.0, 0.5), 2.0, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor=c, edgecolor=c, lw=1, alpha=0.15)
        ax1.add_patch(box2)
        ax1.text(xpos, 1.0,
                 f"$q_{i+1}(\\theta;\\,\\mu_{i+1},\\sigma_{i+1})$",
                 ha="center", va="center", fontsize=11, color=c)

    # --- Bottom: Amortized VI ---
    ax2.set_title("Amortized VI: one optimization for all datasets",
                  fontsize=11, fontweight="bold", loc="left", pad=10)

    for i, (xpos, label, c) in enumerate([
        (1.5, "$y_1$", colors[0]),
        (6.0, "$y_2$", colors[1]),
        (10.5, "$y_3$", colors[2]),
    ]):
        box = FancyBboxPatch((xpos - 0.8, 2.2), 1.6, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor="#F0F0F0", edgecolor=c, lw=2)
        ax2.add_patch(box)
        ax2.text(xpos, 2.6, label, ha="center", va="center",
                 fontsize=13, color=c, fontweight="bold")
        ax2.annotate("", xy=(6.0, 1.7), xytext=(xpos, 2.1),
                     arrowprops=dict(arrowstyle="-|>", lw=1.2, color=c,
                                     alpha=0.6))

    center_box = FancyBboxPatch((3.5, 0.5), 5.0, 1.0,
                                  boxstyle="round,pad=0.15",
                                  facecolor="#D6EAF8",
                                  edgecolor="#2C7FB8", lw=2)
    ax2.add_patch(center_box)
    ax2.text(6.0, 1.0,
             r"$q(\theta;\, \mu(x),\, \sigma(x))$",
             ha="center", va="center", fontsize=13, color="#2C7FB8")
    ax2.text(6.0, 1.85, "single optimization", ha="center",
             fontsize=9, color="gray", fontstyle="italic")


def plot_training_recovery(bkd, mean_func, log_stdev_func,
                           all_obs, labels_np, theta_true,
                           conjugate, K, axes):
    """vi_amortized.qmd -> fig-training-recovery

    Amortized VI posteriors at training groups vs exact conjugate.
    """
    from ._style import apply_style

    theta_grid = np.linspace(-4, 5, 300)

    for k, ax in enumerate(axes):
        # Exact
        conjugate.compute(all_obs[k])
        exact_mean = float(conjugate.posterior_mean()[0, 0])
        exact_std = math.sqrt(float(conjugate.posterior_covariance()[0, 0]))
        exact_pdf = norm.pdf(theta_grid, exact_mean, exact_std)

        # VI
        label_k = bkd.asarray([[labels_np[k]]])
        vi_mu = float(mean_func(label_k)[0, 0])
        vi_std = math.exp(float(log_stdev_func(label_k)[0, 0]))
        vi_pdf = norm.pdf(theta_grid, vi_mu, vi_std)

        ax.plot(theta_grid, exact_pdf, color="#E67E22", lw=2.5,
                label="Exact conjugate")
        ax.fill_between(theta_grid, exact_pdf, alpha=0.12, color="#E67E22")
        ax.plot(theta_grid, vi_pdf, color="#2C7FB8", lw=2.5, ls="--",
                label="Amortized VI")
        ax.axvline(theta_true[k], color="#27AE60", ls="--", lw=1.5,
                   alpha=0.7)
        ax.set_xlabel(r"$\theta$", fontsize=12)
        ax.set_title(f"Group {k} (label = {labels_np[k]:.2f})",
                     fontsize=11)
        apply_style(ax)

    axes[0].set_ylabel("Density", fontsize=12)
    axes[-1].legend(fontsize=9, loc="upper right")


def plot_generalization(bkd, mean_func, log_stdev_func,
                        labels_np, K,
                        test_obs_list, test_labels_list, theta_test, K_test,
                        conjugate, theta_grid, ax1, ax2):
    """vi_amortized.qmd -> fig-generalization

    Learned parameter functions and test-point posterior comparison.
    """
    from ._style import apply_style

    x_eval = np.linspace(-1, 1, 100)
    mu_curve = []
    sig_curve = []
    for x in x_eval:
        xv = bkd.asarray([[x]])
        mu_curve.append(float(mean_func(xv)[0, 0]))
        sig_curve.append(math.exp(float(log_stdev_func(xv)[0, 0])))

    # Left: parameter functions
    ax1_twin = ax1.twinx()
    ln1, = ax1.plot(x_eval, mu_curve, color="#E67E22", lw=2.5,
                    label=r"$\mu(x)$")
    ln2, = ax1_twin.plot(x_eval, sig_curve, color="#8E44AD", lw=2.5,
                          ls="--", label=r"$\sigma(x)$")

    # Training labels
    for k in range(K):
        label_k = bkd.asarray([[labels_np[k]]])
        mu_k = float(mean_func(label_k)[0, 0])
        sig_k = math.exp(float(log_stdev_func(label_k)[0, 0]))
        ax1.plot(labels_np[k], mu_k, "o", ms=10, color="#E67E22",
                 markeredgecolor="k", lw=1.5, zorder=5)
        ax1_twin.plot(labels_np[k], sig_k, "o", ms=10, color="#8E44AD",
                      markeredgecolor="k", lw=1.5, zorder=5)

    # Test labels
    for t in range(K_test):
        label_test_val = test_labels_list[t]
        if abs(label_test_val) <= 1.0:
            xv = bkd.asarray([[label_test_val]])
            mu_t = float(mean_func(xv)[0, 0])
            sig_t = math.exp(float(log_stdev_func(xv)[0, 0]))
            ax1.plot(label_test_val, mu_t, "*", ms=14, color="#E67E22",
                     markeredgecolor="k", lw=1, zorder=5)
            ax1_twin.plot(label_test_val, sig_t, "*", ms=14,
                          color="#8E44AD",
                          markeredgecolor="k", lw=1, zorder=5)

    ax1.set_xlabel("Label $x$", fontsize=12)
    ax1.set_ylabel(r"$\mu(x)$", fontsize=12, color="#E67E22")
    ax1_twin.set_ylabel(r"$\sigma(x)$", fontsize=12, color="#8E44AD")
    ax1.set_title("Learned parameter functions", fontsize=11)
    lns = [ln1, ln2]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.15)
    ax1.text(0.98, 0.05, "● train   ★ test", transform=ax1.transAxes,
             ha="right", fontsize=9, color="gray")

    # Right: example test posterior
    conjugate.compute(test_obs_list[0])
    ex_mean = float(conjugate.posterior_mean()[0, 0])
    ex_std = math.sqrt(float(conjugate.posterior_covariance()[0, 0]))

    label_ex = test_labels_list[0]
    xv_ex = bkd.asarray([[label_ex]])
    vi_mu_ex = float(mean_func(xv_ex)[0, 0])
    vi_std_ex = math.exp(float(log_stdev_func(xv_ex)[0, 0]))

    exact_pdf = norm.pdf(theta_grid, ex_mean, ex_std)
    vi_pdf = norm.pdf(theta_grid, vi_mu_ex, vi_std_ex)

    ax2.plot(theta_grid, exact_pdf, color="#E67E22", lw=2.5,
             label="Exact conjugate")
    ax2.fill_between(theta_grid, exact_pdf, alpha=0.12, color="#E67E22")
    ax2.plot(theta_grid, vi_pdf, color="#2C7FB8", lw=2.5, ls="--",
             label="Amortized VI")
    ax2.axvline(theta_test[0], color="#27AE60", ls="--", lw=1.5, alpha=0.7,
                label=rf"True $\theta$ = {theta_test[0]}")
    ax2.set_xlabel(r"$\theta$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(f"Test group (label = {label_ex:.2f})", fontsize=11)
    ax2.legend(fontsize=9)
    apply_style(ax2)
