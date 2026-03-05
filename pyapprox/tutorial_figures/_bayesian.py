"""Plotting functions for Bayesian inference tutorials.

Covers: bayesian_inference_intro.qmd, mcmc_sampling.qmd, dram_mcmc.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# bayesian_inference_intro.qmd — all echo:false -> Convention A
# ---------------------------------------------------------------------------

def plot_forward_inverse_schematic(fig, axes):
    """bayesian_inference_intro.qmd -> fig-forward-inverse-schematic

    Forward UQ (top, faded) vs Bayesian inference (bottom, vivid) schematic.
    """
    from scipy.stats import norm
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch

    # --- Shared data for the schematic ---
    E_grid_s = np.linspace(6_000, 18_000, 300)
    mu_s, sig_s = 12_000, 2_000
    prior_s = norm.pdf(E_grid_s, mu_s, sig_s)

    q_grid = np.linspace(-1, 5, 300)
    pdf_out = 0.7 * norm.pdf(q_grid, 1.5, 0.8) + 0.3 * norm.pdf(q_grid, 3.0, 0.5)

    mu_post, sig_post = 10_200, 800
    posterior_s = norm.pdf(E_grid_s, mu_post, sig_post)

    # =====================================================================
    # TOP ROW: Forward UQ (faded)
    # =====================================================================
    fade = 0.35

    ax = axes[0, 0]
    ax.fill_between(E_grid_s, prior_s, alpha=0.15, color="#2C7FB8")
    ax.plot(E_grid_s, prior_s, color="#2C7FB8", lw=2, alpha=fade)
    ax.set_xlabel(r"$E$", fontsize=11, alpha=fade)
    ax.set_ylabel("Density", fontsize=11, alpha=fade)
    ax.set_title(r"Prior $p(E)$", fontsize=11, alpha=fade)
    ax.set_yticks([])
    ax.grid(True, alpha=0.1)
    ax.tick_params(colors=(0, 0, 0, fade))
    for spine in ax.spines.values():
        spine.set_alpha(fade)

    ax = axes[0, 1]
    ax.axis("off")
    box = FancyBboxPatch(
        (0.15, 0.35), 0.7, 0.3, boxstyle="round,pad=0.05",
        facecolor="#F0F0F0", edgecolor="gray", linewidth=1.5, alpha=fade,
    )
    ax.add_patch(box)
    ax.text(0.5, 0.5, r"$f(E)$", ha="center", va="center",
            fontsize=14, color="gray", alpha=fade, transform=ax.transAxes)
    ax.annotate("", xy=(0.12, 0.5), xytext=(-0.15, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="#2C7FB8",
                                alpha=fade))
    ax.annotate("", xy=(1.15, 0.5), xytext=(0.88, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="#2C7FB8",
                                alpha=fade))

    ax = axes[0, 2]
    ax.fill_between(q_grid, pdf_out, alpha=0.1, color="#E67E22")
    ax.plot(q_grid, pdf_out, color="#E67E22", lw=2, alpha=fade)
    ax.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=11, alpha=fade)
    ax.set_ylabel("Density", fontsize=11, alpha=fade)
    ax.set_title("Push-forward distribution", fontsize=11, alpha=fade)
    ax.set_yticks([])
    ax.grid(True, alpha=0.1)
    ax.tick_params(colors=(0, 0, 0, fade))
    for spine in ax.spines.values():
        spine.set_alpha(fade)

    # =====================================================================
    # BOTTOM ROW: Bayesian inference (vivid)
    # =====================================================================

    ax = axes[1, 0]
    ax.fill_between(E_grid_s, prior_s, alpha=0.12, color="#2C7FB8")
    ax.plot(E_grid_s, prior_s, color="#2C7FB8", lw=1.5, alpha=0.3,
            label="Prior (faded)")
    ax.fill_between(E_grid_s, posterior_s, alpha=0.3, color="#E67E22")
    ax.plot(E_grid_s, posterior_s, color="#E67E22", lw=2.5,
            label=r"Posterior $p(E \mid y_{\mathrm{obs}})$")
    ax.set_xlabel(r"$E$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Updated belief", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    ax.axis("off")
    box = FancyBboxPatch(
        (0.15, 0.35), 0.7, 0.3, boxstyle="round,pad=0.05",
        facecolor="#F0F0F0", edgecolor="k", linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(0.5, 0.5, r"$f(E)$", ha="center", va="center",
            fontsize=14, transform=ax.transAxes)
    ax.annotate("", xy=(-0.15, 0.5), xytext=(0.12, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="#E67E22"))
    ax.annotate("", xy=(0.88, 0.5), xytext=(1.15, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="#C0392B"))
    ax.text(0.5, 0.15, "Bayes' theorem", ha="center", va="center",
            fontsize=10, fontstyle="italic", color="#333",
            transform=ax.transAxes)

    ax = axes[1, 2]
    y_obs_s = 2.0
    noise_std_s = 0.25
    obs_bell = norm.pdf(q_grid, y_obs_s, noise_std_s)
    obs_bell_scaled = obs_bell / obs_bell.max() * 0.8
    ax.fill_between(q_grid, obs_bell_scaled, alpha=0.2, color="#C0392B")
    ax.plot(q_grid, obs_bell_scaled, color="#C0392B", lw=1.5, ls="--",
            label="Noise model")
    ax.plot(y_obs_s, 0, "o", color="#C0392B", ms=10, zorder=5,
            label=r"$y_{\mathrm{obs}}$", clip_on=False)
    ax.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Observed data", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_yticks([])
    ax.set_xlim(q_grid[0], q_grid[-1])
    ax.grid(True, alpha=0.2)

    # Connecting element: dashed line from top-right to bottom-right
    con = ConnectionPatch(
        xyA=(y_obs_s, 0), coordsA=axes[0, 2].transData,
        xyB=(y_obs_s, obs_bell_scaled.max()), coordsB=axes[1, 2].transData,
        color="gray", ls="--", lw=1.5, alpha=0.6,
    )
    fig.add_artist(con)
    mid_fig_y = 0.5 * (
        axes[0, 2].get_position().y0 + axes[1, 2].get_position().y1
    )
    fig.text(0.92, mid_fig_y, "observe", ha="center", va="center",
             fontsize=9, color="gray", fontstyle="italic", rotation=90)


def plot_point_estimate_problem(ax, model_likelihood, prior, E_true,
                                E_point_estimate, q0, L, H, bkd):
    """bayesian_inference_intro.qmd -> fig-point-estimate-problem

    Point estimate vs prior, with repeated noisy inversions.
    """
    I_rect = H**3 / 12.0
    n_repeat = 20
    y_repeats = model_likelihood.rvs(
        np.full((1, n_repeat), E_true),
    )[0, :]
    E_repeats = (11 * q0 * L**4) / (120 * y_repeats * I_rect)

    E_grid = np.linspace(6_000, 18_000, 500)
    prior_pdf = prior.pdf(E_grid[np.newaxis, :])[0, :]

    ax.plot(E_grid, prior_pdf, color="#2C7FB8", lw=2, label="Prior")
    ax.axvline(E_true, color="#27AE60", ls="-", lw=2,
               label=f"True $E$ = {E_true}")
    ax.axvline(E_point_estimate, color="#C0392B", ls="--", lw=2,
               label=f"Point estimate = {E_point_estimate:.0f}")
    for i, E_r in enumerate(E_repeats):
        ax.axvline(E_r, color="#8E44AD", ls="-", lw=1, alpha=0.4,
                   label="Repeat estimates" if i == 0 else None)

    ax.set_xlabel(r"$E$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Point estimates from repeated experiments", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.1)


def plot_likelihood(ax, E_grid, prior_pdf, lik_grid_scaled, E_true):
    """bayesian_inference_intro.qmd -> fig-likelihood

    Prior vs scaled likelihood, showing disagreement.
    """
    ax.plot(E_grid, prior_pdf, color="#2C7FB8", lw=2, label="Prior $p(E)$")
    ax.plot(E_grid, lik_grid_scaled, color="#C0392B", lw=2,
            label="Likelihood $\\mathcal{L}(E)$ (scaled)")
    ax.axvline(E_true, color="#27AE60", ls="--", lw=1.5, alpha=0.7,
               label=f"True $E$ = {E_true}")
    ax.set_xlabel(r"$E$", fontsize=12)
    ax.set_ylabel("Density / scaled likelihood", fontsize=12)
    ax.set_title("Prior vs. likelihood", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)


def plot_bayes_update(ax, E_grid, prior_vals, likelihood_vals,
                      posterior_vals, E_true):
    """bayesian_inference_intro.qmd -> fig-bayes-update

    Prior, scaled likelihood, and posterior overlaid.
    """
    ax.plot(E_grid, prior_vals, color="#2C7FB8", lw=2, label="Prior $p(E)$")
    ax.fill_between(E_grid, prior_vals, alpha=0.1, color="#2C7FB8")

    lik_scaled = (
        likelihood_vals / likelihood_vals.max() * posterior_vals.max() * 0.9
    )
    ax.plot(E_grid, lik_scaled, color="#C0392B", lw=1.5, ls="--",
            label="Likelihood (scaled)")

    ax.plot(E_grid, posterior_vals, color="#E67E22", lw=2.5,
            label="Posterior $p(E \\mid y_{\\mathrm{obs}})$")
    ax.fill_between(E_grid, posterior_vals, alpha=0.2, color="#E67E22")

    ax.axvline(E_true, color="#27AE60", ls="--", lw=1.5, alpha=0.8,
               label=f"True $E$ = {E_true}")

    ax.set_xlabel(r"$E$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Bayesian update: prior -> posterior", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)


def plot_prior_likelihood_balance(ax1, ax2, E_grid, likelihood_vals,
                                  mu_prior, E_true):
    """bayesian_inference_intro.qmd -> fig-prior-likelihood-balance

    Vague vs tight prior: effect on posterior.
    """
    from scipy.stats import norm

    for ax, sig_p, title in [
        (ax1, 5_000, "Vague prior ($\\sigma_{\\text{prior}} = 5{,}000$)"),
        (ax2, 500, "Tight prior ($\\sigma_{\\text{prior}} = 500$)"),
    ]:
        prior_v = norm.pdf(E_grid, mu_prior, sig_p)
        post_unnorm = likelihood_vals * prior_v
        post_v = post_unnorm / np.trapz(post_unnorm, E_grid)

        ax.plot(E_grid, prior_v / prior_v.max() * post_v.max(),
                color="#2C7FB8", lw=1.5, ls=":", label="Prior (scaled)")
        lik_sc = (
            likelihood_vals / likelihood_vals.max() * post_v.max() * 0.9
        )
        ax.plot(E_grid, lik_sc, color="#C0392B", lw=1.5, ls="--",
                label="Likelihood (scaled)")
        ax.plot(E_grid, post_v, color="#E67E22", lw=2.5, label="Posterior")
        ax.fill_between(E_grid, post_v, alpha=0.15, color="#E67E22")
        ax.axvline(E_true, color="#27AE60", ls="--", lw=1.5, alpha=0.7)
        ax.set_xlabel(r"$E$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)


def plot_multiple_observations(ax, E_grid, prior_vals, post_1, post_2,
                               post_5, E_true):
    """bayesian_inference_intro.qmd -> fig-multiple-observations

    Sequential Bayesian update with 1, 2, and 5 observations.
    """
    ax.plot(E_grid, prior_vals, color="#2C7FB8", lw=2, label="Prior")
    ax.fill_between(E_grid, prior_vals, alpha=0.08, color="#2C7FB8")
    ax.plot(E_grid, post_1, color="#F5B041", lw=2, label="Posterior (1 obs.)")
    ax.fill_between(E_grid, post_1, alpha=0.15, color="#F5B041")
    ax.plot(E_grid, post_2, color="#E67E22", lw=2, label="Posterior (2 obs.)")
    ax.fill_between(E_grid, post_2, alpha=0.15, color="#E67E22")
    ax.plot(E_grid, post_5, color="#C0392B", lw=2.5,
            label="Posterior (5 obs.)")
    ax.fill_between(E_grid, post_5, alpha=0.15, color="#C0392B")
    ax.axvline(E_true, color="#27AE60", ls="--", lw=1.5, alpha=0.8,
               label=f"True $E$ = {E_true}")

    ax.set_xlabel(r"$E$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Sequential Bayesian update with increasing data",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_posterior_pushforward(ax1, ax2, beam_model, prior, post_5, E_grid,
                               E_true, delta_true, rng):
    """bayesian_inference_intro.qmd -> fig-posterior-pushforward

    Prior vs posterior push-forward distributions.
    """
    nsamples = 10_000

    np.random.seed(0)
    E_prior_samples = prior.rvs(nsamples)[0, :]
    E_prior_samples = E_prior_samples[E_prior_samples > 0]
    delta_prior = beam_model(E_prior_samples[np.newaxis, :])[0, :]

    post_cdf = np.cumsum(post_5) * (E_grid[1] - E_grid[0])
    post_cdf = post_cdf / post_cdf[-1]
    u_samples = rng.uniform(size=nsamples)
    E_post_samples = np.interp(u_samples, post_cdf, E_grid)
    delta_post = beam_model(E_post_samples[np.newaxis, :])[0, :]

    ax1.hist(E_prior_samples, bins=60, density=True, color="#2C7FB8",
             alpha=0.4, edgecolor="none", label="Prior")
    ax1.hist(E_post_samples, bins=60, density=True, color="#E67E22",
             alpha=0.5, edgecolor="none", label="Posterior (5 obs.)")
    ax1.axvline(E_true, color="#27AE60", ls="--", lw=1.5)
    ax1.set_xlabel(r"$E$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Parameter distributions", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    ax2.hist(delta_prior, bins=60, density=True, color="#2C7FB8", alpha=0.4,
             edgecolor="none", label="Prior push-forward")
    ax2.hist(delta_post, bins=60, density=True, color="#E67E22", alpha=0.5,
             edgecolor="none", label="Posterior push-forward")
    ax2.axvline(delta_true, color="#27AE60", ls="--", lw=1.5,
                label=f"True $\\delta_{{\\text{{tip}}}}$")
    ax2.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Push-forward distributions", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# mcmc_sampling.qmd — all echo:false -> Convention A
# ---------------------------------------------------------------------------

def plot_random_walk_concept(fig, ax):
    """mcmc_sampling.qmd -> fig-random-walk-concept

    Random walker exploring a 2D posterior distribution.
    """
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(42)

    mu_post = np.array([0.5, -0.3])
    cov_post = np.array([[0.15, 0.06],
                          [0.06, 0.10]])
    rv_post = multivariate_normal(mu_post, cov_post)

    n_vis = 150
    chain_vis = np.empty((n_vis, 2))
    chain_vis[0] = mu_post + rng.normal(0, 0.2, size=2)
    sigma_prop_vis = 0.3

    for t in range(1, n_vis):
        proposal = chain_vis[t - 1] + rng.normal(0, sigma_prop_vis, size=2)
        log_alpha = rv_post.logpdf(proposal) - rv_post.logpdf(chain_vis[t - 1])
        if np.log(rng.uniform()) < log_alpha:
            chain_vis[t] = proposal
        else:
            chain_vis[t] = chain_vis[t - 1]

    g1 = np.linspace(-1.0, 2.0, 200)
    g2 = np.linspace(-1.5, 1.0, 200)
    G1, G2 = np.meshgrid(g1, g2)
    pos = np.dstack((G1, G2))
    Z = rv_post.pdf(pos)

    ax.contourf(G1, G2, Z, levels=12, cmap="Blues", alpha=0.6)
    ax.contour(G1, G2, Z, levels=8, colors="steelblue",
               linewidths=0.5, alpha=0.5)

    colors = np.arange(n_vis)
    ax.plot(chain_vis[:, 0], chain_vis[:, 1], "-", color="gray", lw=0.5,
            alpha=0.4, zorder=2)
    sc = ax.scatter(chain_vis[:, 0], chain_vis[:, 1], c=colors, cmap="YlOrRd",
                    s=15, edgecolors="k", lw=0.3, zorder=3)
    ax.scatter(*chain_vis[0], marker="*", s=200, color="#27AE60",
               edgecolors="k", lw=1, zorder=4, label="Start")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Step number")

    ax.set_xlabel(r"$\xi_1$", fontsize=12)
    ax.set_ylabel(r"$\xi_2$", fontsize=12)
    ax.set_title("Random walker exploring a 2D posterior", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.15)


def plot_mh_single_step(axes, posterior_1d, tip_model_1d, noise_likelihood_1d,
                        y_obs_1d, sigma_noise_1d, bkd):
    """mcmc_sampling.qmd -> fig-mh-single-step

    Three MH accept/reject scenarios on the 1D beam posterior.
    """
    from scipy.stats import norm

    xi_grid = np.linspace(-4.0, 4.0, 500)
    prior_vals = norm.pdf(xi_grid, 0, 1)
    tip_preds = bkd.to_numpy(
        tip_model_1d(bkd.asarray(xi_grid[np.newaxis, :]))
    )[0, :]
    lik_vals = norm.pdf(y_obs_1d, tip_preds, sigma_noise_1d)
    post_vals = lik_vals * prior_vals
    post_vals = post_vals / np.trapz(post_vals, xi_grid)

    scenarios = [
        {"current": 1.0, "proposed": 0.4,
         "title": "Proposal into higher density -> accept",
         "accept_text": r"$\alpha = 1$: always accept"},
        {"current": 0.4, "proposed": 1.5,
         "title": "Proposal into lower density -> sometimes accept",
         "accept_text": None},
        {"current": 0.4, "proposed": 3.0,
         "title": "Proposal into far tail -> almost always reject",
         "accept_text": None},
    ]

    for ax, sc in zip(axes, scenarios):
        xi_curr = sc["current"]
        xi_prop = sc["proposed"]

        p_curr = np.interp(xi_curr, xi_grid, post_vals)
        p_prop = np.interp(xi_prop, xi_grid, post_vals)
        alpha = min(1.0, p_prop / p_curr) if p_curr > 0 else 1.0

        ax.fill_between(xi_grid, post_vals, alpha=0.15, color="#E67E22")
        ax.plot(xi_grid, post_vals, color="#E67E22", lw=2)

        sigma_prop_demo = 1.0
        prop_dist = norm.pdf(xi_grid, xi_curr, sigma_prop_demo)
        prop_dist_scaled = (
            prop_dist / prop_dist.max() * post_vals.max() * 0.3
        )
        ax.plot(xi_grid, prop_dist_scaled, color="gray", ls="--", lw=1,
                label="Proposal distribution")

        ax.plot(xi_curr, p_curr, "o", ms=12, color="#2C7FB8", zorder=5,
                markeredgecolor="k", markeredgewidth=1.5)
        ax.annotate("current", (xi_curr, p_curr),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=9, color="#2C7FB8",
                    fontweight="bold")

        ax.plot(xi_prop, p_prop, "s", ms=12, zorder=5,
                markeredgecolor="k", markeredgewidth=1.5,
                color="#27AE60" if alpha > 0.5 else "#C0392B")
        ax.annotate("proposed", (xi_prop, p_prop),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=9,
                    color="#27AE60" if alpha > 0.5 else "#C0392B",
                    fontweight="bold")

        ax.annotate("", xy=(xi_prop, p_prop * 0.5 + p_curr * 0.5),
                    xytext=(xi_curr, p_prop * 0.5 + p_curr * 0.5),
                    arrowprops=dict(
                        arrowstyle="->", lw=1.5,
                        color="#27AE60" if alpha > 0.5 else "#C0392B",
                    ))

        text = sc["accept_text"] or rf"$\alpha = {alpha:.3f}$"
        outcome = "accept" if alpha > 0.5 else "likely reject"
        ax.text(0.98, 0.88, f"{text}\n{outcome}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#d5f5d5" if alpha > 0.5 else "#f5d5d5",
                          ec="gray", lw=0.5))

        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(sc["title"], fontsize=10)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel(r"$\xi$", fontsize=12)


def plot_trace_and_histogram(ax1, ax2, chain, xi_grid, post_vals,
                             xi_true, n_steps, burnin=200):
    """mcmc_sampling.qmd -> fig-trace-and-histogram

    Trace plot and histogram of 1D MCMC chain vs exact posterior.
    """
    chain_post = chain[burnin:]

    ax1.plot(chain, lw=0.5, color="#2C7FB8", alpha=0.8)
    ax1.axhspan(-5, 5, xmin=0, xmax=burnin / n_steps,
                color="gray", alpha=0.15)
    ax1.axvline(burnin, color="gray", ls="--", lw=1, label="End of burn-in")
    ax1.axhline(xi_true, color="#27AE60", ls="--", lw=1.5, alpha=0.7,
                label=rf"True $\xi$ = {xi_true}")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel(r"$\xi$", fontsize=12)
    ax1.set_title("Trace plot", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.15)

    ax2.hist(chain_post, bins=50, density=True, color="#2C7FB8", alpha=0.6,
             edgecolor="k", lw=0.3, label="MCMC samples")
    ax2.plot(xi_grid, post_vals, color="#E67E22", lw=2.5,
             label="Exact posterior (grid)")
    ax2.axvline(xi_true, color="#27AE60", ls="--", lw=1.5, alpha=0.7)
    ax2.set_xlabel(r"$\xi$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Posterior: MCMC vs. exact", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.15)


def plot_2d_chain(ax, chain_2d, posterior_2d, xi_true_2d, bkd,
                  burnin_2d=500):
    """mcmc_sampling.qmd -> fig-2d-chain

    MCMC trajectory on 2D beam posterior with contour background.
    """
    chain_2d_post = chain_2d[burnin_2d:]

    n_grid_2d = 100
    xi1_g = np.linspace(-3.0, 3.0, n_grid_2d)
    xi2_g = np.linspace(-3.0, 3.0, n_grid_2d)
    xi1_m, xi2_m = np.meshgrid(xi1_g, xi2_g)
    grid_samples = bkd.asarray(np.vstack([xi1_m.ravel(), xi2_m.ravel()]))
    log_post_flat = bkd.to_numpy(posterior_2d(grid_samples))
    log_post_grid = log_post_flat.reshape(n_grid_2d, n_grid_2d)
    post_grid = np.exp(log_post_grid - np.nanmax(log_post_grid))

    ax.contourf(xi1_m, xi2_m, post_grid, levels=10, cmap="Blues", alpha=0.5)
    ax.contour(xi1_m, xi2_m, post_grid, levels=8, colors="steelblue",
               linewidths=0.5, alpha=0.4)

    ax.plot(chain_2d[:burnin_2d, 0], chain_2d[:burnin_2d, 1], "-",
            color="gray", lw=0.3, alpha=0.5)
    ax.scatter(chain_2d[:burnin_2d, 0], chain_2d[:burnin_2d, 1],
               c="gray", s=3, alpha=0.3, zorder=2)

    step_colors = np.arange(len(chain_2d_post))
    ax.plot(chain_2d_post[:200, 0], chain_2d_post[:200, 1], "-",
            color="gray", lw=0.2, alpha=0.3)
    ax.scatter(chain_2d_post[:, 0], chain_2d_post[:, 1],
               c=step_colors, cmap="YlOrRd", s=5, alpha=0.5,
               edgecolors="none", zorder=3)

    ax.scatter(*chain_2d[0], marker="*", s=200, color="#27AE60",
               edgecolors="k", lw=1, zorder=5, label="Start (prior mean)")
    ax.scatter(float(xi_true_2d[0, 0]), float(xi_true_2d[1, 0]),
               marker="D", s=80, color="#C0392B",
               edgecolors="k", lw=1, zorder=5, label="True parameters")

    ax.set_xlabel(r"$\xi_1$", fontsize=12)
    ax.set_ylabel(r"$\xi_2$", fontsize=12)
    ax.set_title("2D MCMC trajectory", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15)


def plot_2d_marginals(ax1, ax2, chain_2d_post, posterior_2d, xi_true_2d,
                      bkd):
    """mcmc_sampling.qmd -> fig-2d-marginals

    Marginal posteriors from the 2D chain vs exact grid marginals.
    """
    n_grid_2d = 100
    xi1_g = np.linspace(-3.0, 3.0, n_grid_2d)
    xi2_g = np.linspace(-3.0, 3.0, n_grid_2d)
    xi1_m, xi2_m = np.meshgrid(xi1_g, xi2_g)
    grid_samples = bkd.asarray(np.vstack([xi1_m.ravel(), xi2_m.ravel()]))
    log_post_flat = bkd.to_numpy(posterior_2d(grid_samples))
    log_post_grid = log_post_flat.reshape(n_grid_2d, n_grid_2d)
    post_grid = np.exp(log_post_grid - np.nanmax(log_post_grid))

    post_grid_normed = post_grid / np.trapz(
        np.trapz(post_grid, xi2_g, axis=0), xi1_g,
    )
    marginal_xi1_exact = np.trapz(post_grid_normed, xi2_g, axis=0)
    marginal_xi2_exact = np.trapz(post_grid_normed, xi1_g, axis=1)

    ax1.hist(chain_2d_post[:, 0], bins=50, density=True, color="#2C7FB8",
             alpha=0.6, edgecolor="k", lw=0.3, label="MCMC")
    ax1.plot(xi1_g, marginal_xi1_exact, color="#E67E22", lw=2.5,
             label="Exact (grid)")
    ax1.axvline(float(xi_true_2d[0, 0]), color="#27AE60", ls="--", lw=1.5)
    ax1.axvline(0.0, color="gray", ls=":", lw=1, label="Prior mean")
    ax1.set_xlabel(r"$\xi_1$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(r"Marginal posterior: $\xi_1$", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.15)

    ax2.hist(chain_2d_post[:, 1], bins=50, density=True, color="#2C7FB8",
             alpha=0.6, edgecolor="k", lw=0.3, label="MCMC")
    ax2.plot(xi2_g, marginal_xi2_exact, color="#E67E22", lw=2.5,
             label="Exact (grid)")
    ax2.axvline(float(xi_true_2d[1, 0]), color="#27AE60", ls="--", lw=1.5)
    ax2.axvline(0.0, color="gray", ls=":", lw=1, label="Prior mean")
    ax2.set_xlabel(r"$\xi_2$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(r"Marginal posterior: $\xi_2$", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.15)


def plot_posterior_predictive(ax, chain_2d_post, tip_model_2d,
                              y_obs_2d, delta_true_2d, bkd):
    """mcmc_sampling.qmd -> fig-posterior-predictive

    Posterior predictive distribution vs observed and true values.
    """
    chain_2d_post_arr = bkd.asarray(chain_2d_post.T)
    delta_predictive = bkd.to_numpy(tip_model_2d(chain_2d_post_arr))[0, :]

    ax.hist(delta_predictive, bins=60, density=True, color="#2C7FB8",
            alpha=0.6, edgecolor="k", lw=0.3, label="Posterior predictions")
    ax.axvline(y_obs_2d, color="#C0392B", ls="--", lw=2,
               label=f"Observed $y_{{\\text{{obs}}}}$")
    ax.axvline(delta_true_2d, color="#27AE60", ls="--", lw=1.5,
               label=f"True $\\delta_{{\\text{{tip}}}}$")
    ax.set_xlabel(r"$\delta_{\mathrm{tip}}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Posterior predictive distribution", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# dram_mcmc.qmd — all echo:false -> Convention A
# ---------------------------------------------------------------------------

def plot_proposal_width(axes, run_mh, xi1_true, xi2_true):
    """dram_mcmc.qmd -> fig-proposal-width

    Effect of proposal width on chain behavior (narrow, tuned, wide).
    """
    n_steps = 2000
    configs = [
        {"sigma": np.array([0.05, 0.05]), "label": "Too narrow",
         "color": "#2C7FB8"},
        {"sigma": np.array([0.5, 0.5]), "label": "Well-tuned",
         "color": "#27AE60"},
        {"sigma": np.array([5.0, 5.0]), "label": "Too wide",
         "color": "#C0392B"},
    ]

    for row, cfg in enumerate(configs):
        chain_cfg, acc_rate = run_mh(n_steps, cfg["sigma"])
        burnin = 500
        chain_post = chain_cfg[burnin:]

        ax_trace = axes[row, 0]
        ax_trace.plot(chain_cfg[:, 0], lw=0.4, color=cfg["color"], alpha=0.8)
        ax_trace.axhline(xi1_true, color="#27AE60", ls="--", lw=1, alpha=0.5)
        ax_trace.axvspan(0, burnin, color="gray", alpha=0.1)
        ax_trace.set_ylabel(r"$\xi_1$", fontsize=11)
        ax_trace.set_title(
            f'{cfg["label"]} (accept rate: {acc_rate:.0%})',
            fontsize=10, color=cfg["color"], fontweight="bold",
        )
        ax_trace.grid(True, alpha=0.15)
        if row == 2:
            ax_trace.set_xlabel("Iteration", fontsize=11)

        ax_scatter = axes[row, 1]
        ax_scatter.plot(chain_cfg[:200, 0], chain_cfg[:200, 1], "-",
                        color="gray", lw=0.3, alpha=0.4)
        ax_scatter.scatter(chain_post[:, 0], chain_post[:, 1], s=3,
                           alpha=0.3, color=cfg["color"], edgecolors="none")
        ax_scatter.scatter(xi1_true, xi2_true, marker="D", s=60, color="k",
                           edgecolors="k", lw=1, zorder=5)
        ax_scatter.set_ylabel(r"$\xi_2$", fontsize=11)
        ax_scatter.set_title(f"2D scatter -- {cfg['label']}", fontsize=10)
        ax_scatter.grid(True, alpha=0.15)
        if row == 2:
            ax_scatter.set_xlabel(r"$\xi_1$", fontsize=11)


def plot_acceptance_rate(ax1, ax2, run_mh, effective_sample_size):
    """dram_mcmc.qmd -> fig-acceptance-rate

    Acceptance rate and ESS vs proposal width multiplier.
    """
    sigma_multipliers = np.logspace(-1.5, 1.2, 12)
    base_sigma = np.array([0.5, 0.5])
    acc_rates = []
    ess_values = []

    for mult in sigma_multipliers:
        ch, ar = run_mh(1500, base_sigma * mult, seed=123)
        acc_rates.append(ar)
        ess_values.append(effective_sample_size(ch[500:, 0]))

    ax1.semilogx(sigma_multipliers, acc_rates, "o-", ms=5, color="#2C7FB8",
                 lw=1.5, label="Acceptance rate")
    ax1.axhspan(0.2, 0.4, color="#27AE60", alpha=0.1)
    ax1.axhline(0.234, color="#27AE60", ls="--", lw=1, alpha=0.6,
                label="Optimal ~ 23.4% (theory)")
    ax1.set_xlabel("Proposal width multiplier", fontsize=12)
    ax1.set_ylabel("Acceptance rate", fontsize=12, color="#2C7FB8")
    ax1.tick_params(axis="y", labelcolor="#2C7FB8")
    ax1.set_ylim(-0.02, 1.02)

    ax2.semilogx(sigma_multipliers, ess_values, "s-", ms=5, color="#E67E22",
                 lw=1.5, label=r"ESS ($\xi_1$)")
    ax2.set_ylabel("Effective sample size", fontsize=12, color="#E67E22")
    ax2.tick_params(axis="y", labelcolor="#E67E22")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
               loc="center right")
    ax1.grid(True, alpha=0.15)
    ax1.set_title("Tuning the proposal width", fontsize=11)


def plot_burnin(ax, run_mh, xi1_true, burnin_val=500):
    """dram_mcmc.qmd -> fig-burnin

    Effect of burn-in removal on posterior histogram estimate.
    """
    chain_demo, _ = run_mh(2000, np.array([0.5, 0.5]), seed=7)

    bins = np.linspace(chain_demo[:, 0].min(), chain_demo[:, 0].max(), 50)

    ax.hist(chain_demo[:, 0], bins=bins, density=True, color="#C0392B",
            alpha=0.35, edgecolor="#C0392B", lw=0.5,
            label="All samples (with burn-in)")
    ax.hist(chain_demo[burnin_val:, 0], bins=bins, density=True,
            color="#2C7FB8", alpha=0.45, edgecolor="#2C7FB8", lw=0.5,
            label="After burn-in removed")
    ax.axvline(xi1_true, color="#27AE60", ls="--", lw=2,
               label=r"True $\xi_1$")
    ax.set_xlabel(r"$\xi_1$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Burn-in skews the posterior estimate", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    ax_in = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
    ax_in.plot(chain_demo[:, 0], lw=0.4, color="#C0392B", alpha=0.7)
    ax_in.axvspan(0, burnin_val, color="gray", alpha=0.2)
    ax_in.axhline(xi1_true, color="#27AE60", ls="--", lw=1, alpha=0.5)
    ax_in.set_ylabel(r"$\xi_1$", fontsize=7)
    ax_in.tick_params(labelsize=6)


def plot_autocorrelation(ax, run_mh, configs, max_lag=200):
    """dram_mcmc.qmd -> fig-autocorrelation

    Autocorrelation of xi_1 chain for three proposal widths.
    """
    for cfg in configs:
        chain_acf, _ = run_mh(3000, cfg["sigma"], seed=99)
        x = chain_acf[500:, 0]
        x = x - np.mean(x)
        acf_full = np.correlate(x, x, mode="full")
        acf_full = acf_full[len(x) - 1:]
        acf_full = acf_full / acf_full[0]
        lags = np.arange(max_lag)
        ax.plot(lags, acf_full[:max_lag], lw=1.5, color=cfg["color"],
                label=cfg["label"])

    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(0.05, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.set_title(r"Autocorrelation of $\xi_1$ chain", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0, max_lag)


def plot_delayed_rejection(ax1, ax2, tip_model_1d, y_obs_1d, sigma_noise_1d,
                           bkd):
    """dram_mcmc.qmd -> fig-delayed-rejection

    DR mechanism: rejected large step, accepted smaller fallback.
    """
    from scipy.stats import norm as sp_norm

    xi_grid_1d = np.linspace(-4.0, 4.0, 500)
    prior_vals_1d = sp_norm.pdf(xi_grid_1d, 0, 1)
    tip_preds_1d = bkd.to_numpy(
        tip_model_1d(bkd.asarray(xi_grid_1d[np.newaxis, :]))
    )[0, :]
    lik_vals_1d = sp_norm.pdf(y_obs_1d, tip_preds_1d, sigma_noise_1d)
    post_vals_1d = lik_vals_1d * prior_vals_1d
    post_vals_1d = post_vals_1d / np.trapz(post_vals_1d, xi_grid_1d)

    xi_current = 0.3
    xi_prop1 = 2.5
    xi_prop2 = 0.8

    p_curr = np.interp(xi_current, xi_grid_1d, post_vals_1d)
    p_prop1 = np.interp(xi_prop1, xi_grid_1d, post_vals_1d)
    p_prop2 = np.interp(xi_prop2, xi_grid_1d, post_vals_1d)

    for ax, xi_prop, p_prop, stage, color, outcome in [
        (ax1, xi_prop1, p_prop1, "Stage 1 (large step)", "#C0392B",
         "Rejected"),
        (ax2, xi_prop2, p_prop2, "Stage 2 (smaller step)", "#27AE60",
         "Accepted"),
    ]:
        ax.fill_between(xi_grid_1d, post_vals_1d, alpha=0.12,
                         color="#E67E22")
        ax.plot(xi_grid_1d, post_vals_1d, color="#E67E22", lw=2)

        ax.plot(xi_current, p_curr, "o", ms=12, color="#2C7FB8",
                markeredgecolor="k", markeredgewidth=1.5, zorder=5)
        ax.annotate("current", (xi_current, p_curr),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=9, color="#2C7FB8",
                    fontweight="bold")

        ax.plot(xi_prop, p_prop, "s", ms=12, color=color,
                markeredgecolor="k", markeredgewidth=1.5, zorder=5)

        y_arrow = max(p_curr, p_prop) * 0.3
        ax.annotate("", xy=(xi_prop, y_arrow),
                    xytext=(xi_current, y_arrow),
                    arrowprops=dict(arrowstyle="->", lw=2, color=color))

        sigma_sketch = 2.0 if "Stage 1" in stage else 0.6
        prop_sketch = sp_norm.pdf(xi_grid_1d, xi_current, sigma_sketch)
        prop_sketch = (
            prop_sketch / prop_sketch.max() * post_vals_1d.max() * 0.2
        )
        ax.plot(xi_grid_1d, prop_sketch, "--", color="gray", lw=1,
                label=f"Proposal ($\\sigma = {sigma_sketch}$)")

        ax.text(0.98, 0.85, f"{stage}\n{outcome}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#f5d5d5" if "Rejected" in outcome
                          else "#d5f5d5",
                          ec="gray", lw=0.5))

        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.15)

    ax2.set_xlabel(r"$\xi$", fontsize=12)


def plot_adaptive_proposal(ax1, ax2, chain_am, cov_early, cov_late,
                           posterior_2d, xi1_true, xi2_true, bkd):
    """dram_mcmc.qmd -> fig-adaptive-proposal

    Early isotropic vs late adapted proposal ellipse on 2D posterior.
    """
    from matplotlib.patches import Ellipse

    def _plot_proposal_ellipse(ax, center, cov, color, label, scale=2):
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        w = 2 * scale * np.sqrt(eigvals[1])
        h = 2 * scale * np.sqrt(eigvals[0])
        ell = Ellipse(
            xy=center, width=w, height=h, angle=angle,
            facecolor="none", edgecolor=color, lw=2.5, ls="--",
            label=label,
        )
        ax.add_patch(ell)

    n_g = 80
    xi1_g = np.linspace(-4.0, 4.0, n_g)
    xi2_g = np.linspace(-4.0, 4.0, n_g)
    xi1_m, xi2_m = np.meshgrid(xi1_g, xi2_g)
    grid_samples = bkd.asarray(np.vstack([xi1_m.ravel(), xi2_m.ravel()]))
    lp_flat = bkd.to_numpy(posterior_2d(grid_samples))
    lp_grid = lp_flat.reshape(n_g, n_g)
    p_grid = np.exp(lp_grid - np.max(lp_grid))

    for ax, title, chain_slice, cov_ell, c_color, ell_label in [
        (ax1, "Early: isotropic proposal", chain_am[:600], cov_early,
         "#C0392B", "Initial proposal"),
        (ax2, "Late: adapted proposal", chain_am[1000:], cov_late,
         "#27AE60", "Adapted proposal"),
    ]:
        ax.contourf(xi1_m, xi2_m, p_grid, levels=10, cmap="Blues", alpha=0.4)
        ax.contour(xi1_m, xi2_m, p_grid, levels=8, colors="steelblue",
                   linewidths=0.4, alpha=0.4)
        ax.scatter(chain_slice[:, 0], chain_slice[:, 1], s=3, alpha=0.4,
                   color=c_color, edgecolors="none")
        ax.scatter(xi1_true, xi2_true, marker="D", s=60, color="k",
                   edgecolors="k", lw=1, zorder=5)

        center = np.mean(chain_slice, axis=0)
        _plot_proposal_ellipse(ax, center, cov_ell, c_color, ell_label)

        ax.set_xlabel(r"$\xi_1$", fontsize=11)
        ax.set_ylabel(r"$\xi_2$", fontsize=11)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)


def plot_dram_comparison(axes, methods, xi1_true, xi2_true,
                         effective_sample_size, burnin_comp=500):
    """dram_mcmc.qmd -> fig-dram-comparison

    Four-column comparison of MH, AM, DR, DRAM chains.
    """
    for col, (name, ch, acc, color) in enumerate(methods):
        ax_t = axes[0, col]
        ax_t.plot(ch[:, 0], lw=0.3, color=color, alpha=0.8)
        ax_t.axhline(xi1_true, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax_t.axvspan(0, burnin_comp, color="gray", alpha=0.08)
        ax_t.set_title(f"{name} (acc: {acc:.0%})", fontsize=10,
                       color=color, fontweight="bold")
        if col == 0:
            ax_t.set_ylabel(r"$\xi_1$", fontsize=10)
        ax_t.grid(True, alpha=0.15)
        ax_t.set_xlabel("Iteration", fontsize=9)

        ax_s = axes[1, col]
        ch_post = ch[burnin_comp:]
        ax_s.scatter(ch_post[:, 0], ch_post[:, 1], s=2, alpha=0.3,
                     color=color, edgecolors="none")
        ax_s.scatter(xi1_true, xi2_true, marker="D", s=50, color="k",
                     zorder=5)
        ax_s.set_xlabel(r"$\xi_1$", fontsize=10)
        if col == 0:
            ax_s.set_ylabel(r"$\xi_2$", fontsize=10)
        ax_s.grid(True, alpha=0.15)

        ess = effective_sample_size(ch_post[:, 0])
        ax_s.text(0.05, 0.92, f"ESS = {ess:.0f}",
                  transform=ax_s.transAxes, fontsize=9, fontweight="bold",
                  color=color,
                  bbox=dict(fc="white", ec="gray", lw=0.5, alpha=0.8))
