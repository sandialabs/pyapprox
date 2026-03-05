"""Plotting functions for PCE and GP-based UQ tutorials.

Covers: polynomial_chaos_surrogates.qmd, pce_sensitivity.qmd,
        practical_pce_fitting.qmd, sparse_pce_fitting.qmd,
        uq_with_pce.qmd, uq_with_gp.qmd
"""

import numpy as np

# ---------------------------------------------------------------------------
# polynomial_chaos_surrogates.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_pce_predictions(pce, samples_test, values_test, qoi_labels, bkd,
                         axes):
    """polynomial_chaos_surrogates.qmd → fig-pce-predictions

    Predicted-vs-true scatter for each QoI on the test set.
    """
    from ._style import apply_style

    preds_np = bkd.to_numpy(pce(samples_test))
    vals_np = bkd.to_numpy(values_test)
    nqoi = vals_np.shape[0]

    for k, ax in enumerate(axes[:nqoi]):
        ax.scatter(vals_np[k, :], preds_np[k, :], s=10, alpha=0.5,
                   color="#2C7FB8", edgecolors="k", lw=0.2)
        lims = [min(vals_np[k, :].min(), preds_np[k, :].min()),
                max(vals_np[k, :].max(), preds_np[k, :].max())]
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_xlabel(f"True {qoi_labels[k]}", fontsize=11)
        ax.set_ylabel(f"PCE {qoi_labels[k]}", fontsize=11)
        l2_err = np.linalg.norm(vals_np[k, :] - preds_np[k, :])
        l2_ref = np.linalg.norm(vals_np[k, :])
        ax.set_title(
            f"{qoi_labels[k]},  rel. $L_2$ error = {l2_err / l2_ref:.2e}",
            fontsize=10,
        )
        ax.set_aspect("equal")
        apply_style(ax)


# ---------------------------------------------------------------------------
# pce_sensitivity.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_pce_convergence(func, prior, marginals, nvars, gt_main, gt_total,
                         bkd, axes):
    """pce_sensitivity.qmd → fig-convergence

    Sobol index convergence with PCE degree (first-order and total-order).
    """
    from pyapprox.sensitivity import PolynomialChaosSensitivityAnalysis
    from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
        LeastSquaresFitter,
    )
    from pyapprox.surrogates.affine.expansions.pce import (
        create_pce_from_marginals,
    )

    from ._style import apply_style

    ls_fitter = LeastSquaresFitter(bkd)
    degrees = [2, 3, 4, 5, 6, 8, 10]
    oversampling = 2
    S_vs_degree = {i: [] for i in range(nvars)}
    T_vs_degree = {i: [] for i in range(nvars)}

    for deg in degrees:
        pce_deg = create_pce_from_marginals(marginals, deg, bkd, nqoi=1)
        n_train_deg = max(oversampling * pce_deg.nterms(), 50)
        np.random.seed(42)
        s = prior.rvs(n_train_deg)
        v = func(s)
        try:
            fitted = ls_fitter.fit(pce_deg, s, v).surrogate()
            sa_deg = PolynomialChaosSensitivityAnalysis(fitted)
            me = bkd.to_numpy(sa_deg.main_effects()).flatten()
            te = bkd.to_numpy(sa_deg.total_effects()).flatten()
            for i in range(nvars):
                S_vs_degree[i].append(me[i])
                T_vs_degree[i].append(te[i])
        except Exception:
            for i in range(nvars):
                S_vs_degree[i].append(np.nan)
                T_vs_degree[i].append(np.nan)

    ax1, ax2 = axes[0], axes[1]
    colors = ["#2C7FB8", "#E67E22", "#27AE60"]
    input_labels = [f"$x_{i+1}$" for i in range(nvars)]

    for i in range(nvars):
        ax1.plot(degrees, S_vs_degree[i], "o-", color=colors[i],
                 label=input_labels[i])
        ax1.axhline(gt_main[i], color=colors[i], linestyle="--", alpha=0.5)
    ax1.set_xlabel("PCE degree", fontsize=11)
    ax1.set_ylabel("$S_i$", fontsize=11)
    ax1.set_title("First-order indices vs. degree", fontsize=12)
    ax1.legend(fontsize=9)
    apply_style(ax1)

    for i in range(nvars):
        ax2.plot(degrees, T_vs_degree[i], "o-", color=colors[i],
                 label=input_labels[i])
        ax2.axhline(gt_total[i], color=colors[i], linestyle="--", alpha=0.5)
    ax2.set_xlabel("PCE degree", fontsize=11)
    ax2.set_ylabel("$S_i^T$", fontsize=11)
    ax2.set_title("Total-order indices vs. degree", fontsize=12)
    ax2.legend(fontsize=9)
    apply_style(ax2)


# ---------------------------------------------------------------------------
# practical_pce_fitting.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_truncation_error(model, prior, marginals, nvars, nqoi,
                          samples_test, values_test, bkd, ax):
    """practical_pce_fitting.qmd → fig-truncation-error

    Test-set L2 error vs training size for several total-degree PCEs.
    """
    from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
        LeastSquaresFitter,
    )
    from pyapprox.surrogates.affine.expansions.pce import (
        create_pce_from_marginals,
    )
    from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices

    ls_fitter = LeastSquaresFitter(bkd)
    vals_np = bkd.to_numpy(values_test)

    degrees_to_test = [4, 6, 8]
    N_train_values = [50, 75, 100, 150, 200, 300, 500]
    n_reps = 5
    colors_deg = ["#2C7FB8", "#E67E22", "#C0392B"]

    for deg, color in zip(degrees_to_test, colors_deg):
        indices_deg = compute_hyperbolic_indices(nvars, deg, 1.0, bkd)
        nterms = indices_deg.shape[1]
        mean_errors = []
        min_errors = []
        max_errors = []
        valid_N = []

        for N_tr in N_train_values:
            if N_tr <= nterms + 2:
                continue
            valid_N.append(N_tr)
            rep_errors = []
            for rep in range(n_reps):
                np.random.seed(2000 + N_tr * 100 + rep)
                s_tr = prior.rvs(N_tr)
                v_tr = model(s_tr)

                pce_rep = create_pce_from_marginals(
                    marginals, deg, bkd, nqoi=nqoi,
                )
                fitted_rep = ls_fitter.fit(pce_rep, s_tr, v_tr).surrogate()

                preds = bkd.to_numpy(fitted_rep(samples_test))
                l2_err = np.linalg.norm(vals_np[0, :] - preds[0, :])
                l2_ref = np.linalg.norm(vals_np[0, :])
                rep_errors.append(l2_err / l2_ref)

            mean_errors.append(np.mean(rep_errors))
            min_errors.append(np.min(rep_errors))
            max_errors.append(np.max(rep_errors))

        ax.fill_between(valid_N, min_errors, max_errors, alpha=0.15,
                         color=color)
        ax.semilogy(valid_N, mean_errors, "o-", ms=5, color=color, lw=1.5,
                     label=f"Degree {deg} ({nterms} terms)")

    ax.set_xlabel("Number of training samples $N$", fontsize=11)
    ax.set_ylabel("Relative $L_2$ error", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")


def plot_train_test_cv(res, nterms_list, N_train, axes):
    """practical_pce_fitting.qmd → fig-train-test-cv

    Test error and CV score vs degree with condition-number subplot.
    """
    degs = res["degrees"]

    ax, ax_cond = axes[0], axes[1]

    # --- Top panel: errors ---
    ax.fill_between(degs, res["te_min"], res["te_max"], alpha=0.15,
                    color="#C0392B")
    ax.semilogy(degs, res["te_mean"], "s--", ms=6, lw=2, color="#C0392B",
                label="Test error")

    cv_ok = res["cv_any_valid"]
    cv_degs = np.array(degs)[cv_ok]
    ax.fill_between(cv_degs, res["cv_min"][cv_ok], res["cv_max"][cv_ok],
                    alpha=0.15, color="#E67E22")
    ax.semilogy(cv_degs, res["cv_mean"][cv_ok], "D-.", ms=6, lw=2,
                color="#E67E22", label="5-fold CV score")

    best_i = np.argmin(res["cv_mean"][cv_ok])
    ax.plot(cv_degs[best_i], res["cv_mean"][cv_ok][best_i], "*", ms=18,
            color="#E67E22", zorder=5, markeredgecolor="k",
            markeredgewidth=0.5)

    # Annotate nterms
    for i, (d, nt) in enumerate(zip(degs, nterms_list)):
        ax.annotate(f"P={nt}", (d, res["te_mean"][i]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="#C0392B")

    # Mark the underdetermined boundary
    for i, (d, nt) in enumerate(zip(degs, nterms_list)):
        if nt >= N_train:
            ax.axvline(d - 0.5, color="gray", ls=":", lw=1, alpha=0.6)
            ax.text(d - 0.35, ax.get_ylim()[1] * 0.3,
                    r"$P \geq N$", fontsize=9, color="gray", va="top")
            break

    ax.set_ylabel("Relative $L_2$ error", fontsize=11)
    ax.set_title(
        f"N = {N_train} training samples  (mean over 10 random draws)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")

    # --- Bottom panel: condition number ---
    ax_cond.semilogy(degs, res["cond_mean"], "^-", ms=5, lw=1.5,
                     color="#7B2D8E")
    ax_cond.set_xlabel("Total polynomial degree", fontsize=11)
    ax_cond.set_ylabel("cond($\\mathbf{\\Phi}$)", fontsize=11)
    ax_cond.grid(True, alpha=0.2, which="both")


# ---------------------------------------------------------------------------
# practical_pce_fitting.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_train_test_cv_2panel(results_left, results_right, nterms_list,
                              N_left, N_right, axes):
    """practical_pce_fitting.qmd → fig-train-test-cv-2panel

    Side-by-side test error and CV score vs degree for two training sizes.
    """
    for ax, res, N_tr in [
        (axes[0], results_left, N_left),
        (axes[1], results_right, N_right),
    ]:
        degs = res["degrees"]

        ax.fill_between(degs, res["te_min"], res["te_max"], alpha=0.15,
                        color="#C0392B")
        ax.semilogy(degs, res["te_mean"], "s--", ms=6, lw=2, color="#C0392B",
                    label="Test error")

        cv_ok = res["cv_any_valid"]
        cv_d = np.array(degs)[cv_ok]
        ax.fill_between(cv_d, res["cv_min"][cv_ok], res["cv_max"][cv_ok],
                        alpha=0.15, color="#E67E22")
        ax.semilogy(cv_d, res["cv_mean"][cv_ok], "D-.", ms=6, lw=2,
                    color="#E67E22", label="5-fold CV score")

        best_i = np.argmin(res["cv_mean"][cv_ok])
        ax.plot(cv_d[best_i], res["cv_mean"][cv_ok][best_i], "*", ms=18,
                color="#E67E22", zorder=5, markeredgecolor="k",
                markeredgewidth=0.5)

        # Annotate nterms
        for i, (d, nt) in enumerate(zip(degs, nterms_list)):
            ax.annotate(f"P={nt}", (d, res["te_mean"][i]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color="#C0392B")

        ax.set_xlabel("Total polynomial degree", fontsize=11)
        ax.set_ylabel("Relative $L_2$ error", fontsize=11)
        ax.set_title(f"N = {N_tr} training samples", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, which="both")


# ---------------------------------------------------------------------------
# sparse_pce_fitting.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_coefficient_decay(sorted_abs, degree, nterms, N_train, ax):
    """sparse_pce_fitting.qmd → fig-coefficient-decay

    Sorted coefficient magnitudes (descending, log scale) for a LS PCE.
    """
    from ._style import apply_style

    ax.bar(range(len(sorted_abs)), sorted_abs, color="#2C7FB8", alpha=0.7,
           width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Rank (sorted by magnitude)", fontsize=11)
    ax.set_ylabel("|Coefficient|", fontsize=11)
    ax.set_title(f"Degree-{degree} LS PCE: {nterms} terms, "
                 f"N = {N_train} samples", fontsize=11)
    apply_style(ax)


def plot_omp_residual(res_hist, ax):
    """sparse_pce_fitting.qmd → fig-omp-residual

    OMP residual norm vs number of selected terms.
    """
    from ._style import apply_style

    ax.semilogy(range(1, len(res_hist) + 1), res_hist, "o-", ms=6, lw=1.5,
                color="#E67E22")
    ax.set_xlabel("Number of selected terms", fontsize=11)
    ax.set_ylabel("Residual norm", fontsize=11)
    ax.set_title("OMP residual history", fontsize=11)
    apply_style(ax)


def plot_sparse_coefficients(results, fitter_names, bkd, axes):
    """sparse_pce_fitting.qmd → fig-sparse-coefficients

    Coefficient magnitudes for LS, OMP, and BPDN side by side.
    """
    colors = ["#2C7FB8", "#E67E22", "#C0392B"]

    for ax, name, result, color in zip(axes, fitter_names, results, colors):
        coef = bkd.to_numpy(result.params())[:, 0]
        ax.bar(range(len(coef)), np.abs(coef), color=color, alpha=0.7,
               width=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("Basis term index", fontsize=11)
        ax.set_title(f"{name}", fontsize=11)
        n_nz = np.sum(np.abs(coef) > 1e-12)
        ax.annotate(
            f"{n_nz} non-zero", xy=(0.95, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
        )

    axes[0].set_ylabel("|Coefficient|", fontsize=11)


def plot_accuracy_vs_n(N_values, ls3_errors, omp5_errors, ax):
    """sparse_pce_fitting.qmd → fig-accuracy-vs-n

    Test-set error vs training size for LS degree-3 and OMP degree-5.
    """
    from ._style import apply_style

    ax.semilogy(N_values, ls3_errors, "o-", ms=7, lw=2, color="#2C7FB8",
                label="LS degree-3 (10 terms)")
    ax.semilogy(N_values, omp5_errors, "s--", ms=7, lw=2, color="#E67E22",
                label="OMP degree-5 (sparse)")
    ax.set_xlabel("Number of training samples $N$", fontsize=12)
    ax.set_ylabel("Relative $L_2$ test error", fontsize=12)
    ax.set_title("Dense vs. sparse fitting", fontsize=11)
    ax.legend(fontsize=10)
    apply_style(ax)


# ---------------------------------------------------------------------------
# uq_with_pce.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_pce_marginal_densities(vals_mc, preds_pce, N_mc, N_train, M_pce,
                                qoi_labels, axes):
    """uq_with_pce.qmd → fig-marginal-densities

    Marginal density histograms at equal wall-clock cost (MC vs PCE).
    """
    from ._style import apply_style

    nqoi = vals_mc.shape[0]
    for k, ax in enumerate(axes[:nqoi]):
        ax.hist(vals_mc[k, :], bins=40, density=True, alpha=0.5,
                color="#2C7FB8", edgecolor="k", lw=0.2,
                label=f"MC ({N_mc} beam evals)")
        ax.hist(preds_pce[k, :], bins=80, density=True, alpha=0.4,
                color="#E67E22", edgecolor="k", lw=0.1,
                label=f"PCE ({N_train}+{M_pce:,} evals)")
        ax.set_xlabel(qoi_labels[k], fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
        apply_style(ax)


# ---------------------------------------------------------------------------
# uq_with_pce.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_pce_vs_mc(model, prior, marginals, ls_fitter, bkd,
                   nvars, nqoi, qoi_labels, axes):
    """uq_with_pce.qmd → fig-pce-vs-mc

    PCE vs MC efficiency: relative error in mean and variance vs budget.
    """
    from pyapprox.surrogates.affine.expansions.pce import (
        create_pce_from_marginals,
    )
    from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
    from pyapprox.surrogates.quadrature import (
        TensorProductQuadratureRule,
        gauss_quadrature_rule,
    )

    # True moments via tensor-product Gauss quadrature
    nquad = 20
    quad_rules = [
        lambda n, m=m: gauss_quadrature_rule(m, n, bkd) for m in marginals
    ]
    quad_rule = TensorProductQuadratureRule(bkd, quad_rules, [nquad] * nvars)
    quad_pts, quad_wts = quad_rule()
    qoi_quad = model(quad_pts)
    qoi_quad_np = bkd.to_numpy(qoi_quad)
    quad_wts_np = bkd.to_numpy(quad_wts)
    qoi_plot = 0
    true_mean = float(quad_wts_np @ qoi_quad_np[qoi_plot])
    true_var = float(quad_wts_np @ (qoi_quad_np[qoi_plot] - true_mean) ** 2)

    # Build (degree, N_budget) pairs with oversampling ratio = 2
    oversampling = 2
    degrees = [1, 2, 3, 4, 5]
    budget_degree_pairs = []
    for deg in degrees:
        nterms = compute_hyperbolic_indices(nvars, deg, 1.0, bkd).shape[1]
        N_bud = oversampling * nterms
        budget_degree_pairs.append((N_bud, deg))

    N_budget_values = [pair[0] for pair in budget_degree_pairs]
    n_reps = 20

    mc_mean_errors = {N: [] for N in N_budget_values}
    mc_var_errors = {N: [] for N in N_budget_values}
    pce_mean_errors = {N: [] for N in N_budget_values}
    pce_var_errors = {N: [] for N in N_budget_values}

    for N_bud, deg in budget_degree_pairs:
        for rep in range(n_reps):
            np.random.seed(1000 * N_bud + rep)
            s = prior.rvs(N_bud)
            v = bkd.to_numpy(model(s))

            mc_mu = np.mean(v[qoi_plot, :])
            mc_va = np.var(v[qoi_plot, :], ddof=1)
            mc_mean_errors[N_bud].append(
                abs(mc_mu - true_mean) / abs(true_mean)
            )
            mc_var_errors[N_bud].append(
                abs(mc_va - true_var) / abs(true_var)
            )

            pce_rep = create_pce_from_marginals(
                marginals, deg, bkd, nqoi=nqoi,
            )
            try:
                rep_result = ls_fitter.fit(pce_rep, s, bkd.asarray(v))
                fitted_rep = rep_result.surrogate()
                pce_mu = bkd.to_numpy(fitted_rep.mean())[qoi_plot]
                pce_va = bkd.to_numpy(fitted_rep.variance())[qoi_plot]
                pce_mean_errors[N_bud].append(
                    abs(pce_mu - true_mean) / abs(true_mean)
                )
                pce_var_errors[N_bud].append(
                    abs(pce_va - true_var) / abs(true_var)
                )
            except Exception:
                pce_mean_errors[N_bud].append(np.nan)
                pce_var_errors[N_bud].append(np.nan)

    def _plot_convergence(ax, N_vals, mc_errs, pce_errs, ylabel, title,
                          deg_labels):
        mc_med = [np.nanmedian(mc_errs[N]) for N in N_vals]
        mc_lo = [np.nanpercentile(mc_errs[N], 10) for N in N_vals]
        mc_hi = [np.nanpercentile(mc_errs[N], 90) for N in N_vals]
        pce_med = [np.nanmedian(pce_errs[N]) for N in N_vals]
        pce_lo = [np.nanpercentile(pce_errs[N], 10) for N in N_vals]
        pce_hi = [np.nanpercentile(pce_errs[N], 90) for N in N_vals]

        ax.fill_between(N_vals, mc_lo, mc_hi, alpha=0.15, color="#2C7FB8")
        ax.semilogy(N_vals, mc_med, "o-", ms=5, color="#2C7FB8", lw=1.5,
                    label="Monte Carlo")
        ax.fill_between(N_vals, pce_lo, pce_hi, alpha=0.15, color="#E67E22")
        ax.semilogy(N_vals, pce_med, "s-", ms=5, color="#E67E22", lw=1.5,
                    label="PCE")
        for N, deg_lbl in zip(N_vals, deg_labels):
            idx = N_vals.index(N)
            ax.annotate(f"p={deg_lbl}", (N, pce_med[idx]),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=7, color="#E67E22")
        ax.set_xlabel("Number of model evaluations $N$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, which="both")

    deg_labels = [str(d) for _, d in budget_degree_pairs]
    ax1, ax2 = axes[0], axes[1]
    _plot_convergence(
        ax1, N_budget_values, mc_mean_errors, pce_mean_errors,
        "Relative error in mean",
        f"Mean of {qoi_labels[qoi_plot]}",
        deg_labels,
    )
    _plot_convergence(
        ax2, N_budget_values, mc_var_errors, pce_var_errors,
        "Relative error in variance",
        f"Variance of {qoi_labels[qoi_plot]}",
        deg_labels,
    )


# ---------------------------------------------------------------------------
# uq_with_gp.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_gp_marginal_density(mu_mc, gp_mean, gp_std, ax):
    """uq_with_gp.qmd → fig-marginal-density

    Histogram of GP posterior mean samples with mean line and +/-1sigma band.
    """
    from ._style import apply_style

    ax.hist(mu_mc, bins=60, density=True, alpha=0.7, color="#2C7FB8",
            edgecolor="k", linewidth=0.2, label="GP surrogate")

    ax.axvline(
        gp_mean, color="#084594", lw=1.5, linestyle="--",
        label=(rf"$\mathbb{{E}}_{{\boldsymbol{{\xi}}}}[\mu^*]"
               rf" = {gp_mean:.4f}$"),
    )
    ax.axvspan(
        gp_mean - gp_std, gp_mean + gp_std, alpha=0.15, color="#084594",
        label=rf"$\pm 1\sigma$ ($\sigma = {gp_std:.4f}$)",
    )

    ax.set_xlabel("Tip deflection", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    apply_style(ax)
