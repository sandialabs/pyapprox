"""Plotting functions for SVGP and DGP concept/analysis tutorials.

Covers: svgp_concept.qmd (and later: dgp_concept.qmd, dgp_quadrature_analysis.qmd, ...)
"""

import numpy as np


# ---------------------------------------------------------------------------
# svgp_concept.qmd — Convention A (echo:false)
# ---------------------------------------------------------------------------


def _rbf_kernel(X1, X2, lengthscale=0.25, variance=1.0):
    """Plain numpy RBF kernel for self-contained illustrations.

    X1: (d, n1), X2: (d, n2). Returns (n1, n2).
    """
    diff = X1[:, :, None] - X2[:, None, :]
    sq = np.sum(diff ** 2, axis=0)
    return variance * np.exp(-0.5 * sq / (lengthscale ** 2))


def _exact_gp_posterior(X_train, y_train, X_test, lengthscale, noise_var):
    """Closed-form exact GP posterior at test inputs.

    Returns latent posterior mean and standard deviation.
    """
    K = _rbf_kernel(X_train, X_train, lengthscale)
    n = K.shape[0]
    A = K + noise_var * np.eye(n)
    L = np.linalg.cholesky(A + 1e-10 * np.eye(n))

    K_star = _rbf_kernel(X_train, X_test, lengthscale)   # (n_train, n_test)
    K_starstar_diag = np.full(X_test.shape[1], 1.0)      # variance=1 default

    alpha = np.linalg.solve(L, y_train.ravel())
    mean = K_star.T @ np.linalg.solve(L.T, alpha)

    v = np.linalg.solve(L, K_star)
    var = K_starstar_diag - np.sum(v * v, axis=0)
    var = np.clip(var, 1e-12, None)
    std = np.sqrt(var)
    return mean, std


def _svgp_titsias_posterior(
    X_train, y_train, Z, X_test, lengthscale, noise_var,
):
    """Closed-form Titsias-collapsed SVGP posterior at test inputs.

    Z: (d, M) inducing locations.
    Returns latent posterior mean and std.
    """
    M = Z.shape[1]
    N = X_train.shape[1]
    K_uu = _rbf_kernel(Z, Z, lengthscale) + 1e-6 * np.eye(M)
    K_uf = _rbf_kernel(Z, X_train, lengthscale)
    K_us = _rbf_kernel(Z, X_test, lengthscale)

    L_uu = np.linalg.cholesky(K_uu)

    A = (K_uu + (1.0 / noise_var) * (K_uf @ K_uf.T))
    L_A = np.linalg.cholesky(A + 1e-10 * np.eye(M))

    rhs = K_uu @ np.linalg.solve(
        L_A.T, np.linalg.solve(L_A, K_uf @ y_train.ravel() / noise_var)
    )
    mean = (
        K_us.T @ np.linalg.solve(L_uu.T, np.linalg.solve(L_uu, rhs))
    )

    Sigma = K_uu @ np.linalg.solve(L_A.T, np.linalg.solve(L_A, K_uu))

    K_starstar_diag = np.full(X_test.shape[1], 1.0)
    qf_minus_kf = K_us.T @ np.linalg.solve(
        L_uu.T, np.linalg.solve(L_uu, Sigma)
    ) @ np.linalg.solve(L_uu.T, np.linalg.solve(L_uu, K_us))
    var_q = (
        K_starstar_diag
        - np.einsum(
            "ij,ji->i",
            K_us.T,
            np.linalg.solve(L_uu.T, np.linalg.solve(L_uu, K_us)),
        )
        + np.diag(qf_minus_kf)
    )
    var_q = np.clip(var_q, 1e-12, None)
    return mean, np.sqrt(var_q)


def _svgp_elbo(X_train, y_train, Z, lengthscale, noise_var):
    """Titsias collapsed ELBO. Used to plot ELBO vs M curves."""
    M = Z.shape[1]
    N = X_train.shape[1]

    K_uu = _rbf_kernel(Z, Z, lengthscale) + 1e-6 * np.eye(M)
    K_uf = _rbf_kernel(Z, X_train, lengthscale)
    K_ff_diag = np.full(N, 1.0)  # variance=1

    L_uu = np.linalg.cholesky(K_uu)
    A_inner = np.linalg.solve(L_uu, K_uf)            # (M, N)
    A = np.eye(M) + (1.0 / noise_var) * (A_inner @ A_inner.T)
    L_A = np.linalg.cholesky(A + 1e-10 * np.eye(M))

    c = (1.0 / noise_var) * np.linalg.solve(
        L_A, A_inner @ y_train.ravel()
    )

    yty = y_train.ravel() @ y_train.ravel() / noise_var
    quad = yty - c @ c

    log_det = (
        N * np.log(noise_var)
        + 2.0 * np.sum(np.log(np.diag(L_A)))
    )

    Q_ff_diag = np.sum(A_inner * A_inner, axis=0)
    trace = np.sum(K_ff_diag - Q_ff_diag) / noise_var

    elbo = (
        -0.5 * N * np.log(2 * np.pi)
        - 0.5 * log_det
        - 0.5 * quad
        - 0.5 * trace
    )
    return elbo


def _exact_log_marginal_likelihood(
    X_train, y_train, lengthscale, noise_var,
):
    N = X_train.shape[1]
    K = _rbf_kernel(X_train, X_train, lengthscale)
    A = K + noise_var * np.eye(N)
    L = np.linalg.cholesky(A + 1e-10 * np.eye(N))
    alpha = np.linalg.solve(L, y_train.ravel())
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    quad = alpha @ alpha
    return -0.5 * (N * np.log(2 * np.pi) + log_det + quad)


def plot_scaling_wall(ax):
    """svgp_concept.qmd -> fig-scaling-wall

    Cost vs N for exact GP (cubic) and SVGP (linear in N for fixed M).
    """
    from ._style import apply_style

    N_vals = np.logspace(1, 5.5, 60)
    M_fixed = 128

    # Notional FLOP counts; constants chosen so curves are comparable
    exact_flops = N_vals ** 3
    svgp_flops = N_vals * (M_fixed ** 2)

    # Scale to seconds via a reference: assume 1e9 flop/s
    flop_rate = 1e9
    exact_secs = exact_flops / flop_rate
    svgp_secs = svgp_flops / flop_rate

    ax.loglog(
        N_vals, exact_secs, lw=2.2, color="#C0392B",
        label=r"Exact GP: $\mathcal{O}(N^3)$",
    )
    ax.loglog(
        N_vals, svgp_secs, lw=2.2, color="#2C7FB8",
        label=rf"SVGP: $\mathcal{{O}}(N M^2),\ M={M_fixed}$",
    )

    # Mark practical cutoffs
    one_minute = 60.0
    one_hour = 3600.0
    ax.axhline(one_minute, color="k", ls=":", alpha=0.4, lw=1)
    ax.text(
        N_vals[0] * 1.2, one_minute * 1.4, "1 minute",
        fontsize=8, color="k", alpha=0.6,
    )
    ax.axhline(one_hour, color="k", ls=":", alpha=0.4, lw=1)
    ax.text(
        N_vals[0] * 1.2, one_hour * 1.4, "1 hour",
        fontsize=8, color="k", alpha=0.6,
    )

    ax.set_xlabel("Training set size $N$", fontsize=11)
    ax.set_ylabel("Compute time (seconds, illustrative)", fontsize=11)
    ax.set_title(
        "Cost scaling: exact GP vs SVGP at fixed inducing budget",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper left")
    apply_style(ax)


def plot_inducing_intuition(axes):
    """svgp_concept.qmd -> fig-inducing-intuition

    1D dataset and three SVGP fits at increasing M, showing the
    inducing summary improves toward the exact GP.
    """
    from ._style import apply_style

    rng = np.random.default_rng(3)

    # Build a synthetic 1D dataset with a non-trivial response
    N = 400
    X = np.sort(rng.uniform(-3.0, 3.0, N)).reshape(1, -1)
    f_true = np.sin(1.5 * X[0]) + 0.4 * np.cos(3.0 * X[0])
    noise_std = 0.1
    y = f_true + noise_std * rng.standard_normal(N)

    lengthscale = 0.45
    noise_var = noise_std ** 2

    X_grid = np.linspace(-3.5, 3.5, 250).reshape(1, -1)

    M_list = [4, 16, 64]
    titles = [
        "M = 4 inducing points",
        "M = 16 inducing points",
        "M = 64 inducing points",
    ]

    # Exact GP reference (latent)
    mean_exact, std_exact = _exact_gp_posterior(
        X, y.reshape(1, -1), X_grid, lengthscale, noise_var,
    )

    for ax, M, title in zip(axes, M_list, titles):
        # Place inducing locations evenly across the input range
        Z = np.linspace(X.min() + 0.1, X.max() - 0.1, M).reshape(1, -1)

        mean_svgp, std_svgp = _svgp_titsias_posterior(
            X, y.reshape(1, -1), Z, X_grid, lengthscale, noise_var,
        )

        # Draw data faintly
        ax.scatter(
            X[0], y, s=8, color="#888888", alpha=0.35,
            label=f"Data ($N={N}$)" if M == M_list[0] else None,
        )

        # Exact GP latent mean as a thin reference
        ax.plot(
            X_grid[0], mean_exact, color="k", lw=1.0, alpha=0.55,
            label="Exact GP" if M == M_list[0] else None,
        )

        # SVGP fit
        ax.plot(
            X_grid[0], mean_svgp, color="#2C7FB8", lw=1.8,
            label="SVGP" if M == M_list[0] else None,
        )
        ax.fill_between(
            X_grid[0],
            mean_svgp - 2 * std_svgp,
            mean_svgp + 2 * std_svgp,
            color="#2C7FB8", alpha=0.18,
            label=r"SVGP $\pm 2\sigma$" if M == M_list[0] else None,
        )

        # Mark inducing locations along the bottom
        y_min = (mean_svgp - 2 * std_svgp).min() - 0.3
        ax.scatter(
            Z[0], np.full(M, y_min), marker="^", s=30, color="#E67E22",
            edgecolors="k", linewidths=0.4, zorder=5,
            label="Inducing locations $Z$" if M == M_list[0] else None,
        )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x$", fontsize=10)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-2.8, 2.5)
        apply_style(ax)

    axes[0].set_ylabel("$f(x)$", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)


def plot_elbo_vs_M(ax):
    """svgp_concept.qmd -> fig-elbo-vs-M

    SVGP collapsed ELBO vs M with reference horizontal line at the
    exact-GP log marginal likelihood. Y-axis is clipped to keep the
    near-bound asymptote readable; off-screen ELBO values at very
    small M are annotated with their numeric value.
    """
    from ._style import apply_style

    rng = np.random.default_rng(5)
    N = 200
    X = np.sort(rng.uniform(-3.0, 3.0, N)).reshape(1, -1)
    f_true = np.sin(1.2 * X[0]) + 0.3 * np.cos(2.5 * X[0])
    noise_std = 0.1
    y = f_true + noise_std * rng.standard_normal(N)

    lengthscale = 0.5
    noise_var = noise_std ** 2

    M_vals = np.array([4, 8, 12, 20, 30, 50, 80, 120, 160, 200])
    elbos = []
    for M in M_vals:
        if M >= N:
            Z = X.copy()
        else:
            Z = np.linspace(X.min() + 0.05, X.max() - 0.05, M).reshape(1, -1)
        elbos.append(
            _svgp_elbo(X, y.reshape(1, -1), Z, lengthscale, noise_var)
        )
    elbos = np.array(elbos)

    log_p_y = _exact_log_marginal_likelihood(
        X, y.reshape(1, -1), lengthscale, noise_var,
    )

    ax.axhline(
        log_p_y, color="#C0392B", lw=1.8, ls="--",
        label=r"Exact GP $\log p(\mathbf{y})$",
    )
    ax.plot(
        M_vals, elbos, "o-", color="#2C7FB8", lw=2, ms=6,
        label=r"SVGP ELBO $\mathcal{L}(M)$",
    )

    # Clip the y-axis so the near-bound asymptote is visible. The
    # ELBO at very small M can be many tens of nats below the bound,
    # which compresses the asymptote out of view. Annotate any
    # off-screen points with their actual numeric values.
    visible_below = 25.0    # nats below the bound to show
    headroom = 5.0          # nats above the bound for breathing space
    y_max = log_p_y + headroom
    y_min = log_p_y - visible_below
    ax.set_ylim(y_min, y_max)

    for m, e in zip(M_vals, elbos):
        if e < y_min:
            ax.annotate(
                "",
                xy=(m, y_min + 1.0),
                xytext=(m, y_min + 4.5),
                arrowprops=dict(
                    arrowstyle="->", color="#2C7FB8", alpha=0.7, lw=1.2,
                ),
            )
            ax.text(
                m, y_min + 5.5, f"{e:.0f}",
                ha="center", fontsize=7, color="#2C7FB8",
            )

    ax.set_xlabel("Number of inducing points $M$", fontsize=11)
    ax.set_ylabel("Log marginal evidence (nats)", fontsize=11)
    ax.set_title(
        rf"ELBO converges from below to $\log p(\mathbf{{y}})$ "
        rf"as $M \to N = {N}$",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    apply_style(ax)


def plot_elbo_decomposition(ax):
    """svgp_concept.qmd -> fig-elbo-decomposition

    Bound gap (log p(y) - ELBO) vs M on a log y-axis. The gap is
    strictly positive (the ELBO is a lower bound) and shrinks
    exponentially in M; log scale makes the convergence rate
    directly visible as a slope.
    """
    from ._style import apply_style

    rng = np.random.default_rng(7)
    N = 200
    X = np.sort(rng.uniform(-3.0, 3.0, N)).reshape(1, -1)
    f_true = np.sin(1.2 * X[0]) + 0.3 * np.cos(2.5 * X[0])
    noise_std = 0.1
    y = f_true + noise_std * rng.standard_normal(N)

    lengthscale = 0.5
    noise_var = noise_std ** 2

    M_vals = np.array([4, 8, 16, 32, 64, 128, 200])

    elbos = []
    for M in M_vals:
        if M >= N:
            Z = X.copy()
        else:
            Z = np.linspace(X.min() + 0.05, X.max() - 0.05, M).reshape(1, -1)
        elbos.append(
            _svgp_elbo(X, y.reshape(1, -1), Z, lengthscale, noise_var)
        )
    elbos = np.array(elbos)
    log_p_y = _exact_log_marginal_likelihood(
        X, y.reshape(1, -1), lengthscale, noise_var,
    )

    gap = log_p_y - elbos  # strictly >= 0 (ELBO is a lower bound)
    gap = np.maximum(gap, 1e-6)  # floor for log scale at M = N

    ax.plot(
        M_vals, gap, "o-", color="#E67E22", lw=2, ms=7,
        markeredgecolor="k", markeredgewidth=0.5,
        label=r"Gap: $\log p(\mathbf{y}) - \mathcal{L}(M)$",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of inducing points $M$", fontsize=11)
    ax.set_ylabel("Bound gap (nats, log scale)", fontsize=11)
    ax.set_title(
        "Approximation error of the variational bound shrinks with $M$",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    apply_style(ax)
