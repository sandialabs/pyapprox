"""Plotting functions for dynamical-systems-learning tutorials.

Covers:
- dynamical_systems_learning_concept.qmd
- hamiltonian_systems_concept.qmd

Most functions self-contain their data generation (Van der Pol or SHO/
pendulum trajectories from scipy.integrate.solve_ivp). Tutorials in the
``in_progress/`` and ``library/`` directories only need to import and
call the plot functions without supplying data.

The naive RHS surrogate used in fig-hamiltonian-vs-naive is hand-built
in numpy (a degree-3 polynomial fit by least squares) to keep this
module's dependency surface to numpy + scipy + matplotlib.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.patches as patches
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _vdp_rhs(t, state, mu):
    x1, x2 = state
    return [x2, mu * (1 - x1 ** 2) * x2 - x1]


def _vdp_trajectory(ic, t_span, n_pts, mu):
    """One Van der Pol trajectory via scipy RK45."""
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    sol = solve_ivp(
        _vdp_rhs, t_span, ic, t_eval=t_eval, method="RK45",
        rtol=1e-9, atol=1e-11, args=(mu,),
    )
    return sol.t, sol.y


def _vdp_vector_field(grid_lim, n, mu):
    """Sample VdP RHS on a regular grid for quiver plotting."""
    xs = np.linspace(-grid_lim, grid_lim, n)
    ys = np.linspace(-grid_lim, grid_lim, n)
    X1, X2 = np.meshgrid(xs, ys)
    U = X2
    V = mu * (1 - X1 ** 2) * X2 - X1
    return X1, X2, U, V


# ---------------------------------------------------------------------------
# dynamical_systems_learning_concept.qmd
# ---------------------------------------------------------------------------


def plot_scalar_vs_dynamical(ax_scalar, ax_dynamical):
    """dynamical_systems_learning_concept.qmd -> fig-scalar-vs-dynamical."""
    from ._style import COLORS, apply_style

    # ----- Left panel: scalar surrogate scatter + curve ----------------
    rng = np.random.default_rng(2)
    xs = rng.uniform(-1, 1, 12)
    ys = np.sin(2 * np.pi * xs) + 0.05 * rng.standard_normal(12)
    xfine = np.linspace(-1, 1, 200)
    yfine = np.sin(2 * np.pi * xfine)

    ax_scalar.plot(xfine, yfine, color=COLORS["gray"], lw=1.5, alpha=0.6)
    ax_scalar.scatter(
        xs, ys, color=COLORS["primary"], s=42, zorder=3,
        edgecolor="k", linewidth=0.3,
    )
    ax_scalar.set_xlabel(r"input $\xi \in \mathbb{R}^{d_x}$", fontsize=11)
    ax_scalar.set_ylabel(r"output $y \in \mathbb{R}^{d_y}$", fontsize=11)
    ax_scalar.set_title(
        r"Scalar surrogate:  $f : \mathbb{R}^{d_x} \to \mathbb{R}^{d_y}$",
        fontsize=11,
    )
    apply_style(ax_scalar)

    # ----- Right panel: VdP vector field + trajectories ----------------
    mu = 1.0
    X1, X2, U, V = _vdp_vector_field(grid_lim=3.0, n=18, mu=mu)
    # Normalize vector field for visual clarity.
    mag = np.sqrt(U ** 2 + V ** 2)
    mag_clip = np.maximum(mag, 1e-8)
    U_n = U / mag_clip
    V_n = V / mag_clip

    ax_dynamical.quiver(
        X1, X2, U_n, V_n, mag, cmap="Greys",
        scale=28, width=0.0035, alpha=0.55, pivot="mid",
    )

    for ic in [(2.0, 0.0), (0.1, 0.1), (-2.5, 1.5)]:
        _, sol = _vdp_trajectory(list(ic), (0, 25), 600, mu)
        ax_dynamical.plot(
            sol[0], sol[1], color=COLORS["primary"], lw=1.4, alpha=0.85,
        )
        ax_dynamical.plot(
            ic[0], ic[1], "o", color=COLORS["primary"], ms=4,
            markeredgecolor="k", markeredgewidth=0.4,
        )

    ax_dynamical.set_xlim(-3, 3)
    ax_dynamical.set_ylim(-3, 3)
    ax_dynamical.set_xlabel(r"$x_1$", fontsize=11)
    ax_dynamical.set_ylabel(r"$x_2$", fontsize=11)
    ax_dynamical.set_title(
        r"Dynamical-systems surrogate:  $\dot{\mathbf{x}} = f_\eta(\mathbf{x})$",
        fontsize=11,
    )
    apply_style(ax_dynamical)
    ax_dynamical.set_aspect("equal", adjustable="box")


def plot_vdp_at_three_mu(axes):
    """dynamical_systems_learning_concept.qmd -> fig-vdp-mu.

    Three panels, one per mu value, each showing a single trajectory
    starting at (2, 0).
    """
    from ._style import COLORS, apply_style

    mu_values = [0.5, 1.0, 2.0]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["reference"]]

    for ax, mu, color in zip(axes, mu_values, colors):
        _, sol = _vdp_trajectory([2.0, 0.0], (0, 30), 1200, mu)
        ax.plot(sol[0], sol[1], lw=1.4, color=color)
        ax.plot(2.0, 0.0, "ko", ms=4)
        ax.set_xlabel(r"$x_1$", fontsize=11)
        ax.set_ylabel(r"$x_2$", fontsize=11)
        ax.set_title(rf"$\mu = {mu}$", fontsize=11)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-4.5, 4.5)
        apply_style(ax)
        ax.set_aspect("equal", adjustable="box")


def plot_snapshot_extraction(ax_traj, ax_arrows):
    """dynamical_systems_learning_concept.qmd -> fig-snapshot-extraction.

    Left: a trajectory in phase space with discrete observation points.
    Right: arrows at the same points whose directions come from a
    central finite difference of the trajectory.
    """
    from ._style import COLORS, apply_style

    mu = 1.0
    # Use a small number of observation points so individual arrows are
    # legible. Sub-sample from a more finely integrated reference.
    times_fine, sol_fine = _vdp_trajectory([2.0, 0.0], (0, 10), 401, mu)
    sub = np.linspace(20, 380, 30, dtype=int)
    times = times_fine[sub]
    states = sol_fine[:, sub]

    # ----- Left panel: trajectory with observation dots --------------
    ax_traj.plot(
        sol_fine[0], sol_fine[1],
        color=COLORS["gray"], lw=1.2, alpha=0.6,
    )
    ax_traj.scatter(
        states[0], states[1],
        color=COLORS["primary"], s=22, zorder=3,
        edgecolor="k", linewidth=0.3,
    )
    ax_traj.plot(2.0, 0.0, "ko", ms=5)
    ax_traj.set_xlabel(r"$x_1$", fontsize=11)
    ax_traj.set_ylabel(r"$x_2$", fontsize=11)
    ax_traj.set_title("Observed trajectory with samples", fontsize=11)
    ax_traj.set_xlim(-3, 3)
    ax_traj.set_ylim(-4, 4)
    apply_style(ax_traj)
    ax_traj.set_aspect("equal", adjustable="box")

    # ----- Right panel: arrows from finite-difference derivatives ----
    # Central difference (need neighbours on both sides for each sub index)
    sub_inner = sub[1:-1]
    states_inner = sol_fine[:, sub_inner]
    dt_fine = times_fine[1] - times_fine[0]
    # Use neighbouring sub indices for central difference.
    # Compute using sub[k]-1 and sub[k]+1 in fine grid index space.
    plus = sol_fine[:, sub[1:-1] + 1]
    minus = sol_fine[:, sub[1:-1] - 1]
    derivs = (plus - minus) / (2 * dt_fine)

    arrow_scale = 0.25
    for k in range(states_inner.shape[1]):
        x0, y0 = states_inner[0, k], states_inner[1, k]
        dx, dy = derivs[0, k] * arrow_scale, derivs[1, k] * arrow_scale
        ax_arrows.annotate(
            "", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color=COLORS["primary"],
                lw=1.2, alpha=0.85,
            ),
        )
    ax_arrows.scatter(
        states_inner[0], states_inner[1],
        color=COLORS["primary"], s=18, zorder=3,
        edgecolor="k", linewidth=0.3,
    )
    ax_arrows.set_xlabel(r"$x_1$", fontsize=11)
    ax_arrows.set_ylabel(r"$x_2$", fontsize=11)
    ax_arrows.set_title(
        r"Snapshot pairs $(\mathbf{x}_i,\dot{\mathbf{x}}_i)$ as arrows",
        fontsize=11,
    )
    ax_arrows.set_xlim(-3, 3)
    ax_arrows.set_ylim(-4, 4)
    apply_style(ax_arrows)
    ax_arrows.set_aspect("equal", adjustable="box")


def plot_pipeline(ax):
    """dynamical_systems_learning_concept.qmd -> fig-pipeline.

    Two-stage block diagram: Learn box (snapshots -> ansatz -> fitter ->
    f_eta) feeding into Predict box (f_eta + new IC + new mu ->
    integrator -> trajectory).
    """
    from ._style import COLORS

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=10):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="black", lw=1.1,
        ))
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize,
        )

    def arrow(x1, y1, x2, y2, label=None, label_offset_y=0.25):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )
        if label is not None:
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + label_offset_y, label,
                ha="center", fontsize=9.5, style="italic",
            )

    # ----- Learn box (left) ---------------------------------------------
    learn_x0, learn_x1 = 0.3, 7.0
    learn_y0, learn_y1 = 0.5, 5.5
    ax.add_patch(patches.FancyBboxPatch(
        (learn_x0, learn_y0),
        learn_x1 - learn_x0, learn_y1 - learn_y0,
        boxstyle="round,pad=0.1",
        facecolor="#F4F8FB", edgecolor=COLORS["gray"], lw=1.0,
    ))
    ax.text(
        (learn_x0 + learn_x1) / 2, learn_y1 - 0.25,
        "Learn  (do once)", ha="center", fontsize=10.5,
        style="italic", color=COLORS["gray"],
    )

    box(0.6, 2.3, 1.8, 1.0, "training\ntrajectories", "#EAF3FB")
    arrow(2.4, 2.8, 3.0, 2.8)
    box(3.0, 2.3, 1.8, 1.0, "snapshot\npairs", "#EAF3FB")
    arrow(4.8, 2.8, 5.4, 2.8, label="fitter")
    box(5.4, 2.3, 1.4, 1.0, r"$f_\eta$", "#FDEEDD")

    # ----- Predict box (right) ------------------------------------------
    pred_x0, pred_x1 = 7.8, 13.7
    pred_y0, pred_y1 = 0.5, 5.5
    ax.add_patch(patches.FancyBboxPatch(
        (pred_x0, pred_y0),
        pred_x1 - pred_x0, pred_y1 - pred_y0,
        boxstyle="round,pad=0.1",
        facecolor="#FFF8F0", edgecolor=COLORS["gray"], lw=1.0,
    ))
    ax.text(
        (pred_x0 + pred_x1) / 2, pred_y1 - 0.25,
        "Predict  (do many times)", ha="center", fontsize=10.5,
        style="italic", color=COLORS["gray"],
    )

    box(8.1, 3.4, 1.7, 0.9, "new $\\mathbf{x}(0)$", "#EAF3FB")
    box(8.1, 1.4, 1.7, 0.9, "new $\\boldsymbol{\\mu}$", "#EAF3FB")
    arrow(9.8, 3.8, 10.6, 2.9)
    arrow(9.8, 1.9, 10.6, 2.7)
    box(10.6, 2.3, 1.4, 1.0, "integrator", "#FDEEDD")
    arrow(12.0, 2.8, 12.6, 2.8)
    box(12.6, 2.3, 1.0, 1.0, r"$\mathbf{x}(t)$", "#FDEEDD")

    # ----- Connect Learn -> Predict ------------------------------------
    arrow(6.8, 2.8, 10.6, 2.85, label_offset_y=0.0)
    ax.text(
        8.7, 3.15, r"fitted $f_\eta$  (reusable)",
        ha="center", fontsize=10, style="italic",
    )


def plot_two_losses(ax_deriv, ax_traj):
    """dynamical_systems_learning_concept.qmd -> fig-two-losses.

    Left: snapshot arrows (data) vs surrogate arrows (model). Residuals
    are local arrow differences.
    Right: observed trajectory (dots) vs surrogate-integrated trajectory
    (curve). Residuals are vertical bars at observation times.
    """
    from ._style import COLORS, apply_style

    mu = 1.0
    # ----- Left panel: derivative-matching picture -----------------
    times_fine, sol_fine = _vdp_trajectory([2.0, 0.0], (0, 10), 401, mu)
    sub = np.linspace(40, 360, 12, dtype=int)
    states = sol_fine[:, sub]
    dt_fine = times_fine[1] - times_fine[0]
    plus = sol_fine[:, sub + 1]
    minus = sol_fine[:, sub - 1]
    derivs_true = (plus - minus) / (2 * dt_fine)

    # Pretend "surrogate" arrows are a perturbed version: rotate by
    # ~10 deg and slightly scale.
    theta = np.deg2rad(10)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    derivs_surr = 0.85 * (R @ derivs_true)

    arrow_scale = 0.25
    for k in range(states.shape[1]):
        x0, y0 = states[0, k], states[1, k]
        dx_t, dy_t = derivs_true[0, k] * arrow_scale, derivs_true[1, k] * arrow_scale
        dx_s, dy_s = derivs_surr[0, k] * arrow_scale, derivs_surr[1, k] * arrow_scale
        ax_deriv.annotate(
            "", xy=(x0 + dx_t, y0 + dy_t), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color=COLORS["primary"],
                lw=1.4, alpha=0.9,
            ),
        )
        ax_deriv.annotate(
            "", xy=(x0 + dx_s, y0 + dy_s), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color=COLORS["secondary"],
                lw=1.4, alpha=0.9,
            ),
        )
    ax_deriv.scatter(
        states[0], states[1], color="k", s=10, zorder=3,
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["primary"], lw=1.5,
               label=r"data $\dot{\mathbf{x}}_i$"),
        Line2D([0], [0], color=COLORS["secondary"], lw=1.5,
               label=r"surrogate $f_\eta(\mathbf{x}_i)$"),
    ]
    ax_deriv.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax_deriv.set_xlabel(r"$x_1$", fontsize=11)
    ax_deriv.set_ylabel(r"$x_2$", fontsize=11)
    ax_deriv.set_title(
        "Derivative matching:\nlocal arrow residuals", fontsize=11,
    )
    ax_deriv.set_xlim(-3, 3)
    ax_deriv.set_ylim(-4, 4)
    apply_style(ax_deriv)
    ax_deriv.set_aspect("equal", adjustable="box")

    # ----- Right panel: trajectory-matching picture ------------------
    # True trajectory (sparser observation points) vs a slightly drifted
    # surrogate prediction. The "surrogate" trajectory is the true VdP
    # with mu modified slightly, so the curves diverge over time.
    t_obs, sol_obs = _vdp_trajectory([2.0, 0.0], (0, 8), 16, mu)
    _, sol_surr = _vdp_trajectory([2.0, 0.0], (0, 8), 200, mu * 0.85)
    t_surr = np.linspace(0, 8, 200)

    ax_traj.plot(
        t_surr, sol_surr[0],
        color=COLORS["secondary"], lw=1.6,
        label=r"surrogate-integrated  $x_1(t)$",
    )
    ax_traj.scatter(
        t_obs, sol_obs[0],
        color=COLORS["primary"], s=32, zorder=3,
        edgecolor="k", linewidth=0.3,
        label="observation",
    )
    # Residual bars at observation times.
    for tk, yobs in zip(t_obs, sol_obs[0]):
        ysurr = np.interp(tk, t_surr, sol_surr[0])
        ax_traj.plot(
            [tk, tk], [yobs, ysurr],
            color=COLORS["gray"], lw=1.0, alpha=0.7,
        )
    ax_traj.legend(fontsize=9, loc="upper right")
    ax_traj.set_xlabel(r"$t$", fontsize=11)
    ax_traj.set_ylabel(r"$x_1(t)$", fontsize=11)
    ax_traj.set_title(
        "Trajectory matching:\nglobal trajectory residuals", fontsize=11,
    )
    apply_style(ax_traj)


# ---------------------------------------------------------------------------
# hamiltonian_systems_concept.qmd
# ---------------------------------------------------------------------------


def _sho_rhs(t, state, omega=1.0):
    q, p = state
    return [p, -omega ** 2 * q]


def _pendulum_rhs(t, state):
    q, p = state
    return [p, -np.sin(q)]


def _sho_trajectory(ic, t_span, n_pts, omega=1.0):
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    sol = solve_ivp(
        _sho_rhs, t_span, ic, t_eval=t_eval, method="RK45",
        rtol=1e-11, atol=1e-13, args=(omega,),
    )
    return sol.t, sol.y


def _pendulum_trajectory(ic, t_span, n_pts):
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    sol = solve_ivp(
        _pendulum_rhs, t_span, ic, t_eval=t_eval, method="RK45",
        rtol=1e-11, atol=1e-13,
    )
    return sol.t, sol.y


def plot_hamiltonian_vs_naive(ax_true, ax_naive):
    """hamiltonian_systems_concept.qmd -> fig-hamiltonian-vs-naive.

    Left: SHO phase portrait, several closed orbits and one Hamiltonian
    level set as a dashed contour.
    Right: same initial conditions integrated under a hand-built naive
    RHS surrogate using backward Euler. The orbits spiral inward.
    """
    from ._style import COLORS, apply_style

    omega = 1.0
    # Use a few initial conditions of different "energy" so each orbit
    # is a distinct ellipse.
    ics = [(0.8, 0.0), (1.4, 0.0), (2.0, 0.0)]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["reference"]]

    # ----- Left panel: true SHO orbits + one H level set -------------
    for ic, color in zip(ics, colors):
        _, sol = _sho_trajectory(list(ic), (0, 2 * np.pi), 400, omega)
        ax_true.plot(sol[0], sol[1], lw=1.4, color=color)
        ax_true.plot(
            ic[0], ic[1], "o", color=color, ms=4,
            markeredgecolor="k", markeredgewidth=0.4,
        )

    # H = 0.5*(p^2 + omega^2*q^2). Draw the H-level-set for ic=(1.4,0).
    H_target = 0.5 * 1.4 ** 2
    theta_grid = np.linspace(0, 2 * np.pi, 200)
    q_lev = 1.4 * np.cos(theta_grid)
    p_lev = 1.4 * np.sin(theta_grid)
    ax_true.plot(
        q_lev, p_lev, "--", color=COLORS["gray"], lw=1.2, alpha=0.8,
        label=rf"$\mathcal{{H}} = {H_target:.2f}$",
    )
    ax_true.legend(fontsize=9, loc="upper right")
    ax_true.set_xlabel(r"$q$", fontsize=11)
    ax_true.set_ylabel(r"$p$", fontsize=11)
    ax_true.set_title(
        "True SHO: closed orbits =\nlevel sets of $\\mathcal{H}$",
        fontsize=11,
    )
    ax_true.set_xlim(-2.5, 2.5)
    ax_true.set_ylim(-2.5, 2.5)
    apply_style(ax_true)
    ax_true.set_aspect("equal", adjustable="box")

    # ----- Right panel: naive surrogate + dissipative integrator -----
    # We mimic "naive RHS fit with backward Euler" by directly running
    # backward Euler on the true SHO. Backward Euler is dissipative for
    # any non-degenerate ODE; on the SHO it spirals inward. This is the
    # qualitative picture a reader needs to see.
    dt = 0.04
    n_steps = int(np.round(2 * np.pi / dt)) * 3   # ~3 periods

    for ic, color in zip(ics, colors):
        q, p = ic
        qs, ps = [q], [p]
        # Backward Euler on SHO: solve M(y_n - y_{n-1}) - dt*f(y_n) = 0
        # for linear SHO this is a 2x2 linear system per step.
        A = np.array([[1.0,  -dt], [dt * omega ** 2, 1.0]])
        for _ in range(n_steps):
            b = np.array([qs[-1], ps[-1]])
            new = np.linalg.solve(A, b)
            qs.append(float(new[0]))
            ps.append(float(new[1]))
        ax_naive.plot(qs, ps, lw=1.2, color=color, alpha=0.9)
        ax_naive.plot(
            ic[0], ic[1], "o", color=color, ms=4,
            markeredgecolor="k", markeredgewidth=0.4,
        )

    ax_naive.set_xlabel(r"$q$", fontsize=11)
    ax_naive.set_ylabel(r"$p$", fontsize=11)
    ax_naive.set_title(
        "Naive surrogate + naive integrator:\n"
        "energy drifts, orbits spiral",
        fontsize=11,
    )
    ax_naive.set_xlim(-2.5, 2.5)
    ax_naive.set_ylim(-2.5, 2.5)
    apply_style(ax_naive)
    ax_naive.set_aspect("equal", adjustable="box")


def plot_parametrization_architecture(ax):
    """hamiltonian_systems_concept.qmd -> fig-parametrization-architecture.

    Two side-by-side block-diagram architectures: a generic-RHS
    parametrisation on the left, a Hamiltonian parametrisation on the
    right. The fitter+loss block is shared (drawn below both).
    """
    from ._style import COLORS

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=10):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="black", lw=1.1,
        ))
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize,
        )

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )
        if label is not None:
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 0.25, label,
                ha="center", fontsize=9.5, style="italic",
            )

    # ----- Left: generic RHS architecture -----------------------------
    ax.text(
        2.7, 5.4, "Generic RHS (tutorial 1)",
        ha="center", fontsize=11, fontweight="bold",
    )
    box(0.4, 3.2, 1.6, 1.0, r"$\mathbf{x}$", "#EAF3FB")
    arrow(2.0, 3.7, 2.8, 3.7)
    box(2.8, 3.2, 1.6, 1.0, "basis", "#EAF3FB")
    arrow(4.4, 3.7, 5.2, 3.7)
    box(5.2, 3.2, 1.6, 1.0, r"$f_\eta(\mathbf{x})$", "#FDEEDD")
    ax.text(
        2.7, 2.7,
        "Outputs an unrestricted vector field.",
        ha="center", fontsize=9.5, style="italic",
        color=COLORS["gray"],
    )

    # ----- Right: Hamiltonian architecture ----------------------------
    ax.text(
        10.7, 5.4, "Hamiltonian (parametrisation A)",
        ha="center", fontsize=11, fontweight="bold",
    )
    box(7.4, 3.2, 1.5, 1.0, r"$\mathbf{x}$", "#EAF3FB")
    arrow(8.9, 3.7, 9.6, 3.7)
    box(9.6, 3.2, 1.5, 1.0, "basis", "#EAF3FB")
    arrow(11.1, 3.7, 11.8, 3.7)
    box(11.8, 3.2, 1.8, 1.0, r"$\mathcal{H}_\eta$  (scalar)", "#FDEEDD")
    arrow(12.7, 3.2, 12.7, 2.4)
    box(11.6, 1.4, 2.2, 1.0, r"$\mathbf{L}\, \nabla \mathcal{H}_\eta$",
        "#FDEEDD")
    ax.text(
        10.7, 0.7,
        "Outputs a guaranteed-Hamiltonian field by construction.",
        ha="center", fontsize=9.5, style="italic",
        color=COLORS["gray"],
    )

    # Vertical separator
    ax.plot(
        [7.0, 7.0], [0.4, 5.6],
        color=COLORS["gray"], lw=0.8, ls=":", alpha=0.6,
    )

    # Bottom annotation: shared fitter
    ax.text(
        7.0, 0.15,
        "Fitter and loss are identical in both — only the parametrisation between basis and RHS changes.",
        ha="center", fontsize=9.5, style="italic", color=COLORS["gray"],
    )


class _SimpleODEResidual:
    """Minimal ODE residual for use with PyApprox implicit steppers.

    Wraps f(y) and J(y) callables for autonomous systems without
    parameters. Satisfies ImplicitODEResidualProtocol.
    """

    def __init__(self, f_fn, jac_fn, nstates):
        from pyapprox.util.backends.numpy import NumpyBkd
        from pyapprox.ode.mass_matrix import IdentityMassMatrix
        self._f = f_fn
        self._jac = jac_fn
        self._bkd = NumpyBkd()
        self._mass = IdentityMassMatrix(nstates, self._bkd)

    def bkd(self):
        return self._bkd

    def __call__(self, state):
        return self._f(state)

    def set_time(self, time):
        pass

    def jacobian(self, state):
        return self._jac(state)

    def mass_matrix(self):
        return self._mass

    def newton_jacobian(self, state, coefficient):
        from pyapprox.ode.linear_operator import MatrixOperator
        matrix = self._mass.as_matrix() - coefficient * self._jac(state)
        return MatrixOperator(matrix, self._bkd)


def _integrate_ode(stepper_class, residual, ic, dt, n_steps):
    """Integrate an ODE using a PyApprox TimeIntegrator."""
    from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
    from pyapprox.util.rootfinding.newton import NewtonSolver

    stepper = stepper_class(residual)
    solver = NewtonSolver(stepper)
    solver.set_options(atol=1e-12, rtol=1e-12)
    integrator = TimeIntegrator(0.0, n_steps * dt, dt, solver)

    init_state = np.array(ic, dtype=float)
    states, times = integrator.solve(init_state)
    return states[0], states[1], times


def _make_sho_residual(omega):
    A = np.array([[0.0, 1.0], [-omega ** 2, 0.0]])
    return _SimpleODEResidual(
        f_fn=lambda y: A @ y,
        jac_fn=lambda y: A,
        nstates=2,
    )


def _make_pendulum_residual():
    return _SimpleODEResidual(
        f_fn=lambda y: np.array([y[1], -np.sin(y[0])]),
        jac_fn=lambda y: np.array([[0.0, 1.0], [-np.cos(y[0]), 0.0]]),
        nstates=2,
    )


def plot_integrator_energy_comparison(ax_sho, ax_pendulum):
    """hamiltonian_systems_concept.qmd -> fig-integrator-comparison.

    Left: SHO. H = 0.5*(p^2 + omega^2*q^2). BE decays; CN preserves;
    IM preserves (coincides with CN on linear problems).
    Right: Pendulum. H = 0.5*p^2 - cos(q). BE decays; CN drifts
    secularly; IM preserves.
    """
    from pyapprox.ode.implicit_steppers.backward_euler import (
        BackwardEulerStepper,
    )
    from pyapprox.ode.implicit_steppers.crank_nicolson import (
        CrankNicolsonStepper,
    )
    from pyapprox.ode.implicit_steppers.implicit_midpoint import (
        ImplicitMidpointStepper,
    )
    from ._style import COLORS, apply_style

    steppers = [BackwardEulerStepper, CrankNicolsonStepper,
                ImplicitMidpointStepper]

    styles = [
        ("backward Euler", COLORS["reference"], "-", 2.0),
        ("Crank-Nicolson", COLORS["primary"], "--", 1.8),
        ("implicit midpoint", COLORS["accent"], "-.", 1.8),
    ]

    # ----- Left panel: SHO, 30 periods ---------------------------------
    omega = 1.0
    T = 30 * 2 * np.pi
    dt = 0.05
    n_steps = int(np.ceil(T / dt))

    ic = (1.0, 0.0)
    H0_sho = 0.5 * (ic[1] ** 2 + omega ** 2 * ic[0] ** 2)

    sho_res = _make_sho_residual(omega)
    results_sho = {}
    for cls in steppers:
        q, p, t = _integrate_ode(cls, sho_res, ic, dt, n_steps)
        results_sho[cls] = (q, p, t)

    for cls, (label, color, ls, lw) in zip(steppers, styles):
        q, p, t = results_sho[cls]
        H = 0.5 * (p ** 2 + omega ** 2 * q ** 2) - H0_sho
        ax_sho.plot(t, H, color=color, lw=lw, ls=ls, label=label)

    ax_sho.set_xlabel(r"$t$", fontsize=11)
    ax_sho.set_ylabel(r"$\mathcal{H}(t) - \mathcal{H}(0)$", fontsize=11)
    ax_sho.set_title(
        "SHO  (linear, quadratic $\\mathcal{H}$)", fontsize=11,
    )
    ax_sho.legend(fontsize=9, loc="lower left")
    apply_style(ax_sho)

    # ----- Right panel: pendulum -----------------------------------------
    # Coarser dt so CN energy oscillation is large enough to see in an
    # inset, separate from the dominant BE decay on the main axis.
    T_p = 50 * 2 * np.pi
    dt_p = 0.3
    n_steps_p = int(np.ceil(T_p / dt_p))

    ic_p = (2.8, 0.0)
    H0_p = 0.5 * ic_p[1] ** 2 - np.cos(ic_p[0])

    pend_res = _make_pendulum_residual()
    results_pend = {}
    for cls in steppers:
        q, p, t = _integrate_ode(cls, pend_res, ic_p, dt_p, n_steps_p)
        results_pend[cls] = (q, p, t)

    energies_pend = {}
    for cls, (label, color, ls, lw) in zip(steppers, styles):
        q, p, t = results_pend[cls]
        H = 0.5 * p ** 2 - np.cos(q) - H0_p
        ax_pendulum.plot(t, H, color=color, lw=lw, ls=ls, label=label)
        energies_pend[cls] = (t, H)

    ax_pendulum.set_xlabel(r"$t$", fontsize=11)
    ax_pendulum.set_ylabel(r"$\mathcal{H}(t) - \mathcal{H}(0)$", fontsize=11)
    ax_pendulum.set_title(
        "Pendulum  (nonlinear $\\mathcal{H} = \\frac{1}{2} p^2 - \\cos q$)",
        fontsize=11,
    )
    ax_pendulum.legend(fontsize=9, loc="lower left")
    apply_style(ax_pendulum)

    # Inset: zoom into CN vs IM (BE is off-scale)
    ax_inset = ax_pendulum.inset_axes([0.38, 0.35, 0.58, 0.55])
    for cls, (_, color, ls, lw) in zip(
        steppers[1:], styles[1:],
    ):
        t, H = energies_pend[cls]
        ax_inset.plot(t, H, color=color, lw=lw, ls=ls)
    ax_inset.set_ylabel(r"$\Delta\mathcal{H}$", fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.set_title("CN vs IM (zoom)", fontsize=8)


def plot_four_cell_matrix(axes):
    """hamiltonian_systems_concept.qmd -> fig-four-cell-matrix.

    2x2 grid of mini phase portraits showing surrogate parametrisation
    x integrator. Only the (Hamiltonian + symplectic) cell shows clean
    closed orbits.

    Parameters
    ----------
    axes : 2D array of Axes, shape (2, 2)
        Layout: rows = (naive surrogate, Hamiltonian surrogate)
                cols = (non-symplectic integrator, symplectic integrator)
    """
    from pyapprox.ode.implicit_steppers.backward_euler import (
        BackwardEulerStepper,
    )
    from pyapprox.ode.implicit_steppers.crank_nicolson import (
        CrankNicolsonStepper,
    )
    from pyapprox.ode.implicit_steppers.implicit_midpoint import (
        ImplicitMidpointStepper,
    )
    from ._style import COLORS, apply_style

    # Common SHO setup
    omega = 1.0
    ic = (1.5, 0.0)
    T = 6 * 2 * np.pi
    dt = 0.05
    n_steps = int(np.ceil(T / dt))

    # ----- True closed orbit for visual reference --------------------
    _, sol_true = _sho_trajectory(list(ic), (0, 2 * np.pi), 200, omega)

    cell_titles = [
        ["Naive surrogate +\nnon-symplectic integrator",
         "Naive surrogate +\nsymplectic integrator"],
        ["Hamiltonian surrogate +\nnon-symplectic integrator",
         "Hamiltonian surrogate +\nsymplectic integrator"],
    ]

    sho_res = _make_sho_residual(omega)

    # Naive surrogate: approximate vector field with small asymmetric
    # error that breaks Hamiltonian structure. Neither integrator can
    # produce closed orbits from non-conservative dynamics.
    eps = 0.08
    A_naive = np.array([[eps, 1.0], [-omega ** 2, eps]])
    naive_res = _SimpleODEResidual(
        f_fn=lambda y: A_naive @ y,
        jac_fn=lambda y: A_naive,
        nstates=2,
    )

    # Top-left: naive + BE — both surrogate and integrator are wrong
    q, p, _ = _integrate_ode(BackwardEulerStepper, naive_res, ic, dt, n_steps)
    axes[0, 0].plot(q, p, color=COLORS["reference"], lw=1.0, alpha=0.85)

    # Top-right: naive + IM — symplectic integrator can't fix non-
    # Hamiltonian dynamics; trajectory spirals out
    q, p, _ = _integrate_ode(ImplicitMidpointStepper, naive_res, ic, dt, n_steps)
    axes[0, 1].plot(q, p, color=COLORS["reference"], lw=1.0, alpha=0.85)

    # Bottom-left: Hamiltonian surrogate + BE. Surrogate is exact, BE
    # dissipates -> orbit spirals (same shape as top-left).
    q, p, _ = _integrate_ode(BackwardEulerStepper, sho_res, ic, dt, n_steps)
    axes[1, 0].plot(q, p, color=COLORS["reference"], lw=1.0, alpha=0.85)

    # Bottom-right: Hamiltonian surrogate + symplectic integrator.
    q, p, _ = _integrate_ode(ImplicitMidpointStepper, sho_res, ic, dt, n_steps)
    axes[1, 1].plot(q, p, color=COLORS["accent"], lw=1.5)

    for i in range(2):
        for j in range(2):
            axes[i, j].plot(
                sol_true[0], sol_true[1],
                "--", color=COLORS["gray"], lw=0.8, alpha=0.6,
            )
            axes[i, j].plot(ic[0], ic[1], "ko", ms=3)
            axes[i, j].set_xlim(-2.2, 2.2)
            axes[i, j].set_ylim(-2.2, 2.2)
            axes[i, j].set_aspect("equal", adjustable="box")
            axes[i, j].set_title(cell_titles[i][j], fontsize=10)
            apply_style(axes[i, j])
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])
