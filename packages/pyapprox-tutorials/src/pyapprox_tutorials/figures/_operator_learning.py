"""Plotting functions for operator learning tutorials.

Covers: operator_learning_concept.qmd, kerneloperator_usage.qmd

All functions follow Convention B: accept pre-computed arrays, only plot.
The shared utilities ``sample_log_diffusion_field`` and ``solve_elliptic_1d``
are used by tutorials to generate training data inline.

# The helpers `sample_log_diffusion_field` and `solve_elliptic_1d` are
# tutorial-scope wrappers over pyapprox's KLE and Galerkin PDE solver.
# They will be replaced by a `build_elliptic_1d_operator` benchmark in
# `pyapprox_benchmarks` in a future change. Keep the function names and
# signatures stable until then.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.patches as patches
import numpy as np

from pyapprox.pde.galerkin import (
    AdvectionDiffusionReaction,
    LagrangeBasis,
    SteadyStateSolver,
    StructuredMesh1D,
)
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.surrogates.kernels import SquaredExponentialKernel
from pyapprox.surrogates.kle import MeshKLE
from pyapprox.util.backends.numpy import NumpyBkd


# ---------------------------------------------------------------------------
# Shared utilities (1D elliptic operator and Gaussian random fields)
# ---------------------------------------------------------------------------

_BKD = NumpyBkd()


def sample_log_diffusion_field(
    grid: np.ndarray,
    N: int,
    length_scale: float = 0.15,
    sigma: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample N log-diffusion fields u(x) on the given 1D grid.

    Uses pyapprox's MeshKLE with a squared-exponential kernel.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    bkd = _BKD
    mesh_coords = bkd.array(grid[None, :])
    kernel = SquaredExponentialKernel(
        bkd.array([length_scale]), (1e-3, 10.0), 1, bkd,
    )
    ngrid = len(grid)
    dx = grid[1] - grid[0]
    w = np.ones(ngrid) * dx
    w[0] /= 2
    w[-1] /= 2
    kle = MeshKLE(
        mesh_coords, kernel, sigma=sigma, nterms=ngrid,
        quad_weights=bkd.array(w), bkd=bkd,
    )
    coef = bkd.array(rng.standard_normal((ngrid, N)))
    return bkd.to_numpy(kle(coef))


def solve_elliptic_1d(u: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Solve -(e^u v')' = 1 on [0,1] with v(0)=v(1)=0.

    Uses pyapprox's Galerkin FEM solver (skfem) with P1 elements on a
    uniform mesh matching the input grid.
    """
    bkd = _BKD
    ngrid, N = u.shape
    mesh = StructuredMesh1D(nx=ngrid - 1, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)

    v = np.zeros_like(u)
    for k in range(N):
        diff_vals = np.exp(u[:, k])

        def diffusivity(x, _dv=diff_vals, _grid=grid):
            return np.interp(x[0], _grid, _dv)

        def forcing(x):
            return np.ones(x.shape[1])

        bc_left = DirichletBC(basis, "left", 0.0, bkd)
        bc_right = DirichletBC(basis, "right", 0.0, bkd)
        physics = AdvectionDiffusionReaction(
            basis=basis, diffusivity=diffusivity,
            forcing=forcing, bkd=bkd,
            boundary_conditions=[bc_left, bc_right],
        )

        solver = SteadyStateSolver(physics, tol=1e-12)
        result = solver.solve(bkd.zeros(physics.nstates()))
        v[:, k] = bkd.to_numpy(result.solution)
    return v


# ---------------------------------------------------------------------------
# operator_learning_concept.qmd
# ---------------------------------------------------------------------------

def plot_scalar_vs_operator(ax_scalar, ax_operator):
    """operator_learning_concept.qmd -> fig-scalar-vs-operator."""
    from matplotlib.lines import Line2D

    from ._style import COLORS, apply_style

    rng = np.random.default_rng(2)
    xs = rng.uniform(0, 1, 12)
    ys = np.sin(2 * np.pi * xs) + 0.05 * rng.standard_normal(12)
    xfine = np.linspace(0, 1, 200)
    yfine = np.sin(2 * np.pi * xfine)
    ax_scalar.plot(xfine, yfine, color=COLORS["gray"], lw=1.5, alpha=0.6)
    ax_scalar.scatter(
        xs, ys, color=COLORS["primary"], s=40, zorder=3,
        edgecolor="k", linewidth=0.3,
    )
    ax_scalar.set_xlabel(r"input $\xi \in \mathbb{R}^{d_x}$", fontsize=11)
    ax_scalar.set_ylabel(r"output $y \in \mathbb{R}^{d_y}$", fontsize=11)
    ax_scalar.set_title(
        r"Scalar surrogate:  $f : \mathbb{R}^{d_x} \to \mathbb{R}^{d_y}$",
        fontsize=11,
    )
    ax_scalar.set_xlim(-0.02, 1.02)
    apply_style(ax_scalar)

    grid = np.linspace(0, 1, 80)
    u_samples = sample_log_diffusion_field(grid, 4, length_scale=0.2, rng=rng)
    v_samples = solve_elliptic_1d(u_samples, grid)
    for k in range(4):
        ax_operator.plot(
            grid, u_samples[:, k], color=COLORS["primary"],
            alpha=0.55, lw=1.2,
        )
        ax_operator.plot(
            grid, v_samples[:, k] * 4.0,
            color=COLORS["secondary"], alpha=0.65, lw=1.5,
        )
    ax_operator.set_xlabel(r"$x$", fontsize=11)
    ax_operator.set_ylabel("function values", fontsize=11)
    ax_operator.set_title(
        r"Operator surrogate:  "
        r"$\mathcal{G} : \mathcal{U} \to \mathcal{V}$",
        fontsize=11,
    )
    handles = [
        Line2D([0], [0], color=COLORS["primary"], lw=1.5,
               label=r"input function $u(x)$"),
        Line2D([0], [0], color=COLORS["secondary"], lw=1.5,
               label=r"output function $v(x)$  (scaled)"),
    ]
    ax_operator.legend(handles=handles, fontsize=9, loc="upper right")
    ax_operator.set_xlim(0, 1)
    apply_style(ax_operator)


def plot_data_sampling_pipeline(fig, axes):
    """operator_learning_concept.qmd -> fig-data-sampling-pipeline.

    Four-panel pipeline: (a) continuous input functions, (b) encoded
    input codes (dots), (c) simulator output codes (dots), (d) decoded
    continuous output functions. Panels (b) and (c) are visually grouped
    by a rounded rectangle with a G^dagger arrow between them.

    Parameters
    ----------
    fig : matplotlib Figure
        Used to access settled axes positions and to add the grouping
        artists in figure coordinates.
    axes : array-like of Axes
        Length-4 array; axes[0]..axes[3] are panels (a) (b) (c) (d).
    """
    from ._style import COLORS, apply_style

    grid_fine = np.linspace(0, 1, 100)
    grid_in = np.linspace(0, 1, 25)
    grid_out = np.linspace(0, 1, 10)
    rng = np.random.default_rng(7)

    u_fine = sample_log_diffusion_field(
        grid_fine, 5, length_scale=0.15, rng=rng,
    )
    u_in = np.zeros((len(grid_in), 5))
    for k in range(5):
        u_in[:, k] = np.interp(grid_in, grid_fine, u_fine[:, k])
    v_fine = solve_elliptic_1d(u_fine, grid_fine)
    v_out = np.zeros((len(grid_out), 5))
    for k in range(5):
        v_out[:, k] = np.interp(grid_out, grid_fine, v_fine[:, k])

    palette = [
        COLORS["primary"], COLORS["secondary"], COLORS["accent"],
        COLORS["purple"], COLORS["reference"],
    ]

    # (a) continuous input functions — y-axis ticks on left
    ax = axes[0]
    for k in range(5):
        ax.plot(grid_fine, u_fine[:, k], color=palette[k], lw=1.4, alpha=0.85)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title(
        "(a) Sample input functions" "\n" r"from prior $\mu$",
        fontsize=10.5,
    )
    apply_style(ax)

    # (b) encode — dots only, no y-axis ticks
    ax = axes[1]
    for k in range(5):
        ax.plot(
            grid_in, u_in[:, k], color=palette[k],
            marker="o", ms=4, lw=0, alpha=0.95,
        )
    ax.set_xlabel(r"$x$")
    ax.set_title(
        r"(b) Encode  $U_i = \phi(u_i)$"
        "\n"
        r"(point evaluation at grid $X$)",
        fontsize=10.5,
    )
    ax.set_yticklabels([])
    apply_style(ax)

    # (c) simulator output — dots on coarser grid, no y-axis ticks
    ax = axes[2]
    for k in range(5):
        ax.plot(
            grid_out, v_out[:, k], color=palette[k],
            marker="s", ms=5, lw=0, alpha=0.95,
        )
    ax.set_xlabel(r"$x$")
    ax.set_title(
        r"(c) Simulator  $V_i \in \mathbb{R}^{d_{\mathrm{out}}}$"
        "\n"
        r"on output grid $Y$",
        fontsize=10.5,
    )
    ax.set_yticklabels([])
    apply_style(ax)

    # (d) decoded continuous output functions — y-axis ticks on right
    ax = axes[3]
    for k in range(5):
        ax.plot(
            grid_fine, v_fine[:, k], color=palette[k], lw=1.4, alpha=0.85,
        )
    ax.set_xlabel(r"$x$")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(r"$v(x)$")
    ax.set_title(
        r"(d) Decode  $v_i = \chi(V_i)$"
        "\n"
        r"continuous output functions",
        fontsize=10.5,
    )
    apply_style(ax)

    # ---- Group panels (b) and (c) with a box + arrow in figure coords ----
    fig.canvas.draw()
    pos_b = axes[1].get_position()
    pos_c = axes[2].get_position()

    margin_x = 0.012
    margin_top = 0.14
    margin_bot = 0.06
    box_x0 = pos_b.x0 - margin_x
    box_x1 = pos_c.x1 + margin_x
    box_y0 = min(pos_b.y0, pos_c.y0) - margin_bot
    box_y1 = max(pos_b.y1, pos_c.y1) + margin_top

    grouping_box = patches.FancyBboxPatch(
        (box_x0, box_y0),
        box_x1 - box_x0,
        box_y1 - box_y0,
        boxstyle="round,pad=0.005",
        linewidth=1.0,
        edgecolor=COLORS["gray"],
        facecolor="none",
        transform=fig.transFigure,
        zorder=0,
    )
    fig.add_artist(grouping_box)

    fig.text(
        (box_x0 + box_x1) / 2.0,
        box_y1 + 0.01,
        r"one training pair $(U_i, V_i)$",
        ha="center", va="bottom", fontsize=10, style="italic",
        color=COLORS["gray"],
        transform=fig.transFigure,
    )

    # Arrow between panels (b) and (c)
    y_mid = 0.5 * (pos_b.y0 + pos_b.y1)
    arrow_x0 = pos_b.x1 + 0.004
    arrow_x1 = pos_c.x0 - 0.004
    arrow = patches.FancyArrowPatch(
        (arrow_x0, y_mid),
        (arrow_x1, y_mid),
        arrowstyle="->",
        mutation_scale=14,
        color="black",
        lw=1.4,
        transform=fig.transFigure,
        zorder=5,
    )
    fig.add_artist(arrow)
    fig.text(
        0.5 * (arrow_x0 + arrow_x1),
        y_mid + 0.025,
        r"$\mathcal{G}^\dagger$",
        ha="center", va="bottom", fontsize=11, style="italic",
        transform=fig.transFigure,
    )


def plot_encode_regress_decode(ax):
    """operator_learning_concept.qmd -> fig-encode-regress-decode."""
    from ._style import COLORS

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.6, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color, ec="black"):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor=ec, lw=1.2,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
        )
        if label is not None:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.25, label,
                    ha="center", fontsize=10.5, style="italic")

    box(0.4, 4.0, 2.6, 1.4, r"$u \in \mathcal{U}$", color="#EAF3FB")
    box(9.0, 4.0, 2.6, 1.4, r"$v \in \mathcal{V}$", color="#FDEEDD")
    arrow(3.0, 4.7, 9.0, 4.7, label=r"$\mathcal{G}^\dagger$")

    box(0.4, 0.4, 2.6, 1.4, r"$U \in \mathbb{R}^{d_{\mathrm{in}}}$",
        color="#EAF3FB")
    box(9.0, 0.4, 2.6, 1.4, r"$V \in \mathbb{R}^{d_{\mathrm{out}}}$",
        color="#FDEEDD")
    arrow(3.0, 1.1, 9.0, 1.1, label=r"$\bar f$  (regress)")

    arrow(1.7, 4.0, 1.7, 1.8, label=r"$\phi$  (encode)")
    arrow(10.3, 1.8, 10.3, 4.0, label=r"$\chi$  (decode)")

    ax.text(
        6.0, -0.4,
        "Encode: linear functionals on $u$.   "
        "Regress: kernel / NN / spectral.   "
        "Decode: paired basis to $\\phi$.",
        ha="center", fontsize=9.5, style="italic", color=COLORS["gray"],
    )


def plot_grid_independence(ax):
    """operator_learning_concept.qmd -> fig-grid-independence."""
    from ._style import COLORS, apply_style

    rng = np.random.default_rng(11)
    grid_fine = np.linspace(0, 1, 200)
    u_fine = sample_log_diffusion_field(
        grid_fine, 1, length_scale=0.12, rng=rng,
    )[:, 0]

    for grid_n, color in [
        (20, COLORS["primary"]),
        (50, COLORS["secondary"]),
        (200, COLORS["accent"]),
    ]:
        x = np.linspace(0, 1, grid_n)
        u_sampled = np.interp(x, grid_fine, u_fine)
        ax.plot(
            x, u_sampled, color=color, lw=1.2, marker="o", ms=3,
            label=f"$d = {grid_n}$ grid points",
        )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title("One input function sampled at three resolutions",
                 fontsize=11)
    ax.legend(fontsize=9.5, loc="upper right")
    apply_style(ax)


# ---------------------------------------------------------------------------
# kerneloperator_usage.qmd
# ---------------------------------------------------------------------------

def plot_training_pairs(grid, u_train, v_train, axes, n_show=8):
    """kerneloperator_usage.qmd -> fig-training-pairs."""
    import matplotlib.pyplot as plt

    from ._style import apply_style

    colors = plt.cm.viridis(np.linspace(0, 0.8, n_show))
    ax_in, ax_out = axes
    for k in range(min(n_show, u_train.shape[1])):
        ax_in.plot(grid, u_train[:, k], color=colors[k], lw=1.2, alpha=0.9)
        ax_out.plot(grid, v_train[:, k], color=colors[k], lw=1.2, alpha=0.9)
    ax_in.set_xlabel(r"$x$")
    ax_in.set_ylabel(r"$u(x)$")
    ax_in.set_title("Input: log-diffusion field", fontsize=11)
    apply_style(ax_in)

    ax_out.set_xlabel(r"$x$")
    ax_out.set_ylabel(r"$v(x)$")
    ax_out.set_title(
        r"Output: PDE solution"
        "\n"
        r"$-(\,e^u v'\,)' = 1$,  $v(0)=v(1)=0$",
        fontsize=11,
    )
    apply_style(ax_out)


def plot_pca_spectra(sing_in, sing_out, ax_in, ax_out, threshold=0.95):
    """kerneloperator_usage.qmd -> fig-pca-spectra."""
    from ._style import COLORS, apply_style

    for ax, sing, label in [
        (ax_in, sing_in, "input field $u$"),
        (ax_out, sing_out, "output field $v$"),
    ]:
        var = sing ** 2
        cum = np.cumsum(var) / var.sum()
        nmax = len(sing)
        ax.plot(
            np.arange(1, nmax + 1), cum, color=COLORS["primary"], lw=2,
        )
        ax.axhline(threshold, ls="--", color=COLORS["reference"], lw=1)
        n_threshold = int(np.searchsorted(cum, threshold)) + 1
        ax.axvline(n_threshold, ls=":", color=COLORS["reference"], lw=1)
        ax.text(
            n_threshold + 0.5, 0.5,
            f"$d_{{\\mathrm{{codes}}}} = {n_threshold}$\n"
            f"at {int(threshold * 100)}%",
            fontsize=10, color=COLORS["reference"],
        )
        ax.set_xlabel("Number of PCA modes")
        ax.set_ylabel("Cumulative variance")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(1, min(nmax, 40))
        ax.set_ylim(0, 1.02)
        apply_style(ax)


def plot_function_predictions(
    grid, v_true, v_pred, v_std, axes, n_show=3,
):
    """kerneloperator_usage.qmd -> fig-function-predictions."""
    from ._style import COLORS, apply_style

    palette_colors = [
        COLORS["primary"], COLORS["secondary"], COLORS["accent"],
    ]
    for i, ax in enumerate(axes[:n_show]):
        ax.fill_between(
            grid,
            v_pred[:, i] - 2 * v_std[:, i],
            v_pred[:, i] + 2 * v_std[:, i],
            color=palette_colors[i], alpha=0.18,
            label=r"$\pm 2\sigma$",
        )
        ax.plot(grid, v_pred[:, i], color=palette_colors[i], lw=1.8,
                label="predicted mean")
        ax.plot(grid, v_true[:, i], "k--", lw=1.2, label="truth")
        ax.set_xlabel(r"$x$")
        if i == 0:
            ax.set_ylabel(r"$v(x)$")
        ax.set_title(f"Test sample {i + 1}", fontsize=11)
        apply_style(ax)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")


def plot_parity_and_calibration(
    v_test, v_pred, v_std, ax_parity, ax_calib,
):
    """kerneloperator_usage.qmd -> fig-parity-and-calibration."""
    from scipy.stats import norm

    from ._style import COLORS, apply_style

    true_flat = v_test.flatten()
    pred_flat = v_pred.flatten()
    std_flat = v_std.flatten()
    lo = min(true_flat.min(), pred_flat.min())
    hi = max(true_flat.max(), pred_flat.max())
    sc = ax_parity.scatter(
        true_flat, pred_flat, c=std_flat, cmap="viridis",
        s=6, alpha=0.6,
    )
    ax_parity.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    rel = np.linalg.norm(true_flat - pred_flat) / np.linalg.norm(true_flat)
    ax_parity.set_title(
        "Pointwise parity"
        "\n"
        f"rel. $L_2$ error = {rel:.2e}",
        fontsize=11,
    )
    ax_parity.set_xlabel("true $v(x)$")
    ax_parity.set_ylabel("predicted $v(x)$")
    apply_style(ax_parity)
    cb = ax_parity.figure.colorbar(
        sc, ax=ax_parity, fraction=0.046, pad=0.04,
    )
    cb.set_label(r"$\sigma^*(x)$", fontsize=9)

    nominals = np.linspace(0.05, 0.99, 19)
    z = norm.ppf(0.5 + nominals / 2.0)
    empirical = []
    for z_val in z:
        inside = np.abs(true_flat - pred_flat) <= z_val * std_flat
        empirical.append(inside.mean())
    empirical = np.array(empirical)
    ax_calib.fill_between(
        [0, 1], [0, 1], [1, 1], alpha=0.07, color=COLORS["accent"],
        label="under-confident",
    )
    ax_calib.fill_between(
        [0, 1], [0, 0], [0, 1], alpha=0.07, color=COLORS["reference"],
        label="over-confident",
    )
    ax_calib.plot([0, 1], [0, 1], "k--", lw=1)
    ax_calib.plot(
        nominals, empirical, "o-", color=COLORS["primary"], lw=1.5, ms=5,
        label="surrogate",
    )
    ax_calib.set_xlim(0, 1)
    ax_calib.set_ylim(0, 1)
    ax_calib.set_xlabel("Nominal coverage")
    ax_calib.set_ylabel("Empirical coverage")
    ax_calib.set_title("Calibration", fontsize=11)
    ax_calib.legend(fontsize=9, loc="lower right")
    apply_style(ax_calib)


def plot_convergence_with_ncodes(
    ncodes_list, errors, ax,
):
    """kerneloperator_usage.qmd -> fig-convergence-ncodes.

    Parameters
    ----------
    ncodes_list : list of int
        Number of input PCA modes for the x-axis.
    errors : list or dict
        If a list, plot a single curve (backwards compatible).
        If a dict mapping str labels to lists of errors, plot one
        curve per entry.
    ax : matplotlib Axes
    """
    from ._style import COLORS, apply_style

    if isinstance(errors, dict):
        palette = [
            COLORS["primary"], COLORS["secondary"], COLORS["accent"],
            COLORS["reference"], COLORS["purple"],
        ]
        for ii, (label, err) in enumerate(errors.items()):
            ax.semilogy(
                ncodes_list, err, "o-",
                color=palette[ii % len(palette)], lw=2, ms=6, label=label,
            )
        ax.legend(fontsize=10)
    else:
        ax.semilogy(
            ncodes_list, errors, "o-",
            color=COLORS["primary"], lw=2, ms=6,
        )
    ax.set_xlabel(r"Number of input PCA modes $d_{\mathrm{in}}$")
    ax.set_ylabel("Test relative $L_2$ error")
    ax.set_title("Encoder truncation controls error", fontsize=11)
    apply_style(ax)


def plot_convergence_with_N(N_list, errors, ax):
    """kerneloperator_usage.qmd -> fig-convergence-N."""
    from ._style import COLORS, apply_style

    ax.loglog(
        N_list, errors, "o-", color=COLORS["secondary"], lw=2, ms=6,
    )
    ref = errors[0] * (np.array(N_list) / N_list[0]) ** (-0.5)
    ax.loglog(
        N_list, ref, "k--", lw=1, alpha=0.6, label=r"$N^{-1/2}$ reference",
    )
    ax.set_xlabel("Number of training samples $N$")
    ax.set_ylabel("Test relative $L_2$ error")
    ax.set_title("More data, lower error", fontsize=11)
    ax.legend(fontsize=10)
    apply_style(ax)
