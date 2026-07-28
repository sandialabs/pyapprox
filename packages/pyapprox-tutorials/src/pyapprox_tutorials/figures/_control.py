"""Plotting functions for PDE control tutorials.

Covers: pde_control_usage.qmd
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.tri import Triangulation

from ._style import COLORS

# Concentration colormap following the pyapprox site convention:
# neon blue glow on a black background, zero = black.
NEON_CMAP = LinearSegmentedColormap.from_list(
    "pyapprox_neon",
    ["#000000", "#001a4d", "#0057d9", "#00b3ff", "#7df9ff", "#e8ffff"],
)
# Positivity alarm: anything below the zero level renders magenta.
NEON_CMAP.set_under("#ff00ff")

# Obstruction blocks of the obstructed-flow substrate (drawn as filled
# rectangles on every domain plot).
_BLOCKS = [
    ((4 / 7, 0.0), 1 / 7, 0.25),
    ((2 / 7, 0.25), 1 / 7, 0.25),
    ((4 / 7, 0.5), 1 / 7, 0.25),
]


def _draw_domain(ax, problem, extraction_rates=None, dark=False):
    """Blocks, zone outline, release marker, extraction-device markers.

    Without rates, devices are drawn as HOLLOW layout markers (where
    devices sit, none active). With rates, devices are filled circles
    sized proportionally to their rate with NO minimum size — a device
    at zero rate disappears, so marker area honestly reflects effort.
    ``dark`` switches marker/outline colors for black-background
    concentration panels.
    """
    accent = "white" if dark else COLORS["primary"]
    active_color = "#ff9f1c" if dark else COLORS["secondary"]
    star_color = "white" if dark else COLORS["purple"]
    for (x0, y0), width, height in _BLOCKS:
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), width, height, facecolor="0.55",
                edgecolor="0.3", zorder=3,
            )
        )
    outline = problem.zone_weight().outline_vertices()
    ax.plot(
        outline[0], outline[1], "--", color=COLORS["reference"],
        lw=1.8, zorder=4,
    )
    ax.plot(
        *problem.release_center(), marker="*", color=star_color,
        markersize=14, zorder=5,
    )
    centers = problem.actuator_centers()
    if extraction_rates is None:
        ax.plot(
            centers[0], centers[1], "o", markerfacecolor="none",
            markeredgecolor=accent, markersize=6, linestyle="none",
            zorder=5,
        )
    else:
        sizes = 18.0 * np.abs(extraction_rates) / max(
            np.abs(extraction_rates).max(), 1e-12
        )
        for kk in range(centers.shape[1]):
            ax.plot(
                centers[0, kk], centers[1, kk], "o",
                color=active_color, markersize=sizes[kk], zorder=5,
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def _triangulation(problem, bkd):
    """Degree-1 DOFs are mesh vertices (doflocs equals mesh.p); quad
    meshes are split into triangles for tricontourf, preserving vertex
    indices."""
    skfem_mesh = problem.basis().skfem_basis().mesh
    if hasattr(skfem_mesh, "to_meshtri"):
        skfem_mesh = skfem_mesh.to_meshtri()
    points = np.asarray(skfem_mesh.p)
    return Triangulation(points[0], points[1], np.asarray(skfem_mesh.t).T)


def plot_frozen_flow(problem, bkd, ax, density=1.4):
    """pde_control_usage.qmd -> fig-frozen-flow

    Streamlines of the frozen Navier-Stokes velocity colored by speed,
    threading the staggered blocks; zone, release, and actuators
    marked.
    """
    ngrid = 120
    grid_1d = np.linspace(0.0, 1.0, ngrid)
    grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    points = np.vstack([grid_x.ravel(), grid_y.ravel()])
    # Points inside the blocks are OUTSIDE the mesh: mask them BEFORE
    # probing (the element finder raises on them), with a small
    # inflation so near-boundary slivers do not trip it either.
    tol = 1e-9
    inside_block = np.zeros_like(grid_x, dtype=bool)
    for (x0, y0), width, height in _BLOCKS:
        inside_block |= (
            (grid_x >= x0 - tol)
            & (grid_x <= x0 + width + tol)
            & (grid_y >= y0 - tol)
            & (grid_y <= y0 + height + tol)
        )
    valid = ~inside_block.ravel()
    vel_flat = np.full((2, points.shape[1]), np.nan)
    vel_flat[:, valid] = problem.velocity()(points[:, valid])
    vel_x = vel_flat[0].reshape(ngrid, ngrid)
    vel_y = vel_flat[1].reshape(ngrid, ngrid)
    speed = np.hypot(vel_x, vel_y)
    stream = ax.streamplot(
        grid_1d, grid_1d, vel_x, vel_y, color=speed, cmap="viridis",
        density=density, linewidth=0.9, arrowsize=0.8,
    )
    _draw_domain(ax, problem)
    ax.set_title("Frozen flow: streamlines colored by speed")
    return stream


def plot_amplitudes(problem, amplitudes, ax_bar, ax_domain):
    """pde_control_usage.qmd -> fig-amplitudes

    Optimized extraction rates as a bar chart in domain-order beneath
    a mini domain sketch aligning bars to device positions.
    """
    centers = problem.actuator_centers()
    labels = problem.actuator_labels()
    order = np.argsort(centers[0])
    positions = np.arange(order.shape[0])
    ax_bar.bar(positions, amplitudes[order], color=COLORS["primary"])
    ax_bar.axhline(0.0, color="0.3", lw=0.8)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(
        [labels[ii].replace("_", "\n") for ii in order], fontsize=7
    )
    ax_bar.set_ylabel(r"$p_k$")
    _draw_domain(ax_domain, problem, extraction_rates=amplitudes)
    for rank, ii in enumerate(order):
        ax_domain.annotate(
            str(rank),
            (centers[0, ii], centers[1, ii]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )


def plot_convergence(histories, labels, gif_iteration, ax):
    """pde_control_usage.qmd -> fig-convergence

    J vs iteration per optimizer (log y), with a marker at the iterate
    whose field the bookend GIF displays.
    """
    for history, label in zip(histories, labels):
        ax.semilogy(np.arange(len(history)), history, "-o", label=label)
    ax.axvline(
        gif_iteration, color=COLORS["reference"], ls="--", lw=1.2,
        label="GIF iterate",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$J(p)$")
    ax.legend()


def save_control_gif(
    problem,
    bkd,
    sols_uncontrolled,
    sols_controlled,
    times,
    amplitudes,
    path,
    fps=8,
    max_frames=40,
    color_bound=None,
    cmap=NEON_CMAP,
    nlevels=41,
    gamma=1.0,
):
    """pde_control_usage.qmd -> the bookend GIF

    Side-by-side concentration evolution, "No control" vs "Optimized
    control", fixed colorbar across panels and frames; zone outlined
    on both. The LEFT panel shows the device layout as hollow markers
    (installed, inactive); the RIGHT panel draws devices sized by
    their extraction rate with no minimum size. Writes ``path``
    (.gif) plus first/last static PNGs.

    Returns the (first_png, last_png) paths.
    """
    sols_unc = bkd.to_numpy(sols_uncontrolled)
    sols_ctl = bkd.to_numpy(sols_controlled)
    times_np = bkd.to_numpy(times)
    ntimes = times_np.shape[0]
    frame_idx = np.unique(
        np.linspace(0, ntimes - 1, min(max_frames, ntimes)).astype(int)
    )
    tri = _triangulation(problem, bkd)
    # Concentration is nonnegative (proportional extraction preserves
    # the maximum principle), so the scale runs from zero = black; any
    # negative value saturates into the colormap's under-color — a
    # visual positivity alarm. ``color_bound`` caps the scale (values
    # beyond it saturate, shown by the colorbar arrow) — for e.g.
    # pulse releases whose brief injection spike would otherwise
    # compress the traveling puff into invisibility. Any matplotlib
    # colormap can be swapped in via ``cmap``. ``gamma`` < 1 applies a
    # power-law color scale (``PowerNorm``): a linear scale spends
    # nearly the whole colormap on the bright source region, leaving
    # dilute late-time tracer in the bottom band or two; gamma
    # brightens low concentrations while the single fixed colorbar
    # stays honest (tick VALUES are real, only their spacing bends).
    # Levels are gamma-spaced too, so each contour band spans an equal
    # color increment (band resolution concentrates where gamma puts
    # the color resolution).
    if color_bound is None:
        color_bound = max(sols_unc.max(), sols_ctl.max())
    levels = color_bound * np.linspace(0.0, 1.0, nlevels) ** (1.0 / gamma)
    norm = (
        None if gamma == 1.0
        else PowerNorm(gamma, vmin=0.0, vmax=color_bound)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))

    def _draw_frame(index):
        contours = None
        for ax, sols, title, amps in (
            (axes[0], sols_unc, "No control", None),
            (axes[1], sols_ctl, "Optimized control", amplitudes),
        ):
            ax.clear()
            contours = ax.tricontourf(
                tri, sols[:, index], levels=levels, cmap=cmap,
                extend="both", norm=norm,
            )
            _draw_domain(ax, problem, extraction_rates=amps, dark=True)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"$t = {times_np[index]:.1f}$")
        return contours

    contours = _draw_frame(frame_idx[0])
    # Fixed levels: one static colorbar serves every frame.
    fig.colorbar(contours, ax=axes, shrink=0.85, label="concentration $u$")

    first_png = str(path).replace(".gif", "_first.png")
    fig.savefig(first_png, dpi=110, bbox_inches="tight")

    animation = FuncAnimation(
        fig, _draw_frame, frames=frame_idx, interval=1000 / fps
    )
    animation.save(str(path), writer=PillowWriter(fps=fps))

    _draw_frame(frame_idx[-1])
    last_png = str(path).replace(".gif", "_last.png")
    fig.savefig(last_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return first_png, last_png
