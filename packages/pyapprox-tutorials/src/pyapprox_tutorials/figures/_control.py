"""Plotting functions for PDE control tutorials.

Covers: pde_control_usage.qmd
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import PowerNorm
from matplotlib.tri import Triangulation

from ._galerkin import _mp4_writer
from ._style import COLORS, NEON_CMAP

# Obstruction blocks of the obstructed-flow substrate (drawn as filled
# rectangles on every domain plot).
_BLOCKS = [
    ((4 / 7, 0.0), 1 / 7, 0.25),
    ((2 / 7, 0.25), 1 / 7, 0.25),
    ((4 / 7, 0.5), 1 / 7, 0.25),
]


def _draw_domain(
    ax, problem, extraction_rates=None, dark=False, size_ref=None,
    accent=None, filled=False,
):
    """Blocks, zone outline, release marker, extraction-device markers.

    Without rates, devices are drawn as HOLLOW layout markers (where
    devices sit, none active). With rates, devices are filled circles
    sized proportionally to their rate with NO minimum size — a device
    at zero rate disappears, so marker area honestly reflects effort.
    ``dark`` switches marker/outline colors for black-background
    concentration panels.

    ``accent`` overrides the hollow-marker and outline color, for
    panels whose background already spends the default. ``filled``
    fills the layout markers, for a busy background where a thin ring
    disappears; only for panels that show no rates at all, since where
    rates ARE drawn the hollow/filled distinction is what separates an
    installed device from a running one.

    ``size_ref`` is the rate that draws a full-size marker. It defaults
    to the largest rate in THIS call, which is right for a single
    figure but wrong across the frames of an animation: each frame
    would renormalize to its own maximum, so a device pulling 10% of
    peak effort in a busy frame would draw the same size as one pulling
    100% in a quiet frame — hiding exactly the variation a time-varying
    control is meant to show. Pass the maximum over all frames to make
    sizes comparable.
    """
    if accent is None:
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
            centers[0], centers[1], "o",
            markerfacecolor=accent if filled else "none",
            markeredgecolor=accent, markersize=7 if filled else 6,
            linestyle="none", zorder=5,
        )
    else:
        reference = (
            np.abs(extraction_rates).max() if size_ref is None
            else size_ref
        )
        sizes = 18.0 * np.abs(extraction_rates) / max(reference, 1e-12)
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

    Speed of the frozen Navier-Stokes velocity as a filled neon field,
    with streamlines threading the staggered blocks drawn over it in
    neon orange; zone, release, and actuators marked.

    Two channels, two encodings: magnitude is the field (zero = black,
    the house convention every scalar field in the series uses), and
    DIRECTION is the streamlines. Coloring the lines by speed instead
    would spend both channels on the same quantity and leave the
    direction to be inferred from line shape alone.
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
    # Speed is a magnitude, so the sequential house map applies with
    # zero = black as everywhere else in the series. Levels start at
    # EXACTLY zero and the range is not extended: NEON_CMAP paints
    # under-range values magenta as a negativity alarm, and a magnitude
    # can never trip it honestly. Extending the range would fire it on
    # the no-slip walls and the stagnation pockets behind the blocks
    # --- 7.7% of the domain here --- reporting an error where there is
    # only slow water. Block interiors are NaN and stay unpainted, so
    # the blocks read as holes rather than as still water.
    field = ax.contourf(
        grid_x, grid_y, speed, cmap=NEON_CMAP,
        levels=np.linspace(0.0, float(np.nanmax(speed)), 41),
    )
    # Streamlines over a dark field need the light accent; the neon blue
    # they used to carry is invisible against the low end of its own
    # colormap.
    ax.streamplot(
        grid_1d, grid_1d, vel_x, vel_y, color="#ff9f1c",
        density=density, linewidth=0.8, arrowsize=0.8,
    )
    # Magenta for the device markers: it is the one hue the neon map
    # never produces, so the markers cannot be mistaken for the field
    # they sit on, and the orange is already spent on the streamlines.
    _draw_domain(ax, problem, dark=True, accent="#ff00ff", filled=True)
    ax.set_title("Frozen flow: speed, with streamlines")
    return field


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
    amplitudes_at=None,
    size_ref=None,
    bitrate=1400,
):
    """pde_control_usage.qmd -> the bookend GIF

    Side-by-side concentration evolution, "No control" vs "Optimized
    control", fixed colorbar across panels and frames; zone outlined
    on both. The LEFT panel shows the device layout as hollow markers
    (installed, inactive); the RIGHT panel draws devices sized by
    their extraction rate with no minimum size. Writes ``path`` --- an
    ``.mp4`` (H.264, the house format) or a ``.gif``, chosen by the
    extension --- plus first/last static PNGs.

    ``amplitudes`` are the steady rates, drawn identically on every
    frame. For a control that varies in time, pass ``amplitudes_at``,
    a callable ``time -> rates``, and the markers pulse with the
    schedule; ``amplitudes`` is then ignored. Supply ``size_ref`` (the
    largest rate over the whole horizon) so marker sizes stay
    comparable from frame to frame and between the two GIFs — without
    it each frame renormalizes to its own maximum and the schedule
    becomes invisible.

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
        frame_amps = (
            amplitudes if amplitudes_at is None
            else amplitudes_at(float(times_np[index]))
        )
        for ax, sols, title, amps in (
            (axes[0], sols_unc, "No control", None),
            (axes[1], sols_ctl, "Optimized control", frame_amps),
        ):
            ax.clear()
            contours = ax.tricontourf(
                tri, sols[:, index], levels=levels, cmap=cmap,
                extend="both", norm=norm,
            )
            _draw_domain(
                ax, problem, extraction_rates=amps, dark=True,
                size_ref=size_ref,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"$t = {times_np[index]:.1f}$")
        return contours

    contours = _draw_frame(frame_idx[0])
    # Fixed levels: one static colorbar serves every frame.
    fig.colorbar(contours, ax=axes, shrink=0.85, label="concentration $u$")

    stem = str(path).rsplit(".", 1)[0]
    first_png = f"{stem}_first.png"
    fig.savefig(first_png, dpi=110, bbox_inches="tight")

    animation = FuncAnimation(
        fig, _draw_frame, frames=frame_idx, interval=1000 / fps
    )
    # H.264 for an .mp4 target (the house format for tutorial video --- a
    # 40-frame GIF of two contour panels is an order of magnitude
    # larger), Pillow for a .gif.
    writer = (
        _mp4_writer(fps, bitrate) if str(path).endswith(".mp4")
        else PillowWriter(fps=fps)
    )
    animation.save(str(path), writer=writer)

    _draw_frame(frame_idx[-1])
    last_png = f"{stem}_last.png"
    fig.savefig(last_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return first_png, last_png
