"""Plotting and measurement helpers for the Galerkin ADR usage tutorial.

Covers: galerkin_adr_usage.qmd

The convergence study measures error in a true :math:`L^2(\\Omega)` norm,
evaluated by element quadrature. A root-mean-square over nodal values is
cheaper but superconverges for degree-2 elements --- it reports a rate
near four where the :math:`L^2` theory predicts three --- so it cannot be
checked against the theoretical order.
"""

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

from ._style import COLORS, NEON_CMAP


def triangulation(basis):
    """Triangulate a degree-1 basis for tricontourf.

    Degree-1 DOFs are mesh vertices (doflocs equals mesh.p); quad meshes
    are split into triangles, preserving vertex indices. Higher-degree
    bases carry DOFs that are not mesh vertices, so plotting those
    requires interpolation rather than this shortcut.
    """
    skfem_mesh = basis.skfem_basis().mesh
    if hasattr(skfem_mesh, "to_meshtri"):
        skfem_mesh = skfem_mesh.to_meshtri()
    points = np.asarray(skfem_mesh.p)
    return Triangulation(points[0], points[1], np.asarray(skfem_mesh.t).T)


def l2_error(basis, solution, exact_func):
    """Return the L2 norm of the discretization error.

    Computes :math:`\\left(\\int_\\Omega (u_h - u)^2\\right)^{1/2}` by
    element quadrature, so the result is a norm of the error field rather
    than a sample of it at DOF locations.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis the solution was computed on.
    solution : array
        Solution DOF values, shape (ndofs,).
    exact_func : Callable
        Exact solution, taking coordinates of shape (ndim, npts).
    """
    import skfem

    skfem_basis = basis.skfem_basis()

    @skfem.Functional
    def squared_error(w):
        # w.x is a skfem DiscreteField of shape (ndim, nelems, nquad);
        # the manufactured solution callable wants (ndim, npts).
        pts = np.asarray(w.x)
        exact = np.asarray(
            exact_func(pts.reshape(pts.shape[0], -1))
        ).reshape(pts.shape[1:])
        return (w["uh"] - exact) ** 2

    integral = squared_error.assemble(
        skfem_basis, uh=skfem_basis.interpolate(np.asarray(solution))
    )
    return float(np.sqrt(integral))


def plot_mesh_and_field(basis, solution, bkd, axes):
    """galerkin_adr_usage.qmd -> fig-mesh-field

    The mesh on the left, the steady solution it produced on the right.
    """
    tri = triangulation(basis)
    values = bkd.to_numpy(solution)

    axes[0].triplot(tri, color=COLORS["primary"], linewidth=0.5)
    axes[0].set_title("Mesh")

    contours = axes[1].tricontourf(tri, values, levels=40, cmap=NEON_CMAP)
    axes[1].get_figure().colorbar(contours, ax=axes[1])
    axes[1].set_title("Steady solution")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")


def plot_bc_triptych(bases, solutions, labels, bkd, axes):
    """galerkin_adr_usage.qmd -> fig-bc-swap

    The same physics under three boundary-condition combinations, on a
    shared color scale so the boundary behavior is what differs.
    """
    values = [bkd.to_numpy(sol) for sol in solutions]
    vmin = min(v.min() for v in values)
    vmax = max(v.max() for v in values)
    levels = np.linspace(vmin, vmax, 40)

    for ax, basis, vals, label in zip(axes, bases, values, labels):
        contours = ax.tricontourf(
            triangulation(basis), vals, levels=levels, cmap=NEON_CMAP
        )
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[-1].get_figure().colorbar(contours, ax=axes, shrink=0.8)


def plot_spatial_convergence(mesh_sizes, errors_by_degree, ax):
    """galerkin_adr_usage.qmd -> fig-spatial-convergence

    Measured L2 error against element size, with reference slopes for the
    theoretical order of each element degree.
    """
    hs = 1.0 / np.asarray(mesh_sizes, dtype=float)
    palette = [COLORS["primary"], COLORS["secondary"]]

    for (degree, errors), color in zip(sorted(errors_by_degree.items()),
                                       palette):
        errs = np.asarray(errors, dtype=float)
        ax.loglog(hs, errs, "o-", color=color, label=f"degree {degree}")
        # Reference slope anchored at the coarsest mesh, then shifted down
        # a little: the measured rate matches theory closely enough that an
        # unshifted guide would hide underneath the data.
        order = degree + 1
        ax.loglog(
            hs, 0.35 * errs[0] * (hs / hs[0]) ** order, "--",
            color=color, alpha=0.9, label=f"$h^{order}$ (slope)",
        )

    ax.set_xlabel("element size $h$")
    ax.set_ylabel(r"$\|u_h - u\|_{L^2}$")
    ax.legend()
    ax.grid(True, alpha=0.2, which="both")


def convergence_rates(errors):
    """Return observed orders from errors on successively halved meshes."""
    errs = np.asarray(errors, dtype=float)
    return np.log2(errs[:-1] / errs[1:])


# Obstacle rectangles of the obstructed-flow domain, as
# ((x0, y0), width, height). Points inside them lie outside the mesh.
_BLOCKS = [
    ((4 / 7, 0.0), 1 / 7, 0.25),
    ((2 / 7, 0.25), 1 / 7, 0.25),
    ((4 / 7, 0.5), 1 / 7, 0.25),
]


def plot_velocity_field(velocity, ax, ngrid=44, density=1.3):
    """galerkin_adr_usage.qmd -> fig-velocity

    Streamlines of the flow, colored by speed. Points inside the blocks
    are outside the mesh, so they are masked before probing: the element
    finder raises on them otherwise.
    """
    grid_1d = np.linspace(0.0, 1.0, ngrid)
    grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    points = np.vstack([grid_x.ravel(), grid_y.ravel()])

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

    values = np.full((2, points.shape[1]), np.nan)
    values[:, valid] = np.asarray(velocity(points[:, valid]))
    vel_x = values[0].reshape(ngrid, ngrid)
    vel_y = values[1].reshape(ngrid, ngrid)
    speed = np.hypot(vel_x, vel_y)

    # Speed is non-negative, so the sequential house map applies; viridis
    # here was the last off-palette colormap in the series.
    stream = ax.streamplot(
        grid_1d, grid_1d, vel_x, vel_y, color=speed, cmap=NEON_CMAP,
        density=density, linewidth=0.9, arrowsize=0.8,
    )
    for (x0, y0), width, height in _BLOCKS:
        ax.add_patch(
            Rectangle((x0, y0), width, height, facecolor="0.55",
                      edgecolor="0.3", zorder=3)
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return stream


def plot_roundoff_errors(mesh_sizes, errors_by_label, ax):
    """mms_verification_usage.qmd -> fig-t1-roundoff

    Tier-1 errors against mesh size. An exactly representable solution
    produces a flat line at machine precision: refining cannot improve an
    error that is already round-off, which is the visual signature the
    tier checks for. The dotted reference marks double-precision epsilon.
    """
    hs = 1.0 / np.asarray(mesh_sizes, dtype=float)
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    for (label, errors), color in zip(errors_by_label.items(), palette):
        ax.loglog(hs, np.asarray(errors, dtype=float), "o-", color=color,
                  label=label)

    ax.axhline(np.finfo(float).eps, color=COLORS["gray"], linestyle=":",
               label=r"machine $\epsilon$")
    # Discretization error is absent, so the scale is pure round-off; fix
    # the limits or matplotlib zooms into meaningless noise.
    ax.set_ylim(1e-18, 1e-12)
    ax.set_xlabel("element size $h$")
    ax.set_ylabel(r"$\|u_h - u\|_{L^2}$")
    ax.legend()
    ax.grid(True, alpha=0.2, which="both")


def plot_field_realizations(basis, fields, solutions, bkd, axes, decades=3):
    """parameterized_forward_usage.qmd -> fig-realizations

    Diffusivity realizations on the top row, the concentrations they
    produce on the bottom. Each ROW shares a color scale so panels are
    comparable across realizations; the rows do not share one with each
    other because they are different quantities.

    The diffusivity row is linear --- the draws differ by a factor of
    about three, which a linear scale shows plainly. The concentration
    row is logarithmic: those fields are dominated by the bright core at
    the release, and on a linear scale that core swamps the differences
    the diffusivity produces, rendering three visibly distinct inputs as
    three near-identical pictures.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator

    tri = triangulation(basis)
    field_values = [np.asarray(bkd.to_numpy(f)) for f in fields]
    solution_values = [np.asarray(bkd.to_numpy(s)) for s in solutions]

    levels = np.linspace(
        min(v.min() for v in field_values),
        max(v.max() for v in field_values),
        40,
    )
    for col, vals in enumerate(field_values):
        contours = axes[0, col].tricontourf(
            tri, vals, levels=levels, cmap=NEON_CMAP
        )
        axes[0, col].set_aspect("equal")
        axes[0, col].set_title(f"realization {col + 1}")
    bar = axes[0, -1].get_figure().colorbar(
        contours, ax=list(axes[0, :]), shrink=0.85
    )
    bar.set_label(r"$\kappa$")

    vmax = max(v.max() for v in solution_values)
    floor = vmax * 10.0 ** (-decades)
    log_levels = np.logspace(np.log10(floor), np.log10(vmax), 60)
    # Below-range means "very dilute" here, not the positivity alarm the
    # shared palette reserves magenta for.
    cmap = NEON_CMAP.copy()
    cmap.set_under(cmap(0.0))
    for col, vals in enumerate(solution_values):
        contours = axes[1, col].tricontourf(
            tri, np.clip(vals, floor, None), levels=log_levels,
            norm=LogNorm(vmin=floor, vmax=vmax), cmap=cmap,
        )
        axes[1, col].set_aspect("equal")
        axes[1, col].set_xlabel("$x$")
    bar = axes[1, -1].get_figure().colorbar(
        contours, ax=list(axes[1, :]), shrink=0.85,
        ticks=LogLocator(base=10),
    )
    bar.set_label("$u$")

    for row in (0, 1):
        axes[row, 0].set_ylabel("$y$")


def plot_qoi_histogram(values, ax):
    """parameterized_forward_usage.qmd -> fig-qoi-histogram

    Push-forward distribution of a scalar output, with the sample mean
    and a one-standard-deviation band marked.
    """
    values = np.asarray(values)
    mean = values.mean()
    std = values.std(ddof=1)

    ax.hist(values, bins=24, color=COLORS["primary"], alpha=0.75,
            edgecolor="white")
    ax.axvline(mean, color=COLORS["reference"], linewidth=2,
               label=f"mean = {mean:.3g}")
    ax.axvspan(mean - std, mean + std, color=COLORS["secondary"],
               alpha=0.18, label=rf"$\pm$ 1 std = {std:.2g}")

    ax.set_xlabel("downstream concentration")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.2)


def plot_concentration_comparison(basis, solutions, titles, bkd, axes,
                                  decades=3):
    """galerkin_adr_usage.qmd -> fig-coupled

    Two concentration fields on a SHARED LOGARITHMIC color scale.

    Shared, because separate scales would rescale each panel to its own
    peak and hide the very difference being shown. Logarithmic, because
    the interesting contrast is in the dilute tail --- how far material
    reaches --- which spans several decades and which a linear scale
    renders as uniform black.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator

    values = [np.asarray(bkd.to_numpy(sol)) for sol in solutions]
    vmax = max(v.max() for v in values)
    floor = vmax * 10.0 ** (-decades)
    levels = np.logspace(np.log10(floor), np.log10(vmax), 60)

    # The shared palette flags below-range values in magenta as a
    # positivity alarm; here below-range only means "very dilute", so the
    # under-color is muted to the darkest level instead.
    cmap = NEON_CMAP.copy()
    cmap.set_under(cmap(0.0))

    tri = triangulation(basis)
    for ax, vals, title in zip(axes, values, titles):
        contours = ax.tricontourf(
            tri, np.clip(vals, floor, None), levels=levels,
            norm=LogNorm(vmin=floor, vmax=vmax), cmap=cmap,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")

    bar = axes[-1].get_figure().colorbar(
        contours, ax=list(axes), shrink=0.85, ticks=LogLocator(base=10)
    )
    bar.set_label("concentration")


def vorticity(vel_basis, vel_dofs, scalar_basis):
    """Recover vorticity from a velocity field by L2 projection.

    Vorticity ``w = dv/dx - du/dy`` is a DERIVATIVE of the discrete
    velocity, so it does not live in the velocity space: differentiating
    a P2 field gives something discontinuous across element boundaries.
    The variationally consistent recovery solves

        (w, q) = (dv/dx - du/dy, q)    for all test functions q

    i.e. a mass-matrix solve against the weak curl. This uses the
    basis's own quadrature and shape-function gradients, so the P2
    velocity is differentiated exactly rather than being reduced to
    vertex values and differenced by hand.

    Parameters
    ----------
    vel_basis : skfem Basis
        Vector basis the velocity dofs belong to (e.g. P2 vector).
    vel_dofs : array
        Velocity coefficients, length ``vel_basis.N``.
    scalar_basis : skfem Basis
        Scalar basis to project onto (e.g. P1). Must be built on the
        same mesh; its quadrature is used for the assembly.

    Returns
    -------
    array
        Nodal vorticity, length ``scalar_basis.N``.
    """
    from skfem import BilinearForm, LinearForm, asm, solve

    @BilinearForm
    def _mass(u, v, w):
        return u * v

    @LinearForm
    def _weak_curl(q, w):
        # w["uh"].grad[i][j] = d u_i / d x_j
        return (w["uh"].grad[1][0] - w["uh"].grad[0][1]) * q

    # The projection basis must integrate on the SAME quadrature points
    # as the velocity, or skfem rejects the interpolated field. Rebuild
    # it here rather than requiring every caller to know that.
    from skfem import Basis
    quad_basis = Basis(
        scalar_basis.mesh, scalar_basis.elem, quadrature=vel_basis.quadrature
    )
    uh = vel_basis.interpolate(np.asarray(vel_dofs))
    return solve(asm(_mass, quad_basis),
                 asm(_weak_curl, quad_basis, uh=uh))


def vorticity_animation(vel_basis, scalar_basis, solutions, times, geometry,
                        path, fps=12, stride=2, nrefs=2, bitrate=900,
                        width_in=12.0, xmax_diameters=13.0):
    """galerkin_transient_usage.qmd -> the vortex-shedding animation.

    Writes an MP4 (H.264, web-playable) of the vorticity field over a
    trajectory. MP4 rather than GIF: the same clip is roughly 17x
    smaller and does not band the smooth colour gradients.

    Frames are drawn one at a time rather than accumulated, so peak
    memory is a single frame regardless of trajectory length.

    Parameters
    ----------
    vel_basis, scalar_basis : skfem Basis
        Vector velocity basis and the scalar basis to recover vorticity
        onto; see :func:`vorticity`.
    solutions : array (nstates, ntimes)
        Trajectory. Only the velocity block is read.
    times : array (ntimes,)
    geometry : dict
        ``L``, ``H``, ``cx``, ``cy``, ``R`` -- domain and cylinder.
    path : str
        Output ``.mp4``.
    nrefs : int
        Display refinement. Vorticity is recovered on ``scalar_basis``
        and then subdivided for plotting; without this the picture is
        piecewise-linear on the original elements and looks faceted.

    Returns
    -------
    str
        ``path``, so a Quarto cell can hand it straight to an embed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from ._style import NEON_DIVERGING

    cx, cy, R = geometry["cx"], geometry["cy"], geometry["R"]
    H = geometry["H"]
    xmax = min(geometry["L"], cx + xmax_diameters)
    nvel = vel_basis.N
    frames = np.arange(0, len(times), stride)

    def frame_field(index):
        nodal = vorticity(vel_basis, solutions[:nvel, index], scalar_basis)
        refined, values = scalar_basis.refinterp(nodal, nrefs=nrefs)
        tri = Triangulation(refined.p[0], refined.p[1], refined.t.T)
        xc = refined.p[0][refined.t].mean(axis=0)
        yc = refined.p[1][refined.t].mean(axis=0)
        tri.set_mask(np.hypot(xc - cx, yc - cy) < R * 1.001)
        return tri, values

    # Colour scale from a sample of frames: the full trajectory would
    # mean holding every field in memory at once.
    sample = [frame_field(k)[1] for k in frames[::max(1, len(frames) // 8)]]
    vmax = float(np.median([np.percentile(np.abs(v), 97) for v in sample]))
    levels = np.linspace(-vmax, vmax, 41)

    fontsize = max(6.5, 11.0 * width_in / 13.0)
    fig, ax = plt.subplots(
        figsize=(width_in, width_in * H / xmax), facecolor="black"
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.005)

    def draw(i):
        ax.clear()
        tri, values = frame_field(frames[i])
        ax.tricontourf(
            tri, values, levels=levels, cmap=NEON_DIVERGING, extend="both"
        )
        ax.add_patch(
            plt.Circle((cx, cy), R, facecolor="0.12", edgecolor="0.75",
                       lw=1.3, zorder=6)
        )
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Vorticity   t = {times[frames[i]]:.1f}",
            color="white", fontsize=fontsize,
        )
        return []

    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(
        path,
        writer=_mp4_writer(fps, bitrate),
        savefig_kwargs={"facecolor": "black"},
    )
    plt.close(fig)
    return path


def _mp4_writer(fps, bitrate):
    """FFMpegWriter configured for web playback.

    Even dimensions are required by H.264; ``faststart`` puts the
    metadata first so a browser can begin playing before the whole file
    has downloaded.

    The encoder comes from ``imageio-ffmpeg``, which bundles an ffmpeg
    binary -- CI runners have none. It sits in this package's
    ``docs-build`` extra rather than its base dependencies, so importing
    the figure helpers does not pull a large wheel. Every path that
    encodes goes through here, which is why this is the only place the
    dependency is checked.
    """
    import matplotlib
    from matplotlib.animation import FFMpegWriter

    try:
        import imageio_ffmpeg
    except ImportError as err:
        # pyapprox.util.import_optional_dependency is not used here: it
        # names pyapprox in its install hint, and this extra is on
        # pyapprox-tutorials.
        raise ImportError(
            "Tutorial animations require the optional dependency "
            "'imageio-ffmpeg'. Install it with: "
            "pip install 'pyapprox-tutorials[docs-build]'"
        ) from err

    matplotlib.rcParams["animation.ffmpeg_path"] = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )
    return FFMpegWriter(
        fps=fps, bitrate=bitrate, codec="libx264",
        extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-profile:v", "baseline",
                    "-movflags", "+faststart"],
    )


def concentration_animation(basis, solutions, times, path, bkd,
                            fps=12, stride=1, bitrate=900, nlevels=41,
                            label="concentration $u$"):
    """galerkin_transient_usage.qmd -> the transport-evolution animation.

    A scalar field on a degree-1 basis, drawn with a fixed colour scale
    so brightness is comparable across frames -- a per-frame scale would
    make a decaying field look constant.

    Frames are drawn one at a time rather than accumulated, so peak
    memory is a single frame regardless of trajectory length.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Degree-1 basis the solution lives on.
    solutions : array (ndofs, ntimes)
    times : array (ntimes,)
    path : str
        Output ``.mp4``.

    Returns
    -------
    str
        ``path``, so a Quarto cell can hand it straight to an embed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # NEON_CMAP renders anything below the lowest level in magenta -- a
    # deliberate alarm for genuinely negative concentration. A discrete
    # solution dips to round-off below zero (order 1e-11 here), which
    # would speckle the whole domain magenta for no physical reason, so
    # clip at zero and let the alarm mean what it is for.
    values = np.clip(bkd.to_numpy(solutions), 0.0, None)
    times_np = bkd.to_numpy(times)
    frames = np.arange(0, len(times_np), stride)
    tri = triangulation(basis)

    vmax = float(np.percentile(np.abs(values), 99.5))
    levels = np.linspace(0.0, vmax, nlevels)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    def draw(i):
        ax.clear()
        contours = ax.tricontourf(
            tri, values[:, frames[i]], levels=levels,
            cmap=NEON_CMAP, extend="both",
        )
        for (x0, y0), width, height in _BLOCKS:
            ax.add_patch(
                Rectangle((x0, y0), width, height, facecolor="0.55",
                          edgecolor="0.3", zorder=3)
            )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$t = {times_np[frames[i]]:.2f}$")
        return contours

    contours = draw(0)
    fig.colorbar(contours, ax=ax, shrink=0.85, label=label)
    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(
        path, writer=_mp4_writer(fps, bitrate)
    )
    plt.close(fig)
    return path
