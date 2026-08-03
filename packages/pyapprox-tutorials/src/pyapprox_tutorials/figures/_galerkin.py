"""Plotting and measurement helpers for the Galerkin ADR usage tutorial.

Covers: galerkin_adr_usage.qmd

The convergence study measures error in a true :math:`L^2(\\Omega)` norm,
evaluated by element quadrature. A root-mean-square over nodal values is
cheaper but superconverges for degree-2 elements --- it reports a rate
near four where the :math:`L^2` theory predicts three --- so it cannot be
checked against the theoretical order.
"""

import numpy as np
from matplotlib.colors import ListedColormap
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


def plot_rows_to_columns(jacobian, constrained, ax_pair, bkd):
    """adjoint_steady_pde_concept.qmd -> fig-rows-to-columns

    A small Jacobian beside its transpose, with the constrained degrees
    of freedom marked. A constraint ROW of the forward matrix is a row
    of the identity; transposing turns it into a COLUMN, and the
    couplings that lived in the constrained COLUMN become a row. That is
    why forward quantities are corrected by zeroing rows and transposed
    ones by zeroing columns.

    Plots the sparsity pattern rather than the magnitudes. A constraint
    row holds a single 1.0 among zeros while the interior couplings are
    far larger, so a magnitude scale renders those rows as near-blank ---
    reading as absent structure when they are the most structured rows
    present.
    """
    import scipy.sparse as sp

    dense = sp.csr_matrix(bkd.to_numpy(jacobian)).toarray()
    nonzero = (np.abs(dense) > 0.0).astype(float)
    idx = np.asarray(bkd.to_numpy(constrained), dtype=int)

    for ax, matrix, title, mark in (
        (ax_pair[0], nonzero, r"$J$: constraint ROWS", "row"),
        (ax_pair[1], nonzero.T, r"$J^{T}$: constraint COLUMNS", "col"),
    ):
        ax.imshow(matrix, cmap=NEON_CMAP, vmin=0.0, vmax=1.0)
        for d in idx:
            if mark == "row":
                ax.axhline(d, color=COLORS["secondary"], lw=2.5, alpha=0.75)
            else:
                ax.axvline(d, color=COLORS["secondary"], lw=2.5, alpha=0.75)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_forward_and_adjoint(basis, state, adjoint, bkd, axes, probe_xy):
    """adjoint_steady_pde_concept.qmd -> fig-forward-adjoint

    The forward solution beside the adjoint, with the sensor marked on
    both. They answer different questions and peak in different places:
    the concentration is largest at the release and spreads downstream,
    while the adjoint is largest AT the sensor and decays upstream. The
    sensor is the only feature the two pictures share.

    Both use the sequential map. The adjoint here is one-signed apart
    from the constrained dofs, whose reaction is 0.3% of the range --
    a diverging map would spend half its span on values that are
    indistinguishable from zero, and its near-white extreme would hide
    the sensor marker exactly where the adjoint peaks.
    """
    forward = bkd.to_numpy(state)
    lam = bkd.to_numpy(adjoint)
    tri = triangulation(basis)

    fwd_max = float(np.percentile(forward, 99.5))
    contours_f = axes[0].tricontourf(
        tri, np.clip(forward, 0.0, None),
        levels=np.linspace(0.0, fwd_max, 41), cmap=NEON_CMAP, extend="max",
    )
    axes[0].set_title("Forward: where the contaminant is")

    lam_max = float(np.percentile(lam, 99.5))
    contours_a = axes[1].tricontourf(
        tri, np.clip(lam, 0.0, None),
        levels=np.linspace(0.0, lam_max, 41), cmap=NEON_CMAP, extend="max",
    )
    axes[1].set_title("Adjoint: what the sensor can see")

    for ax in axes:
        for (x0, y0), width, height in _BLOCKS:
            ax.add_patch(
                Rectangle((x0, y0), width, height, facecolor="0.55",
                          edgecolor="0.3", zorder=3)
            )
        # The sensor, marked identically on both so the eye can compare.
        # Orange rather than white: the map runs to near-white at its top
        # end, and the adjoint peaks AT the sensor, so a white marker
        # vanishes exactly where it is most needed.
        ax.plot(*probe_xy, "o", markersize=11, markerfacecolor="none",
                markeredgecolor=COLORS["secondary"], markeredgewidth=2.2,
                zorder=6)
        # Offset left: the sensor sits near the right edge of the domain,
        # so a rightward label runs off the axes.
        ax.annotate("sensor", probe_xy, textcoords="offset points",
                    xytext=(-12, 12), ha="right",
                    color=COLORS["secondary"],
                    fontsize=9, fontweight="bold", zorder=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    return contours_f, contours_a


def plot_temporal_convergence(deltats, errors_by_case, ax):
    """adjoint_transient_pde_concept.qmd -> fig-temporal-convergence

    Error in the quantity of interest AND in its gradient against time
    step, for two schemes. The point of plotting both together is that
    the gradient tracks the scheme's order rather than losing one:
    the discrete adjoint differentiates the discrete solve exactly, so
    it inherits the scheme's temporal accuracy.

    Parameters
    ----------
    deltats : array_like
        Time steps, one per measurement.
    errors_by_case : dict
        Maps ``(scheme_label, quantity_label)`` to an error sequence.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    """
    dts = np.asarray(deltats, dtype=float)
    styles = {"$Q$": "o-", r"$\nabla Q$": "s--"}
    # Fixed per scheme, so the slope guides below cannot end up a
    # different color from the curves they belong to.
    palette = [COLORS["primary"], COLORS["secondary"]]
    schemes = list(dict.fromkeys(key[0] for key in errors_by_case))
    colors = {
        scheme: palette[ii % len(palette)]
        for ii, scheme in enumerate(schemes)
    }
    for (scheme, quantity), errs in errors_by_case.items():
        ax.loglog(
            dts, np.asarray(errs, dtype=float),
            styles.get(quantity, "o-"), color=colors[scheme],
            label=f"{scheme}, {quantity}",
        )

    # Reference slopes anchored at the coarsest step of each scheme's Q
    # curve, offset down so they do not hide under the data. Each guide
    # takes its scheme's color: two same-colored dotted lines of
    # different slope are not tellable apart in the legend.
    for scheme, order in (("backward Euler", 1), ("Crank-Nicolson", 2)):
        key = (scheme, "$Q$")
        if key not in errors_by_case:
            continue
        base = float(np.asarray(errors_by_case[key], dtype=float)[0])
        ax.loglog(
            dts, 0.3 * base * (dts / dts[0]) ** order, ":",
            color=colors.get(scheme, "0.45"), alpha=0.8,
            label=rf"$\Delta t^{order}$ (slope)",
        )

    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=8)


def plot_mass_coupling(mass, constrained, coords, axes):
    """adjoint_transient_pde_concept.qmd -> fig-mass-coupling

    Left: the mass matrix sparsity, with the constrained rows and
    columns marked. Right: the mesh nodes, distinguishing constrained
    nodes from the interior nodes that couple to them through
    :math:`M_{id}`.

    The pairing is the argument: :math:`M_{id}` is not a boundary
    artifact confined to the boundary, it reaches one layer of interior
    nodes inward, and those are the nodes whose evolution feels the
    boundary's rate of change.
    """
    dense = np.asarray(mass)
    idx = np.asarray(constrained, dtype=int)
    interior = np.array(
        [k for k in range(dense.shape[0]) if k not in set(idx.tolist())]
    )

    # Three-level map rather than gridlines over the whole matrix: with
    # this many constrained dofs, ruled lines cover more of the picture
    # than they annotate. 0 = zero, 1 = interior-interior nonzero,
    # 2 = a nonzero in a constrained row or column.
    pattern = (np.abs(dense) > 1e-14).astype(float)
    touches = np.zeros_like(pattern, dtype=bool)
    touches[idx, :] = True
    touches[:, idx] = True
    pattern[(pattern > 0) & touches] = 2.0
    axes[0].imshow(
        pattern,
        cmap=ListedColormap(["black", "#7df9ff", COLORS["secondary"]]),
        vmin=0.0, vmax=2.0, interpolation="nearest",
    )
    axes[0].set_title(
        "$M$: nonzeros (orange touches a constrained dof)", fontsize=10
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    coupling = np.abs(dense[np.ix_(interior, idx)]) > 1e-14
    couples = interior[coupling.any(axis=1)]
    isolated = interior[~coupling.any(axis=1)]

    pts = np.asarray(coords)
    axes[1].plot(
        pts[0, isolated], pts[1, isolated], "o", markersize=6,
        color="0.75", label="interior, no coupling",
    )
    axes[1].plot(
        pts[0, couples], pts[1, couples], "o", markersize=7,
        color=COLORS["primary"], label="interior, couples to boundary",
    )
    axes[1].plot(
        pts[0, idx], pts[1, idx], "s", markersize=6,
        color=COLORS["secondary"], label="constrained",
    )
    axes[1].set_aspect("equal")
    axes[1].set_title(r"nodes reached by $M_{id}$", fontsize=10)
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    # Outside the axes: every interior position holds a node, so any
    # in-axes legend covers the data it explains.
    axes[1].legend(
        fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0, frameon=False,
    )


def forward_adjoint_animation(basis, solutions, adjoints, times, path, bkd,
                              probe_xy, fps=12, stride=1, bitrate=1400,
                              nlevels=41):
    """adjoint_transient_pde_concept.qmd -> the forward/adjoint animation.

    The forward solution and the adjoint side by side, played as the
    solver actually runs them: first a forward sweep with the adjoint
    panel dark, then --- once the trajectory is complete and the
    functional has seeded the adjoint at the final time --- a second
    sweep with the clock running BACKWARDS and the forward field held at
    its final state.

    Showing the two phases in sequence rather than on one clock is the
    point. The adjoint cannot start until the forward solve has
    finished, because the recursion needs the trajectory it linearizes
    about; that ordering is why the whole forward solution must be
    stored, and it is invisible if both panels advance together.

    BOTH panels are drawn on a log scale, and the adjoint over
    :math:`|\\lambda|`. Neither choice is cosmetic. The forward peak
    saturates within the first few frames and then changes by a few
    percent, while the plume that actually reaches the sensor is two
    orders of magnitude fainter --- on a linear scale the panel looks
    frozen after t is small, because everything still evolving is
    crushed into the bottom of the range. The adjoint spans three orders
    of magnitude across the trajectory, being seeded at the final time
    and growing as the recursion runs backwards, and is mostly negative,
    since a residual perturbation enters the reading with a minus sign.

    Frames are drawn one at a time rather than accumulated, so peak
    memory is a single frame regardless of trajectory length.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Degree-1 basis both fields live on.
    solutions : array (ndofs, ntimes)
        Forward trajectory.
    adjoints : array (ndofs, ntimes)
        Adjoint trajectory, same time ordering as ``solutions``.
    times : array (ntimes,)
    path : str
        Output ``.mp4``.
    probe_xy : tuple of float
        Sensor location, marked on both panels.

    Returns
    -------
    str
        ``path``, so a Quarto cell can hand it straight to an embed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LogNorm

    fwd = np.clip(bkd.to_numpy(solutions), 0.0, None)
    adj = np.abs(bkd.to_numpy(adjoints))
    times_np = bkd.to_numpy(times)
    frames = np.arange(0, len(times_np), stride)
    tri = triangulation(basis)

    def decade_levels(values, ndecades):
        """Log levels spanning ndecades below the field's peak.

        Floored at a fixed fraction of the peak rather than at the true
        minimum, which is round-off: a scale reaching down to 1e-16
        would spend most of its range on noise.
        """
        top = float(np.percentile(values, 99.9))
        return np.logspace(np.log10(top) - ndecades, np.log10(top),
                           nlevels)

    fwd_levels = decade_levels(fwd, 4.0)
    adj_levels = decade_levels(adj, 5.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # Phase 1 walks the forward trajectory with the adjoint panel dark;
    # phase 2 holds the forward field at its final state and walks the
    # adjoint backwards. Each entry is (forward index, adjoint index or
    # None).
    schedule = [(k, None) for k in frames]
    schedule += [(frames[-1], k) for k in frames[::-1]]

    def draw(i):
        fwd_index, adj_index = schedule[i]
        for ax in axes:
            ax.clear()
        # Clip up to the floor rather than extending below it: NEON_CMAP
        # paints under-range values magenta as an alarm for genuinely
        # negative concentration, and a log floor set by choice is not
        # that.
        forward = axes[0].tricontourf(
            tri, np.clip(fwd[:, fwd_index], fwd_levels[0], None),
            levels=fwd_levels,
            norm=LogNorm(vmin=fwd_levels[0], vmax=fwd_levels[-1]),
            cmap=NEON_CMAP, extend="max",
        )
        # Before the forward sweep finishes there is no adjoint to draw.
        # Flooring the field at the scale's own minimum renders the panel
        # in the map's darkest colour, so "not yet computed" looks like
        # what it is rather than like a field of zeros.
        adjoint_values = (
            np.full(adj.shape[0], adj_levels[0]) if adj_index is None
            else np.clip(adj[:, adj_index], adj_levels[0], None)
        )
        adjoint = axes[1].tricontourf(
            tri, adjoint_values, levels=adj_levels,
            norm=LogNorm(vmin=adj_levels[0], vmax=adj_levels[-1]),
            cmap=NEON_CMAP, extend="max",
        )
        axes[0].set_title(
            "forward: where the contaminant is"
            + ("" if adj_index is None else "  (final state, held)"),
            fontsize=10,
        )
        axes[1].set_title(
            r"adjoint $|\lambda|$: what the sensor can see"
            if adj_index is not None
            else "adjoint: waits for the forward solve",
            fontsize=10,
        )
        for ax in axes:
            for (x0, y0), width, height in _BLOCKS:
                ax.add_patch(
                    Rectangle((x0, y0), width, height, facecolor="0.55",
                              edgecolor="0.3", zorder=3)
                )
            ax.plot(*probe_xy, "o", markersize=9, markerfacecolor="none",
                    markeredgecolor=COLORS["secondary"], markeredgewidth=2.0,
                    zorder=6)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        clock = fwd_index if adj_index is None else adj_index
        phase = (
            "forward sweep  $\\rightarrow$" if adj_index is None
            else "$\\leftarrow$  adjoint sweep"
        )
        fig.suptitle(
            f"{phase}     $t = {times_np[clock]:.2f}$", fontsize=11
        )
        return forward, adjoint

    forward, adjoint = draw(0)
    # Decade ticks explicitly: contour levels on a LogNorm otherwise
    # leave the colorbar with a single label.
    for mappable, levels, ax, label in (
        (forward, fwd_levels, axes[0], "$u$"),
        (adjoint, adj_levels, axes[1], r"$|\lambda|$"),
    ):
        decades = 10.0 ** np.arange(
            np.ceil(np.log10(levels[0])), np.floor(np.log10(levels[-1])) + 1
        )
        bar = fig.colorbar(mappable, ax=ax, shrink=0.85, label=label)
        bar.set_ticks(decades)
        bar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(top=0.86, bottom=0.04)
    FuncAnimation(fig, draw, frames=len(schedule), blit=False).save(
        path, writer=_mp4_writer(fps, bitrate)
    )
    plt.close(fig)
    return path
