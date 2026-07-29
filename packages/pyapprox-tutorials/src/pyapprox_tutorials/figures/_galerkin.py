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

    stream = ax.streamplot(
        grid_1d, grid_1d, vel_x, vel_y, color=speed, cmap="viridis",
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
