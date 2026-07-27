"""Obstructed-channel flow substrate: mesh, frozen Stokes flow, velocity.

The domain is the unit square with three staggered rectangular blocks
whose gaps force a serpentine flow path — a substrate for transport,
control, and OED problems that need a nontrivial frozen velocity field.

Provides:

- :func:`build_obstructed_mesh` — the obstructed ``[0, 1]^2`` mesh,
  optionally inserting subdomain boundary coordinates as grid lines so
  no element straddles the subdomain edge.
- Picklable BC callable adapters :class:`ParabolicInlet` and
  :class:`ZeroVelocity` (stored callables must pickle, so these are
  classes rather than closures).
- :func:`solve_obstructed_stokes` — steady Navier-Stokes solve with a
  parabolic inlet and no-slip walls/obstacles.
- :func:`extract_velocity_callable` — pulls the velocity field from a
  Stokes solution vector and returns a callable evaluating it at
  arbitrary points (preserving skfem's ``(ndim, nquad, nelem)``
  quadrature-point convention).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _insert_grid_line(vals: np.ndarray, new_val: float) -> np.ndarray:
    """Insert ``new_val`` into ``vals`` if not already present (within tol)."""
    tol = 1e-12
    if np.any(np.abs(vals - new_val) <= tol):
        return vals.copy()
    merged = np.concatenate([vals, [new_val]])
    return np.sort(merged)


def _recompute_obstruction_indices(
    old_xintervals: np.ndarray,
    old_yintervals: np.ndarray,
    old_obstruction_indices: np.ndarray,
    new_xintervals: np.ndarray,
    new_yintervals: np.ndarray,
) -> np.ndarray:
    """Re-map obstruction cell indices after inserting new grid lines.

    Obstruction indices are row-major with x varying fastest:
        ``idx = row * (nx - 1) + col``
    where ``nx`` is the number of x grid lines. Each obstruction cell
    has fixed ``(xlo, xhi, ylo, yhi)`` coordinates which we recover
    from the old grid, then locate in the new grid.
    """
    old_ncols = old_xintervals.shape[0] - 1
    new_ncols = new_xintervals.shape[0] - 1
    tol = 1e-12

    new_indices = []
    for idx in old_obstruction_indices:
        row = int(idx) // old_ncols
        col = int(idx) % old_ncols
        xlo = old_xintervals[col]
        ylo = old_yintervals[row]
        new_col = int(np.argmin(np.abs(new_xintervals[:-1] - xlo)))
        new_row = int(np.argmin(np.abs(new_yintervals[:-1] - ylo)))
        if (
            abs(new_xintervals[new_col] - xlo) > tol
            or abs(new_yintervals[new_row] - ylo) > tol
        ):
            raise RuntimeError(
                f"Could not recover obstruction cell {idx} at "
                f"({xlo}, {ylo}) in the refined grid."
            )
        new_indices.append(new_row * new_ncols + new_col)
    return np.array(new_indices, dtype=int)


def build_obstructed_mesh(
    bkd: Backend[Array],
    nrefine: int,
    subdomain: Optional[Tuple[float, float, float, float]] = None,
) -> "ObstructedMesh2D[Array]":
    """Create the obstructed-channel mesh on ``[0, 1]^2``.

    The base grid has x-lines ``{0, 2/7, 3/7, 4/7, 5/7, 1}`` and five
    uniform y-lines; the obstruction cells form three staggered
    rectangular blocks (boundary labels ``obs0``/``obs1``/``obs2``
    alongside ``left``/``right``/``bottom``/``top``).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    nrefine : int
        Number of uniform refinements of the base grid.
    subdomain : tuple of 4 floats, optional
        ``(xmin, xmax, ymin, ymax)`` whose boundary coordinates are
        inserted into the interval grids so subdomain edges are mesh
        grid lines. Uniform refinement only halves existing cells, so
        no element ever straddles the subdomain boundary. Obstruction
        cell indices are re-mapped automatically.
    """
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D

    xintervals = np.array([0, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 1.0])
    yintervals = np.linspace(0, 1, 5)
    obstruction_indices = np.array([3, 6, 13], dtype=int)

    if subdomain is not None:
        xmin, xmax, ymin, ymax = subdomain
        if not (
            xintervals[0] <= xmin < xmax <= xintervals[-1]
            and yintervals[0] <= ymin < ymax <= yintervals[-1]
        ):
            raise ValueError(
                f"subdomain {subdomain} must lie inside the base "
                f"domain [{xintervals[0]}, {xintervals[-1]}] x "
                f"[{yintervals[0]}, {yintervals[-1]}] with xmin<xmax and "
                f"ymin<ymax."
            )
        new_x = xintervals.copy()
        new_y = yintervals.copy()
        for v in (xmin, xmax):
            new_x = _insert_grid_line(new_x, float(v))
        for v in (ymin, ymax):
            new_y = _insert_grid_line(new_y, float(v))
        obstruction_indices = _recompute_obstruction_indices(
            xintervals, yintervals, obstruction_indices, new_x, new_y,
        )
        xintervals = new_x
        yintervals = new_y

    return ObstructedMesh2D(
        xintervals,
        yintervals,
        obstruction_indices,
        bkd,
        nrefine=nrefine,
    )


class ParabolicInlet:
    """Inlet velocity profile ``y^(a-1) * (1-y)^(b-1)`` in x-direction.

    Picklable (a class, not a closure: stored BC callables must
    survive ``pickle.dumps``). Callable signature matches skfem BC
    expectations: ``(x: np.ndarray, time: float = 0.0) -> np.ndarray``.
    """

    def __init__(self, a: float, b: float) -> None:
        self._a = float(a)
        self._b = float(b)

    def __call__(
        self, x: np.ndarray, time: float = 0.0,
    ) -> np.ndarray:
        y = x[1]
        vals = np.zeros((x.shape[1], 2))
        vals[:, 0] = y ** (self._a - 1) * (1 - y) ** (self._b - 1)
        return vals


class ZeroVelocity:
    """Zero-velocity BC (a picklable class, not a closure)."""

    def __call__(
        self, x: np.ndarray, time: float = 0.0,
    ) -> np.ndarray:
        return np.zeros((x.shape[1], 2))


def solve_obstructed_stokes(
    mesh: "ObstructedMesh2D[Array]",
    bkd: Backend[Array],
    reynolds_num: float,
    vel_shape_params: List[float],
) -> Tuple[
    Array,
    "StokesPhysics[Array]",
    "VectorLagrangeBasis[Array]",
    "LagrangeBasis[Array]",
]:
    """Solve steady Navier-Stokes on the obstructed mesh.

    Parabolic inlet on the left wall (shape parameters ``(a, b)``),
    no-slip on bottom/top and all obstruction boundaries, natural
    outflow on the right; viscosity is ``1 / reynolds_num``.

    Returns
    -------
    tuple
        ``(sol, stokes, vel_basis, pres_basis)``: the solution vector,
        the physics object, and the degree-2 velocity / degree-1
        pressure bases.
    """
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
    from pyapprox.pde.galerkin.time_integration.galerkin_model import (
        GalerkinModel,
    )

    vel_basis = VectorLagrangeBasis(mesh, degree=2)
    pres_basis = LagrangeBasis(mesh, degree=1)

    a, b = vel_shape_params
    inlet_func = ParabolicInlet(a, b)
    zero_vel = ZeroVelocity()

    # No-slip on obs, bottom, top; parabolic inlet on left.
    vel_bcs: List[Tuple[str, Callable[..., Any]]] = [
        ("left", inlet_func),
        ("bottom", zero_vel),
        ("top", zero_vel),
        ("obs0", zero_vel),
        ("obs1", zero_vel),
        ("obs2", zero_vel),
    ]

    viscosity = 1.0 / reynolds_num

    stokes = StokesPhysics(
        vel_basis,
        pres_basis,
        bkd,
        navier_stokes=True,
        viscosity=viscosity,
        vel_dirichlet_bcs=vel_bcs,
    )

    model = GalerkinModel(stokes, bkd)
    init_guess = stokes.init_guess(0.0)
    sol = model.solve_steady(init_guess, tol=1e-10, maxiter=50)

    return sol, stokes, vel_basis, pres_basis


def extract_velocity_callable(
    sol: Array,
    stokes: "StokesPhysics[Array]",
    vel_basis: "VectorLagrangeBasis[Array]",
    pres_basis: "LagrangeBasis[Array]",
    adr_basis: "LagrangeBasis[Array]",
    bkd: Backend[Array],
    probes_cache: Optional[Dict[int, Any]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Extract velocity from a Stokes solution as a callable field.

    Returns a callable that accepts points of shape ``(2, ...)`` and
    returns velocity of shape ``(2, ...)``, preserving any extra
    dimensions (e.g., quadrature-point structure from skfem:
    ``(2, nqpts, nelem)``).

    Parameters
    ----------
    probes_cache
        Shared ``{npts: probes_csr}`` dict. If provided, the returned
        callable stores and retrieves the skfem ``probes`` sparse
        matrix here instead of in a private dict. Pass the same cache
        to other interpolators on the same basis so the expensive
        element-finder is run at most once per distinct point count.
    """
    vel_ndofs = stokes.vel_ndofs()
    vel_state_np = bkd.to_numpy(sol[:vel_ndofs])

    # skfem ElementVector DOFs are interleaved:
    # ``[comp0_dof0, comp1_dof0, comp0_dof1, comp1_dof1, ...]``
    vel_x = vel_state_np[0::2]
    vel_y = vel_state_np[1::2]

    vel_skfem = vel_basis.skfem_basis()
    from skfem import Basis

    scalar_elem = vel_skfem.elem.elem
    scalar_vel_basis = Basis(
        vel_basis.mesh().skfem_mesh(),
        scalar_elem,
        intorder=4,
    )

    # Interpolate Stokes velocity onto ADR DOF locations (done once).
    adr_skfem = adr_basis.skfem_basis()
    vel_x_at_dofs = scalar_vel_basis.interpolator(vel_x)(adr_skfem.doflocs)
    vel_y_at_dofs = scalar_vel_basis.interpolator(vel_y)(adr_skfem.doflocs)

    # Probes-matrix cache: keyed by number of evaluation points.
    # The matrix depends only on the mesh and the evaluation points
    # (quadrature coords), which are identical across all time steps
    # and Newton iterations for a given basis.
    cache: Dict[int, Any] = probes_cache if probes_cache is not None else {}

    def velocity_field(x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape[1:]
        x_flat = x.reshape(2, -1)
        npts = x_flat.shape[1]

        if npts not in cache:
            cache[npts] = adr_skfem.probes(x_flat).tocsr()

        probes_csr = cache[npts]
        vx = (probes_csr @ vel_x_at_dofs).reshape(orig_shape)
        vy = (probes_csr @ vel_y_at_dofs).reshape(orig_shape)
        return np.stack([vx, vy], axis=0)

    return velocity_field
