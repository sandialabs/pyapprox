"""Mesh construction helpers for the obstructed advection-diffusion problem.

Pure functions for building the obstructed ``[0, 1]^2`` mesh with three
rectangular obstacles, optionally inserting KLE-subdomain boundary
coordinates as grid lines so no element straddles the subdomain edge.
No Stokes, no KLE, no OED state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D

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


def _build_obstructed_mesh(
    bkd: Backend[Array],
    nrefine: int,
    kle_subdomain: Optional[Tuple[float, float, float, float]] = None,
) -> "ObstructedMesh2D[Array]":
    """Create the obstructed domain mesh.

    If ``kle_subdomain`` is provided, its boundary coordinates are
    inserted into the x/y interval grids so that subdomain edges are
    mesh grid lines. This prevents any element from straddling the
    subdomain boundary after uniform refinement (which only halves
    existing cells). Obstruction cell indices are re-mapped
    automatically to account for the inserted grid lines.
    """
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D

    xintervals = np.array([0, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 1.0])
    yintervals = np.linspace(0, 1, 5)
    obstruction_indices = np.array([3, 6, 13], dtype=int)

    if kle_subdomain is not None:
        xmin, xmax, ymin, ymax = kle_subdomain
        if not (
            xintervals[0] <= xmin < xmax <= xintervals[-1]
            and yintervals[0] <= ymin < ymax <= yintervals[-1]
        ):
            raise ValueError(
                f"kle_subdomain {kle_subdomain} must lie inside the base "
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
