"""Obstructed 2D mesh for domains with rectangular obstacles.

Creates a structured quad mesh with selected cells removed to represent
rectangular obstacles. Boundaries are automatically named: "left",
"right", "bottom", "top", and "obs0", "obs1", ... for each obstacle.
"""

from functools import partial
from typing import Generic, List

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import MeshQuad
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class ObstructedMesh2D(Generic[Array]):
    """Structured 2D quad mesh with rectangular obstacles removed.

    Parameters
    ----------
    xintervals : np.ndarray
        Grid lines in x direction. Shape: (nx,)
    yintervals : np.ndarray
        Grid lines in y direction. Shape: (ny,)
    obstruction_indices : np.ndarray
        Indices of cells to remove (0-based, row-major order with
        x varying fastest). Shape: (nobstructions,)
    bkd : Backend[Array]
        Computational backend.
    nrefine : int, optional
        Number of uniform refinements to apply. Default: 0.
    """

    def __init__(
        self,
        xintervals: np.ndarray,
        yintervals: np.ndarray,
        obstruction_indices: np.ndarray,
        bkd: Backend[Array],
        nrefine: int = 0,
    ) -> None:
        self._bkd = bkd
        self._xintervals = xintervals
        self._yintervals = yintervals
        self._obstruction_indices = obstruction_indices

        nx = xintervals.shape[0]
        ny = yintervals.shape[0]

        # Full vertices via cartesian product (x varies fastest)
        xx, yy = np.meshgrid(xintervals, yintervals, indexing="xy")
        self._full_vertices = np.vstack([xx.ravel(), yy.ravel()]).astype(np.float64)

        # Full connectivity (4, ncells) — clockwise from bottom-left
        self._full_connectivity = self._generate_connectivity(nx, ny)

        # Remove obstructed cells
        mask = np.ones(self._full_connectivity.shape[1], dtype=bool)
        mask[obstruction_indices] = False
        connectivity = self._full_connectivity[:, mask]

        # Build skfem mesh
        mesh = MeshQuad(self._full_vertices, connectivity)
        bndry_defs = self._build_boundary_defs()
        self._skfem_mesh = mesh.with_boundaries(bndry_defs)

        if nrefine > 0:
            self._skfem_mesh = self._skfem_mesh.refined(nrefine)

        self._nodes = bkd.asarray(
            self._skfem_mesh.p.astype(np.float64),
        )

    def _generate_connectivity(
        self,
        nx: int,
        ny: int,
    ) -> np.ndarray:
        """Generate full quad connectivity for an nx × ny vertex grid."""
        t = []
        for row in range(ny - 1):
            for col in range(nx - 1):
                bl = row * nx + col
                br = bl + 1
                tl = bl + nx
                tr = tl + 1
                t.append([bl, br, tr, tl])
        return np.array(t, dtype=np.int64).T

    def _obstruction_boundary(
        self,
        obstruction_idx: int,
        x: np.ndarray,
    ) -> np.ndarray:
        """Test whether points lie on the boundary of an obstruction."""
        eps = 1e-8
        vertex_indices = self._full_connectivity[:, obstruction_idx]
        vertices = self._full_vertices[:, vertex_indices]
        return (
            (x[0] >= (vertices[0, 0] - eps))
            & (x[0] <= (vertices[0, 1] + eps))
            & (x[1] >= (vertices[1, 0] - eps))
            & (x[1] <= (vertices[1, 2] + eps))
        )

    def _build_boundary_defs(self) -> dict:
        """Build boundary definitions for skfem with_boundaries."""
        defs: dict = {
            "left": lambda x: np.isclose(x[0], self._xintervals[0]),
            "right": lambda x: np.isclose(x[0], self._xintervals[-1]),
            "bottom": lambda x: np.isclose(x[1], self._yintervals[0]),
            "top": lambda x: np.isclose(x[1], self._yintervals[-1]),
        }
        for ii, idx in enumerate(self._obstruction_indices):
            defs[f"obs{ii}"] = partial(
                self._obstruction_boundary,
                idx,
            )
        return defs

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return spatial dimension."""
        return 2

    def nelements(self) -> int:
        """Return total number of mesh elements."""
        return self._skfem_mesh.nelements

    def nnodes(self) -> int:
        """Return total number of mesh nodes."""
        return self._skfem_mesh.nvertices

    def nodes(self) -> Array:
        """Return mesh node coordinates. Shape: (2, nnodes)"""
        return self._nodes

    def elements(self) -> Array:
        """Return element connectivity. Shape: (4, nelements)"""
        return self._bkd.asarray(
            self._skfem_mesh.t.astype(np.int64),
        )

    def skfem_mesh(self) -> MeshQuad:
        """Return the underlying skfem mesh object."""
        return self._skfem_mesh

    def boundary_nodes(self, boundary_id: str) -> Array:
        """Return node indices on a named boundary.

        Parameters
        ----------
        boundary_id : str
            Boundary name: "left", "right", "bottom", "top",
            "obs0", "obs1", etc.

        Returns
        -------
        Array
            Node indices. Shape: (nboundary_nodes,)
        """
        nodes_np = self._bkd.to_numpy(self._nodes)
        defs = self._build_boundary_defs()
        if boundary_id not in defs:
            raise ValueError(
                f"Unknown boundary_id '{boundary_id}'. "
                f"Valid options: {list(defs.keys())}"
            )
        mask = defs[boundary_id](nodes_np)
        return self._bkd.asarray(np.where(mask)[0].astype(np.int64))

    def boundary_names(self) -> List[str]:
        """Return all boundary names."""
        names = ["left", "right", "bottom", "top"]
        for ii in range(len(self._obstruction_indices)):
            names.append(f"obs{ii}")
        return names

    def __repr__(self) -> str:
        return (
            f"ObstructedMesh2D("
            f"nx={self._xintervals.shape[0]}, "
            f"ny={self._yintervals.shape[0]}, "
            f"nobstructions={len(self._obstruction_indices)}, "
            f"nelements={self.nelements()}, "
            f"nnodes={self.nnodes()})"
        )
