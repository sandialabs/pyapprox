"""Unstructured mesh implementations for Galerkin finite element methods.

Loads 2D unstructured meshes from JSON files with named boundaries
(as facet indices) and subdomains (as element indices).
"""

import json
from typing import Dict, Generic, List, Optional, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import MeshQuad
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class UnstructuredMesh2D(Generic[Array]):
    """2D unstructured quad mesh loaded from a JSON file.

    The JSON file must contain:
    - ``p``: node coordinates, list of [x, y] pairs (nnodes x 2)
    - ``t``: element connectivity, list of 4-node lists (nelems x 4)
    - ``boundaries``: dict mapping boundary names to lists of facet indices
    - ``subdomains``: dict mapping subdomain names to lists of element indices

    Parameters
    ----------
    json_path : str
        Path to the JSON mesh file.
    bkd : Backend[Array]
        Computational backend.
    rescale_origin : Optional[Tuple[float, float]]
        If provided, shift coordinates so the minimum corner maps to this
        point. For example, ``(0.0, 0.0)`` shifts the mesh so all
        coordinates are non-negative.
    """

    def __init__(
        self,
        json_path: str,
        bkd: Backend[Array],
        rescale_origin: Optional[Tuple[float, float]] = None,
    ):
        self._bkd = bkd

        with open(json_path) as f:
            data = json.load(f)

        pts = np.array(data["p"], dtype=np.float64)  # (nnodes, 2)
        elems = np.array(data["t"], dtype=np.int64)  # (nelems, 4)

        if rescale_origin is not None:
            shift = pts.min(axis=0) - np.array(rescale_origin, dtype=np.float64)
            pts -= shift

        # Transpose to skfem convention: (2, nnodes) and (4, nelems)
        pts_T = np.ascontiguousarray(pts.T)
        elems_T = np.ascontiguousarray(elems.T)

        # Build boundary dict: name -> facet index array
        boundaries: Dict[str, np.ndarray] = {}
        for name, facet_indices in data.get("boundaries", {}).items():
            boundaries[name] = np.array(facet_indices, dtype=np.int32)

        # Build subdomain dict: name -> element index array
        subdomains: Dict[str, np.ndarray] = {}
        for name, elem_indices in data.get("subdomains", {}).items():
            subdomains[name] = np.array(elem_indices, dtype=np.int64)

        self._subdomains = subdomains

        # Construct skfem mesh with boundaries and subdomains
        self._skfem_mesh = MeshQuad(
            pts_T,
            elems_T,
            _boundaries=boundaries if boundaries else None,
            _subdomains=subdomains if subdomains else None,
        )

        # Cache nodes as backend array
        self._nodes = bkd.asarray(self._skfem_mesh.p.astype(np.float64))

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return spatial dimension."""
        return 2

    def nelements(self) -> int:
        """Return total number of mesh elements."""
        return int(self._skfem_mesh.nelements)

    def nnodes(self) -> int:
        """Return total number of mesh nodes."""
        return int(self._skfem_mesh.nvertices)

    def nodes(self) -> Array:
        """Return mesh node coordinates.

        Returns
        -------
        Array
            Node coordinates. Shape: (2, nnodes)
        """
        return self._nodes

    def elements(self) -> Array:
        """Return element connectivity.

        Returns
        -------
        Array
            Element-to-node connectivity. Shape: (4, nelements)
        """
        return self._bkd.asarray(self._skfem_mesh.t.astype(np.int64))

    def skfem_mesh(self) -> MeshQuad:
        """Return the underlying skfem mesh object."""
        return self._skfem_mesh

    def boundary_nodes(self, boundary_id: str) -> Array:
        """Return node indices on a named boundary.

        Parameters
        ----------
        boundary_id : str
            Boundary identifier (must match a key in the JSON boundaries).

        Returns
        -------
        Array
            Node indices on the boundary. Shape: (nboundary_nodes,)
        """
        if (
            self._skfem_mesh.boundaries is None
            or boundary_id not in self._skfem_mesh.boundaries
        ):
            available = (
                list(self._skfem_mesh.boundaries.keys())
                if self._skfem_mesh.boundaries
                else []
            )
            raise ValueError(
                f"Unknown boundary_id '{boundary_id}'. Available: {available}"
            )
        facet_indices = self._skfem_mesh.boundaries[boundary_id]
        node_indices = np.unique(self._skfem_mesh.facets[:, facet_indices])
        return self._bkd.asarray(node_indices.astype(np.int64))

    def subdomain_elements(self, name: str) -> np.ndarray:
        """Return element indices for a named subdomain.

        Parameters
        ----------
        name : str
            Subdomain name (must match a key in the JSON subdomains).

        Returns
        -------
        np.ndarray
            Element indices. Shape: (nsubdomain_elements,)
        """
        if name not in self._subdomains:
            raise ValueError(
                f"Unknown subdomain '{name}'. "
                f"Available: {list(self._subdomains.keys())}"
            )
        return self._subdomains[name]

    def subdomain_names(self) -> List[str]:
        """Return names of all subdomains."""
        return list(self._subdomains.keys())

    def boundary_names(self) -> List[str]:
        """Return names of all boundaries."""
        if self._skfem_mesh.boundaries is None:
            return []
        return list(self._skfem_mesh.boundaries.keys())

    def __repr__(self) -> str:
        return (
            f"UnstructuredMesh2D(nnodes={self.nnodes()}, "
            f"nelements={self.nelements()}, "
            f"boundaries={self.boundary_names()}, "
            f"subdomains={self.subdomain_names()})"
        )
