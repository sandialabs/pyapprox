"""Subdomain-localized lognormal KLE forcing for advection-diffusion.

Builds a lognormal KLE on the subset of ADR mesh nodes that falls
inside a rectangular subdomain, scattered back into a full-length
nodal vector. Elements outside the subdomain are exactly zero at
every ADR mesh node.

No Stokes, no OED state, no inference plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional, Tuple

if TYPE_CHECKING:
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis

import numpy as np
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.util.backends.protocols import Array, Backend


class _ScatteredFieldMap(Generic[Array]):
    """Scatter a subdomain field into a full-length ADR nodal vector.

    Wraps an inner :class:`TransformedFieldMap` that evaluates on a
    subset of ADR mesh nodes (the KLE subdomain) and produces a
    length-``ntotal`` output in which entries **outside** the
    subdomain are exactly zero.

    Because the ADR problem is NumPy-only (skfem dependency), this
    class uses direct NumPy indexed assignment and then ``bkd.asarray``
    back. No sparse/dense scatter matrix is needed.
    """

    def __init__(
        self,
        inner: TransformedFieldMap[Array],
        global_node_indices: np.ndarray,
        ntotal: int,
        bkd: Backend[Array],
    ) -> None:
        self._inner = inner
        self._bkd = bkd
        self._ntotal = ntotal
        self._global_node_indices = np.asarray(
            global_node_indices, dtype=np.int64,
        )

    def nvars(self) -> int:
        return self._inner.nvars()

    def __call__(self, params_1d: Array) -> Array:
        sub_vals = self._bkd.to_numpy(self._inner(params_1d))
        full = np.zeros(self._ntotal, dtype=np.float64)
        full[self._global_node_indices] = sub_vals
        return self._bkd.asarray(full)


def _compute_lumped_mass_weights(
    skfem_mesh: Any,
    elem_idx: np.ndarray,
    global_nodes: np.ndarray,
) -> np.ndarray:
    """Lumped mass weights on subdomain nodes via shoelace element area.

    Distributes each element's area uniformly across its ``nverts``
    vertices. Matches the pattern used in ``cantilever_beam.py``.
    """
    nverts = skfem_mesh.t.shape[0]
    n_nodes = global_nodes.shape[0]
    node_to_local = {int(n): i for i, n in enumerate(global_nodes)}
    sub_w = np.zeros(n_nodes)
    for e in elem_idx:
        verts = skfem_mesh.p[:, skfem_mesh.t[:, e]]
        x, y = verts[0], verts[1]
        area = 0.5 * abs(
            sum(
                x[i] * y[(i + 1) % nverts] - x[(i + 1) % nverts] * y[i]
                for i in range(nverts)
            )
        )
        for n in skfem_mesh.t[:, e]:
            sub_w[node_to_local[int(n)]] += area / nverts
    return sub_w


def _create_subdomain_kle_forcing(
    adr_basis: "LagrangeBasis[Array]",
    nkle_terms: int,
    correlation_length: float,
    sigma: float,
    subdomain: Optional[Tuple[float, float, float, float]],
    bkd: Backend[Array],
) -> "_ScatteredFieldMap[Array]":
    """Create lognormal KLE forcing localized to a rectangular subdomain.

    Builds a lognormal KLE on the ADR mesh nodes that fall inside the
    given rectangle (via element-centroid predicate), using lumped-mass
    quadrature weights on the subdomain nodes. Outside the subdomain,
    the forcing is exactly zero at every ADR node.

    If ``subdomain`` is ``None``, the KLE is constructed on the full
    ADR mesh (backward-compatible behavior).
    """
    from pyapprox.pde.field_maps.kle_factory import (
        create_lognormal_kle_field_map,
    )

    skfem_mesh = adr_basis.mesh().skfem_mesh()
    ntotal = int(skfem_mesh.p.shape[1])

    if subdomain is None:
        elem_idx = np.arange(skfem_mesh.t.shape[1])
        global_nodes = np.arange(ntotal)
    else:
        xmin, xmax, ymin, ymax = subdomain
        # Select elements whose centroid lies inside the rectangle.
        # NOTE: The caller is responsible for ensuring subdomain
        # bounds coincide with mesh grid lines (see
        # ``_build_obstructed_mesh``'s ``kle_subdomain``-aware
        # ``xintervals``/``yintervals``), so no elements straddle the
        # subdomain boundary.
        centroids = skfem_mesh.p[:, skfem_mesh.t].mean(axis=1)
        mask = (
            (centroids[0] >= xmin)
            & (centroids[0] <= xmax)
            & (centroids[1] >= ymin)
            & (centroids[1] <= ymax)
        )
        elem_idx = np.where(mask)[0]
        if elem_idx.size == 0:
            raise ValueError(
                f"KLE subdomain {subdomain} contains no mesh elements. "
                f"Check bounds against the domain extents."
            )
        global_nodes = np.unique(skfem_mesh.t[:, elem_idx].ravel())

    nsub = int(global_nodes.shape[0])
    if nsub < nkle_terms:
        raise ValueError(
            f"KLE subdomain has only {nsub} mesh nodes, but nkle_terms="
            f"{nkle_terms} was requested. Reduce nkle_terms or enlarge "
            f"the subdomain / increase mesh refinement."
        )

    coords_sub = skfem_mesh.p[:, global_nodes]
    sub_w = _compute_lumped_mass_weights(skfem_mesh, elem_idx, global_nodes)

    mean_log = bkd.zeros((nsub,))
    inner = create_lognormal_kle_field_map(
        mesh_coords=bkd.asarray(coords_sub),
        mean_log_field=mean_log,
        bkd=bkd,
        correlation_length=correlation_length,
        num_kle_terms=nkle_terms,
        sigma=sigma,
        quad_weights=bkd.asarray(sub_w),
    )

    return _ScatteredFieldMap(inner, global_nodes, ntotal, bkd)
