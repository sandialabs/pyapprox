"""Deterministic sensor and QoI location selection for OED benchmarks.

Provides greedy maximin distance selection for placing candidate sensor
locations and downstream QoI prediction points on a finite element mesh.
All functions are fully deterministic (no random number generator).
"""

import numpy as np
from numpy.typing import NDArray


def select_maximin_locations(
    nodes: NDArray[np.floating],
    feasible_mask: NDArray[np.bool_],
    n_locations: int,
) -> NDArray[np.intp]:
    """Select well-spaced locations via greedy maximin distance.

    Starts with the feasible node closest to the centroid of all
    feasible nodes, then iteratively adds the feasible node that
    maximizes the minimum distance to all already-selected nodes.

    Parameters
    ----------
    nodes : NDArray
        Mesh node coordinates. Shape: (ndim, nnodes).
    feasible_mask : NDArray
        Boolean mask identifying feasible nodes. Shape: (nnodes,).
    n_locations : int
        Number of locations to select.

    Returns
    -------
    NDArray[np.intp]
        Sorted indices into ``nodes``. Shape: (n_locations,).

    Raises
    ------
    ValueError
        If fewer feasible nodes are available than requested.
    """
    feasible_idx = np.where(feasible_mask)[0]
    if len(feasible_idx) < n_locations:
        raise ValueError(
            f"Requested {n_locations} locations but only "
            f"{len(feasible_idx)} feasible nodes available."
        )

    feasible_nodes = nodes[:, feasible_idx]  # (ndim, nfeasible)
    nfeasible = feasible_nodes.shape[1]

    # Start with the node closest to the centroid
    centroid = feasible_nodes.mean(axis=1, keepdims=True)  # (ndim, 1)
    dists_to_centroid = np.linalg.norm(
        feasible_nodes - centroid, axis=0,
    )
    first = int(np.argmin(dists_to_centroid))

    selected = [first]
    # min_dist_to_selected[j] = min distance from feasible node j to any
    # already-selected node
    min_dist = np.linalg.norm(
        feasible_nodes - feasible_nodes[:, first:first + 1], axis=0,
    )
    min_dist[first] = -1.0  # exclude already selected

    for _ in range(1, n_locations):
        # Pick the feasible node with the largest min-distance
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        min_dist[next_idx] = -1.0

        # Update min distances
        new_dists = np.linalg.norm(
            feasible_nodes - feasible_nodes[:, next_idx:next_idx + 1],
            axis=0,
        )
        # Only update where new distance is smaller (and node not selected)
        mask = (new_dists < min_dist) & (min_dist >= 0)
        min_dist[mask] = new_dists[mask]

    # Map back to global node indices and sort
    global_indices = feasible_idx[np.array(selected)]
    return np.sort(global_indices)


def select_qoi_locations(
    nodes: NDArray[np.floating],
    feasible_mask: NDArray[np.bool_],
    n_qoi: int,
    exclude_indices: NDArray[np.intp],
    x_threshold: float = 5.0 / 7.0,
) -> NDArray[np.intp]:
    """Select downstream QoI prediction locations via greedy maximin.

    Selects from feasible nodes with x >= x_threshold that are not in
    ``exclude_indices`` (typically the candidate sensor locations).

    Parameters
    ----------
    nodes : NDArray
        Mesh node coordinates. Shape: (ndim, nnodes).
    feasible_mask : NDArray
        Boolean mask identifying feasible nodes. Shape: (nnodes,).
    n_qoi : int
        Number of QoI locations to select.
    exclude_indices : NDArray[np.intp]
        Node indices to exclude (e.g., candidate sensor indices).
    x_threshold : float
        Minimum x-coordinate for downstream region. Default: 5/7.

    Returns
    -------
    NDArray[np.intp]
        Sorted indices into ``nodes``. Shape: (n_qoi,).

    Raises
    ------
    ValueError
        If fewer eligible nodes are available than requested.
    """
    nnodes = nodes.shape[1]
    # Build mask: feasible AND downstream AND not excluded
    downstream_mask = nodes[0, :] >= x_threshold
    exclude_set = set(exclude_indices.tolist())
    not_excluded = np.array(
        [i not in exclude_set for i in range(nnodes)], dtype=bool,
    )
    combined_mask = feasible_mask & downstream_mask & not_excluded

    return select_maximin_locations(nodes, combined_mask, n_qoi)


def get_feasible_mask(
    nodes: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Return a feasibility mask for mesh nodes.

    For ObstructedMesh2D, all existing mesh nodes are feasible because
    obstacle cells are removed from the mesh — no node lies inside an
    obstacle.

    Parameters
    ----------
    nodes : NDArray
        Mesh node coordinates. Shape: (ndim, nnodes).

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask. Shape: (nnodes,). All True for obstructed meshes.
    """
    return np.ones(nodes.shape[1], dtype=bool)
