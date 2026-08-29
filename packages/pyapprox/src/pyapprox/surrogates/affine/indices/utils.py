"""Utility functions for multi-index operations.

This module provides utility functions for working with multi-indices,
including hashing, sorting, and computing hyperbolic index sets.
"""

import itertools
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend


def hash_index(array: Array, bkd: Backend[Array]) -> int:
    """Compute a hash for a multi-index array.

    Parameters
    ----------
    array : Array
        Multi-index to hash. Shape: (nvars,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    int
        Hash value for the array.
    """
    np_array = bkd.to_numpy(array)
    return hash(np_array.tobytes())


def _unique_values_per_row(a: NDArray[np.intp]) -> NDArray[np.intp]:
    """Count unique values per row using bincount.

    Internal utility for computing hyperbolic indices.
    """
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:, None] * N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0] * N).reshape(-1, N)


def anisotropy_penalties_from_importance(
    importance: Array, bkd: Backend[Array]
) -> Array:
    r"""Convert per-dimension importance into anisotropy penalties.

    Index generators take *penalties*, where a large value admits fewer
    terms. Importance runs the other way: a large value should admit
    more. This inverts and rescales, so the most important dimension
    gets a penalty of one and is resolved in full.

    Use it when the natural input decays with importance, such as KLE
    eigenvalues, variance contributions, or sensitivity indices.

    Parameters
    ----------
    importance : Array
        Per-dimension importance, all strictly positive. Shape: (nvars,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Penalties :math:`\gamma_i = \max_j(s_j) / s_i`, the smallest
        being one. Shape: (nvars,)

    Notes
    -----
    Importance spanning orders of magnitude — as KLE eigenvalues often
    do — produces penalties just as large, and a dimension with penalty
    :math:`\gamma_i` gets no terms until the level reaches
    :math:`\gamma_i`. To resolve such dimensions at a practical level,
    compress the range first, for example by passing a fractional power
    of the eigenvalues.
    """
    if importance.ndim != 1:
        raise ValueError(
            f"importance must be 1D, got shape {importance.shape}"
        )
    if bkd.to_numpy(importance).min() <= 0:
        raise ValueError("importance must be strictly positive")
    return bkd.max(importance) / importance


def _normalized_anisotropy_penalties(
    nvars: int, penalties: Array, bkd: Backend[Array]
) -> NDArray[np.double]:
    """Validate anisotropy penalties and rescale the smallest to one.

    Only the ratios between penalties carry meaning; the overall scale
    is the job of the level, so it is normalized away. Normalizing by
    the minimum leaves every penalty at least one, so the weighted norm
    never falls below the unweighted one and the anisotropic set is
    always a subset of the isotropic set at the same level.
    """
    if penalties.shape != (nvars,):
        raise ValueError(
            f"penalties has wrong shape {penalties.shape}, "
            f"expected ({nvars},)"
        )
    np_penalties: NDArray[np.double] = bkd.to_numpy(penalties).astype(
        np.double
    )
    if np.any(np_penalties <= 0):
        raise ValueError("penalties must be strictly positive")
    normalized: NDArray[np.double] = np_penalties / np_penalties.min()
    return normalized


def compute_hyperbolic_level_indices(
    nvars: int,
    level: int,
    pnorm: float,
    bkd: Backend[Array],
) -> Array:
    """Compute multi-indices at a specific hyperbolic level.

    Parameters
    ----------
    nvars : int
        Number of variables.
    level : int
        Hyperbolic level.
    pnorm : float
        p-norm exponent for hyperbolic cross.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Multi-indices at this level. Shape: (nvars, nindices)
    """
    indices = _total_degree_shell(nvars, level)
    if indices is None:
        return bkd.zeros((nvars, 1), dtype=bkd.int64_dtype())
    keep = _within_norm_bound(indices, pnorm, level, None)
    return bkd.asarray(indices[:, keep], dtype=bkd.int64_dtype())


def _total_degree_shell(nvars: int, level: int) -> Optional[NDArray[np.intp]]:
    """Enumerate the multi-indices whose entries sum to ``level``.

    Returns None for the zero level, whose single index the callers
    build directly in their own backend.
    """
    if level == 0:
        return None
    # Generate combinations using numpy (backend-agnostic for this step)
    tmp = np.asarray(
        list(itertools.combinations_with_replacement(np.arange(nvars), level))
    )
    # Count occurrences to get multi-indices
    shell: NDArray[np.intp] = _unique_values_per_row(tmp).T
    return shell


def _within_norm_bound(
    indices: NDArray[np.intp],
    pnorm: float,
    norm_bound: float,
    np_penalties: Optional[NDArray[np.double]],
) -> NDArray[np.intp]:
    """Return the columns satisfying the (optionally penalized) p-norm."""
    eps = 1000 * np.finfo(np.double).eps
    scaled = indices if np_penalties is None else np_penalties[:, None] * indices
    p_norms = np.sum(scaled**pnorm, axis=0) ** (1.0 / pnorm)
    kept: NDArray[np.intp] = np.where(p_norms <= norm_bound + eps)[0]
    return kept


def compute_hyperbolic_indices(
    nvars: int,
    max_level: int,
    pnorm: float,
    bkd: Backend[Array],
    penalties: Optional[Array] = None,
) -> Array:
    r"""Compute all multi-indices up to a maximum hyperbolic level.

    Returns every :math:`\lambda` with
    :math:`\|\gamma \odot \lambda\|_p \le k`, where :math:`k` is
    ``max_level`` and :math:`\gamma` are the penalties, defaulting to
    one. Penalties are normalized to a minimum of one, so the
    anisotropic set is a subset of the isotropic set at the same level
    and no admissible index has total degree above :math:`k` — sweeping
    total-degree shells to ``max_level`` is therefore complete. Each
    shell is tested against ``max_level`` rather than its own degree,
    since a penalized index can exceed its degree in weighted norm.

    Parameters
    ----------
    nvars : int
        Number of variables.
    max_level : int
        Maximum hyperbolic level.
    pnorm : float
        p-norm exponent for hyperbolic cross.
    bkd : Backend[Array]
        Computational backend.
    penalties : Array, optional
        Anisotropy penalties :math:`\gamma`, all strictly positive, of
        which only the ratios matter — they are rescaled so the
        smallest is one. A large penalty admits *fewer* terms in its
        dimension; see
        :func:`anisotropy_penalties_from_importance` when the natural
        input grows with importance instead. Shape: (nvars,). Defaults
        to uniform penalties, the isotropic set.

    Returns
    -------
    Array
        All multi-indices up to max_level. Shape: (nvars, nindices)
    """
    np_penalties = (
        None
        if penalties is None
        else _normalized_anisotropy_penalties(nvars, penalties, bkd)
    )

    indices_list = []
    for dd in range(max_level + 1):
        shell = _total_degree_shell(nvars, dd)
        if shell is None:
            indices_list.append(bkd.zeros((nvars, 1), dtype=bkd.int64_dtype()))
            continue
        keep = _within_norm_bound(shell, pnorm, max_level, np_penalties)
        indices_list.append(
            bkd.asarray(shell[:, keep], dtype=bkd.int64_dtype())
        )

    if len(indices_list) == 0:
        return bkd.zeros((nvars, 0), dtype=bkd.int64_dtype())

    return bkd.hstack(indices_list)


def argsort_indices_lexiographically(indices: Array, bkd: Backend[Array]) -> Array:
    """Return indices that would sort multi-indices lexicographically.

    Sorts by total level first, then lexicographically.

    Parameters
    ----------
    indices : Array
        Multi-indices to sort. Shape: (nvars, nindices)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Sorting indices. Shape: (nindices,)
    """
    np_indices = bkd.to_numpy(indices)

    # Build tuple for lexsort: last key is primary sort
    keys: List[NDArray[np.generic]] = [
        np_indices[ii, :] for ii in range(np_indices.shape[0])
    ]
    # Add total level as primary sort key
    keys.append(np_indices.sum(axis=0))

    return bkd.asarray(np.lexsort(tuple(keys)), dtype=bkd.int64_dtype())


def sort_indices_lexiographically(indices: Array, bkd: Backend[Array]) -> Array:
    """Sort multi-indices lexicographically.

    Sorts by total level first, then lexicographically within each level.

    Parameters
    ----------
    indices : Array
        Multi-indices to sort. Shape: (nvars, nindices)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Sorted multi-indices. Shape: (nvars, nindices)
    """
    sort_idx = argsort_indices_lexiographically(indices, bkd)
    return indices[:, sort_idx]


def indices_pnorm(indices: Array, pnorm: float, bkd: Backend[Array]) -> Array:
    """Compute p-norm of multi-indices.

    Parameters
    ----------
    indices : Array
        Multi-indices. Shape: (nvars, nindices) or (nvars,)
    pnorm : float
        p-norm exponent.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        p-norms. Shape: (nindices,) or scalar
    """
    if indices.ndim == 1:
        return bkd.sum(indices**pnorm) ** (1.0 / pnorm)
    return bkd.sum(indices**pnorm, axis=0) ** (1.0 / pnorm)


def compute_downward_closure(indices: Array, bkd: Backend[Array]) -> Array:
    """Compute the downward closure of a set of multi-indices.

    The downward closure of a set S of multi-indices is the smallest
    downward-closed set containing S. For each index (l_0, ..., l_{d-1})
    in S, the closure includes all indices (k_0, ..., k_{d-1}) where
    0 <= k_i <= l_i for all dimensions i.

    Parameters
    ----------
    indices : Array
        Multi-indices to compute closure of. Shape: (nvars, nindices)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Downward closure of the input indices. Shape: (nvars, nclosure)
        Sorted lexicographically by total level, then by dimension.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Closure of {(2, 1)} = {(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)}
    >>> indices = bkd.asarray([[2], [1]])
    >>> closure = compute_downward_closure(indices, bkd)
    >>> closure.shape[1]  # 6 indices
    6

    >>> # Closure of {(1, 0), (0, 2)} includes both closures merged
    >>> indices = bkd.asarray([[1, 0], [0, 2]])
    >>> closure = compute_downward_closure(indices, bkd)
    >>> # Result: {(0,0), (1,0), (0,1), (0,2)}
    >>> closure.shape[1]
    4
    """
    nvars = indices.shape[0]

    # Collect all indices in the closure using a set
    closure_set: set[tuple[int, ...]] = set()

    for j in range(indices.shape[1]):
        index = tuple(bkd.to_int(indices[i, j]) for i in range(nvars))

        # Add all predecessors (including the index itself)
        ranges = [range(index[i] + 1) for i in range(nvars)]
        for predecessor in itertools.product(*ranges):
            closure_set.add(predecessor)

    # Convert to array
    nclosure = len(closure_set)
    if nclosure == 0:
        return bkd.zeros((nvars, 0), dtype=bkd.int64_dtype())

    result = bkd.zeros((nvars, nclosure), dtype=bkd.int64_dtype())
    for j, idx in enumerate(closure_set):
        for i in range(nvars):
            result[i, j] = idx[i]

    # Sort lexicographically for consistent output
    return sort_indices_lexiographically(result, bkd)
