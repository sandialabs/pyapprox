"""Utility functions for multi-index operations.

This module provides utility functions for working with multi-indices,
including hashing, sorting, and computing hyperbolic index sets.
"""

import itertools
from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


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


def _unique_values_per_row(a: Array) -> Array:
    """Count unique values per row using bincount.

    Internal utility for computing hyperbolic indices.
    """
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:, None] * N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0] * N).reshape(-1, N)


def compute_hyperbolic_level_indices(
    nvars: int, level: int, pnorm: float, bkd: Backend[Array]
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
    eps = 1000 * np.finfo(np.double).eps
    if level == 0:
        return bkd.zeros((nvars, 1), dtype=bkd.int64_dtype())

    # Generate combinations using numpy (backend-agnostic for this step)
    tmp = np.asarray(
        list(itertools.combinations_with_replacement(np.arange(nvars), level))
    )

    # Count occurrences to get multi-indices
    indices = _unique_values_per_row(tmp).T

    # Filter by p-norm
    p_norms = np.sum(indices**pnorm, axis=0) ** (1.0 / pnorm)
    II = np.where(p_norms <= level + eps)[0]

    return bkd.asarray(indices[:, II], dtype=bkd.int64_dtype())


def compute_hyperbolic_indices(
    nvars: int,
    max_level: int,
    pnorm: float,
    bkd: Backend[Array],
) -> Array:
    """Compute all multi-indices up to a maximum hyperbolic level.

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

    Returns
    -------
    Array
        All multi-indices up to max_level. Shape: (nvars, nindices)
    """
    indices_list = []
    for dd in range(max_level + 1):
        new_indices = compute_hyperbolic_level_indices(nvars, dd, pnorm, bkd)
        indices_list.append(new_indices)

    if len(indices_list) == 0:
        return bkd.zeros((nvars, 0), dtype=bkd.int64_dtype())

    return bkd.hstack(indices_list)


def argsort_indices_lexiographically(
    indices: Array, bkd: Backend[Array]
) -> Array:
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
    index_tuple = (np_indices[0, :],)
    for ii in range(1, np_indices.shape[0]):
        index_tuple = index_tuple + (np_indices[ii, :],)
    # Add total level as primary sort key
    index_tuple = index_tuple + (np_indices.sum(axis=0),)

    return bkd.asarray(np.lexsort(index_tuple), dtype=bkd.int64_dtype())


def sort_indices_lexiographically(
    indices: Array, bkd: Backend[Array]
) -> Array:
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
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
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
        index = tuple(int(bkd.to_numpy(indices[i, j])) for i in range(nvars))

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
