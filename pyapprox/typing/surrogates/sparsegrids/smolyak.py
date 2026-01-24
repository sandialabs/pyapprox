"""Smolyak combination technique for sparse grids.

This module provides functions for computing Smolyak combination coefficients
using the inclusion-exclusion formula.

The Smolyak combination technique expresses a sparse grid interpolant as a
weighted sum of tensor product interpolants:

    I_L = sum_{k in K} c_k * I_k

where K is a downward-closed index set and c_k are the combination coefficients.

The coefficients are computed using:
    c_k = sum_{e in {0,1}^d} (-1)^|e| * indicator(k + e in K)
"""

from typing import Set, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


def _index_to_tuple(index: Array) -> Tuple[int, ...]:
    """Convert array index to hashable tuple.

    Parameters
    ----------
    index : Array
        1D array of shape (nvars,)

    Returns
    -------
    Tuple[int, ...]
        Hashable tuple representation of the index
    """
    # Iterate directly over elements - works for 1D arrays
    return tuple(int(index[i]) for i in range(index.shape[0]))


def compute_smolyak_coefficients(
    subspace_indices: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute Smolyak combination coefficients.

    Uses the inclusion-exclusion formula:
    c_k = sum_{e in {0,1}^d} (-1)^|e| * indicator(k + e in K)

    Parameters
    ----------
    subspace_indices : Array
        Multi-indices of subspaces, shape (nvars, nsubspaces)
    bkd : Backend[Array]
        Computational backend

    Returns
    -------
    Array
        Combination coefficients, shape (nsubspaces,)

    Notes
    -----
    The Smolyak combination technique expresses a sparse grid interpolant
    as a weighted sum of tensor product interpolants:

        I_K = sum_{k in K} c_k * I_k

    where K is a downward-closed index set, I_k is the tensor product
    interpolant for subspace k, and c_k are the combination coefficients.

    Key mathematical properties:

    1. **Sum to one**: sum(c_k) = 1, ensuring constants are reproduced exactly
    2. **Boundary coefficients**: Indices with no forward neighbors have c_k = 1
    3. **Telescoping in 1D**: Only the highest level has non-zero coefficient
    4. **Negative coefficients**: Interior indices can have negative coefficients

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # 2D isotropic sparse grid level 2
    >>> indices = bkd.asarray([[0, 1, 0, 2, 1, 0],
    ...                        [0, 0, 1, 0, 1, 2]])
    >>> coefs = compute_smolyak_coefficients(indices, bkd)
    """
    nvars = subspace_indices.shape[0]
    nsubspaces = subspace_indices.shape[1]
    nshifts = 2**nvars

    # Build set of index tuples for fast O(1) lookup
    index_set: Set[Tuple[int, ...]] = set()
    for j in range(nsubspaces):
        index_set.add(_index_to_tuple(subspace_indices[:, j]))

    # Precompute all 2^nvars shift vectors and their signs
    # shifts: shape (nvars, nshifts) - each column is a binary shift vector
    # signs: shape (nshifts,) - (-1)^|shift| for each shift
    shifts = bkd.zeros((nvars, nshifts), dtype=bkd.int64_dtype())
    signs = bkd.zeros((nshifts,))

    for shift_int in range(nshifts):
        temp = shift_int
        parity = 0
        for dim in range(nvars):
            bit = temp % 2
            shifts[dim, shift_int] = bit
            parity += bit
            temp //= 2
        signs[shift_int] = (-1.0) ** parity

    # Vectorized coefficient computation
    # For each shift, compute all shifted indices at once and check membership
    # shifted_indices: shape (nvars, nsubspaces) for each shift
    coefficients = bkd.zeros((nsubspaces,))

    for shift_idx in range(nshifts):
        shift = shifts[:, shift_idx]  # shape (nvars,)
        sign = signs[shift_idx]

        # Compute shifted indices for all subspaces at once
        # shift needs to be reshaped to (nvars, 1) for broadcasting
        shift_col = bkd.reshape(shift, (nvars, 1))
        shifted = subspace_indices + shift_col  # shape (nvars, nsubspaces)

        # Check membership for each shifted index
        # This loop is O(nsubspaces) but uses fast set lookup
        for j in range(nsubspaces):
            shifted_tuple = _index_to_tuple(shifted[:, j])
            if shifted_tuple in index_set:
                coefficients[j] = coefficients[j] + sign

    return coefficients


def is_downward_closed(subspace_indices: Array, bkd: Backend[Array]) -> bool:
    """Check if index set is downward closed.

    An index set K is downward closed if for every k in K,
    all indices k' with k'_i <= k_i for all i are also in K.

    This property is required for valid Smolyak combination.

    Parameters
    ----------
    subspace_indices : Array
        Multi-indices of subspaces, shape (nvars, nsubspaces)
    bkd : Backend[Array]
        Computational backend

    Returns
    -------
    bool
        True if the index set is downward closed

    Notes
    -----
    A downward-closed (or lower) set is essential for Smolyak combination.
    It ensures that all "predecessor" subspaces required for interpolation
    are available. Without this property, the combination would be incomplete.

    Mathematically, K is downward-closed if:
        k in K and k' <= k (componentwise) implies k' in K
    """
    nvars = subspace_indices.shape[0]
    nsubspaces = subspace_indices.shape[1]

    # Build set of index tuples for fast lookup
    index_set: Set[Tuple[int, ...]] = set()
    for j in range(nsubspaces):
        index_set.add(_index_to_tuple(subspace_indices[:, j]))

    # Check each index
    for j in range(nsubspaces):
        index = subspace_indices[:, j]

        # Check all predecessors (indices with one coordinate decremented)
        for dim in range(nvars):
            if int(index[dim]) > 0:
                predecessor = list(_index_to_tuple(index))
                predecessor[dim] -= 1
                if tuple(predecessor) not in index_set:
                    return False

    return True


def get_subspace_neighbors(
    index: Array,
    bkd: Backend[Array],
) -> Array:
    """Get forward neighbors of a subspace index.

    Forward neighbors are indices with one coordinate incremented by 1.

    Parameters
    ----------
    index : Array
        Multi-index of shape (nvars,)
    bkd : Backend[Array]
        Computational backend

    Returns
    -------
    Array
        Neighbor indices of shape (nvars, nvars)
    """
    nvars = index.shape[0]
    neighbors = bkd.zeros((nvars, nvars), dtype=bkd.int64_dtype())

    for dim in range(nvars):
        neighbors[:, dim] = index
        neighbors[dim, dim] = index[dim] + 1

    return neighbors


def check_admissibility(
    candidate: Array,
    existing_indices: Array,
    bkd: Backend[Array],
) -> bool:
    """Check if adding candidate maintains downward closure.

    A candidate index is admissible if all its predecessors
    (indices with one coordinate decremented) are already in the set.

    Parameters
    ----------
    candidate : Array
        Candidate multi-index of shape (nvars,)
    existing_indices : Array
        Current multi-indices, shape (nvars, nsubspaces)
    bkd : Backend[Array]
        Computational backend

    Returns
    -------
    bool
        True if candidate can be added while maintaining downward closure

    Notes
    -----
    Admissibility is used during adaptive refinement to ensure that adding
    a new index maintains the downward-closed property. A candidate is
    admissible if and only if all its immediate predecessors (indices with
    exactly one coordinate decremented by 1) are already in the set.
    """
    nvars = candidate.shape[0]
    nsubspaces = existing_indices.shape[1] if existing_indices.ndim > 1 else 0

    # Build set of existing indices
    index_set: Set[Tuple[int, ...]] = set()
    for j in range(nsubspaces):
        index_set.add(_index_to_tuple(existing_indices[:, j]))

    # Check all predecessors
    for dim in range(nvars):
        if int(candidate[dim]) > 0:
            predecessor = list(_index_to_tuple(candidate))
            predecessor[dim] -= 1
            if tuple(predecessor) not in index_set:
                return False

    return True
