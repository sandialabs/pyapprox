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

from typing import (
    Dict,
    Iterable,
    List,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

import numpy as np

from pyapprox.surrogates.sparsegrids.smolyak_dispatch import (
    get_smolyak_impl,
)
from pyapprox.util.backends.protocols import Array, Backend

# A subspace multi-index as a hashable key, including any config dims.
SubspaceKey = Tuple[int, ...]


def _index_to_tuple(
    index: Array, bkd: Backend[Array]
) -> Tuple[int, ...]:
    """Convert array index to hashable tuple.

    Parameters
    ----------
    index : Array
        1D array of shape (nvars,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Tuple[int, ...]
        Hashable tuple representation of the index
    """
    return tuple(bkd.to_int(index[i]) for i in range(index.shape[0]))


def _unit_box_shifts(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Enumerate the corners of the unit box {0,1}^n with their signs.

    Parameters
    ----------
    n : int
        Number of dimensions to enumerate over.

    Returns
    -------
    shifts : np.ndarray
        Corner offsets of shape (n, 2**n); row d holds bit d of the
        column's index, so column j is the binary expansion of j.
    signs : np.ndarray
        (-1)**|e| for each corner, shape (2**n,).
    """
    shift_ints = np.arange(2**n, dtype=np.int64)
    # Broadcast the per-dimension shift amounts (n, 1) against the corner
    # indices (2**n,) so row d is bit d of every corner at once.
    bit_positions = np.arange(n, dtype=np.int64)[:, None]
    shifts = (shift_ints >> bit_positions) & 1
    signs = (-1.0) ** shifts.sum(axis=0)
    return shifts, signs


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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # 2D isotropic sparse grid level 2
    >>> indices = bkd.asarray([[0, 1, 0, 2, 1, 0],
    ...                        [0, 0, 1, 0, 1, 2]])
    >>> coefs = compute_smolyak_coefficients(indices, bkd)
    """
    nvars = subspace_indices.shape[0]
    nsubspaces = subspace_indices.shape[1]

    # Work in numpy for fast tuple hashing and integer arithmetic
    np_indices = np.asarray(
        bkd.to_numpy(subspace_indices), dtype=np.int64
    )  # (nvars, nsubspaces)

    # Precompute all 2^nvars shift vectors and their signs
    np_shifts, np_signs = _unit_box_shifts(nvars)  # (nvars, nshifts), (nshifts,)
    nshifts = np_signs.shape[0]

    # Dispatch to best available implementation
    impl = get_smolyak_impl()
    np_coefs = impl(np_indices, np_shifts, np_signs, nvars, nsubspaces, nshifts)

    return bkd.asarray(np_coefs)


def backward_box(key: SubspaceKey) -> List[Tuple[int, SubspaceKey]]:
    """Enumerate the backward box of a subspace key with its signs.

    The box is {key - e : e in {0,1}^d restricted to the dimensions where
    key[i] > 0}, paired with the sign (-1)^|e|. It has 2^nnz entries,
    where nnz is the number of nonzero components, and the first entry is
    always (+1, key) itself. The entries with |e| = 1 are the backward
    neighbours.

    Adding an admissible key to a downward-closed set changes the Smolyak
    coefficients exactly on this box, by these signs:

        Delta c_{key - e} = (-1)^|e|

    so the box is the full support of the update.

    Parameters
    ----------
    key : SubspaceKey
        Multi-index of the subspace, as a tuple of levels.

    Returns
    -------
    List[Tuple[int, SubspaceKey]]
        (sign, key - e) pairs, with (+1, key) first.
    """
    nonzero_dims = [i for i, level in enumerate(key) if level > 0]
    shifts, signs = _unit_box_shifts(len(nonzero_dims))

    box: List[Tuple[int, SubspaceKey]] = []
    for corner in range(shifts.shape[1]):
        shifted = list(key)
        for row, dim in enumerate(nonzero_dims):
            shifted[dim] -= int(shifts[row, corner])
        box.append((int(signs[corner]), tuple(shifted)))
    return box


@runtime_checkable
class EvaluableProtocol(Protocol[Array]):
    """Anything that maps samples to values, such as a subspace."""

    def __call__(self, samples: Array) -> Array: ...


def evaluate_box(
    terms: Sequence[Tuple[int, EvaluableProtocol[Array]]],
    samples: Array,
) -> Array:
    """Evaluate the signed sum of subspace interpolants over a box.

    Returns sum_e (-1)^|e| I_{k-e}(x), which is the change the
    interpolant undergoes when subspace k is added:

        Delta I(x) = sum_e (-1)^|e| I_{k-e}(x)

    Only the box's subspaces are touched, so the cost is independent of
    how many subspaces the grid already holds.

    Parameters
    ----------
    terms : Sequence[Tuple[int, EvaluableProtocol[Array]]]
        (sign, subspace) pairs, as produced from a backward box.
    samples : Array
        Evaluation points, shape (nvars, npoints).

    Returns
    -------
    Array
        Signed sum of interpolant values, shape (nqoi, npoints).

    Raises
    ------
    ValueError
        If terms is empty, since the result's shape is unknown.
    """
    if len(terms) == 0:
        raise ValueError("cannot evaluate an empty box")
    sign, subspace = terms[0]
    total = sign * subspace(samples)
    for sign, subspace in terms[1:]:
        total = total + sign * subspace(samples)
    return total


class IncrementalSmolyakCoefficients:
    """Smolyak coefficients of a downward-closed set, updated in place.

    Recomputing the coefficients from scratch costs O(n 2^d) in the size
    of the set. Adding one admissible index instead touches only its
    backward box, which is O(2^nnz) and independent of how large the set
    has grown.

    Coefficients are exact Python ints: the inclusion-exclusion sum is
    integral, and keeping it so avoids the drift a float accumulator
    would pick up over many updates.

    Parameters
    ----------
    nvars : int
        Length of every subspace key, including any config dimensions.

    Examples
    --------
    >>> coefs = IncrementalSmolyakCoefficients(2)
    >>> coefs.add((0, 0))
    >>> coefs.add((1, 0))
    >>> coefs.coefficient((0, 0))
    0
    >>> coefs.coefficient((1, 0))
    1
    """

    def __init__(self, nvars: int) -> None:
        if nvars < 1:
            raise ValueError(f"nvars must be positive, got {nvars}")
        self._nvars = nvars
        # Insertion-ordered: keys() must return the order subspaces were
        # added, so callers can align coefficients with their own lists.
        self._coefs: Dict[SubspaceKey, int] = {}

    def _validate(self, key: SubspaceKey) -> None:
        """Raise if the key has the wrong length or negative levels."""
        if len(key) != self._nvars:
            raise ValueError(
                f"key {key} has length {len(key)}, expected {self._nvars}"
            )
        if any(level < 0 for level in key):
            raise ValueError(f"key {key} has a negative level")

    def can_add(self, key: SubspaceKey) -> bool:
        """Return whether adding the key keeps the set downward closed.

        True when the key is absent and every backward neighbour (each
        |e| = 1 entry of its box) is already present. Named ``can_add``
        rather than ``is_admissible`` to keep it distinct from
        ``AdmissibilityCriteria``, which additionally applies level caps.

        Parameters
        ----------
        key : SubspaceKey
            Candidate multi-index.

        Returns
        -------
        bool
            True if the key may be added.
        """
        self._validate(key)
        if key in self._coefs:
            return False
        for dim, level in enumerate(key):
            if level > 0:
                neighbor = key[:dim] + (level - 1,) + key[dim + 1 :]
                if neighbor not in self._coefs:
                    return False
        return True

    def delta(self, key: SubspaceKey) -> List[Tuple[SubspaceKey, int]]:
        """Return the coefficient change from adding the key, without adding.

        Every entry of the backward box other than the key itself is
        already in the set, so this is the complete set of coefficients
        that move.

        Parameters
        ----------
        key : SubspaceKey
            Multi-index to be added.

        Returns
        -------
        List[Tuple[SubspaceKey, int]]
            (key - e, (-1)^|e|) pairs, the key itself first.

        Raises
        ------
        ValueError
            If the key cannot be added.
        """
        if not self.can_add(key):
            raise ValueError(
                f"key {key} cannot be added: it is already present or a "
                "backward neighbour is missing"
            )
        return [(box_key, sign) for sign, box_key in backward_box(key)]

    def add(self, key: SubspaceKey) -> None:
        """Add an admissible key, updating coefficients on its box.

        Parameters
        ----------
        key : SubspaceKey
            Multi-index to add.

        Raises
        ------
        ValueError
            If the key cannot be added.
        """
        changes = self.delta(key)
        # Insert first so the key keeps its insertion position even though
        # its own delta is applied alongside the rest of the box.
        self._coefs[key] = 0
        for box_key, sign in changes:
            self._coefs[box_key] += sign

    def with_added(self, keys: Iterable[SubspaceKey]) -> (
        "IncrementalSmolyakCoefficients"
    ):
        """Return a copy with the given keys added, leaving self unchanged.

        The keys are added in order of increasing level sum, so a set
        that is jointly admissible may be passed in any order: every
        backward neighbour of a key has a strictly smaller level sum and
        is therefore added first.

        Parameters
        ----------
        keys : Iterable[SubspaceKey]
            Keys to add. Order does not matter, but each must be
            admissible once the lower-level-sum keys are present.

        Returns
        -------
        IncrementalSmolyakCoefficients
            A new object; self is not modified.

        Raises
        ------
        ValueError
            If a key is already present, or is still inadmissible when
            reached.
        """
        clone = IncrementalSmolyakCoefficients(self._nvars)
        clone._coefs = dict(self._coefs)
        for key in sorted(keys, key=sum):
            clone.add(key)
        return clone

    def keys(self) -> List[SubspaceKey]:
        """Return all keys in insertion order."""
        return list(self._coefs)

    def nonzero_items(self) -> List[Tuple[SubspaceKey, int]]:
        """Return (key, coefficient) for keys with a nonzero coefficient."""
        return [(key, c) for key, c in self._coefs.items() if c != 0]

    def coefficient(self, key: SubspaceKey) -> int:
        """Return the coefficient of a key, or 0 if it is absent."""
        return self._coefs.get(key, 0)

    def coefficient_list(self, keys: Sequence[SubspaceKey]) -> List[int]:
        """Return coefficients aligned with the given keys."""
        return [self._coefs.get(key, 0) for key in keys]

    def nterms(self) -> int:
        """Return the number of keys held, including zero-coefficient ones."""
        return len(self._coefs)

    def nvars(self) -> int:
        """Return the length of every key."""
        return self._nvars

    def __repr__(self) -> str:
        return (
            f"IncrementalSmolyakCoefficients(nvars={self._nvars}, "
            f"nterms={len(self._coefs)})"
        )


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
        index_set.add(_index_to_tuple(subspace_indices[:, j], bkd))

    # Check each index
    for j in range(nsubspaces):
        index = subspace_indices[:, j]

        # Check all predecessors (indices with one coordinate decremented)
        for dim in range(nvars):
            if bkd.to_int(index[dim]) > 0:
                predecessor = list(_index_to_tuple(index, bkd))
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
        index_set.add(_index_to_tuple(existing_indices[:, j], bkd))

    # Check all predecessors
    for dim in range(nvars):
        if bkd.to_int(candidate[dim]) > 0:
            predecessor = list(_index_to_tuple(candidate, bkd))
            predecessor[dim] -= 1
            if tuple(predecessor) not in index_set:
                return False

    return True
