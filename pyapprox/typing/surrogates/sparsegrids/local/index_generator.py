"""Local (hierarchical) index generator for adaptive sparse grids.

This module provides index generation for locally-adaptive sparse grids
that refine individual basis functions using hierarchical splitting.
"""

from typing import Dict, Generic, List, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


class LocalIndexGenerator(Generic[Array]):
    """Generator for hierarchical local basis indices.

    Manages basis function indices for locally-adaptive sparse grids.
    Uses a hierarchical structure where each basis function can be
    refined into children by splitting its support interval.

    For the standard piecewise linear hierarchical basis:
    - Level 0: single basis at x=0
    - Level 1: bases at x=1, x=2 (left/right children of 0)
    - Level n: basis at index k has children at 2k-1 and 2k

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> gen = LocalIndexGenerator(bkd, nvars=2)
    >>> # Get children of the root basis function
    >>> root = bkd.zeros((2,), dtype=bkd.int64_dtype())
    >>> children = gen.get_children(root)
    """

    def __init__(self, bkd: Backend[Array], nvars: int):
        self._bkd = bkd
        self._nvars = nvars

        # Tracking indices
        self._basis_indices: Optional[Array] = None
        self._basis_indices_dict: Dict[Tuple[int, ...], int] = {}
        self._sel_basis_indices_dict: Dict[Tuple[int, ...], int] = {}
        self._cand_basis_indices_dict: Dict[Tuple[int, ...], int] = {}

    @property
    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def _hash_index(self, index: Array) -> Tuple[int, ...]:
        """Convert index array to hashable tuple."""
        return tuple(int(index[i]) for i in range(self._nvars))

    def get_level(self, basis_index: Array) -> int:
        """Get hierarchical level of a basis index.

        The level is computed as the maximum level across all dimensions.
        For each dimension:
        - Index 0 is level 0
        - Index 1, 2 are level 1
        - Index 3, 4 are level 2
        - Index 2^n - 1 to 2^(n+1) - 2 are level n

        Parameters
        ----------
        basis_index : Array
            Local index identifying a basis function. Shape: (nvars,)

        Returns
        -------
        int
            Hierarchical level (maximum across dimensions)
        """
        max_level = 0
        for d in range(self._nvars):
            idx = int(basis_index[d])
            if idx == 0:
                level = 0
            elif idx <= 2:
                level = 1
            else:
                # Level n has indices from 2^n - 1 to 2^(n+1) - 2
                import math
                level = int(math.floor(math.log2(idx + 1)))
            max_level = max(max_level, level)
        return max_level

    def _left_child(self, basis_index: Array, dim: int) -> Optional[Array]:
        """Get left child index in given dimension.

        Parameters
        ----------
        basis_index : Array
            Parent basis index. Shape: (nvars,)
        dim : int
            Dimension to refine.

        Returns
        -------
        Optional[Array]
            Left child index, or None if not valid.
        """
        idx = int(basis_index[dim])
        if idx == 1:
            # Index 1 has no left child (it's the left boundary)
            return None

        child = self._bkd.copy(basis_index)
        if idx == 0:
            child[dim] = 1
        elif idx == 2:
            child[dim] = 4
        else:
            child[dim] = 2 * idx - 1
        return child

    def _right_child(self, basis_index: Array, dim: int) -> Optional[Array]:
        """Get right child index in given dimension.

        Parameters
        ----------
        basis_index : Array
            Parent basis index. Shape: (nvars,)
        dim : int
            Dimension to refine.

        Returns
        -------
        Optional[Array]
            Right child index, or None if not valid.
        """
        idx = int(basis_index[dim])
        if idx == 2:
            # Index 2 has no right child (it's the right boundary)
            return None

        child = self._bkd.copy(basis_index)
        if idx == 0:
            child[dim] = 2
        elif idx == 1:
            child[dim] = 3
        else:
            child[dim] = 2 * idx
        return child

    def get_children(self, basis_index: Array) -> List[Array]:
        """Get all child indices for refinement.

        Each dimension can contribute up to 2 children (left and right).
        Only returns valid children that haven't been added yet.

        Parameters
        ----------
        basis_index : Array
            Parent basis index. Shape: (nvars,)

        Returns
        -------
        List[Array]
            List of child indices.
        """
        children: List[Array] = []
        for dim in range(self._nvars):
            left = self._left_child(basis_index, dim)
            if left is not None:
                key = self._hash_index(left)
                if (key not in self._sel_basis_indices_dict and
                    key not in self._cand_basis_indices_dict):
                    children.append(left)

            right = self._right_child(basis_index, dim)
            if right is not None:
                key = self._hash_index(right)
                if (key not in self._sel_basis_indices_dict and
                    key not in self._cand_basis_indices_dict):
                    children.append(right)

        return children

    def _parent(self, basis_index: Array, dim: int) -> Array:
        """Get parent index in given dimension.

        Parameters
        ----------
        basis_index : Array
            Child basis index. Shape: (nvars,)
        dim : int
            Dimension to get parent in.

        Returns
        -------
        Array
            Parent index.
        """
        parent = self._bkd.copy(basis_index)
        idx = int(basis_index[dim])
        if idx <= 2:
            parent[dim] = 0
        else:
            parent[dim] = (idx + (idx % 2)) // 2
        return parent

    def initialize(self) -> Array:
        """Initialize with root basis function.

        Returns
        -------
        Array
            Initial basis indices of shape (nvars, 1).
        """
        root = self._bkd.zeros((self._nvars,), dtype=self._bkd.int64_dtype())
        self._basis_indices = root[:, None]
        key = self._hash_index(root)
        self._basis_indices_dict[key] = 0
        self._sel_basis_indices_dict[key] = 0
        return self._basis_indices

    def add_candidates(self, basis_index: Array) -> List[Array]:
        """Add candidate children for a selected basis index.

        Parameters
        ----------
        basis_index : Array
            Selected basis index to generate children from. Shape: (nvars,)

        Returns
        -------
        List[Array]
            New candidate basis indices.
        """
        children = self.get_children(basis_index)
        new_candidates: List[Array] = []

        for child in children:
            key = self._hash_index(child)
            if key not in self._basis_indices_dict:
                idx = self._basis_indices.shape[1]
                self._basis_indices = self._bkd.hstack(
                    (self._basis_indices, child[:, None])
                )
                self._basis_indices_dict[key] = idx
                self._cand_basis_indices_dict[key] = idx
                new_candidates.append(child)

        return new_candidates

    def select_candidate(self, basis_index: Array) -> None:
        """Move a candidate to selected status.

        Parameters
        ----------
        basis_index : Array
            Candidate basis index to select. Shape: (nvars,)
        """
        key = self._hash_index(basis_index)
        if key in self._cand_basis_indices_dict:
            idx = self._cand_basis_indices_dict[key]
            del self._cand_basis_indices_dict[key]
            self._sel_basis_indices_dict[key] = idx

    def nselected(self) -> int:
        """Return number of selected basis indices."""
        return len(self._sel_basis_indices_dict)

    def ncandidates(self) -> int:
        """Return number of candidate basis indices."""
        return len(self._cand_basis_indices_dict)

    def get_selected_indices(self) -> Array:
        """Return selected basis indices.

        Returns
        -------
        Array
            Selected indices of shape (nvars, nselected).
        """
        if self.nselected() == 0:
            return self._bkd.zeros(
                (self._nvars, 0), dtype=self._bkd.int64_dtype()
            )
        idxs = list(self._sel_basis_indices_dict.values())
        return self._basis_indices[:, idxs]

    def get_candidate_indices(self) -> Optional[Array]:
        """Return candidate basis indices.

        Returns
        -------
        Optional[Array]
            Candidate indices of shape (nvars, ncandidates), or None if empty.
        """
        if self.ncandidates() == 0:
            return None
        idxs = list(self._cand_basis_indices_dict.values())
        return self._basis_indices[:, idxs]

    def __repr__(self) -> str:
        return (
            f"LocalIndexGenerator(nvars={self._nvars}, "
            f"nselected={self.nselected()}, ncandidates={self.ncandidates()})"
        )
