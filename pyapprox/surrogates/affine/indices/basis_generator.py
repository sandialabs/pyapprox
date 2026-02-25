"""Basis index generator for mapping subspaces to basis functions.

This module provides the BasisIndexGenerator class that maps multi-indices
representing subspaces to actual basis function indices. Used in both
PCE and sparse grid contexts.
"""

import math
from typing import Dict, Generic, List, Optional, Tuple, Union

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.protocols.index import (
    IndexGrowthRuleProtocol,
)
from pyapprox.surrogates.affine.indices.growth_rules import (
    LinearGrowthRule,
)
from pyapprox.surrogates.affine.indices.utils import hash_index


class BasisIndexGenerator(Generic[Array]):
    """Maps subspace indices to basis function indices.

    For each subspace (identified by a multi-index), computes:
    - Number of univariate basis functions per dimension
    - Sample points within the subspace
    - Mapping from subspace to global basis indices

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables.
    growth_rules : IndexGrowthRuleProtocol or List[IndexGrowthRuleProtocol], optional
        Growth rule(s) for mapping levels to basis counts. If a single rule,
        it is used for all dimensions. If a list, each element applies to
        the corresponding dimension.
        Default: LinearGrowthRule(scale=1, shift=1) for all dimensions.
    nrefinement_vars : int, optional
        Number of refinement variables (for multi-fidelity).
        Default: nvars.
    all_nested : bool, optional
        Whether all quadrature rules are nested. If True, basis index hashing
        is used for deduplication. If False, sample coordinate hashing is used.
        Default: False.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> gen = BasisIndexGenerator(bkd, nvars=2)
    >>> index = bkd.asarray([2, 1])
    >>> gen.nunivariate_basis(index)
    [3, 2]
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        growth_rules: Optional[
            Union[IndexGrowthRuleProtocol, List[IndexGrowthRuleProtocol]]
        ] = None,
        nrefinement_vars: Optional[int] = None,
        all_nested: bool = False,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._nrefinement_vars = nrefinement_vars or nvars
        self._all_nested = all_nested

        # Handle growth rules - normalize to list for uniform handling
        if growth_rules is None:
            default_rule = LinearGrowthRule(scale=1, shift=1)
            self._growth_rules: List[IndexGrowthRuleProtocol] = [
                default_rule for _ in range(nvars)
            ]
        elif isinstance(growth_rules, list):
            if len(growth_rules) != nvars:
                raise ValueError(
                    f"growth_rules list length ({len(growth_rules)}) must match "
                    f"nvars ({nvars})"
                )
            self._growth_rules = growth_rules
        else:
            # Single rule - replicate for all dimensions
            self._growth_rules = [growth_rules for _ in range(nvars)]

        # Deduplication tracking for sparse grids
        # Maps hash -> (global_idx, first_subspace_idx)
        self._basis_indices_dict: Dict[int, Tuple[int, int]] = {}
        # Per-subspace mapping: local sample index -> global value index
        self._subspace_basis_idx: List[Array] = []
        # Per-subspace: list of local indices that are unique (first occurrence)
        self._unique_subspace_basis_idx: List[List[int]] = []

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nrefinement_vars(self) -> int:
        """Return the number of refinement variables.

        For standard approximations, this equals nvars.
        For multi-fidelity, this may differ.

        Returns
        -------
        int
            Number of refinement variables.
        """
        return self._nrefinement_vars

    def nunivariate_basis(self, subspace_index: Array) -> List[int]:
        """Return number of univariate basis functions per dimension.

        Parameters
        ----------
        subspace_index : Array
            Multi-index specifying the subspace. Shape: (nvars,)

        Returns
        -------
        List[int]
            Number of basis functions in each dimension.
        """
        index_np = self._bkd.to_numpy(subspace_index)
        return [
            self._growth_rules[dim](int(index_np[dim]))
            for dim in range(self._nvars)
        ]

    def nsubspace_basis(self, subspace_index: Array) -> int:
        """Return total number of basis functions in subspace.

        Parameters
        ----------
        subspace_index : Array
            Multi-index specifying the subspace. Shape: (nvars,)

        Returns
        -------
        int
            Total number of basis functions (product of univariate counts).
        """
        counts = self.nunivariate_basis(subspace_index)
        result = 1
        for c in counts:
            result *= c
        return result

    def refine_subspace_index(self, subspace_index: Array) -> Array:
        """Return children indices for refinement.

        Children are obtained by incrementing each dimension by 1.

        Parameters
        ----------
        subspace_index : Array
            Multi-index to refine. Shape: (nvars,)

        Returns
        -------
        Array
            Children indices. Shape: (nvars, nvars)
            Each column is a child index.
        """
        children = []
        for dim in range(self._nrefinement_vars):
            child = self._bkd.copy(subspace_index)
            child[dim] = child[dim] + 1
            children.append(child)

        return self._bkd.stack(children, axis=1)

    def get_basis_indices(self, subspace_index: Array) -> Array:
        """Return all basis function indices within a subspace.

        Parameters
        ----------
        subspace_index : Array
            Multi-index specifying the subspace. Shape: (nvars,)

        Returns
        -------
        Array
            Basis indices within subspace. Shape: (nvars, nbasis)
            Each column is a basis function multi-index.
        """
        counts = self.nunivariate_basis(subspace_index)
        nbasis = self.nsubspace_basis(subspace_index)

        if nbasis == 0:
            return self._bkd.zeros(
                (self._nvars, 0), dtype=self._bkd.int64_dtype()
            )

        # Generate tensor product of indices
        indices = self._bkd.zeros(
            (self._nvars, nbasis), dtype=self._bkd.int64_dtype()
        )

        # Compute stride for each dimension
        stride = 1
        for dim in range(self._nvars - 1, -1, -1):
            npts = counts[dim]
            for idx in range(nbasis):
                indices[dim, idx] = (idx // stride) % npts
            stride *= npts

        return indices

    def get_hierarchical_surplus_indices(
        self, subspace_index: Array
    ) -> Array:
        """Return indices of hierarchical surplus basis functions.

        For a subspace at level l, the hierarchical surplus contains
        basis functions at level l but not at level l-1.

        Parameters
        ----------
        subspace_index : Array
            Multi-index specifying the subspace. Shape: (nvars,)

        Returns
        -------
        Array
            Hierarchical surplus indices. Shape: (nvars, nsurplus)
        """
        # Get all basis indices in this subspace
        all_indices = self.get_basis_indices(subspace_index)

        # For simplest case (linear growth), surplus is the outer boundary
        # This is a simplified implementation
        return all_indices

    # -------------------------------------------------------------------------
    # Deduplication methods for sparse grids
    # -------------------------------------------------------------------------

    def _hash_sample_coords(
        self, sample: Array, tolerance: float = 1e-12
    ) -> int:
        """Hash sample coordinates with tolerance for floating-point comparison.

        Used for non-nested quadrature rules where basis indices don't uniquely
        identify sample points across subspaces.

        Parameters
        ----------
        sample : Array
            Sample coordinates. Shape: (nvars,)
        tolerance : float, optional
            Tolerance for rounding. Default: 1e-12.

        Returns
        -------
        int
            Hash of the rounded coordinates.
        """
        decimals = int(-math.log10(tolerance))
        np_sample = self._bkd.to_numpy(sample)
        rounded = tuple(round(float(x), decimals) for x in np_sample)
        return hash(rounded)

    def _set_unique_subspace_basis_indices(
        self,
        subspace_index: Array,
        subspace_idx: int,
        subspace_samples: Optional[Array] = None,
    ) -> None:
        """Build basis index to global sample index mappings for a subspace.

        For nested quadrature rules (all_nested=True), deduplication is based
        on integer basis indices. For non-nested rules, deduplication uses
        sample coordinates with tolerance-based hashing.

        Parameters
        ----------
        subspace_index : Array
            Multi-index specifying the subspace. Shape: (nvars,)
        subspace_idx : int
            Index of this subspace in the subspace list.
        subspace_samples : Array, optional
            Sample coordinates for this subspace. Shape: (nvars, nsamples)
            Required when all_nested=False.

        Raises
        ------
        ValueError
            If all_nested=False and subspace_samples is None.
        """
        if not self._all_nested and subspace_samples is None:
            raise ValueError(
                "subspace_samples required for non-nested quadrature rules"
            )

        unique_subspace_sample_idx: List[int] = []
        global_basis_idx: List[int] = []
        idx = len(self._basis_indices_dict)

        basis_indices = self.get_basis_indices(subspace_index)
        nsamples = basis_indices.shape[1]

        for sample_idx in range(nsamples):
            if self._all_nested:
                # NESTED PATH: Hash basis indices (integers)
                basis_index = basis_indices[:, sample_idx]
                key = hash_index(basis_index, self._bkd)
            else:
                # NON-NESTED PATH: Hash sample coordinates with tolerance
                assert subspace_samples is not None
                sample = subspace_samples[:, sample_idx]
                key = self._hash_sample_coords(sample)

            if key not in self._basis_indices_dict:
                self._basis_indices_dict[key] = (idx, subspace_idx)
                unique_subspace_sample_idx.append(sample_idx)
                global_basis_idx.append(idx)
                idx += 1
            else:
                global_basis_idx.append(self._basis_indices_dict[key][0])

        self._unique_subspace_basis_idx.append(unique_subspace_sample_idx)
        self._subspace_basis_idx.append(
            self._bkd.asarray(global_basis_idx, dtype=self._bkd.int64_dtype())
        )

    def n_unique_samples(self) -> int:
        """Return total number of unique samples across all subspaces.

        Returns
        -------
        int
            Number of unique samples.
        """
        return len(self._basis_indices_dict)

    def get_unique_local_indices(self, subspace_idx: int) -> List[int]:
        """Return local indices of unique samples for a subspace.

        Parameters
        ----------
        subspace_idx : int
            Index of the subspace.

        Returns
        -------
        List[int]
            Local sample indices that are unique (first occurrence).
        """
        return self._unique_subspace_basis_idx[subspace_idx]

    def get_subspace_value_indices(self, subspace_idx: int) -> Array:
        """Return global value indices for all samples in a subspace.

        Parameters
        ----------
        subspace_idx : int
            Index of the subspace.

        Returns
        -------
        Array
            Array mapping local sample index to global value index.
            Shape: (nsamples_in_subspace,)
        """
        return self._subspace_basis_idx[subspace_idx]

    def __repr__(self) -> str:
        return (
            f"BasisIndexGenerator(nvars={self._nvars}, "
            f"growth_rule={self._growth_rule})"
        )
