"""Basis index generator for mapping subspaces to basis functions.

This module provides the BasisIndexGenerator class that maps multi-indices
representing subspaces to actual basis function indices. Used in both
PCE and sparse grid contexts.
"""

from typing import Generic, List, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols.index import (
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.affine.indices.growth_rules import (
    LinearGrowthRule,
)


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
    growth_rule : IndexGrowthRuleProtocol, optional
        Growth rule for mapping levels to basis counts.
        Default: LinearGrowthRule(scale=1, shift=1).
    nrefinement_vars : int, optional
        Number of refinement variables (for multi-fidelity).
        Default: nvars.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
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
        growth_rule: Optional[IndexGrowthRuleProtocol] = None,
        nrefinement_vars: Optional[int] = None,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._growth_rule = growth_rule or LinearGrowthRule(scale=1, shift=1)
        self._nrefinement_vars = nrefinement_vars or nvars

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
        return [self._growth_rule(int(level)) for level in index_np]

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

    def __repr__(self) -> str:
        return (
            f"BasisIndexGenerator(nvars={self._nvars}, "
            f"growth_rule={self._growth_rule})"
        )
