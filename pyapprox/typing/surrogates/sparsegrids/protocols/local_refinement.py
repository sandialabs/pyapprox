"""Protocols for local (hierarchical) sparse grid refinement.

This module defines protocols for locally-adaptive sparse grids
that refine individual basis functions rather than entire subspaces.
"""

from typing import Generic, List, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend

from .sparse_grid import SubspaceProtocol


@runtime_checkable
class LocalIndexGeneratorProtocol(Protocol, Generic[Array]):
    """Protocol for hierarchical local index generation.

    Local indices identify individual basis functions within a subspace.
    For piecewise polynomial bases, refinement creates children by
    splitting intervals.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def get_children(self, local_index: Array) -> List[Array]:
        """Get child indices for refinement.

        Parameters
        ----------
        local_index : Array
            Local index identifying a basis function

        Returns
        -------
        List[Array]
            Child indices created by refinement
        """
        ...

    def get_level(self, local_index: Array) -> int:
        """Get hierarchical level of a local index.

        Parameters
        ----------
        local_index : Array
            Local index identifying a basis function

        Returns
        -------
        int
            Hierarchical level (depth in refinement tree)
        """
        ...


@runtime_checkable
class LocalRefinementCriteriaProtocol(Protocol, Generic[Array]):
    """Protocol for local (basis-level) refinement criteria.

    Determines priority for refining individual basis functions
    within a subspace, typically based on hierarchical surplus.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(
        self,
        subspace: SubspaceProtocol[Array],
        local_index: Array,
    ) -> float:
        """Compute refinement priority for a local index.

        Parameters
        ----------
        subspace : SubspaceProtocol[Array]
            The subspace containing the basis function
        local_index : Array
            Local index identifying the basis function

        Returns
        -------
        float
            Refinement priority (higher = refine first)
        """
        ...
