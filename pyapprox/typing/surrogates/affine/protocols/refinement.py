"""Protocols for adaptive refinement.

This module defines protocols for refinement criteria and cost functions
used in adaptive sparse grids and polynomial chaos expansions.

Protocol Hierarchy:
    CostFunctionProtocol - computes cost of refining an index
    RefinementCriteriaProtocol - computes error and priority for refinement
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array


@runtime_checkable
class CostFunctionProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for computing refinement costs.

    Cost functions estimate the computational cost of refining a given
    multi-index, used to prioritize refinement by error/cost ratio.

    Methods
    -------
    __call__(index: Array) -> float
        Return the cost of refining the given index.
    """

    def __call__(self, index: Array) -> float:
        """Compute the cost of refining an index.

        Parameters
        ----------
        index : Array
            Multi-index to evaluate. Shape: (nvars,)

        Returns
        -------
        float
            Estimated cost of refining this index.
        """
        ...


@runtime_checkable
class RefinementCriteriaProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for refinement criteria.

    Refinement criteria determine which indices should be refined based
    on error estimates and priorities. Used in adaptive algorithms.

    Methods
    -------
    __call__(index: Array) -> Tuple[float, float]
        Return (error, priority) for the given index.
    cost(index: Array) -> float
        Return the cost of refining the given index.
    """

    def __call__(self, index: Array) -> Tuple[float, float]:
        """Compute error estimate and priority for an index.

        Parameters
        ----------
        index : Array
            Multi-index to evaluate. Shape: (nvars,)

        Returns
        -------
        Tuple[float, float]
            (error_estimate, priority) where:
            - error_estimate: Contribution to approximation error
            - priority: Refinement priority (higher = refine first)
        """
        ...

    def cost(self, index: Array) -> float:
        """Compute the cost of refining an index.

        Parameters
        ----------
        index : Array
            Multi-index to evaluate. Shape: (nvars,)

        Returns
        -------
        float
            Estimated cost of refining this index.
        """
        ...
