"""Protocols for adaptive iteration.

This module defines protocols for adaptive iteration patterns used in
sparse grids and polynomial chaos expansion refinement.

Protocol Hierarchy:
    AdaptiveIteratorProtocol - step_samples/step_values iteration pattern
    PrioritizedCandidateQueueProtocol - priority queue for candidates
    BasisIndexGeneratorProtocol - maps subspace indices to basis indices
"""

from typing import Generic, List, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class AdaptiveIteratorProtocol(Protocol, Generic[Array]):
    """Protocol for adaptive iterators.

    Adaptive iterators follow a step_samples/step_values pattern:
    1. Call step_samples() to get new sample locations
    2. Evaluate function at samples
    3. Call step_values(values) to incorporate results
    4. Repeat until converged

    Methods
    -------
    step_samples() -> Optional[Array]
        Get new sample locations for the next refinement step.
    step_values(values: Array) -> None
        Incorporate function values at the sampled locations.
    error() -> float
        Return current error estimate.
    """

    def step_samples(self) -> Optional[Array]:
        """Get sample locations for the next refinement step.

        Returns
        -------
        Array or None
            Sample locations. Shape: (nvars, nsamples)
            Returns None if iteration is complete.
        """
        ...

    def step_values(self, values: Array) -> None:
        """Incorporate function values at sample locations.

        Parameters
        ----------
        values : Array
            Function values at samples from step_samples().
            Shape: (nsamples, nqoi)
        """
        ...

    def error(self) -> float:
        """Return current error estimate.

        Returns
        -------
        float
            Estimate of approximation error.
        """
        ...


@runtime_checkable
class PrioritizedCandidateQueueProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for priority queues of candidate indices.

    Priority queues manage candidate indices for adaptive refinement,
    ordering them by error/priority for efficient selection.

    Methods
    -------
    empty() -> bool
        Return True if the queue is empty.
    put(priority: float, error: float, index_id: int) -> None
        Add a candidate to the queue.
    get() -> Tuple[float, float, int]
        Remove and return the highest priority candidate.
    """

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        ...

    def put(self, priority: float, error: float, index_id: int) -> None:
        """Add a candidate to the queue.

        Parameters
        ----------
        priority : float
            Refinement priority (higher = refine first).
        error : float
            Error estimate for this candidate.
        index_id : int
            Identifier for the candidate index.
        """
        ...

    def get(self) -> Tuple[float, float, int]:
        """Remove and return the highest priority candidate.

        Returns
        -------
        Tuple[float, float, int]
            (priority, error, index_id) for the highest priority candidate.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        ...


@runtime_checkable
class BasisIndexGeneratorProtocol(Protocol, Generic[Array]):
    """Protocol for mapping subspace indices to basis function indices.

    Basis index generators map multi-indices representing subspaces
    to the actual basis function indices within each subspace.
    Supports both PCE and sparse grid use cases.

    Methods
    -------
    bkd() -> Backend[Array]
        Return the computational backend.
    nvars() -> int
        Return the number of variables.
    nrefinement_vars() -> int
        Return the number of refinement variables (for multi-fidelity).
    nunivariate_basis(subspace_index: Array) -> List[int]
        Return number of univariate basis functions per dimension.
    refine_subspace_index(subspace_index: Array) -> Array
        Return children indices for refinement.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def nrefinement_vars(self) -> int:
        """Return the number of refinement variables.

        For standard approximations, this equals nvars.
        For multi-fidelity, this may differ.

        Returns
        -------
        int
            Number of refinement variables.
        """
        ...

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
        ...

    def refine_subspace_index(self, subspace_index: Array) -> Array:
        """Return children indices for refinement.

        Parameters
        ----------
        subspace_index : Array
            Multi-index to refine. Shape: (nvars,)

        Returns
        -------
        Array
            Children indices. Shape: (nvars, nchildren)
        """
        ...
