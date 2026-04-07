"""Protocols for multi-index generation.

This module defines protocols for index generation used in polynomial
chaos expansions and other tensor-product approximations.

Protocol Hierarchy:
    IndexGeneratorProtocol (base) - basic index generation
        ↓
    IterativeIndexGeneratorProtocol - adaptive index refinement

Admissibility protocols:
    AdmissibilityCriteriaProtocol - criteria for admissible indices

Growth rule protocols:
    IndexGrowthRuleProtocol - univariate level-to-degree mapping
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class IndexGeneratorProtocol(Protocol, Generic[Array]):
    """Protocol for multi-index generators.

    Multi-index generators produce sets of multi-indices that define
    the terms in a polynomial chaos expansion or similar approximation.

    Methods
    -------
    bkd() -> Backend[Array]
        Return the computational backend.
    nvars() -> int
        Return the number of variables.
    nindices() -> int
        Return the number of indices.
    get_indices() -> Array
        Return the multi-indices.

    Notes
    -----
    Indices shape: (nvars, nindices)
    Each column is a multi-index specifying degrees for each variable.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def nindices(self) -> int:
        """Return the number of multi-indices."""
        ...

    def get_indices(self) -> Array:
        """Return the multi-indices.

        Returns
        -------
        Array
            Multi-indices. Shape: (nvars, nindices)
        """
        ...


@runtime_checkable
class IterativeIndexGeneratorProtocol(Protocol, Generic[Array]):
    """Protocol for iterative/adaptive index generators.

    Iterative generators maintain separate selected and candidate index sets,
    allowing for adaptive refinement of the approximation.

    Methods
    -------
    get_selected_indices() -> Array
        Return indices currently in the approximation.
    get_candidate_indices() -> Array
        Return indices available for refinement.
    refine_index(index: Array) -> Array
        Add an index to the selected set, return new candidates.
    step() -> None
        Advance the generator by one level.
    """

    def nselected_indices(self) -> int:
        """Return the number of selected indices."""
        ...

    def ncandidate_indices(self) -> int:
        """Return the number of candidate indices."""
        ...

    def get_selected_indices(self) -> Array:
        """Return the selected indices.

        Returns
        -------
        Array
            Selected multi-indices. Shape: (nvars, nselected)
        """
        ...

    def get_candidate_indices(self) -> Optional[Array]:
        """Return the candidate indices.

        Returns
        -------
        Array or None
            Candidate multi-indices. Shape: (nvars, ncandidate)
            Returns None if no candidates available.
        """
        ...

    def refine_index(self, index: Array) -> Array:
        """Move an index from candidates to selected.

        Parameters
        ----------
        index : Array
            Multi-index to refine. Shape: (nvars,)

        Returns
        -------
        Array
            New candidate indices generated. Shape: (nvars, nnew)
        """
        ...

    def step(self) -> None:
        """Advance the generator by one level/iteration."""
        ...


@runtime_checkable
class AdmissibilityCriteriaProtocol(Protocol, Generic[Array]):  # type: ignore[misc]
    """Protocol for index admissibility criteria.

    Admissibility criteria determine which indices can be added to
    a multi-index set while maintaining desired properties like
    downward closure or bounded level.

    Methods
    -------
    __call__(index: Array) -> bool
        Check if an index is admissible.
    """

    def __call__(self, index: Array) -> bool:
        """Check if an index is admissible.

        Parameters
        ----------
        index : Array
            Multi-index to check. Shape: (nvars,)

        Returns
        -------
        bool
            True if the index is admissible.
        """
        ...


@runtime_checkable
class IndexGrowthRuleProtocol(Protocol):
    """Protocol for univariate index growth rules.

    Growth rules map polynomial levels to the number of basis functions
    at that level, controlling the growth of tensor-product approximations.

    Methods
    -------
    __call__(level: int) -> int
        Return the number of basis functions at the given level.
    """

    def __call__(self, level: int) -> int:
        """Return the number of basis functions at the given level.

        Parameters
        ----------
        level : int
            Polynomial level.

        Returns
        -------
        int
            Number of basis functions at this level.
        """
        ...


@runtime_checkable
class IndexSequenceProtocol(Protocol, Generic[Array]):
    """Maps an integer level to a multi-index set.

    Implementations produce index sets of shape ``(nvars, nterms)`` for
    a given level, suitable for sweeping over candidate index sets in
    cross-validation or model selection.

    Methods
    -------
    __call__(level: int) -> Array
        Return the multi-index set for the given level.
    """

    def __call__(self, level: int) -> Array:
        """Return the multi-index set for the given level.

        Parameters
        ----------
        level : int
            Integer level controlling the size of the index set.

        Returns
        -------
        Array
            Multi-indices. Shape: (nvars, nterms)
        """
        ...


@runtime_checkable
class CompositeAdmissibilityCriteriaProtocol(Protocol, Generic[Array]):
    """Protocol for composite admissibility criteria.

    Composite criteria combine multiple individual criteria with AND logic.

    Methods
    -------
    add_criteria(criteria: AdmissibilityCriteriaProtocol) -> None
        Add a criteria to the composite.
    __call__(index: Array) -> bool
        Check if an index satisfies all criteria.
    """

    def add_criteria(self, criteria: AdmissibilityCriteriaProtocol[Array]) -> None:
        """Add a criteria to the composite."""
        ...

    def __call__(self, index: Array) -> bool:
        """Check if an index satisfies all criteria."""
        ...
