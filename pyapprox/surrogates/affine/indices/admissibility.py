"""Composable admissibility criteria for multi-index generation.

This module provides composable criteria for determining which indices
are admissible in adaptive polynomial expansions.

Key design: Criteria are composable via CompositeCriteria, allowing
flexible combinations of constraints.

Example
-------
>>> criteria = CompositeCriteria(
...     MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=NumpyBkd),
...     Max1DLevelsCriteria(max_levels=bkd.array([3, 3, 4]), bkd=NumpyBkd),
... )
>>> criteria(bkd.array([2, 1, 3]))  # Check if admissible
True
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TYPE_CHECKING

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.indices.utils import indices_pnorm

if TYPE_CHECKING:
    from pyapprox.surrogates.affine.indices.generators import (
        IterativeIndexGenerator,
    )


class AdmissibilityCriteria(ABC, Generic[Array]):
    """Abstract base class for admissibility criteria.

    Subclasses implement specific admissibility checks for multi-indices.
    """

    @abstractmethod
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
        raise NotImplementedError

    def failure_message(self) -> str:
        """Return a message explaining why the last check failed."""
        return "Index does not satisfy admissibility criteria"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class MaxLevelCriteria(AdmissibilityCriteria[Array], Generic[Array]):
    """Criteria based on maximum hyperbolic level.

    An index is admissible if its p-norm is at most max_level.

    Parameters
    ----------
    max_level : int
        Maximum allowed hyperbolic level.
    pnorm : float
        p-norm exponent (1.0 = total degree, inf = max degree).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, max_level: int, pnorm: float, bkd: Backend[Array]):
        self._max_level = max_level
        self._pnorm = pnorm
        self._bkd = bkd
        self._last_norm: Optional[float] = None

    @property
    def max_level(self) -> int:
        """Return the maximum level."""
        return self._max_level

    @max_level.setter
    def max_level(self, value: int) -> None:
        """Set the maximum level."""
        self._max_level = value

    def __call__(self, index: Array) -> bool:
        self._last_norm = indices_pnorm(index, self._pnorm, self._bkd)
        return self._last_norm <= self._max_level

    def failure_message(self) -> str:
        return (
            f"Index p-norm ({self._last_norm:.2f}) exceeds "
            f"max_level ({self._max_level})"
        )

    def __repr__(self) -> str:
        return f"MaxLevelCriteria(max_level={self._max_level}, pnorm={self._pnorm})"


class Max1DLevelsCriteria(AdmissibilityCriteria[Array], Generic[Array]):
    """Criteria based on maximum level per dimension.

    An index is admissible if each component is at most the corresponding
    maximum 1D level.

    Parameters
    ----------
    max_levels : Array
        Maximum level per dimension. Shape: (nvars,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, max_levels: Array, bkd: Backend[Array]):
        self._max_levels = max_levels
        self._bkd = bkd
        self._last_violations: Optional[Array] = None

    def __call__(self, index: Array) -> bool:
        violations = index > self._max_levels
        self._last_violations = violations
        return not self._bkd.any_bool(violations)

    def failure_message(self) -> str:
        return "Index exceeds maximum 1D levels in some dimensions"

    def __repr__(self) -> str:
        return f"Max1DLevelsCriteria(max_levels={self._max_levels})"


class MaxIndicesCriteria(AdmissibilityCriteria[Array], Generic[Array]):
    """Criteria based on maximum number of indices.

    An index is admissible if adding it would not exceed the maximum
    number of indices. Requires connection to an index generator.

    Parameters
    ----------
    max_nindices : int
        Maximum number of indices allowed.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, max_nindices: int, bkd: Backend[Array]):
        self._max_nindices = max_nindices
        self._bkd = bkd
        self._generator: Optional["IterativeIndexGenerator"] = None
        self._count: int = 0

    def set_generator(self, generator: "IterativeIndexGenerator") -> None:
        """Set the index generator to track index count."""
        self._generator = generator
        self._count = generator.nindices()

    def set_max_nindices(self, max_nindices: int) -> None:
        """Update the maximum number of indices."""
        if self._generator is not None:
            self._count = self._generator.nindices()
        self._max_nindices = self._count + max_nindices

    def __call__(self, index: Array) -> bool:
        if self._generator is None:
            raise RuntimeError("Must call set_generator before using criteria")
        if self._generator.nindices() + self._count < self._max_nindices:
            self._count += 1
            return True
        return False

    def failure_message(self) -> str:
        return f"Would exceed maximum number of indices ({self._max_nindices})"

    def __repr__(self) -> str:
        return f"MaxIndicesCriteria(max_nindices={self._max_nindices})"


class CompositeCriteria(AdmissibilityCriteria[Array], Generic[Array]):
    """Composite admissibility criteria combining multiple criteria with AND.

    An index is admissible if it satisfies ALL component criteria.

    Parameters
    ----------
    *criteria : AdmissibilityCriteria
        Variable number of criteria to combine.

    Example
    -------
    >>> criteria = CompositeCriteria(
    ...     MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=NumpyBkd),
    ...     Max1DLevelsCriteria(max_levels=bkd.array([3, 3]), bkd=NumpyBkd),
    ... )
    """

    def __init__(self, *criteria: AdmissibilityCriteria[Array]):
        self._criteria: List[AdmissibilityCriteria[Array]] = [
            c for c in criteria if c is not None
        ]
        self._failed_criteria: Optional[AdmissibilityCriteria[Array]] = None

    def add_criteria(self, criteria: AdmissibilityCriteria[Array]) -> None:
        """Add a criteria to the composite."""
        if criteria is not None:
            self._criteria.append(criteria)

    def __call__(self, index: Array) -> bool:
        for criteria in self._criteria:
            if not criteria(index):
                self._failed_criteria = criteria
                return False
        self._failed_criteria = None
        return True

    def failure_message(self) -> str:
        if self._failed_criteria is not None:
            return self._failed_criteria.failure_message()
        return "No failure"

    def __repr__(self) -> str:
        criteria_strs = [repr(c) for c in self._criteria]
        return f"CompositeCriteria({', '.join(criteria_strs)})"
