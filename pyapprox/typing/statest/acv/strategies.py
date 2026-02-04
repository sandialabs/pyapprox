"""Search strategies for ACV estimator configuration.

This module provides strategy classes for:
- RecursionIndexStrategy: Generates recursion indices to search
- ModelSubsetStrategy: Generates model subsets to search
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Generic, List, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.statest.factory.tree_enumeration import (
    get_acv_recursion_indices,
)


class RecursionIndexStrategy(ABC, Generic[Array]):
    """Strategy for generating recursion indices to search."""

    @abstractmethod
    def indices(self, nmodels: int, bkd: Backend[Array]) -> List[Array]:
        """Generate recursion indices to search.

        Parameters
        ----------
        nmodels : int
            Number of models.
        bkd : Backend
            Backend for array operations.

        Returns
        -------
        List[Array]
            List of recursion index arrays, each of shape (nmodels-1,).
        """
        raise NotImplementedError

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this strategy."""
        raise NotImplementedError


@dataclass(frozen=True)
class DefaultRecursionStrategy(RecursionIndexStrategy):
    """Use the default recursion index [0, 1, 2, ..., nmodels-2]."""

    def indices(self, nmodels: int, bkd: Backend[Array]) -> List[Array]:
        return [bkd.arange(nmodels - 1, dtype=int)]

    def description(self) -> str:
        return "default recursion index [0, 1, ..., nmodels-2]"


@dataclass(frozen=True)
class FixedRecursionStrategy(RecursionIndexStrategy):
    """Use a single fixed recursion index."""

    recursion_index: tuple

    def indices(self, nmodels: int, bkd: Backend[Array]) -> List[Array]:
        return [bkd.array(self.recursion_index, dtype=int)]

    def description(self) -> str:
        return f"fixed recursion index {self.recursion_index}"


@dataclass(frozen=True)
class ListRecursionStrategy(RecursionIndexStrategy):
    """Use a custom list of recursion indices."""

    recursion_indices: tuple  # tuple of tuples for hashability

    def indices(self, nmodels: int, bkd: Backend[Array]) -> List[Array]:
        return [bkd.array(idx, dtype=int) for idx in self.recursion_indices]

    def description(self) -> str:
        return f"custom list of {len(self.recursion_indices)} recursion indices"


@dataclass(frozen=True)
class TreeDepthRecursionStrategy(RecursionIndexStrategy):
    """Search all recursion trees up to a maximum depth."""

    max_depth: int

    def indices(self, nmodels: int, bkd: Backend[Array]) -> List[Array]:
        # Use tree enumeration from typing.statest.factory
        return list(get_acv_recursion_indices(nmodels, self.max_depth, bkd))

    def description(self) -> str:
        return f"all recursion trees up to depth {self.max_depth}"


@dataclass(frozen=True)
class HierarchicalPermutationRecursionStrategy(RecursionIndexStrategy):
    """Search all permutations of hierarchical recursion indices.

    For nmodels=5, the base hierarchical index is [0, 1, 2, 3].
    This generates all permutations: [0, 1, 2, 3], [0, 1, 3, 2],
    [0, 2, 1, 3], [1, 0, 2, 3], etc.

    Note: There is a one-to-one mapping between recursion index
    permutations and model orderings in hierarchical structures.

    Warning: Grows as (nmodels-1)! - use only for small nmodels.
    """

    def indices(self, nmodels: int, bkd: Backend[Array]) -> List[Array]:
        base = list(range(nmodels - 1))
        return [bkd.array(list(perm), dtype=int) for perm in permutations(base)]

    def description(self) -> str:
        return "all permutations of hierarchical recursion indices"


# === Model Subset Strategies ===


class ModelSubsetStrategy(ABC):
    """Strategy for generating model subsets to search."""

    @abstractmethod
    def subsets(self, nmodels: int) -> List[List[int]]:
        """Generate model subsets to search.

        Parameters
        ----------
        nmodels : int
            Total number of models available.

        Returns
        -------
        List[List[int]]
            List of model index lists. Each list must include 0
            (the high-fidelity model).
        """
        raise NotImplementedError

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this strategy."""
        raise NotImplementedError


@dataclass(frozen=True)
class AllModelsStrategy(ModelSubsetStrategy):
    """Use all available models (no subset search)."""

    def subsets(self, nmodels: int) -> List[List[int]]:
        return [list(range(nmodels))]

    def description(self) -> str:
        return "all models (no subset search)"


@dataclass(frozen=True)
class FixedSubsetStrategy(ModelSubsetStrategy):
    """Use a single fixed model subset."""

    model_indices: tuple

    def __post_init__(self):
        if 0 not in self.model_indices:
            raise ValueError("model_indices must include 0 (high-fidelity)")

    def subsets(self, nmodels: int) -> List[List[int]]:
        return [list(self.model_indices)]

    def description(self) -> str:
        return f"fixed model subset {self.model_indices}"


@dataclass(frozen=True)
class AllSubsetsStrategy(ModelSubsetStrategy):
    """Generate all model subsets up to max_models."""

    min_models: int = 2
    max_models: Optional[int] = None

    def subsets(self, nmodels: int) -> List[List[int]]:
        max_size = self.max_models if self.max_models is not None else nmodels
        max_size = min(max_size, nmodels)

        result = []
        other_models = list(range(1, nmodels))

        for size in range(self.min_models, max_size + 1):
            for combo in combinations(other_models, size - 1):
                result.append([0] + list(combo))

        return result

    def description(self) -> str:
        max_str = str(self.max_models) if self.max_models else "all"
        return f"all subsets with {self.min_models} to {max_str} models"


@dataclass(frozen=True)
class ListSubsetStrategy(ModelSubsetStrategy):
    """Use a custom list of model subsets."""

    model_subsets: tuple  # tuple of tuples for hashability

    def __post_init__(self):
        for subset in self.model_subsets:
            if 0 not in subset:
                raise ValueError(f"All subsets must include 0: {subset}")

    def subsets(self, nmodels: int) -> List[List[int]]:
        return [list(subset) for subset in self.model_subsets]

    def description(self) -> str:
        return f"custom list of {len(self.model_subsets)} model subsets"
