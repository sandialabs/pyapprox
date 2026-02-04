"""Shared search strategies for multi-fidelity estimators.

This module provides strategy classes for generating subsets to search:
- ModelSubsetStrategy: Generates model subsets
- QoISubsetStrategy: Generates QoI (quantity of interest) subsets
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional


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
    """Use all models (no subset search)."""

    def subsets(self, nmodels: int) -> List[List[int]]:
        return [list(range(nmodels))]

    def description(self) -> str:
        return "all models (no subset search)"


@dataclass(frozen=True)
class FixedSubsetStrategy(ModelSubsetStrategy):
    """Use a single fixed model subset.

    Parameters
    ----------
    model_indices : tuple
        Indices of models to include. Must include 0 (high-fidelity).
    """

    model_indices: tuple

    def __post_init__(self) -> None:
        if 0 not in self.model_indices:
            raise ValueError("model_indices must include 0 (high-fidelity)")

    def subsets(self, nmodels: int) -> List[List[int]]:
        return [list(self.model_indices)]

    def description(self) -> str:
        return f"fixed model subset {self.model_indices}"


@dataclass(frozen=True)
class AllSubsetsStrategy(ModelSubsetStrategy):
    """Generate all model subsets.

    Parameters
    ----------
    min_models : int
        Minimum subset size (default 2).
    max_models : int, optional
        Maximum subset size. If None, uses nmodels.
    """

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
    """Use a custom list of model subsets.

    Parameters
    ----------
    model_subsets : tuple
        Tuple of tuples, each containing model indices. All must include 0.
    """

    model_subsets: tuple

    def __post_init__(self) -> None:
        for subset in self.model_subsets:
            if 0 not in subset:
                raise ValueError(f"All subsets must include 0: {subset}")

    def subsets(self, nmodels: int) -> List[List[int]]:
        return [list(subset) for subset in self.model_subsets]

    def description(self) -> str:
        return f"custom list of {len(self.model_subsets)} model subsets"


class QoISubsetStrategy(ABC):
    """Strategy for generating QoI subsets to search."""

    @abstractmethod
    def subsets(self, nqoi: int) -> List[List[int]]:
        """Generate QoI subsets to search.

        Parameters
        ----------
        nqoi : int
            Total number of QoI available.

        Returns
        -------
        List[List[int]]
            List of QoI index lists.
        """
        raise NotImplementedError

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this strategy."""
        raise NotImplementedError


@dataclass(frozen=True)
class AllQoIStrategy(QoISubsetStrategy):
    """Use all QoI (no subset search)."""

    def subsets(self, nqoi: int) -> List[List[int]]:
        return [list(range(nqoi))]

    def description(self) -> str:
        return "all QoI (no subset search)"


@dataclass(frozen=True)
class FixedQoIStrategy(QoISubsetStrategy):
    """Use a single fixed QoI subset.

    Parameters
    ----------
    qoi_indices : tuple
        Indices of QoI to include.
    """

    qoi_indices: tuple

    def subsets(self, nqoi: int) -> List[List[int]]:
        return [list(self.qoi_indices)]

    def description(self) -> str:
        return f"fixed QoI subset {self.qoi_indices}"


@dataclass(frozen=True)
class AllQoISubsetsStrategy(QoISubsetStrategy):
    """Generate all QoI subsets.

    Parameters
    ----------
    min_qoi : int
        Minimum subset size (default 1).
    max_qoi : int, optional
        Maximum subset size. If None, uses nqoi.
    required_qoi : tuple
        QoI indices that must be in every subset (default empty).
    """

    min_qoi: int = 1
    max_qoi: Optional[int] = None
    required_qoi: tuple = ()

    def subsets(self, nqoi: int) -> List[List[int]]:
        required = set(self.required_qoi)
        optional = [i for i in range(nqoi) if i not in required]

        max_size = self.max_qoi if self.max_qoi is not None else nqoi
        max_size = min(max_size, nqoi)

        result = []
        n_required = len(required)

        for size in range(self.min_qoi, max_size + 1):
            n_optional_needed = size - n_required
            if n_optional_needed < 0:
                continue
            if n_optional_needed > len(optional):
                continue
            for combo in combinations(optional, n_optional_needed):
                result.append(sorted(list(required) + list(combo)))

        return result

    def description(self) -> str:
        req_str = f", required={self.required_qoi}" if self.required_qoi else ""
        max_str = str(self.max_qoi) if self.max_qoi else "all"
        return f"all QoI subsets with {self.min_qoi} to {max_str} QoI{req_str}"


@dataclass(frozen=True)
class ListQoIStrategy(QoISubsetStrategy):
    """Use a custom list of QoI subsets.

    Parameters
    ----------
    qoi_subsets : tuple
        Tuple of tuples, each containing QoI indices.
    """

    qoi_subsets: tuple

    def subsets(self, nqoi: int) -> List[List[int]]:
        return [list(subset) for subset in self.qoi_subsets]

    def description(self) -> str:
        return f"custom list of {len(self.qoi_subsets)} QoI subsets"
