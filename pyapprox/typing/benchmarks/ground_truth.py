"""Ground truth dataclasses for benchmarks.

Each domain (sensitivity, optimization, quadrature, etc.) has its own
ground truth dataclass containing the known/computable values.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SensitivityGroundTruth:
    """Ground truth for sensitivity analysis benchmarks."""

    mean: Optional[float] = None
    variance: Optional[float] = None
    main_effects: Optional[NDArray[np.floating[Any]]] = None
    total_effects: Optional[NDArray[np.floating[Any]]] = None
    sobol_indices: Optional[Dict[Tuple[int, ...], float]] = None

    def available(self) -> Sequence[str]:
        """Return list of available ground truth properties."""
        return [k for k, v in self.__dict__.items() if v is not None]

    def get(self, name: str) -> Any:
        """Get a ground truth value by name."""
        value = getattr(self, name, None)
        if value is None:
            raise ValueError(
                f"Ground truth '{name}' not available. "
                f"Available: {self.available()}"
            )
        return value


@dataclass(frozen=True)
class OptimizationGroundTruth:
    """Ground truth for optimization benchmarks."""

    global_minimum: Optional[float] = None
    global_minimizers: Optional[NDArray[np.floating[Any]]] = None
    local_minima: Optional[NDArray[np.floating[Any]]] = None

    def available(self) -> Sequence[str]:
        """Return list of available ground truth properties."""
        return [k for k, v in self.__dict__.items() if v is not None]

    def get(self, name: str) -> Any:
        """Get a ground truth value by name."""
        value = getattr(self, name, None)
        if value is None:
            raise ValueError(
                f"Ground truth '{name}' not available. "
                f"Available: {self.available()}"
            )
        return value


@dataclass(frozen=True)
class QuadratureGroundTruth:
    """Ground truth for quadrature/integration benchmarks."""

    integral: Optional[float] = None
    integral_formula: Optional[str] = None

    def available(self) -> Sequence[str]:
        """Return list of available ground truth properties."""
        return [k for k, v in self.__dict__.items() if v is not None]

    def get(self, name: str) -> Any:
        """Get a ground truth value by name."""
        value = getattr(self, name, None)
        if value is None:
            raise ValueError(
                f"Ground truth '{name}' not available. "
                f"Available: {self.available()}"
            )
        return value


@dataclass(frozen=True)
class MultifidelityGroundTruth:
    """Ground truth for multifidelity benchmarks."""

    high_fidelity_mean: Optional[float] = None
    high_fidelity_variance: Optional[float] = None
    model_correlations: Optional[NDArray[np.floating[Any]]] = None
    model_costs: Optional[NDArray[np.floating[Any]]] = None
    optimal_allocation: Optional[NDArray[np.floating[Any]]] = None

    def available(self) -> Sequence[str]:
        """Return list of available ground truth properties."""
        return [k for k, v in self.__dict__.items() if v is not None]

    def get(self, name: str) -> Any:
        """Get a ground truth value by name."""
        value = getattr(self, name, None)
        if value is None:
            raise ValueError(
                f"Ground truth '{name}' not available. "
                f"Available: {self.available()}"
            )
        return value


@dataclass(frozen=True)
class InverseGroundTruth:
    """Ground truth for inverse/inference benchmarks."""

    true_parameters: Optional[NDArray[np.floating[Any]]] = None
    posterior_mean: Optional[NDArray[np.floating[Any]]] = None
    posterior_covariance: Optional[NDArray[np.floating[Any]]] = None
    map_estimate: Optional[NDArray[np.floating[Any]]] = None
    evidence: Optional[float] = None

    def available(self) -> Sequence[str]:
        """Return list of available ground truth properties."""
        return [k for k, v in self.__dict__.items() if v is not None]

    def get(self, name: str) -> Any:
        """Get a ground truth value by name."""
        value = getattr(self, name, None)
        if value is None:
            raise ValueError(
                f"Ground truth '{name}' not available. "
                f"Available: {self.available()}"
            )
        return value
