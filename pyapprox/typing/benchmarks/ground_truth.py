"""Ground truth dataclasses for benchmarks.

Each domain (sensitivity, optimization, quadrature, etc.) has its own
ground truth dataclass containing the known/computable values.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence, Tuple, Callable

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


@dataclass(frozen=True)
class ODEGroundTruth:
    """Ground truth for ODE benchmarks.

    Attributes
    ----------
    nstates : int
        Number of state variables in the ODE system.
    nparams : int
        Number of parameters in the ODE system.
    initial_condition : np.ndarray, optional
        Default initial condition for the ODE system. Shape: (nstates,).
    nominal_parameters : np.ndarray, optional
        Nominal parameter values. Shape: (nparams,).
    init_time : float, optional
        Initial time for integration.
    final_time : float, optional
        Final time for integration.
    deltat : float, optional
        Time step for integration.
    reference_solution : np.ndarray, optional
        High-fidelity numerical reference solution. Shape: (nstates, ntimes).
    analytical_solution : Callable, optional
        Analytical solution function if available (e.g., for linear ODEs).
        Signature: (time: float, params: np.ndarray) -> np.ndarray
    steady_state : np.ndarray, optional
        Steady state solution for dissipative systems. Shape: (nstates,).
    """

    nstates: int
    nparams: int
    initial_condition: Optional[NDArray[np.floating[Any]]] = None
    nominal_parameters: Optional[NDArray[np.floating[Any]]] = None
    init_time: Optional[float] = None
    final_time: Optional[float] = None
    deltat: Optional[float] = None
    reference_solution: Optional[NDArray[np.floating[Any]]] = None
    analytical_solution: Optional[
        Callable[[float, NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
    ] = None
    steady_state: Optional[NDArray[np.floating[Any]]] = None

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
