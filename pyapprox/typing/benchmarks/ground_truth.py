"""Ground truth dataclasses for benchmarks.

Each domain (sensitivity, optimization, quadrature, etc.) has its own
ground truth dataclass containing the known/computable values.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence, Tuple, Callable, Generic

from pyapprox.typing.util.backends.protocols import Array


@dataclass(frozen=True)
class SensitivityGroundTruth(Generic[Array]):
    """Ground truth for sensitivity analysis benchmarks."""

    mean: Optional[float] = None
    variance: Optional[float] = None
    main_effects: Optional[Array] = None
    total_effects: Optional[Array] = None
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
class OptimizationGroundTruth(Generic[Array]):
    """Ground truth for optimization benchmarks."""

    global_minimum: Optional[float] = None
    global_minimizers: Optional[Array] = None
    local_minima: Optional[Array] = None

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
class MultifidelityGroundTruth(Generic[Array]):
    """Ground truth for multifidelity benchmarks."""

    high_fidelity_mean: Optional[float] = None
    high_fidelity_variance: Optional[float] = None
    model_correlations: Optional[Array] = None
    model_costs: Optional[Array] = None
    optimal_allocation: Optional[Array] = None

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
class InverseGroundTruth(Generic[Array]):
    """Ground truth for inverse/inference benchmarks."""

    true_parameters: Optional[Array] = None
    posterior_mean: Optional[Array] = None
    posterior_covariance: Optional[Array] = None
    map_estimate: Optional[Array] = None
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
class ODEGroundTruth(Generic[Array]):
    """Ground truth for ODE benchmarks.

    Attributes
    ----------
    nstates : int
        Number of state variables in the ODE system.
    nparams : int
        Number of parameters in the ODE system.
    initial_condition : Array, optional
        Default initial condition for the ODE system. Shape: (nstates, 1).
    nominal_parameters : Array, optional
        Nominal parameter values. Shape: (nparams, 1).
    init_time : float, optional
        Initial time for integration.
    final_time : float, optional
        Final time for integration.
    deltat : float, optional
        Time step for integration.
    reference_solution : Array, optional
        High-fidelity numerical reference solution. Shape: (nstates, ntimes).
    analytical_solution : Callable, optional
        Analytical solution function if available (e.g., for linear ODEs).
        Signature: (time: float, params: Array) -> Array
    steady_state : Array, optional
        Steady state solution for dissipative systems. Shape: (nstates, 1).
    """

    nstates: int
    nparams: int
    initial_condition: Optional[Array] = None
    nominal_parameters: Optional[Array] = None
    init_time: Optional[float] = None
    final_time: Optional[float] = None
    deltat: Optional[float] = None
    reference_solution: Optional[Array] = None
    analytical_solution: Optional[Callable[[float, Array], Array]] = None
    steady_state: Optional[Array] = None

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
