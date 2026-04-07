"""Ground truth dataclasses for benchmarks.

Each domain (sensitivity, optimization, quadrature, etc.) has its own
ground truth dataclass containing the known/computable values.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Sequence, Tuple, Union

from pyapprox.util.backends.protocols import Array

#TODO: Implement a StatisticsGroundTruth, e.g. mean and variance.
# Note Integral Ground Truth is not a statistics ground truth
# because it uses a lebesque integration measure where as statistics
# Benchmark uses an arbitrary prior probability measure.

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
                f"Ground truth '{name}' not available. Available: {self.available()}"
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
                f"Ground truth '{name}' not available. Available: {self.available()}"
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
                f"Ground truth '{name}' not available. Available: {self.available()}"
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
                f"Ground truth '{name}' not available. Available: {self.available()}"
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
                f"Ground truth '{name}' not available. Available: {self.available()}"
            )
        return value

# TODO: for this to really be a ground truth we need either reference_solution
# of analytical_solution to be provided. If both are not we need to raise an
# error, Adding this validation check will likely break existing tests because
# we are incorrectly using ODE ground truth just to provide a
# configuration of an ODE problem instance. Perhaps consider
# splitting this into ODEConfiguration and ODEGroundTruth, An
# analytical ground truth just has the analytical solution, but a
# reference ground truth has all the time configurations, etc
# needed to produce the reference solution. ODE ground truth seems
# pretty useless as is, really only useful for manufactured
# solution testing but benchmarks are more intended for users not
# developers. We really care about using ODEs for parameterized
# problems, eg. ODE solutions for many parameters, but storing
# them would be resource consuming, especially for git repo. I
# think we really only need outerloop benchmark, e.g. sensitivity
# analysis, statistics moment estimation, optimization and problem
# instances that depend on a function instance (like those based
# on ode, pde, algebraic). Are ground truths the best most
# extensible way to support different benchmarks that may have
# multiple ground truths, e.g. moments and sensitivity and
# posterior if a problem instance supports both prior and
# posterior moments how do we handle that. Again recall we want
# registry to find benchmarks with certain properties, e.g. prior
# based moments, or posterior PDF, we also may want to extend
# what a benchmark provides in the future. Currently ground
# truths provided many optional arguments, would it be better to
# make benchmarks classes with functions that provide all
# ground_truths they support and use protocols to determine broad
# categories when searching, or user defined custom searches that
# only care about one ground truth function.
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
                f"Ground truth '{name}' not available. Available: {self.available()}"
            )
        return value

# TODO: Should a benchmark have only one ground_thruth e.g. ODEGroundTruth
# or should we allow multiple, e.g. ODEGroundTruth, OptimizationGroundTruth
# for ODE based optimization benchmark.


@dataclass(frozen=True)
class OEDGroundTruth:
    """Ground truth for OED benchmarks.

    Benchmarks compose this and delegate exact_eig/exact_utility calls to it.

    Parameters
    ----------
    exact_eig : callable or None
        Callable mapping weights -> exact EIG value.
    exact_utility : callable or None
        Callable mapping weights -> exact utility value.
    """

    exact_eig: Optional[Callable[..., float]] = None
    exact_utility: Optional[Callable[..., float]] = None
