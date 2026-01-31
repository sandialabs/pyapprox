"""Registry utilities for statest estimators and statistics.

Provides registry functions for registering and creating estimator and
statistic types, along with protocol validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Type, Optional, Any

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.statest.protocols import EstimatorProtocol
from pyapprox.typing.statest.statistics import (
    MultiOutputStatistic,
    MultiOutputMean,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
)
from pyapprox.typing.statest.acv import (
    GMFEstimator,
    GISEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)


# --- Registry for Estimator Types ---

# Maps estimator type name to (class, requires_recursion_index)
_ESTIMATOR_REGISTRY: Dict[str, tuple] = {}

# Required methods for EstimatorProtocol
_ESTIMATOR_REQUIRED_METHODS = [
    "bkd",
    "allocate_samples",
    "generate_samples_per_model",
    "__call__",
    "optimized_covariance",
    "nsamples_per_model",
]


def _validate_estimator_class(estimator_class: Type) -> None:
    """Validate that a class satisfies EstimatorProtocol.

    Parameters
    ----------
    estimator_class : Type
        The class to validate.

    Raises
    ------
    TypeError
        If the class does not have the required methods.
    """
    missing_methods = []
    for method_name in _ESTIMATOR_REQUIRED_METHODS:
        if not hasattr(estimator_class, method_name):
            missing_methods.append(method_name)

    if missing_methods:
        raise TypeError(
            f"Estimator class {estimator_class.__name__} does not satisfy "
            f"EstimatorProtocol. Missing methods: {missing_methods}"
        )


def register_estimator(
    name: str,
    estimator_class: Type,
    requires_recursion_index: bool = True,
) -> None:
    """Register an estimator type for use in BestEstimatorFactory.

    The estimator class must satisfy EstimatorProtocol (have methods:
    bkd, allocate_samples, generate_samples_per_model, __call__,
    optimized_covariance, nsamples_per_model).

    Parameters
    ----------
    name : str
        Short name for the estimator type (e.g., "gmf", "gis").
    estimator_class : Type
        The estimator class to instantiate. Must satisfy EstimatorProtocol.
    requires_recursion_index : bool, optional
        Whether this estimator requires a recursion index. Default: True.

    Raises
    ------
    TypeError
        If estimator_class does not satisfy EstimatorProtocol.

    Examples
    --------
    >>> from pyapprox.typing.statest.acv import GMFEstimator
    >>> register_estimator("gmf", GMFEstimator, requires_recursion_index=True)
    """
    _validate_estimator_class(estimator_class)
    _ESTIMATOR_REGISTRY[name.lower()] = (estimator_class, requires_recursion_index)


def get_registered_estimators() -> Dict[str, tuple]:
    """Return the registry of estimator types."""
    return dict(_ESTIMATOR_REGISTRY)


def get_estimator_registry() -> Dict[str, tuple]:
    """Return the internal estimator registry (for factory use)."""
    return _ESTIMATOR_REGISTRY


def create_estimator(
    est_type: str,
    stat: MultiOutputStatistic[Array],
    costs: Array,
    recursion_index: Optional[Array] = None,
) -> EstimatorProtocol[Array]:
    """Create an estimator instance from the registry.

    Parameters
    ----------
    est_type : str
        Estimator type name.
    stat : MultiOutputStatistic
        Statistic to estimate.
    costs : Array
        Model costs.
    recursion_index : Optional[Array]
        Recursion index for ACV estimators.

    Returns
    -------
    EstimatorProtocol
        The created estimator.

    Raises
    ------
    ValueError
        If estimator type is not registered.
    """
    key = est_type.lower()
    if key not in _ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown estimator type: {est_type}. "
            f"Registered types: {list(_ESTIMATOR_REGISTRY.keys())}"
        )

    cls, requires_recursion = _ESTIMATOR_REGISTRY[key]

    if requires_recursion:
        return cls(stat, costs, recursion_index=recursion_index)
    else:
        return cls(stat, costs)


# Register default estimator types
register_estimator("gmf", GMFEstimator, requires_recursion_index=True)
register_estimator("gis", GISEstimator, requires_recursion_index=True)
register_estimator("grd", GRDEstimator, requires_recursion_index=True)
register_estimator("mfmc", MFMCEstimator, requires_recursion_index=False)
register_estimator("mlmc", MLMCEstimator, requires_recursion_index=False)


# --- Statistic Type Registry ---

_STATISTIC_REGISTRY: Dict[str, Type] = {
    "MultiOutputMean": MultiOutputMean,
    "MultiOutputVariance": MultiOutputVariance,
    "MultiOutputMeanAndVariance": MultiOutputMeanAndVariance,
}

# Required methods for StatisticProtocol
_STATISTIC_REQUIRED_METHODS = [
    "bkd",
    "nqoi",
    "nstats",
    "sample_estimate",
    "set_pilot_quantities",
]


def _validate_statistic_class(statistic_class: Type) -> None:
    """Validate that a class satisfies StatisticProtocol.

    Parameters
    ----------
    statistic_class : Type
        The class to validate.

    Raises
    ------
    TypeError
        If the class does not have the required methods.
    """
    missing_methods = []
    for method_name in _STATISTIC_REQUIRED_METHODS:
        if not hasattr(statistic_class, method_name):
            missing_methods.append(method_name)

    if missing_methods:
        raise TypeError(
            f"Statistic class {statistic_class.__name__} does not satisfy "
            f"StatisticProtocol. Missing methods: {missing_methods}"
        )


def register_statistic(name: str, statistic_class: Type) -> None:
    """Register a statistic type for subset creation.

    The statistic class must satisfy StatisticProtocol (have methods:
    bkd, nqoi, nstats, sample_estimate, set_pilot_quantities).

    Parameters
    ----------
    name : str
        Class name of the statistic.
    statistic_class : Type
        The statistic class to instantiate. Must satisfy StatisticProtocol.

    Raises
    ------
    TypeError
        If statistic_class does not satisfy StatisticProtocol.
    """
    _validate_statistic_class(statistic_class)
    _STATISTIC_REGISTRY[name] = statistic_class


def get_statistic_registry() -> Dict[str, Type]:
    """Return the internal statistic registry (for factory use)."""
    return _STATISTIC_REGISTRY


# --- Candidate Result ---


@dataclass
class CandidateResult:
    """Result from evaluating a single candidate configuration.

    Attributes
    ----------
    estimator_type : str
        Type of estimator ("gmf", "gis", "grd", etc.).
    model_indices : List[int]
        Indices of models used (subset of full model set).
    recursion_index : Optional[Array]
        Recursion index used, if applicable.
    objective_value : float
        Value of log-det objective (lower is better).
    estimator : Optional[EstimatorProtocol]
        The optimized estimator instance.
    success : bool
        Whether optimization succeeded.
    error_message : Optional[str]
        Error message if failed.
    """

    estimator_type: str
    model_indices: List[int]
    recursion_index: Optional[Any]  # Array type varies
    objective_value: float
    estimator: Optional[Any]  # EstimatorProtocol[Array]
    success: bool
    error_message: Optional[str] = None


# --- Objective Computation ---


def compute_objective(cov: Array, bkd: Backend[Array]) -> float:
    """Compute log-determinant of covariance matrix.

    This is the same objective used in allocate_samples() optimization.

    Parameters
    ----------
    cov : Array
        Covariance matrix.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        Log-determinant value.
    """
    if cov.ndim == 0:
        result = bkd.log(bkd.abs(cov) + 1e-14)
        return float(bkd.to_numpy(bkd.reshape(result, (1,)))[0])

    if cov.shape == (1, 1):
        result = bkd.log(bkd.abs(cov[0, 0]) + 1e-14)
        return float(bkd.to_numpy(bkd.reshape(result, (1,)))[0])

    # For matrix case, compute log-det via Cholesky
    # log(det(A)) = 2 * sum(log(diag(L))) where A = L @ L.T
    L = bkd.cholesky(cov)
    diag_L = bkd.diag(L)
    result = 2 * bkd.sum(bkd.log(bkd.abs(diag_L) + 1e-14))
    return float(bkd.to_numpy(bkd.reshape(result, (1,)))[0])
