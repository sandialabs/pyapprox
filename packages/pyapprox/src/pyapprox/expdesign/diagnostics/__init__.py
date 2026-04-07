"""OED diagnostic utilities for testing and validation.

Diagnostics accept raw sample arrays — sampling is the caller's
responsibility. Use OEDQuadratureSampler + problem.obs_map() to
generate the required arrays.

Shared utilities (MSE decomposition, convergence rates) are in utils.py.
"""

from .kl_diagnostics import KLOEDDiagnostics
from .prediction_diagnostics import (
    PredictionOEDDiagnostics,
    UtilityConfig,
    compute_exact_prediction_utility,
    create_prediction_oed_diagnostics,
    get_registered_utility_types,
    get_utility_factory,
    register_utility,
)
from .utils import compute_convergence_rate, compute_estimator_mse

__all__ = [
    "KLOEDDiagnostics",
    "PredictionOEDDiagnostics",
    "UtilityConfig",
    "compute_convergence_rate",
    "compute_estimator_mse",
    "compute_exact_prediction_utility",
    "create_prediction_oed_diagnostics",
    "get_registered_utility_types",
    "get_utility_factory",
    "register_utility",
]
