"""
OED diagnostic utilities for testing and validation.

This module provides diagnostic tools for analyzing OED estimator
performance, including MSE computation and convergence rate analysis.
"""

from .kl_diagnostics import KLOEDDiagnostics
from .prediction_diagnostics import (
    PredictionOEDDiagnostics,
    create_prediction_oed_diagnostics,
    get_registered_utility_types,
    register_utility,
)

__all__ = [
    "KLOEDDiagnostics",
    "PredictionOEDDiagnostics",
    "create_prediction_oed_diagnostics",
    "register_utility",
    "get_registered_utility_types",
]
