"""Estimator factory and model selection utilities.

This module provides convenient factory functions for creating estimators
and model selection utilities for finding the best estimator.
"""

from pyapprox.typing.stats.factory.estimator_factory import get_estimator
from pyapprox.typing.stats.factory.best_estimator import BestEstimator
from pyapprox.typing.stats.factory.comparison import compare_estimators

__all__ = [
    "get_estimator",
    "BestEstimator",
    "compare_estimators",
]
