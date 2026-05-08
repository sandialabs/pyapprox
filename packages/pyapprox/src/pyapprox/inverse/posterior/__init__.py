"""
Posterior utilities for inverse problems.

This module provides utilities for working with Bayesian posteriors:
- Log unnormalized posterior combining likelihood and prior
- MAP estimation
"""

from .log_unnormalized import LogUnNormalizedPosterior

__all__ = [
    "LogUnNormalizedPosterior",
]
