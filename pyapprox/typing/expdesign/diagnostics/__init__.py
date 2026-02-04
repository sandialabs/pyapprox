"""
OED diagnostic utilities for testing and validation.

This module provides diagnostic tools for analyzing OED estimator
performance, including MSE computation and convergence rate analysis.
"""

from .kl_diagnostics import KLOEDDiagnostics

__all__ = [
    "KLOEDDiagnostics",
]
