"""
Evidence computation for Bayesian OED.

This module provides classes for computing the evidence (marginal likelihood)
and log-evidence, including Jacobians for gradient-based optimization.
"""

from .evidence import Evidence, LogEvidence

__all__ = [
    "Evidence",
    "LogEvidence",
]
