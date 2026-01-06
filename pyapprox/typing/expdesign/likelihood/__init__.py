"""
OED-specific likelihood wrappers.

This module provides likelihood classes specialized for optimal
experimental design, including vectorized evaluation and Jacobians
with respect to design weights.
"""

from .gaussian import (
    GaussianOEDOuterLoopLikelihood,
    GaussianOEDInnerLoopLikelihood,
)

__all__ = [
    "GaussianOEDOuterLoopLikelihood",
    "GaussianOEDInnerLoopLikelihood",
]
