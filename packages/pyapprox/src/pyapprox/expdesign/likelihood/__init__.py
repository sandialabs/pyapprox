"""
OED-specific likelihood wrappers.

This module provides likelihood classes specialized for optimal
experimental design, including vectorized evaluation and Jacobians
with respect to design weights.
"""

from .gaussian import (
    GaussianOEDInnerLoopLikelihood,
    GaussianOEDOuterLoopLikelihood,
)
from .parallel_gaussian import (
    ParallelGaussianOEDInnerLoopLikelihood,
)

__all__ = [
    "GaussianOEDOuterLoopLikelihood",
    "GaussianOEDInnerLoopLikelihood",
    "ParallelGaussianOEDInnerLoopLikelihood",
]
