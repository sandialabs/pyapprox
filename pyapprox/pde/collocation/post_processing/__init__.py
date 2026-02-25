"""Post-processing module for collocation-based PDE solutions."""

from pyapprox.pde.collocation.post_processing.stress import (
    HyperelasticStressPostProcessor2D,
    StressPostProcessor2D,
)

__all__ = [
    "StressPostProcessor2D",
    "HyperelasticStressPostProcessor2D",
]
