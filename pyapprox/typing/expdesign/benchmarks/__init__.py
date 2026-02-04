"""
OED benchmark problems for testing and validation.

This module provides benchmark problems with known analytical solutions
for validating OED implementations.
"""

from .linear_gaussian import LinearGaussianOEDBenchmark

__all__ = [
    "LinearGaussianOEDBenchmark",
]
