"""
OED benchmark problems for testing and validation.

This module provides benchmark problems with known analytical solutions
for validating OED implementations.
"""

from .linear_gaussian import LinearGaussianOEDBenchmark
from .linear_gaussian_model import LinearGaussianOEDModel
from .nonlinear_gaussian import NonLinearGaussianOEDBenchmark
from .linear_gaussian_pred import LinearGaussianPredOEDBenchmark

__all__ = [
    "LinearGaussianOEDBenchmark",
    "LinearGaussianOEDModel",
    "NonLinearGaussianOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
]
