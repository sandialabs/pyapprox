"""
OED benchmark problems for testing and validation.

This module provides benchmark problems with known analytical solutions
for validating OED implementations.
"""

from .advection_diffusion import ObstructedAdvectionDiffusionOEDBenchmark
from .linear_gaussian import LinearGaussianOEDBenchmark
from .linear_gaussian_model import LinearGaussianOEDModel
from .linear_gaussian_pred import LinearGaussianPredOEDBenchmark
from .lotka_volterra import LotkaVolterraOEDBenchmark
from .nonlinear_gaussian import NonLinearGaussianOEDBenchmark

__all__ = [
    "LinearGaussianOEDBenchmark",
    "LinearGaussianOEDModel",
    "NonLinearGaussianOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
    "LotkaVolterraOEDBenchmark",
    "ObstructedAdvectionDiffusionOEDBenchmark",
]
