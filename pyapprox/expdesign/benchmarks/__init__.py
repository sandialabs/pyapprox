"""
OED benchmark problems for testing and validation.

This module provides benchmark problems with known analytical solutions
for validating OED implementations.
"""

import pyapprox.expdesign.benchmarks.instances  # noqa: F401

from .advection_diffusion import ObstructedAdvectionDiffusionOEDBenchmark
from .instances.linear_gaussian import LinearGaussianKLOEDBenchmark
from .instances.linear_gaussian_pred import LinearGaussianPredOEDBenchmark
from .instances.nonlinear_gaussian import NonLinearGaussianPredOEDBenchmark
from .lotka_volterra import LotkaVolterraOEDBenchmark

__all__ = [
    "LinearGaussianKLOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
    "NonLinearGaussianPredOEDBenchmark",
    "LotkaVolterraOEDBenchmark",
    "ObstructedAdvectionDiffusionOEDBenchmark",
]
