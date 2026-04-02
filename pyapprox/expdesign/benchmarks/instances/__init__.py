"""Fixed OED benchmark instances with ground truth."""

from .linear_gaussian import LinearGaussianKLOEDBenchmark
from .linear_gaussian_pred import LinearGaussianPredOEDBenchmark
from .nonlinear_gaussian import NonLinearGaussianPredOEDBenchmark

__all__ = [
    "LinearGaussianKLOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
    "NonLinearGaussianPredOEDBenchmark",
]
