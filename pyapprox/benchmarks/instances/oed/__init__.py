"""OED benchmark instances."""

from pyapprox.benchmarks.instances.oed.linear_gaussian import (
    LinearGaussianKLOEDBenchmark,
    build_linear_gaussian_kl_benchmark,
)
from pyapprox.benchmarks.instances.oed.linear_gaussian_pred import (
    LinearGaussianPredOEDBenchmark,
    build_linear_gaussian_pred_benchmark,
)
from pyapprox.benchmarks.instances.oed.nonlinear_gaussian import (
    NonLinearGaussianPredOEDBenchmark,
    build_nonlinear_gaussian_pred_benchmark,
)

__all__ = [
    "LinearGaussianKLOEDBenchmark",
    "build_linear_gaussian_kl_benchmark",
    "LinearGaussianPredOEDBenchmark",
    "build_linear_gaussian_pred_benchmark",
    "NonLinearGaussianPredOEDBenchmark",
    "build_nonlinear_gaussian_pred_benchmark",
]
