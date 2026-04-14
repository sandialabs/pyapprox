"""OED benchmarks and problem wrappers for experimental design.

Benchmarks (have analytical ground truth):
- ``LinearGaussianKLOEDBenchmark`` — exact EIG via conjugate Gaussian
- ``LinearGaussianPredOEDBenchmark`` — exact prediction utility (linear QoI)
- ``NonLinearGaussianPredOEDBenchmark`` — exact prediction utility (lognormal QoI)

Problem wrappers (no ground truth):
- ``LotkaVolterraPredictionOEDProblem`` — nonlinear ODE-based prediction OED
- ``ObstructedAdvectionDiffusionOEDBenchmark`` — PDE-based OED
- ``FixedVelocityObstructedAdvectionDiffusionOEDBenchmark`` — fixed-velocity variant
"""

from pyapprox_benchmarks.expdesign.linear_gaussian import (
    LinearGaussianKLOEDBenchmark,
    build_linear_gaussian_kl_benchmark,
)
from pyapprox_benchmarks.expdesign.linear_gaussian_pred import (
    LinearGaussianPredOEDBenchmark,
    build_linear_gaussian_pred_benchmark,
)
from pyapprox_benchmarks.expdesign.nonlinear_gaussian import (
    NonLinearGaussianPredOEDBenchmark,
    build_nonlinear_gaussian_pred_benchmark,
)
from pyapprox_benchmarks.expdesign.lotka_volterra import (
    LotkaVolterraPredictionOEDProblem,
)
from pyapprox_benchmarks.expdesign.advection_diffusion import (
    FixedVelocityObstructedAdvectionDiffusionOEDBenchmark,
    ObstructedAdvectionDiffusionOEDBenchmark,
    build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark,
    build_obstructed_advection_diffusion_oed_benchmark,
)

__all__ = [
    "LinearGaussianKLOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
    "NonLinearGaussianPredOEDBenchmark",
    "LotkaVolterraPredictionOEDProblem",
    "ObstructedAdvectionDiffusionOEDBenchmark",
    "FixedVelocityObstructedAdvectionDiffusionOEDBenchmark",
    "build_linear_gaussian_kl_benchmark",
    "build_linear_gaussian_pred_benchmark",
    "build_nonlinear_gaussian_pred_benchmark",
    "build_obstructed_advection_diffusion_oed_benchmark",
    "build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark",
]
