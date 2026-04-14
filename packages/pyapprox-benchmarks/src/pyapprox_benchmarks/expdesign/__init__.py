"""OED benchmarks and problem wrappers for experimental design.

Benchmarks (have analytical ground truth):
- ``LinearGaussianKLOEDBenchmark`` — exact EIG via conjugate Gaussian
- ``LinearGaussianPredOEDBenchmark`` — exact prediction utility (linear QoI)
- ``NonLinearGaussianPredOEDBenchmark`` — exact prediction utility (lognormal QoI)

Problem wrappers (no ground truth):
- ``LotkaVolterraPredictionOEDProblem`` — nonlinear ODE-based prediction OED
- ``ObstructedAdvectionDiffusionOEDProblemWrapper`` — PDE-based OED
- ``FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper``
"""

from pyapprox_benchmarks.expdesign.advection_diffusion import (
    FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper,
    ObstructedAdvectionDiffusionOEDProblemWrapper,
    build_fixed_velocity_obstructed_advection_diffusion_oed_problem,
    build_obstructed_advection_diffusion_oed_problem,
)
from pyapprox_benchmarks.expdesign.cantilever_beam import (
    CantileverBeam2DLoadOEDBenchmark,
    build_cantilever_beam_oed_benchmark,
)
from pyapprox_benchmarks.expdesign.linear_gaussian import (
    LinearGaussianKLOEDBenchmark,
    build_linear_gaussian_kl_benchmark,
)
from pyapprox_benchmarks.expdesign.linear_gaussian_pred import (
    LinearGaussianPredOEDBenchmark,
    build_linear_gaussian_pred_benchmark,
)
from pyapprox_benchmarks.expdesign.lotka_volterra import (
    LotkaVolterraPredictionOEDProblem,
)
from pyapprox_benchmarks.expdesign.nonlinear_gaussian import (
    NonLinearGaussianPredOEDBenchmark,
    build_nonlinear_gaussian_pred_benchmark,
)

__all__ = [
    "LinearGaussianKLOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
    "NonLinearGaussianPredOEDBenchmark",
    "LotkaVolterraPredictionOEDProblem",
    "ObstructedAdvectionDiffusionOEDProblemWrapper",
    "FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper",
    "build_linear_gaussian_kl_benchmark",
    "build_linear_gaussian_pred_benchmark",
    "build_nonlinear_gaussian_pred_benchmark",
    "build_obstructed_advection_diffusion_oed_problem",
    "build_fixed_velocity_obstructed_advection_diffusion_oed_problem",
    "CantileverBeam2DLoadOEDBenchmark",
    "build_cantilever_beam_oed_benchmark",
]
