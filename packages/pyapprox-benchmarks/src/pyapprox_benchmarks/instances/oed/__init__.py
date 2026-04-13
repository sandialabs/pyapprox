"""OED benchmark instances — re-exports from ``expdesign/``.

All classes have moved to ``pyapprox_benchmarks.expdesign``.
This module re-exports them for backward compatibility.
"""

from pyapprox_benchmarks.expdesign.advection_diffusion import (
    FixedVelocityObstructedAdvectionDiffusionOEDBenchmark,
    ObstructedAdvectionDiffusionOEDBenchmark,
    build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark,
    build_obstructed_advection_diffusion_oed_benchmark,
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
    LotkaVolterraOEDBenchmark,
)
from pyapprox_benchmarks.expdesign.nonlinear_gaussian import (
    NonLinearGaussianPredOEDBenchmark,
    build_nonlinear_gaussian_pred_benchmark,
)

__all__ = [
    "FixedVelocityObstructedAdvectionDiffusionOEDBenchmark",
    "LinearGaussianKLOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
    "LotkaVolterraOEDBenchmark",
    "NonLinearGaussianPredOEDBenchmark",
    "ObstructedAdvectionDiffusionOEDBenchmark",
    "build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark",
    "build_linear_gaussian_kl_benchmark",
    "build_linear_gaussian_pred_benchmark",
    "build_nonlinear_gaussian_pred_benchmark",
    "build_obstructed_advection_diffusion_oed_benchmark",
]
