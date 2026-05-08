"""OED problem classes for benchmarks."""

from pyapprox_benchmarks.problems.oed.advection_diffusion import (
    AdvectionDiffusionOEDProblem,
    FixedVelocityAdvectionDiffusionOEDProblem,
)
from pyapprox_benchmarks.problems.oed.kl_problem import KLOEDProblem
from pyapprox_benchmarks.problems.oed.prediction_problem import (
    PredictionOEDProblem,
)

__all__ = [
    "AdvectionDiffusionOEDProblem",
    "FixedVelocityAdvectionDiffusionOEDProblem",
    "KLOEDProblem",
    "PredictionOEDProblem",
]
