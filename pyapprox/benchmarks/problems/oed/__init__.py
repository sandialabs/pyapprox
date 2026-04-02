"""OED problem classes for benchmarks."""

from pyapprox.benchmarks.problems.oed.kl_problem import KLOEDProblem
from pyapprox.benchmarks.problems.oed.prediction_problem import (
    PredictionOEDProblem,
)

__all__ = [
    "KLOEDProblem",
    "PredictionOEDProblem",
]
