"""Inverse problem classes for benchmarks."""

from pyapprox.benchmarks.problems.inverse.inference_problem import (
    BayesianInferenceProblem,
    GaussianInferenceProblem,
    build_linear_gaussian_inference_problem,
)

__all__ = [
    "BayesianInferenceProblem",
    "GaussianInferenceProblem",
    "build_linear_gaussian_inference_problem",
]
