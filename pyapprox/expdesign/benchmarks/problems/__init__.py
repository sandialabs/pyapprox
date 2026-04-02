"""Composable OED problem classes."""

from .inference_problem import BayesianInferenceProblem, GaussianInferenceProblem
from .kl_problem import KLOEDProblem
from .prediction_problem import PredictionOEDProblem

__all__ = [
    "BayesianInferenceProblem",
    "GaussianInferenceProblem",
    "KLOEDProblem",
    "PredictionOEDProblem",
]
