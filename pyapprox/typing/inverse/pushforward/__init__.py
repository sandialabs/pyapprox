"""
Pushforward module for Gaussian distributions.

This module provides tools for computing the mean and covariance
of Gaussian distributions when pushed through linear transformations.
"""

from .gaussian import GaussianPushforward
from .prediction import DenseGaussianPrediction

__all__ = [
    "GaussianPushforward",
    "DenseGaussianPrediction",
]
