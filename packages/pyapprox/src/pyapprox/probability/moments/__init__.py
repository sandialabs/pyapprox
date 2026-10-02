"""Joint moments of targets and observations from weighted samples.

``DenseBlocks`` holds the mean and covariance of the stacked targets and
noise-free observations. ``WeightedAccumulator`` and
``UnbiasedMCAccumulator`` build it from batches of joint outputs and
quadrature weights.
"""

from .accumulators import UnbiasedMCAccumulator, WeightedAccumulator
from .blocks import DenseBlocks
from .protocols import CovarianceBlocksProtocol, MomentAccumulatorProtocol

__all__ = [
    "CovarianceBlocksProtocol",
    "DenseBlocks",
    "MomentAccumulatorProtocol",
    "UnbiasedMCAccumulator",
    "WeightedAccumulator",
]
