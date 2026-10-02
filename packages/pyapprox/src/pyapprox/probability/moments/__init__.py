"""Joint moments of targets and observations from weighted samples.

``DenseBlocks`` holds the mean and covariance of the stacked targets and
noise-free observations. ``WeightedAccumulator`` and
``UnbiasedMCAccumulator`` build it from batches of joint outputs and
quadrature weights. ``QuadratureMoments`` runs a joint evaluator on a
rule's points to get them, and ``CachedMoments`` reuses stored outputs.
``SampledRule`` and ``AtLevel`` turn samplers and level-parameterized
rules into fixed rules.
"""

from .accumulators import UnbiasedMCAccumulator, WeightedAccumulator
from .blocks import DenseBlocks
from .protocols import CovarianceBlocksProtocol, MomentAccumulatorProtocol
from .rules import AtLevel, SampledRule, WeightedRuleProtocol
from .sources import CachedMoments, MomentSourceProtocol, QuadratureMoments

__all__ = [
    "AtLevel",
    "CachedMoments",
    "CovarianceBlocksProtocol",
    "DenseBlocks",
    "MomentAccumulatorProtocol",
    "MomentSourceProtocol",
    "QuadratureMoments",
    "SampledRule",
    "UnbiasedMCAccumulator",
    "WeightedAccumulator",
    "WeightedRuleProtocol",
]
