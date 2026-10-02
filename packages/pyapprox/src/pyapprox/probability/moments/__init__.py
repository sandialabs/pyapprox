"""Joint moments of targets and observations from weighted samples.

``DenseBlocks`` holds the mean and covariance of the stacked targets and
noise-free observations. ``WeightedAccumulator`` and
``UnbiasedMCAccumulator`` build it from batches of joint outputs and
quadrature weights. ``QuadratureMoments`` runs a joint evaluator on a
rule's points to get them, and ``CachedMoments`` reuses stored outputs.
``SampledRule`` and ``AtLevel`` turn samplers and level-parameterized
rules into fixed rules. ``NoRepair`` refuses indefinite blocks and
``EigenClip`` repairs them; consumers default to ``NoRepair``, so repair
happens only when a caller asks for it.
"""

from .accumulators import UnbiasedMCAccumulator, WeightedAccumulator
from .blocks import DenseBlocks
from .protocols import (
    CovarianceBlocksProtocol,
    CovarianceRepairProtocol,
    MomentAccumulatorProtocol,
)
from .repair import EigenClip, NoRepair
from .rules import AtLevel, SampledRule, WeightedRuleProtocol
from .sources import CachedMoments, MomentSourceProtocol, QuadratureMoments

__all__ = [
    "AtLevel",
    "CachedMoments",
    "CovarianceBlocksProtocol",
    "CovarianceRepairProtocol",
    "DenseBlocks",
    "EigenClip",
    "MomentAccumulatorProtocol",
    "MomentSourceProtocol",
    "NoRepair",
    "QuadratureMoments",
    "SampledRule",
    "UnbiasedMCAccumulator",
    "WeightedAccumulator",
    "WeightedRuleProtocol",
]
