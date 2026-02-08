"""
Conditional probability distributions.

This module provides conditional distributions where the distribution
parameters are functions of conditioning variables.
"""

from pyapprox.typing.probability.conditional.protocols import (
    ConditionalDistributionProtocol,
)
from pyapprox.typing.probability.conditional.gaussian import ConditionalGaussian
from pyapprox.typing.probability.conditional.beta import ConditionalBeta
from pyapprox.typing.probability.conditional.gamma import ConditionalGamma
from pyapprox.typing.probability.conditional.joint import (
    ConditionalIndependentJoint,
)
from pyapprox.typing.probability.conditional.multivariate_gaussian import (
    ConditionalDenseCholGaussian,
    ConditionalLowRankCholGaussian,
)

__all__ = [
    "ConditionalDistributionProtocol",
    "ConditionalGaussian",
    "ConditionalBeta",
    "ConditionalGamma",
    "ConditionalIndependentJoint",
    "ConditionalDenseCholGaussian",
    "ConditionalLowRankCholGaussian",
]
