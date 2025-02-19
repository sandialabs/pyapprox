"""The :mod:`pyapprox.expdesign` module implements a number of popular tools
for designing experiments
"""

from pyapprox.expdesign.sequences import SobolSequence, HaltonSequence
from pyapprox.expdesign.bayesian_oed import (
    get_bayesian_oed_optimizer,
    AbstractBayesianOED,
    BayesianBatchKLOED,
    BayesianBatchDeviationOED,
    BayesianSequentialOED,
    BayesianSequentialKLOED,
    BayesianSequentialDeviationOED,
)


__all__ = [
    "SobolSequence",
    "HaltonSequence",
    "get_bayesian_oed_optimizer",
    "BayesianBatchKLOED",
    "BayesianBatchDeviationOED",
    "BayesianSequentialOED",
    "BayesianSequentialKLOED",
    "BayesianSequentialDeviationOED",
    "AbstractBayesianOED",
]
