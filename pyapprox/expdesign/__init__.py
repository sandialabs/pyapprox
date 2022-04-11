"""The :mod:`pyapprox.expdesign` module implements a number of popular tools
for designing experiments
"""

from pyapprox.expdesign.low_discrepancy_sequences import (
    sobol_sequence, halton_sequence
)
from pyapprox.expdesign.linear_oed import (
    optimal_experimental_design, AlphabetOptimalDesign,
    NonLinearAlphabetOptimalDesign
)
from pyapprox.expdesign.bayesian_oed import (
    get_bayesian_oed_optimizer, AbstractBayesianOED,
    BayesianBatchKLOED, BayesianBatchDeviationOED,
    BayesianSequentialOED, BayesianSequentialKLOED,
    BayesianSequentialDeviationOED
)


__all__ = ["sobol_sequence", "halton_sequence", "optimal_experimental_design",
           "AlphabetOptimalDesign", "NonLinearAlphabetOptimalDesign",
           "get_bayesian_oed_optimizer", "BayesianBatchKLOED",
           "BayesianBatchDeviationOED", "BayesianSequentialOED",
           "BayesianSequentialKLOED", "BayesianSequentialDeviationOED",
           "AbstractBayesianOED"]
