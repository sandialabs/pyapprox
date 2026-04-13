"""Problem classes for benchmarks.

Each problem type is a standalone class with exactly the fields it needs.
"""

from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem
from pyapprox_benchmarks.problems.function_over_domain import (
    FunctionOverDomainProblem,
)
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)
from pyapprox_benchmarks.problems.inverse import (
    BayesianInferenceProblem,
    GaussianInferenceProblem,
)
from pyapprox_benchmarks.problems.oed import (
    KLOEDProblem,
    PredictionOEDProblem,
)
from pyapprox_benchmarks.problems.optimization import (
    ConstrainedOptimizationProblem,
)

__all__ = [
    "BayesianInferenceProblem",
    "ConstrainedOptimizationProblem",
    "ForwardUQProblem",
    "FunctionOverDomainProblem",
    "GaussianInferenceProblem",
    "KLOEDProblem",
    "MultifidelityForwardUQProblem",
    "PredictionOEDProblem",
]
