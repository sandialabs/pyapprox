"""Optimization problem definitions: generic types and PDE instances."""

from pyapprox_benchmarks.problems.optimization.constrained import (
    ConstrainedOptimizationProblem,
)
from pyapprox_benchmarks.problems.optimization.obstructed_flow_control import (
    ObstructedFlowControlProblem,
)

__all__ = [
    "ConstrainedOptimizationProblem",
    "ObstructedFlowControlProblem",
]
