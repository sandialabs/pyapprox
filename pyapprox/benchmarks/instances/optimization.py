"""Backward compatibility re-exports.

Use pyapprox.benchmarks.instances.analytic instead.
"""

from pyapprox.benchmarks.instances.analytic.branin import branin_2d
from pyapprox.benchmarks.instances.analytic.rosenbrock import (
    rosenbrock_2d,
    rosenbrock_10d,
)

__all__ = ["rosenbrock_2d", "rosenbrock_10d", "branin_2d"]
