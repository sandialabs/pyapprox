"""Backward compatibility re-exports.

Use pyapprox.typing.benchmarks.instances.analytic instead.
"""

from pyapprox.typing.benchmarks.instances.analytic.rosenbrock import (
    rosenbrock_2d,
    rosenbrock_10d,
)
from pyapprox.typing.benchmarks.instances.analytic.branin import branin_2d

__all__ = ["rosenbrock_2d", "rosenbrock_10d", "branin_2d"]
