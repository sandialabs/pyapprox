"""Fixed benchmark instances with known ground truth."""

from pyapprox.typing.benchmarks.instances.sensitivity import (
    ishigami_3d,
)
from pyapprox.typing.benchmarks.instances.optimization import (
    rosenbrock_2d,
    rosenbrock_10d,
)

__all__ = [
    "ishigami_3d",
    "rosenbrock_2d",
    "rosenbrock_10d",
]
