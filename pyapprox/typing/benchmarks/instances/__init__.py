"""Fixed benchmark instances with known ground truth."""

from pyapprox.typing.benchmarks.instances.sensitivity import (
    ishigami_3d,
    sobol_g_6d,
    sobol_g_4d,
)
from pyapprox.typing.benchmarks.instances.optimization import (
    rosenbrock_2d,
    rosenbrock_10d,
    branin_2d,
)

__all__ = [
    "ishigami_3d",
    "sobol_g_6d",
    "sobol_g_4d",
    "rosenbrock_2d",
    "rosenbrock_10d",
    "branin_2d",
]
