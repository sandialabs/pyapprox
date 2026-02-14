"""Backward compatibility re-exports.

Use pyapprox.typing.benchmarks.instances.analytic instead.
"""

from pyapprox.typing.benchmarks.instances.analytic.ishigami import ishigami_3d
from pyapprox.typing.benchmarks.instances.analytic.sobol_g import (
    sobol_g_6d,
    sobol_g_4d,
)

__all__ = ["ishigami_3d", "sobol_g_6d", "sobol_g_4d"]
