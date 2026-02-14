"""Analytic benchmark instances."""

from pyapprox.typing.benchmarks.instances.analytic.ishigami import ishigami_3d
from pyapprox.typing.benchmarks.instances.analytic.sobol_g import (
    sobol_g_6d,
    sobol_g_4d,
)
from pyapprox.typing.benchmarks.instances.analytic.rosenbrock import (
    rosenbrock_2d,
    rosenbrock_10d,
)
from pyapprox.typing.benchmarks.instances.analytic.branin import branin_2d
from pyapprox.typing.benchmarks.instances.analytic.genz import (
    genz_oscillatory_2d,
    genz_product_peak_2d,
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_oscillatory_5d,
    genz_gaussian_peak_5d,
)

__all__ = [
    "ishigami_3d",
    "sobol_g_6d",
    "sobol_g_4d",
    "rosenbrock_2d",
    "rosenbrock_10d",
    "branin_2d",
    "genz_oscillatory_2d",
    "genz_product_peak_2d",
    "genz_corner_peak_2d",
    "genz_gaussian_peak_2d",
    "genz_oscillatory_5d",
    "genz_gaussian_peak_5d",
]
