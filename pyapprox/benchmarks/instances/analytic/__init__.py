"""Analytic benchmark instances."""

from pyapprox.benchmarks.instances.analytic.ishigami import ishigami_3d
from pyapprox.benchmarks.instances.analytic.sobol_g import (
    sobol_g_6d,
    sobol_g_4d,
)
from pyapprox.benchmarks.instances.analytic.rosenbrock import (
    rosenbrock_2d,
    rosenbrock_10d,
)
from pyapprox.benchmarks.instances.analytic.branin import branin_2d
from pyapprox.benchmarks.instances.analytic.genz import (
    genz_oscillatory_2d,
    genz_product_peak_2d,
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_oscillatory_5d,
    genz_gaussian_peak_5d,
)
from pyapprox.benchmarks.instances.analytic.cantilever_beam import (
    cantilever_beam_1d_analytical,
)
from pyapprox.benchmarks.instances.analytic.cantilever_beam_2d import (
    cantilever_beam_2d_analytical,
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
    "cantilever_beam_1d_analytical",
    "cantilever_beam_2d_analytical",
]
