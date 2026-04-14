"""Fixed benchmark instances with known ground truth."""

from pyapprox_benchmarks.instances.analytic import (
    branin_2d,
    cantilever_beam_1d_analytical,
    cantilever_beam_2d_analytical,
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_gaussian_peak_5d,
    genz_oscillatory_2d,
    genz_oscillatory_5d,
    genz_product_peak_2d,
    ishigami_3d,
    rosenbrock_2d,
    rosenbrock_10d,
    sobol_g_4d,
    sobol_g_6d,
)
from pyapprox_benchmarks.instances.pde import (
    cantilever_beam_1d,
    cantilever_beam_2d_linear,
    cantilever_beam_2d_neohookean,
    elastic_bar_1d,
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
    "elastic_bar_1d",
    "cantilever_beam_1d",
    "cantilever_beam_2d_linear",
    "cantilever_beam_2d_neohookean",
    "cantilever_beam_1d_analytical",
    "cantilever_beam_2d_analytical",
]
