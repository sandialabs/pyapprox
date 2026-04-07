"""Analytic benchmark instances."""

from pyapprox_benchmarks.instances.analytic.branin import branin_2d
from pyapprox_benchmarks.instances.analytic.cantilever_beam import (
    cantilever_beam_1d_analytical,
)
from pyapprox_benchmarks.instances.analytic.cantilever_beam_2d import (
    cantilever_beam_2d_analytical,
)
from pyapprox_benchmarks.instances.analytic.genz import (
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_gaussian_peak_5d,
    genz_oscillatory_2d,
    genz_oscillatory_5d,
    genz_product_peak_2d,
)
from pyapprox_benchmarks.instances.analytic.ishigami import ishigami_3d
from pyapprox_benchmarks.instances.analytic.rosenbrock import (
    rosenbrock_2d,
    rosenbrock_10d,
)
from pyapprox_benchmarks.instances.analytic.sobol_g import (
    sobol_g_4d,
    sobol_g_6d,
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

# TODO: There are no tests of analytical functions/models and
# also no test of registry access at this level at least.
# TODO: decide on naming convention for this package function vs
# model. E.g. function is any mathematical map from inputs to
# outputs and can be based on evaluating a function of a model
# state space. A model is technically a function that maps inputs
# like boundary conditions, initial conditions, to state space
# solution, but is based on solving governing equations. Is this
# distinction worthwhile?
