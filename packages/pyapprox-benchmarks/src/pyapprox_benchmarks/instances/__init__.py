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
from pyapprox_benchmarks.instances.multifidelity import (
    multioutput_ensemble_3x3,
    polynomial_ensemble_3model,
    polynomial_ensemble_5model,
    psd_multioutput_ensemble_3x3,
    tunable_ensemble_3model,
)
from pyapprox_benchmarks.instances.ode import (
    chemical_reaction_surface,
    coupled_springs_2mass,
    hastings_ecology_3species,
    lotka_volterra_3species,
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
    "polynomial_ensemble_5model",
    "polynomial_ensemble_3model",
    "multioutput_ensemble_3x3",
    "psd_multioutput_ensemble_3x3",
    "tunable_ensemble_3model",
    "lotka_volterra_3species",
    "coupled_springs_2mass",
    "hastings_ecology_3species",
    "chemical_reaction_surface",
    "elastic_bar_1d",
    "cantilever_beam_1d",
    "cantilever_beam_2d_linear",
    "cantilever_beam_2d_neohookean",
    "cantilever_beam_1d_analytical",
    "cantilever_beam_2d_analytical",
]
