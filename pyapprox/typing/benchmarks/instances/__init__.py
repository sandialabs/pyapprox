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
from pyapprox.typing.benchmarks.instances.quadrature import (
    genz_oscillatory_2d,
    genz_product_peak_2d,
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_oscillatory_5d,
    genz_gaussian_peak_5d,
)
from pyapprox.typing.benchmarks.instances.multifidelity import (
    polynomial_ensemble_5model,
    polynomial_ensemble_3model,
)
from pyapprox.typing.benchmarks.instances.ode import (
    lotka_volterra_3species,
    coupled_springs_2mass,
    hastings_ecology_3species,
    chemical_reaction_surface,
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
    "lotka_volterra_3species",
    "coupled_springs_2mass",
    "hastings_ecology_3species",
    "chemical_reaction_surface",
]
