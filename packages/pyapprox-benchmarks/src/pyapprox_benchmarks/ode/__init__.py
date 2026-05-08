"""ODE forward UQ problems."""

from pyapprox_benchmarks.ode.chemical_reaction import (
    build_chemical_reaction_surface,
)
from pyapprox_benchmarks.ode.coupled_springs import (
    build_coupled_springs_2mass,
)
from pyapprox_benchmarks.ode.hastings_ecology import (
    build_hastings_ecology_3species,
)
from pyapprox_benchmarks.ode.lotka_volterra import (
    build_lotka_volterra_3species,
)

__all__ = [
    "build_chemical_reaction_surface",
    "build_coupled_springs_2mass",
    "build_hastings_ecology_3species",
    "build_lotka_volterra_3species",
]
