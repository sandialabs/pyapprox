"""ODE benchmark instances."""

from pyapprox_benchmarks.instances.ode.chemical_reaction import (
    chemical_reaction_surface,
)
from pyapprox_benchmarks.instances.ode.coupled_springs import (
    coupled_springs_2mass,
)
from pyapprox_benchmarks.instances.ode.hastings_ecology import (
    hastings_ecology_3species,
)
from pyapprox_benchmarks.instances.ode.lotka_volterra import (
    lotka_volterra_3species,
)

__all__ = [
    "lotka_volterra_3species",
    "coupled_springs_2mass",
    "hastings_ecology_3species",
    "chemical_reaction_surface",
]
