"""PDE benchmark instances."""

from pyapprox.benchmarks.instances.pde.cantilever_beam import (
    cantilever_beam_1d,
    cantilever_beam_2d_linear,
    cantilever_beam_2d_neohookean,
)
from pyapprox.benchmarks.instances.pde.elastic_bar import (
    elastic_bar_1d,
)
from pyapprox.benchmarks.instances.pde.pressurized_cylinder import (
    hyperelastic_pressurized_cylinder_2d,
    pressurized_cylinder_2d,
)

__all__ = [
    "elastic_bar_1d",
    "hyperelastic_pressurized_cylinder_2d",
    "pressurized_cylinder_2d",
    "cantilever_beam_1d",
    "cantilever_beam_2d_linear",
    "cantilever_beam_2d_neohookean",
]
