"""PDE benchmark instances."""

from pyapprox.typing.benchmarks.instances.pde.elastic_bar import (
    elastic_bar_1d,
)
from pyapprox.typing.benchmarks.instances.pde.pressurized_cylinder import (
    hyperelastic_pressurized_cylinder_2d,
    pressurized_cylinder_2d,
)
from pyapprox.typing.benchmarks.instances.pde.cantilever_beam import (
    cantilever_beam_1d,
    cantilever_beam_2d_linear,
    cantilever_beam_2d_neohookean,
)

__all__ = [
    "elastic_bar_1d",
    "hyperelastic_pressurized_cylinder_2d",
    "pressurized_cylinder_2d",
    "cantilever_beam_1d",
    "cantilever_beam_2d_linear",
    "cantilever_beam_2d_neohookean",
]
