"""PDE benchmark instances."""

from pyapprox.typing.benchmarks.instances.pde.elastic_bar import (
    elastic_bar_1d,
)
from pyapprox.typing.benchmarks.instances.pde.pressurized_cylinder import (
    pressurized_cylinder_2d,
)

__all__ = [
    "elastic_bar_1d",
    "pressurized_cylinder_2d",
]
