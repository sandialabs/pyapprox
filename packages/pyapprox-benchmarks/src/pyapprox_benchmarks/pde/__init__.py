"""PDE forward UQ problems."""

from pyapprox_benchmarks.pde.cantilever_beam import (
    MESH_PATHS,
    CantileverBeam2DForwardModel,
    CompositeBeam1DForwardModel,
    CantileverBeam1DKLEForwardModel,
    build_cantilever_beam_1d,
    build_cantilever_beam_1d_spde,
    build_cantilever_beam_2d_linear,
    build_cantilever_beam_2d_linear_spde,
    build_cantilever_beam_2d_neohookean,
    build_cantilever_beam_2d_neohookean_spde,
)
from pyapprox_benchmarks.pde.cantilever_beam_1d_analytical import (
    build_cantilever_beam_1d_analytical,
)
from pyapprox_benchmarks.pde.cantilever_beam_2d_analytical import (
    build_cantilever_beam_2d_analytical,
)
from pyapprox_benchmarks.pde.elastic_bar import (
    build_elastic_bar_1d,
)
from pyapprox_benchmarks.pde.pressurized_cylinder import (
    build_hyperelastic_pressurized_cylinder_2d,
    build_pressurized_cylinder_2d,
)

__all__ = [
    "MESH_PATHS",
    "CantileverBeam2DForwardModel",
    "CompositeBeam1DForwardModel",
    "CantileverBeam1DKLEForwardModel",
    "build_cantilever_beam_1d",
    "build_cantilever_beam_1d_analytical",
    "build_cantilever_beam_1d_spde",
    "build_cantilever_beam_2d_analytical",
    "build_cantilever_beam_2d_linear",
    "build_cantilever_beam_2d_linear_spde",
    "build_cantilever_beam_2d_neohookean",
    "build_cantilever_beam_2d_neohookean_spde",
    "build_elastic_bar_1d",
    "build_hyperelastic_pressurized_cylinder_2d",
    "build_pressurized_cylinder_2d",
]
