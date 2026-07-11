"""PDE-based function builders."""

from pyapprox.util.optional_deps import package_available

from pyapprox_benchmarks.functions.pde.cantilever_beam_obs_map import (
    build_cantilever_beam_design_matrix,
    build_cantilever_beam_obs_map,
)

__all__ = [
    "build_cantilever_beam_design_matrix",
    "build_cantilever_beam_obs_map",
]

if package_available("skfem"):
    from pyapprox_benchmarks.functions.pde.burgers import (
        build_periodic_burgers_physics,
        build_periodic_line_basis,
    )
    from pyapprox_benchmarks.functions.pde.chafee_infante import (
        build_chafee_infante_physics,
        build_line_basis,
    )

    __all__ += [
        "build_chafee_infante_physics",
        "build_line_basis",
        "build_periodic_burgers_physics",
        "build_periodic_line_basis",
    ]
