"""Zoo of pre-configured forward models for common PDE problems."""

from .diffusion import (
    create_steady_diffusion_1d,
    create_transient_diffusion_1d,
)
from .elastic_bar_1d import (
    create_linear_elastic_bar_1d,
)
from .hyperelastic_bar_1d import (
    create_hyperelastic_bar_1d,
)
from .hyperelastic_cylinder_2d import (
    create_hyperelastic_pressurized_cylinder_2d,
)
from .obstructed_flow import (
    ParabolicInlet,
    ZeroVelocity,
    build_obstructed_mesh,
    extract_velocity_callable,
    solve_obstructed_stokes,
)
from .pressurized_cylinder_2d import (
    create_linear_pressurized_cylinder_2d,
)

__all__ = [
    "ParabolicInlet",
    "ZeroVelocity",
    "build_obstructed_mesh",
    "create_steady_diffusion_1d",
    "create_transient_diffusion_1d",
    "create_linear_elastic_bar_1d",
    "create_hyperelastic_bar_1d",
    "create_linear_pressurized_cylinder_2d",
    "create_hyperelastic_pressurized_cylinder_2d",
    "extract_velocity_callable",
    "solve_obstructed_stokes",
]
