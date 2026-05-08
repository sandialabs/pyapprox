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
from .pressurized_cylinder_2d import (
    create_linear_pressurized_cylinder_2d,
)

__all__ = [
    "create_steady_diffusion_1d",
    "create_transient_diffusion_1d",
    "create_linear_elastic_bar_1d",
    "create_hyperelastic_bar_1d",
    "create_linear_pressurized_cylinder_2d",
    "create_hyperelastic_pressurized_cylinder_2d",
]
