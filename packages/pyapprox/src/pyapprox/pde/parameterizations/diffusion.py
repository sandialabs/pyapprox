"""Factory for parameterizing the collocation ADR diffusion field."""

from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.util.backends.protocols import Array, Backend


def create_diffusion_parameterization(
    physics: AdvectionDiffusionReaction[Array],
    bkd: Backend[Array],
    field_map: FieldMapProtocol[Array],
) -> CollocationAdvectionDiffusionParameterization[Array]:
    """Factory: parameterize the diffusion field through the ADR facade.

    Parameters
    ----------
    physics : AdvectionDiffusionReaction
        Collocation physics to bind.
    bkd : Backend
        Computational backend.
    field_map : FieldMapProtocol
        Field map for diffusion.
    """
    return CollocationAdvectionDiffusionParameterization(
        physics, diffusion_map=field_map, bkd=bkd
    )
