"""PDE physics module for spectral collocation methods."""

from pyapprox.typing.pde.collocation.physics.base import (
    AbstractPhysics,
    AbstractScalarPhysics,
    AbstractVectorPhysics,
)
from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
    AdvectionDiffusionReactionWithParam,
    create_steady_diffusion,
    create_advection_diffusion,
)
from pyapprox.typing.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
    create_linear_elasticity,
)

__all__ = [
    # Base classes
    "AbstractPhysics",
    "AbstractScalarPhysics",
    "AbstractVectorPhysics",
    # Advection-Diffusion-Reaction
    "AdvectionDiffusionReaction",
    "AdvectionDiffusionReactionWithParam",
    "create_steady_diffusion",
    "create_advection_diffusion",
    # Linear Elasticity
    "LinearElasticityPhysics",
    "create_linear_elasticity",
]
