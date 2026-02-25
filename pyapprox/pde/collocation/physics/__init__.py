"""PDE physics module for spectral collocation methods."""

from pyapprox.pde.collocation.physics.base import (
    AbstractPhysics,
    AbstractScalarPhysics,
    AbstractVectorPhysics,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
    create_steady_diffusion,
    create_advection_diffusion,
)
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
    create_linear_elasticity,
)
from pyapprox.pde.collocation.physics.helmholtz import (
    HelmholtzPhysics,
    create_helmholtz,
)
from pyapprox.pde.collocation.physics.burgers import (
    BurgersPhysics1D,
    create_burgers_1d,
)
from pyapprox.pde.collocation.physics.shallow_ice import (
    ShallowIcePhysics,
    create_shallow_ice,
)
from pyapprox.pde.collocation.physics.reaction_diffusion import (
    ReactionProtocol,
    SymbolicReactionProtocol,
    TwoSpeciesReactionDiffusionPhysics,
    LinearReaction,
    FitzHughNagumoReaction,
    create_two_species_reaction_diffusion,
)
from pyapprox.pde.collocation.physics.fitzhugh_nagumo import (
    FitzHughNagumoPhysics,
    create_fitzhugh_nagumo,
)
from pyapprox.pde.collocation.physics.shallow_wave import (
    ShallowWavePhysics,
    create_shallow_wave,
)
from pyapprox.pde.collocation.physics.shallow_shelf import (
    ShallowShelfVelocityPhysics,
    ShallowShelfDepthPhysics,
    create_shallow_shelf_velocity,
    create_shallow_shelf_depth,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
    create_hyperelasticity,
)
from pyapprox.pde.collocation.physics.stress_models import (
    StressModelProtocol,
    StressModelWithTangentProtocol,
    SymbolicStressModelProtocol,
    NeoHookeanStress,
    register_stress_model,
    create_stress_model,
    list_stress_models,
)

__all__ = [
    # Base classes
    "AbstractPhysics",
    "AbstractScalarPhysics",
    "AbstractVectorPhysics",
    # Advection-Diffusion-Reaction
    "AdvectionDiffusionReaction",
    "create_steady_diffusion",
    "create_advection_diffusion",
    # Linear Elasticity
    "LinearElasticityPhysics",
    "create_linear_elasticity",
    # Helmholtz
    "HelmholtzPhysics",
    "create_helmholtz",
    # Burgers
    "BurgersPhysics1D",
    "create_burgers_1d",
    # Shallow Ice
    "ShallowIcePhysics",
    "create_shallow_ice",
    # Reaction-Diffusion
    "ReactionProtocol",
    "SymbolicReactionProtocol",
    "TwoSpeciesReactionDiffusionPhysics",
    "LinearReaction",
    "FitzHughNagumoReaction",
    "create_two_species_reaction_diffusion",
    # FitzHugh-Nagumo
    "FitzHughNagumoPhysics",
    "create_fitzhugh_nagumo",
    # Shallow Wave
    "ShallowWavePhysics",
    "create_shallow_wave",
    # Shallow Shelf
    "ShallowShelfVelocityPhysics",
    "ShallowShelfDepthPhysics",
    "create_shallow_shelf_velocity",
    "create_shallow_shelf_depth",
    # Hyperelasticity
    "HyperelasticityPhysics",
    "create_hyperelasticity",
    # Stress models
    "StressModelProtocol",
    "StressModelWithTangentProtocol",
    "SymbolicStressModelProtocol",
    "NeoHookeanStress",
    "register_stress_model",
    "create_stress_model",
    "list_stress_models",
]
