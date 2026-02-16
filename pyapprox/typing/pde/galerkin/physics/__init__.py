"""Physics implementations for Galerkin finite element methods."""

from pyapprox.typing.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.typing.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.typing.pde.galerkin.physics.helpers import ScalarMassAssembler
from pyapprox.typing.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
    LinearAdvectionDiffusionReaction,
)
from pyapprox.typing.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.typing.pde.galerkin.physics.helmholtz import Helmholtz
from pyapprox.typing.pde.galerkin.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.typing.pde.galerkin.physics.composite_hyperelasticity import (
    CompositeHyperelasticityPhysics,
)
from pyapprox.typing.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.typing.pde.galerkin.physics.euler_bernoulli import (
    EulerBernoulliBeamAnalytical,
    EulerBernoulliBeamFEM,
)

# Backward-compatible alias
LinearElasticity = CompositeLinearElasticity
from pyapprox.typing.pde.galerkin.physics.stokes import StokesPhysics

__all__ = [
    "GalerkinBCMixin",
    "GalerkinPhysicsBase",
    "ScalarMassAssembler",
    "AdvectionDiffusionReaction",
    "BurgersPhysics",
    "HyperelasticityPhysics",
    "LinearAdvectionDiffusionReaction",
    "Helmholtz",
    "CompositeHyperelasticityPhysics",
    "CompositeLinearElasticity",
    "EulerBernoulliBeamAnalytical",
    "EulerBernoulliBeamFEM",
    "LinearElasticity",
    "StokesPhysics",
]
