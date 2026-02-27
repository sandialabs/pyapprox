"""Physics implementations for Galerkin finite element methods."""

from pyapprox.util.optional_deps import package_available

if package_available("skfem"):
    from pyapprox.pde.galerkin.physics.advection_diffusion import (
        AdvectionDiffusionReaction,
        LinearAdvectionDiffusionReaction,
    )
    from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
    from pyapprox.pde.galerkin.physics.burgers import BurgersPhysics
    from pyapprox.pde.galerkin.physics.composite_hyperelasticity import (
        CompositeHyperelasticityPhysics,
    )
    from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
        CompositeLinearElasticity,
    )
    from pyapprox.pde.galerkin.physics.euler_bernoulli import (
        EulerBernoulliBeamAnalytical,
        EulerBernoulliBeamFEM,
    )
    from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
    from pyapprox.pde.galerkin.physics.helmholtz import Helmholtz
    from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler
    from pyapprox.pde.galerkin.physics.hyperelasticity import (
        HyperelasticityPhysics,
    )

    # Backward-compatible alias
    LinearElasticity = CompositeLinearElasticity
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics

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
else:
    __all__: list[str] = []
