"""Parameterized models for the Galerkin solver."""

from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithHVPAdapter,
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.models.galerkin.steady import (
    GalerkinStateEquationWithHVPAdapter,
)
from pyapprox.pde.models.galerkin.transient import (
    GalerkinTransientForwardModel,
)

__all__ = [
    "GalerkinPhysicsToODEResidualWithSetParamAdapter",
    "GalerkinPhysicsToODEResidualWithParamJacobianAdapter",
    "GalerkinPhysicsToODEResidualWithHVPAdapter",
    "create_galerkin_physics_ode_residual",
    "GalerkinStateEquationWithHVPAdapter",
    "GalerkinTransientForwardModel",
]
