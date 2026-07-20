"""Parameterized models for the Galerkin solver."""

from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)

__all__ = [
    "GalerkinPhysicsToODEResidualWithSetParamAdapter",
    "GalerkinPhysicsToODEResidualWithParamJacobianAdapter",
    "create_galerkin_physics_ode_residual",
]
