"""Parameterized models for the spectral collocation solver."""

from pyapprox.pde.models.collocation.factory import (
    create_collocation_model,
)
from pyapprox.pde.models.collocation.physics_adapter import (
    CollocationPhysicsToODEResidualWithHVPAdapter,
    CollocationPhysicsToODEResidualWithParamJacobianAdapter,
    CollocationPhysicsToODEResidualWithSetParamAdapter,
    create_collocation_physics_ode_residual,
)
from pyapprox.pde.models.collocation.steady import (
    CollocationStateEquationAdapter,
    SteadyForwardModel,
)
from pyapprox.pde.models.collocation.transient import (
    TransientForwardModel,
)

__all__ = [
    "CollocationPhysicsToODEResidualWithSetParamAdapter",
    "CollocationPhysicsToODEResidualWithParamJacobianAdapter",
    "CollocationPhysicsToODEResidualWithHVPAdapter",
    "create_collocation_physics_ode_residual",
    "create_collocation_model",
    "CollocationStateEquationAdapter",
    "SteadyForwardModel",
    "TransientForwardModel",
]
