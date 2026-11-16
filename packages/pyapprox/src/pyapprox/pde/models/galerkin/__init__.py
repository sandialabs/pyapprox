"""Parameterized models for the Galerkin solver."""

from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    TimeIntegratedEssentialBoundaryFluxFunctional,
    essential_boundary_flux,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
    ParameterizedResidualDerivatives,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithHVPAdapter,
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
    GalerkinSteadyForwardModel,
)
from pyapprox.pde.models.galerkin.transient import (
    GalerkinTransientForwardModel,
)

__all__ = [
    "GalerkinPhysicsToODEResidualWithSetParamAdapter",
    "GalerkinPhysicsToODEResidualWithParamJacobianAdapter",
    "GalerkinPhysicsToODEResidualWithHVPAdapter",
    "create_galerkin_physics_ode_residual",
    "GalerkinParameterizedStateEquation",
    "GalerkinSteadyForwardModel",
    "GalerkinParameterizedResidual",
    "ParameterizedResidualDerivatives",
    "TimeIntegratedEssentialBoundaryFluxFunctional",
    "essential_boundary_flux",
    "GalerkinTransientForwardModel",
]
