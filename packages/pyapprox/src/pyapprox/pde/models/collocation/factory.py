"""Convenience factory for parameterized collocation models."""

from typing import Optional

from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.models.collocation.physics_adapter import (
    create_collocation_physics_ode_residual,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


def create_collocation_model(
    physics: PhysicsProtocol[Array],
    bkd: Backend[Array],
    parameterization: Optional[ParameterizationProtocol[Array]] = None,
) -> CollocationModel[Array]:
    """Create a CollocationModel wired with the widest adapter tier.

    One-liner entry point for parameterized problems: builds the
    fixed-tier ODE-residual adapter from the parameterization's
    ParamDerivatives bundle and injects it into the model. Without a
    parameterization this is equivalent to ``CollocationModel(physics,
    bkd)``.

    Parameters
    ----------
    physics : PhysicsProtocol
        Physics object defining the PDE.
    bkd : Backend
        Computational backend.
    parameterization : ParameterizationProtocol, optional
        Maps parameter vectors to physics coefficients.

    Returns
    -------
    CollocationModel
        Model whose adapter tier matches the parameterization's
        derivative capability.
    """
    if parameterization is None:
        return CollocationModel(physics, bkd)
    adapter = create_collocation_physics_ode_residual(
        physics, bkd, parameterization
    )
    return CollocationModel(physics, bkd, adapter=adapter)
