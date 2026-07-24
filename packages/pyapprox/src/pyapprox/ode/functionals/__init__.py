"""Functionals for transient problems with adjoint and HVP support."""

from pyapprox.ode.functionals.endpoint import (
    EndpointFunctional,
)
from pyapprox.ode.functionals.mse import (
    TransientMSEFunctional,
)
from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianAndHVPProtocol,
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.ode.functionals.weighted_endpoint import (
    WeightedEndpointFunctional,
)

__all__ = [
    # Protocols
    "TransientFunctionalWithJacobianProtocol",
    "TransientFunctionalWithJacobianAndHVPProtocol",
    # Implementations
    "EndpointFunctional",
    "TransientMSEFunctional",
    "WeightedEndpointFunctional",
]
