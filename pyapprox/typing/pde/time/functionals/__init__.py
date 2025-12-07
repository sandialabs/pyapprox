"""Functionals for transient problems with adjoint and HVP support."""

from pyapprox.typing.pde.time.functionals.protocols import (
    TransientFunctionalWithJacobianProtocol,
    TransientFunctionalWithJacobianAndHVPProtocol,
)
from pyapprox.typing.pde.time.functionals.endpoint import (
    EndpointFunctional,
)
from pyapprox.typing.pde.time.functionals.mse import (
    TransientMSEFunctional,
)

__all__ = [
    # Protocols
    "TransientFunctionalWithJacobianProtocol",
    "TransientFunctionalWithJacobianAndHVPProtocol",
    # Implementations
    "EndpointFunctional",
    "TransientMSEFunctional",
]
