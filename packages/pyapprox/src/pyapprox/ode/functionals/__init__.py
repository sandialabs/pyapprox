"""Functionals for transient problems with adjoint and HVP support."""

from pyapprox.ode.functionals.endpoint import (
    EndpointFunctional,
)
from pyapprox.ode.functionals.mse import (
    TransientMSEFunctional,
)
from pyapprox.ode.functionals.protocols import (
    TimeQuadratureAwareFunctionalProtocol,
    TransientFunctionalWithJacobianAndHVPProtocol,
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.ode.functionals.tikhonov import (
    TikhonovAugmentedFunctional,
)
from pyapprox.ode.functionals.time_integrated_weighted_l2 import (
    TimeIntegratedWeightedL2Functional,
)
from pyapprox.ode.functionals.weighted_endpoint import (
    WeightedEndpointFunctional,
)

__all__ = [
    # Protocols
    "TimeQuadratureAwareFunctionalProtocol",
    "TransientFunctionalWithJacobianProtocol",
    "TransientFunctionalWithJacobianAndHVPProtocol",
    # Implementations
    "EndpointFunctional",
    "TikhonovAugmentedFunctional",
    "TimeIntegratedWeightedL2Functional",
    "TransientMSEFunctional",
    "WeightedEndpointFunctional",
]
