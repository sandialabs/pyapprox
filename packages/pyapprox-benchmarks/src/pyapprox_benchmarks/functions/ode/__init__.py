"""ODE QoI function and functional classes."""

from pyapprox_benchmarks.functions.ode.ode_qoi import (
    AllStatesEndpointODEFunctional,
    MaxODEFunctional,
    ODEFunctionalProtocol,
    ODEQoIFunction,
    ODETimeConfig,
    SingleStateEndpointODEFunctional,
)
from pyapprox_benchmarks.functions.ode.time_modulated_quadratic import (
    TimeModulatedQuadraticODE,
)

__all__ = [
    "ODETimeConfig",
    "ODEQoIFunction",
    "ODEFunctionalProtocol",
    "AllStatesEndpointODEFunctional",
    "SingleStateEndpointODEFunctional",
    "MaxODEFunctional",
    "TimeModulatedQuadraticODE",
]
