"""ODE QoI function and functional classes."""

from pyapprox_benchmarks.functions.ode.ode_qoi import (
    AllStatesEndpointODEFunctional,
    MaxODEFunctional,
    ODEFunctionalProtocol,
    ODEQoIFunction,
    ODETimeConfig,
    SingleStateEndpointODEFunctional,
)

__all__ = [
    "ODETimeConfig",
    "ODEQoIFunction",
    "ODEFunctionalProtocol",
    "AllStatesEndpointODEFunctional",
    "SingleStateEndpointODEFunctional",
    "MaxODEFunctional",
]
