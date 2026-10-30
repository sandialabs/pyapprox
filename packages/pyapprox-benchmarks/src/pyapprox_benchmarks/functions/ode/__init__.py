"""ODE QoI function and functional classes."""

from pyapprox_benchmarks.functions.ode.ode_qoi import (
    AllStatesEndpointODEFunctional,
    MaxODEFunctional,
    ODEBenchmarkResidualProtocol,
    ODEFunctionalProtocol,
    ODEQoIFunction,
    ODETimeConfig,
    SelectedStateEntries,
    SingleStateEndpointODEFunctional,
)
from pyapprox_benchmarks.functions.ode.time_modulated_quadratic import (
    TimeModulatedQuadraticODE,
)

__all__ = [
    "ODETimeConfig",
    "ODEQoIFunction",
    "ODEFunctionalProtocol",
    "ODEBenchmarkResidualProtocol",
    "SelectedStateEntries",
    "AllStatesEndpointODEFunctional",
    "SingleStateEndpointODEFunctional",
    "MaxODEFunctional",
    "TimeModulatedQuadraticODE",
]
