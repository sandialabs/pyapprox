"""ODE benchmark wrapper classes."""

from pyapprox.typing.benchmarks.functions.ode.ode_benchmark import (
    ODEBenchmark,
    ODETimeConfig,
    ODEQoIFunction,
    ODEFunctionalProtocol,
    AllStatesEndpointODEFunctional,
    SingleStateEndpointODEFunctional,
    MaxODEFunctional,
)

__all__ = [
    "ODEBenchmark",
    "ODETimeConfig",
    "ODEQoIFunction",
    "ODEFunctionalProtocol",
    "AllStatesEndpointODEFunctional",
    "SingleStateEndpointODEFunctional",
    "MaxODEFunctional",
]
