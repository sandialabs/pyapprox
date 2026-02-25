"""ODE benchmark wrapper classes."""

from pyapprox.benchmarks.functions.ode.ode_benchmark import (
    AllStatesEndpointODEFunctional,
    MaxODEFunctional,
    ODEBenchmark,
    ODEFunctionalProtocol,
    ODEQoIFunction,
    ODETimeConfig,
    SingleStateEndpointODEFunctional,
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
