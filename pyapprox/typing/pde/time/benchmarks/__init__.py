"""Benchmark ODE problems for testing time integration."""

from pyapprox.typing.pde.time.benchmarks.linear_ode import (
    LinearODEResidual,
    QuadraticODEResidual,
)
from pyapprox.typing.pde.time.benchmarks.lotka_volterra import (
    LotkaVolterraResidual,
)

__all__ = [
    "LinearODEResidual",
    "QuadraticODEResidual",
    "LotkaVolterraResidual",
]
