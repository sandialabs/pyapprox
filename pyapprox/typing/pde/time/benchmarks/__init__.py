"""Benchmark ODE problems for testing time integration."""

from pyapprox.typing.pde.time.benchmarks.linear_ode import (
    LinearODEResidual,
    QuadraticODEResidual,
)
from pyapprox.typing.pde.time.benchmarks.lotka_volterra import (
    LotkaVolterraResidual,
)
from pyapprox.typing.pde.time.benchmarks.coupled_springs import (
    CoupledSpringsResidual,
)
from pyapprox.typing.pde.time.benchmarks.hastings_ecology import (
    HastingsEcologyResidual,
)
from pyapprox.typing.pde.time.benchmarks.chemical_reaction import (
    ChemicalReactionResidual,
)

__all__ = [
    "LinearODEResidual",
    "QuadraticODEResidual",
    "LotkaVolterraResidual",
    "CoupledSpringsResidual",
    "HastingsEcologyResidual",
    "ChemicalReactionResidual",
]
