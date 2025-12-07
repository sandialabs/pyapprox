"""Boundary conditions module for spectral collocation methods."""

from pyapprox.typing.pde.collocation.boundary.dirichlet import (
    DirichletBC,
    constant_dirichlet_bc,
    zero_dirichlet_bc,
)
from pyapprox.typing.pde.collocation.boundary.neumann import (
    NeumannBC,
    zero_neumann_bc,
)
from pyapprox.typing.pde.collocation.boundary.robin import (
    RobinBC,
    homogeneous_robin_bc,
)
from pyapprox.typing.pde.collocation.boundary.periodic import (
    PeriodicBC,
)

__all__ = [
    # Dirichlet
    "DirichletBC",
    "constant_dirichlet_bc",
    "zero_dirichlet_bc",
    # Neumann
    "NeumannBC",
    "zero_neumann_bc",
    # Robin
    "RobinBC",
    "homogeneous_robin_bc",
    # Periodic
    "PeriodicBC",
]
