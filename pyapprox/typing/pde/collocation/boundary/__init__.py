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
from pyapprox.typing.pde.collocation.boundary.normal_operators import (
    GradientNormalOperator,
    FluxNormalOperator,
    TractionNormalOperator,
)
from pyapprox.typing.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
    hyperelastic_traction_neumann_bc,
    hyperelastic_traction_robin_bc,
)
from pyapprox.typing.pde.collocation.boundary.robin import (
    RobinBC,
    homogeneous_robin_bc,
    gradient_robin_bc,
    flux_robin_bc,
    gradient_neumann_bc,
    flux_neumann_bc,
    traction_robin_bc,
    traction_neumann_bc,
)
from pyapprox.typing.pde.collocation.boundary.periodic import (
    PeriodicBC,
)

__all__ = [
    # Dirichlet
    "DirichletBC",
    "constant_dirichlet_bc",
    "zero_dirichlet_bc",
    # Neumann (legacy)
    "NeumannBC",
    "zero_neumann_bc",
    # Robin
    "RobinBC",
    "homogeneous_robin_bc",
    # Normal operators
    "GradientNormalOperator",
    "FluxNormalOperator",
    "TractionNormalOperator",
    # Factory functions (new API)
    "gradient_robin_bc",
    "flux_robin_bc",
    "gradient_neumann_bc",
    "flux_neumann_bc",
    "traction_robin_bc",
    "traction_neumann_bc",
    # Hyperelastic traction
    "HyperelasticTractionNormalOperator",
    "hyperelastic_traction_neumann_bc",
    "hyperelastic_traction_robin_bc",
    # Periodic
    "PeriodicBC",
]
