"""Boundary conditions module for spectral collocation methods."""

from pyapprox.pde.collocation.boundary.dirichlet import (
    DirichletBC,
    constant_dirichlet_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
    hyperelastic_traction_neumann_bc,
    hyperelastic_traction_robin_bc,
)
from pyapprox.pde.collocation.boundary.neumann import (
    NeumannBC,
    zero_neumann_bc,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
    TractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic import (
    PeriodicBC,
)
from pyapprox.pde.collocation.boundary.robin import (
    RobinBC,
    flux_neumann_bc,
    flux_robin_bc,
    gradient_neumann_bc,
    gradient_robin_bc,
    homogeneous_robin_bc,
    traction_neumann_bc,
    traction_robin_bc,
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
