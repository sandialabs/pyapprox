"""Dirichlet-to-Neumann domain decomposition for coupled PDEs.

This module provides a framework for solving PDEs on decomposed domains
using a Dirichlet-to-Neumann (DtN) approach. The method:

1. Parameterizes interface functions with polynomial coefficients
2. Solves subdomains independently with Dirichlet BCs on interfaces
3. Uses Newton iteration to find interface values that satisfy flux continuity

The approach is equivalent to computing the Schur complement implicitly,
eliminating interior DOFs and solving for interface unknowns.
"""

from pyapprox.pde.decomposition.protocols import (
    InterfaceBasisProtocol,
    InterfaceProtocol,
    SubdomainSolverProtocol,
    DomainDecompositionProtocol,
)
from pyapprox.pde.decomposition.interface import (
    LegendreInterfaceBasis1D,
    LegendreInterfaceBasis2D,
    Interface1D,
    Interface,
    Interface2D,
    InterpolationOperator,
    RestrictionOperator,
)
from pyapprox.pde.decomposition.subdomain import (
    SubdomainWrapper,
    FluxComputer,
)
from pyapprox.pde.decomposition.solver import (
    DtNResidual,
    DtNJacobian,
    DtNSolver,
    DtNSolverResult,
    create_dtn_solver,
)

__all__ = [
    # Protocols
    "InterfaceBasisProtocol",
    "InterfaceProtocol",
    "SubdomainSolverProtocol",
    "DomainDecompositionProtocol",
    # Interface
    "LegendreInterfaceBasis1D",
    "LegendreInterfaceBasis2D",
    "Interface1D",
    "Interface",
    "Interface2D",
    "InterpolationOperator",
    "RestrictionOperator",
    # Subdomain
    "SubdomainWrapper",
    "FluxComputer",
    # Solver
    "DtNResidual",
    "DtNJacobian",
    "DtNSolver",
    "DtNSolverResult",
    "create_dtn_solver",
]
