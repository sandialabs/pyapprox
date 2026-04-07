"""Protocols for Dirichlet-to-Neumann domain decomposition.

Defines protocols for:
- InterfaceBasisProtocol: polynomial basis on interface
- InterfaceProtocol: interface between subdomains
- SubdomainSolverProtocol: subdomain solver with DtN capabilities
- DomainDecompositionProtocol: full decomposition setup
"""

from pyapprox.pde.decomposition.protocols.decomposition import (
    DomainDecompositionProtocol,
)
from pyapprox.pde.decomposition.protocols.interface import (
    InterfaceBasisProtocol,
    InterfaceProtocol,
)
from pyapprox.pde.decomposition.protocols.subdomain import (
    SubdomainSolverProtocol,
)

__all__ = [
    "InterfaceBasisProtocol",
    "InterfaceProtocol",
    "SubdomainSolverProtocol",
    "DomainDecompositionProtocol",
]
