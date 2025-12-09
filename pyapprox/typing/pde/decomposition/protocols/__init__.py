"""Protocols for Dirichlet-to-Neumann domain decomposition.

Defines protocols for:
- InterfaceBasisProtocol: polynomial basis on interface
- InterfaceProtocol: interface between subdomains
- SubdomainSolverProtocol: subdomain solver with DtN capabilities
- DomainDecompositionProtocol: full decomposition setup
"""

from pyapprox.typing.pde.decomposition.protocols.interface import (
    InterfaceBasisProtocol,
    InterfaceProtocol,
)
from pyapprox.typing.pde.decomposition.protocols.subdomain import (
    SubdomainSolverProtocol,
)
from pyapprox.typing.pde.decomposition.protocols.decomposition import (
    DomainDecompositionProtocol,
)

__all__ = [
    "InterfaceBasisProtocol",
    "InterfaceProtocol",
    "SubdomainSolverProtocol",
    "DomainDecompositionProtocol",
]
