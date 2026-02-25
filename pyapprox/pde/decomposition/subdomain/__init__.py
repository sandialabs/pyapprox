"""Subdomain solver wrapper for domain decomposition.

Provides:
- SubdomainWrapper: Wraps PDE physics for DtN decomposition
- FluxComputer: Helper for computing interface fluxes
"""

from pyapprox.pde.decomposition.subdomain.flux import (
    FluxComputer,
    compute_flux_mismatch,
    flux_mismatch_norm,
)
from pyapprox.pde.decomposition.subdomain.wrapper import (
    SubdomainWrapper,
)

__all__ = [
    "SubdomainWrapper",
    "FluxComputer",
    "compute_flux_mismatch",
    "flux_mismatch_norm",
]
