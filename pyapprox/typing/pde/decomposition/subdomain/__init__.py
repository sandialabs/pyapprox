"""Subdomain solver wrapper for domain decomposition.

Provides:
- SubdomainWrapper: Wraps PDE physics for DtN decomposition
- FluxComputer: Helper for computing interface fluxes
"""

from pyapprox.typing.pde.decomposition.subdomain.wrapper import (
    SubdomainWrapper,
)
from pyapprox.typing.pde.decomposition.subdomain.flux import (
    FluxComputer,
    compute_flux_mismatch,
    flux_mismatch_norm,
)

__all__ = [
    "SubdomainWrapper",
    "FluxComputer",
    "compute_flux_mismatch",
    "flux_mismatch_norm",
]
