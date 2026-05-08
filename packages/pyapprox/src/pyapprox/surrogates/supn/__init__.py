"""Shallow Universal Polynomial Networks (SUPN).

Implements the SUPN surrogate from Morrow et al. (2025, arXiv:2511.21414).
"""

from pyapprox.surrogates.supn.chebyshev import StandardChebyshev1D
from pyapprox.surrogates.supn.losses import SUPNMSELoss
from pyapprox.surrogates.supn.supn import SUPN, create_supn

__all__ = [
    "SUPN",
    "SUPNMSELoss",
    "StandardChebyshev1D",
    "create_supn",
]
