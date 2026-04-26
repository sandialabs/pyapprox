"""Fitters for SUPN surrogates."""

from pyapprox.surrogates.supn.fitters.mse_fitter import SUPNMSEFitter
from pyapprox.surrogates.supn.fitters.results import SUPNFitterResult

__all__ = [
    "SUPNMSEFitter",
    "SUPNFitterResult",
]
