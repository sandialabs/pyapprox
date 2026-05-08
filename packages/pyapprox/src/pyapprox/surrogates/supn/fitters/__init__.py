"""Fitters for SUPN surrogates."""

from pyapprox.surrogates.supn.fitters.mse_fitter import (
    SUPNMSEFitter,
    supn_paper_rol_parameter_list,
)
from pyapprox.surrogates.supn.fitters.results import SUPNFitterResult

__all__ = [
    "SUPNMSEFitter",
    "SUPNFitterResult",
    "supn_paper_rol_parameter_list",
]
