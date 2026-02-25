"""Fitters for FunctionTrain surrogates."""

from pyapprox.surrogates.functiontrain.fitters.mse_fitter import (
    MSEFitter,
)
from pyapprox.surrogates.functiontrain.fitters.results import (
    MSEFitterResult,
)

__all__ = [
    "MSEFitterResult",
    "MSEFitter",
]
