"""Fitters for dynamical systems learning."""

from pyapprox.surrogates.dynamical_systems.fitters.linear_fitter import (
    LinearInParamsFitter,
)
from pyapprox.surrogates.dynamical_systems.fitters.results import (
    DirectSolverResult,
)

__all__ = ["LinearInParamsFitter", "DirectSolverResult"]
