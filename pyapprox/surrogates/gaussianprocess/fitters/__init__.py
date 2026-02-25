"""Fitters for Gaussian Process models.

This module provides fitters that separate fitting algorithms from the GP model
using a fitter-centric pattern:

    result = fitter.fit(gp, X_train, y_train)
    fitted_gp = result.surrogate()

The original GP is not modified during fitting - a deep copy is created
internally and fitted parameters are stored on the copy (immutable pattern).

.. deprecated::
    The legacy ``gp.fit()`` method still works but delegates to these fitters
    internally. Prefer using fitters directly for cleaner separation of
    concerns.
"""

from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPFitResult,
    GPOptimizedFitResult,
)
from pyapprox.surrogates.gaussianprocess.fitters.fixed_hyperparameter_fitter import (
    GPFixedHyperparameterFitter,
)
from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
    GPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.fitters.variational_fitter import (
    VariationalGPFixedHyperparameterFitter,
    VariationalGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.fitters.multioutput_fitter import (
    MultiOutputGPFixedHyperparameterFitter,
    MultiOutputGPMaximumLikelihoodFitter,
)

__all__ = [
    "GPFitResult",
    "GPOptimizedFitResult",
    "GPFixedHyperparameterFitter",
    "GPMaximumLikelihoodFitter",
    "VariationalGPFixedHyperparameterFitter",
    "VariationalGPMaximumLikelihoodFitter",
    "MultiOutputGPFixedHyperparameterFitter",
    "MultiOutputGPMaximumLikelihoodFitter",
]
