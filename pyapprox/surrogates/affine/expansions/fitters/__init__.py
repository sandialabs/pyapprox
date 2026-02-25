"""Fitters for basis expansions.

This module provides fitters that separate surrogate representation from
parameter estimation using a fitter-centric pattern:

    result = fitter.fit(expansion, samples, values)
    fitted_expansion = result.surrogate()

The surrogate is not modified during fitting - instead, a new instance
is returned with the fitted parameters (immutable pattern).
"""

from pyapprox.surrogates.affine.expansions.fitters.adaptive_pce import (
    AdaptivePCEFitter,
    AdaptivePCEResult,
)
from pyapprox.surrogates.affine.expansions.fitters.basis_pursuit import (
    BasisPursuitFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.bayesian import (
    BayesianConjugateFitter,
    BayesianConjugateResult,
)
from pyapprox.surrogates.affine.expansions.fitters.bpdn import (
    BPDNFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.conservative import (
    ConservativeLstSqFitter,
    ConservativeQuantileFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.entropic import (
    EntropicFitter,
    EntropicLoss,
)
from pyapprox.surrogates.affine.expansions.fitters.gradient_enhanced import (
    GradientEnhancedPCEFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.omp import (
    OMPFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.omp_cv import (
    OMPCVFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.pce_cv import (
    PCEDegreeSelectionFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.quantile import (
    QuantileFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
    DirectSolverResult,
    OMPResult,
    SparseResult,
)
from pyapprox.surrogates.affine.expansions.fitters.ridge import (
    RidgeFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.stochastic_dominance import (
    FSDFitter,
    FSDObjective,
    SSDFitter,
    StochasticDominanceConstraint,
)

__all__ = [
    "DirectSolverResult",
    "SparseResult",
    "OMPResult",
    "LeastSquaresFitter",
    "RidgeFitter",
    "BPDNFitter",
    "OMPFitter",
    "BayesianConjugateFitter",
    "BayesianConjugateResult",
    "QuantileFitter",
    "BasisPursuitFitter",
    "GradientEnhancedPCEFitter",
    "EntropicLoss",
    "EntropicFitter",
    "ConservativeLstSqFitter",
    "ConservativeQuantileFitter",
    "FSDObjective",
    "StochasticDominanceConstraint",
    "FSDFitter",
    "SSDFitter",
    "CVSelectionResult",
    "PCEDegreeSelectionFitter",
    "OMPCVFitter",
    "AdaptivePCEFitter",
    "AdaptivePCEResult",
]
