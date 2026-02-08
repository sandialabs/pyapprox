"""Fitters for basis expansions.

This module provides fitters that separate surrogate representation from
parameter estimation using a fitter-centric pattern:

    result = fitter.fit(expansion, samples, values)
    fitted_expansion = result.surrogate()

The surrogate is not modified during fitting - instead, a new instance
is returned with the fitted parameters (immutable pattern).
"""

from pyapprox.typing.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
    SparseResult,
    OMPResult,
    CVSelectionResult,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.ridge import (
    RidgeFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.bpdn import (
    BPDNFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.omp import (
    OMPFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.bayesian import (
    BayesianConjugateFitter,
    BayesianConjugateResult,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.quantile import (
    QuantileFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.basis_pursuit import (
    BasisPursuitFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.gradient_enhanced import (
    GradientEnhancedPCEFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.entropic import (
    EntropicLoss,
    EntropicFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.conservative import (
    ConservativeLstSqFitter,
    ConservativeQuantileFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.stochastic_dominance import (
    FSDObjective,
    StochasticDominanceConstraint,
    FSDFitter,
    SSDFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.pce_cv import (
    PCEDegreeSelectionFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.omp_cv import (
    OMPCVFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.adaptive_pce import (
    AdaptivePCEFitter,
    AdaptivePCEResult,
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
