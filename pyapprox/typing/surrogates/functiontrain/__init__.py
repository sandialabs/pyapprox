"""FunctionTrain tensor decomposition module.

This module provides FunctionTrain surrogates - tensor train decompositions
that represent multivariate functions as sequences of univariate basis
expansions connected via tensor contractions.
"""

from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.typing.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.typing.surrogates.functiontrain.pce_core import (
    PCEFunctionTrainCore,
)
from pyapprox.typing.surrogates.functiontrain.pce_functiontrain import (
    PCEFunctionTrain,
    create_pce_functiontrain,
    create_uniform_pce_functiontrain,
)
from pyapprox.typing.surrogates.functiontrain.additive import (
    create_additive_functiontrain,
    ConstantExpansion,
)
from pyapprox.typing.surrogates.functiontrain.als_fitter import (
    ALSFitter,
    ALSFitterResult,
)
from pyapprox.typing.surrogates.functiontrain.losses import (
    FunctionTrainMSELoss,
)
from pyapprox.typing.surrogates.functiontrain.fitters import (
    MSEFitter,
    MSEFitterResult,
)
from pyapprox.typing.surrogates.functiontrain.statistics.marginalization import (
    FunctionTrainMarginalization,
    FTDimensionReducer,
    marginal_1d,
    marginal_2d,
    all_marginals_1d,
)

__all__ = [
    "FunctionTrainCore",
    "FunctionTrain",
    "PCEFunctionTrainCore",
    "PCEFunctionTrain",
    "create_additive_functiontrain",
    "create_pce_functiontrain",
    "create_uniform_pce_functiontrain",
    "ConstantExpansion",
    "ALSFitter",
    "ALSFitterResult",
    "FunctionTrainMSELoss",
    "MSEFitter",
    "MSEFitterResult",
    "FunctionTrainMarginalization",
    "FTDimensionReducer",
    "marginal_1d",
    "marginal_2d",
    "all_marginals_1d",
]
