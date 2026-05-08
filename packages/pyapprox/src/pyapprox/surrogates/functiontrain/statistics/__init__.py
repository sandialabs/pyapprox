"""FunctionTrain statistics module for PCE-based analytical computations.

Warning
-------
All classes assume orthonormal polynomial bases (e.g., Legendre, Hermite).
Results are mathematically incorrect for non-orthonormal bases.

Currently only supports nqoi=1.
"""

# Re-export from parent module for convenience
from pyapprox.surrogates.functiontrain.pce_core import (
    PCEFunctionTrainCore,
)
from pyapprox.surrogates.functiontrain.pce_functiontrain import (
    PCEFunctionTrain,
)
from pyapprox.surrogates.functiontrain.statistics.marginalization import (
    FunctionTrainMarginalization,
    all_marginals_1d,
    marginal_1d,
    marginal_2d,
)
from pyapprox.surrogates.functiontrain.statistics.moments import (
    FunctionTrainMoments,
)
from pyapprox.surrogates.functiontrain.statistics.sensitivity import (
    FunctionTrainSensitivity,
)

__all__ = [
    "PCEFunctionTrainCore",
    "PCEFunctionTrain",
    "FunctionTrainMoments",
    "FunctionTrainSensitivity",
    "FunctionTrainMarginalization",
    "marginal_1d",
    "marginal_2d",
    "all_marginals_1d",
]
