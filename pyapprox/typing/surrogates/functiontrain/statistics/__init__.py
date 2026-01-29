"""FunctionTrain statistics module for PCE-based analytical computations.

Warning
-------
All classes assume orthonormal polynomial bases (e.g., Legendre, Hermite).
Results are mathematically incorrect for non-orthonormal bases.

Currently only supports nqoi=1.
"""

# Re-export from parent module for convenience
from pyapprox.typing.surrogates.functiontrain.pce_core import (
    PCEFunctionTrainCore,
)
from pyapprox.typing.surrogates.functiontrain.pce_functiontrain import (
    PCEFunctionTrain,
)
from pyapprox.typing.surrogates.functiontrain.statistics.moments import (
    FunctionTrainMoments,
)
from pyapprox.typing.surrogates.functiontrain.statistics.sensitivity import (
    FunctionTrainSensitivity,
)

__all__ = [
    "PCEFunctionTrainCore",
    "PCEFunctionTrain",
    "FunctionTrainMoments",
    "FunctionTrainSensitivity",
]
