"""
Gaussian Process regression implementations.

This module provides type-safe, backend-agnostic implementations of
Gaussian Process regression with comprehensive uncertainty quantification.
"""

from pyapprox.typing.surrogates.gaussianprocess.protocols import (
    GaussianProcessProtocol,
    FittableGPProtocol,
    PredictiveGPProtocol,
    TrainableGPProtocol
)
from pyapprox.typing.surrogates.gaussianprocess.data import GPTrainingData
from pyapprox.typing.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ZeroMean,
    ConstantMean
)
from pyapprox.typing.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess
)
from pyapprox.typing.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP
)
from pyapprox.typing.surrogates.gaussianprocess.loss import (
    NegativeLogMarginalLikelihoodLoss
)
from pyapprox.typing.surrogates.gaussianprocess.inducing_samples import (
    InducingSamples
)
from pyapprox.typing.surrogates.gaussianprocess.variational import (
    VariationalGaussianProcess
)

__all__ = [
    # Protocols
    "GaussianProcessProtocol",
    "FittableGPProtocol",
    "PredictiveGPProtocol",
    "TrainableGPProtocol",
    # Data management
    "GPTrainingData",
    # Mean functions
    "MeanFunction",
    "ZeroMean",
    "ConstantMean",
    # GP implementations
    "ExactGaussianProcess",
    "MultiOutputGP",
    "VariationalGaussianProcess",
    # Inducing samples
    "InducingSamples",
    # Loss functions
    "NegativeLogMarginalLikelihoodLoss",
]
