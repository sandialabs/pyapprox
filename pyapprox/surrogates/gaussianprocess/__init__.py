"""
Gaussian Process regression implementations.

This module provides type-safe, backend-agnostic implementations of
Gaussian Process regression with comprehensive uncertainty quantification.
"""

from pyapprox.surrogates.gaussianprocess.protocols import (
    GaussianProcessProtocol,
    FittableGPProtocol,
    PredictiveGPProtocol,
    TrainableGPProtocol
)
from pyapprox.surrogates.gaussianprocess.data import GPTrainingData
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ZeroMean,
    ConstantMean
)
from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess
)
from pyapprox.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP
)
from pyapprox.surrogates.gaussianprocess.loss import (
    NegativeLogMarginalLikelihoodLoss
)
from pyapprox.surrogates.gaussianprocess.inducing_samples import (
    InducingSamples
)
from pyapprox.surrogates.gaussianprocess.variational import (
    VariationalGaussianProcess
)
from pyapprox.surrogates.gaussianprocess.fitters import (
    GPFitResult,
    GPOptimizedFitResult,
    GPFixedHyperparameterFitter,
    GPMaximumLikelihoodFitter,
    VariationalGPFixedHyperparameterFitter,
    VariationalGPMaximumLikelihoodFitter,
    MultiOutputGPFixedHyperparameterFitter,
    MultiOutputGPMaximumLikelihoodFitter,
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
    # Fitter results
    "GPFitResult",
    "GPOptimizedFitResult",
    # Fitters
    "GPFixedHyperparameterFitter",
    "GPMaximumLikelihoodFitter",
    "VariationalGPFixedHyperparameterFitter",
    "VariationalGPMaximumLikelihoodFitter",
    "MultiOutputGPFixedHyperparameterFitter",
    "MultiOutputGPMaximumLikelihoodFitter",
]
