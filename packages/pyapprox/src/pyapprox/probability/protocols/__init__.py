"""
Protocols for probability module.

This package defines the core interfaces (protocols) for:
- Probability distributions (marginal, joint)
- Covariance operators (sqrt, dense, diagonal)
- Probability transforms (affine, Gaussian, Nataf)
- Likelihood functions (Gaussian, vectorized)

All protocols are @runtime_checkable for duck typing support.
"""

from .covariance import (
    CovarianceOperatorProtocol,
    DiagonalCovarianceOperatorProtocol,
    SqrtCovarianceOperatorProtocol,
)
from .distribution import (
    DistributionHasLogpdfDerivativesProtocol,
    DistributionProtocol,
    JointDistributionProtocol,
    MarginalHasLogpdfJacobianProtocol,
    MarginalHasLogpdfParamJacobianProtocol,
    MarginalHasPdfJacobianProtocol,
    MarginalProtocol,
    MarginalWithHypListProtocol,
    MarginalWithJacobianProtocol,
    MarginalWithParamJacobianProtocol,
    UniformQuadratureRule01Protocol,
)
from .likelihood import (
    GaussianLogLikelihoodProtocol,
    LogLikelihoodHasDesignWeightsProtocol,
    LogLikelihoodHasRVSProtocol,
    LogLikelihoodProtocol,
    VectorizedLogLikelihoodProtocol,
)
from .transform import (
    InvertibleTransformProtocol,
    TransformProtocol,
    TransformWithJacobianProtocol,
)

__all__ = [
    # Distributions
    "DistributionProtocol",
    "DistributionHasLogpdfDerivativesProtocol",
    "MarginalHasLogpdfJacobianProtocol",
    "MarginalHasLogpdfParamJacobianProtocol",
    "MarginalHasPdfJacobianProtocol",
    "MarginalProtocol",
    "MarginalWithHypListProtocol",
    "MarginalWithJacobianProtocol",
    "MarginalWithParamJacobianProtocol",
    "JointDistributionProtocol",
    "UniformQuadratureRule01Protocol",
    # Covariance
    "SqrtCovarianceOperatorProtocol",
    "CovarianceOperatorProtocol",
    "DiagonalCovarianceOperatorProtocol",
    # Transforms
    "TransformProtocol",
    "InvertibleTransformProtocol",
    "TransformWithJacobianProtocol",
    # Likelihoods
    "LogLikelihoodHasDesignWeightsProtocol",
    "LogLikelihoodHasRVSProtocol",
    "LogLikelihoodProtocol",
    "GaussianLogLikelihoodProtocol",
    "VectorizedLogLikelihoodProtocol",
]
