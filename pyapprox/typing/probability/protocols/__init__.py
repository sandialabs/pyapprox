"""
Protocols for probability module.

This package defines the core interfaces (protocols) for:
- Probability distributions (marginal, joint)
- Covariance operators (sqrt, dense, diagonal)
- Probability transforms (affine, Gaussian, Nataf)
- Likelihood functions (Gaussian, vectorized)

All protocols are @runtime_checkable for duck typing support.
"""

from .distribution import (
    DistributionProtocol,
    MarginalProtocol,
    MarginalWithJacobianProtocol,
    JointDistributionProtocol,
)

from .covariance import (
    SqrtCovarianceOperatorProtocol,
    CovarianceOperatorProtocol,
    DiagonalCovarianceOperatorProtocol,
)

from .transform import (
    TransformProtocol,
    InvertibleTransformProtocol,
    TransformWithJacobianProtocol,
)

from .likelihood import (
    LogLikelihoodProtocol,
    GaussianLogLikelihoodProtocol,
    VectorizedLogLikelihoodProtocol,
)

__all__ = [
    # Distributions
    "DistributionProtocol",
    "MarginalProtocol",
    "MarginalWithJacobianProtocol",
    "JointDistributionProtocol",
    # Covariance
    "SqrtCovarianceOperatorProtocol",
    "CovarianceOperatorProtocol",
    "DiagonalCovarianceOperatorProtocol",
    # Transforms
    "TransformProtocol",
    "InvertibleTransformProtocol",
    "TransformWithJacobianProtocol",
    # Likelihoods
    "LogLikelihoodProtocol",
    "GaussianLogLikelihoodProtocol",
    "VectorizedLogLikelihoodProtocol",
]
