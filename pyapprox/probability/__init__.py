"""
Probability module for pyapprox.

This module provides a comprehensive set of probability distributions,
covariance operators, transforms, and likelihood functions for Bayesian
inference and uncertainty quantification.

Subpackages
-----------
protocols
    Core interfaces (protocols) for distributions, covariance, transforms.
covariance
    Covariance operators with square-root (Cholesky) factorization.
univariate
    Univariate probability distributions (Gaussian, SciPy wrappers).
gaussian
    Multivariate Gaussian distributions with various covariance forms.
joint
    Joint distributions composed of marginal distributions.
transforms
    Probability transforms (affine, Gaussian, Nataf, Rosenblatt).
likelihood
    Likelihood functions for Bayesian inference.
risk
    Analytical risk measures for specific distributions.

Examples
--------
>>> import numpy as np
>>> from pyapprox.util.backends.numpy import NumpyBkd
>>> from pyapprox.probability import (
...     GaussianMarginal,
...     IndependentJoint,
...     IndependentGaussianTransform,
... )

Create a joint distribution with independent marginals:

>>> bkd = NumpyBkd()
>>> marginal1 = GaussianMarginal(0.0, 1.0, bkd)
>>> marginal2 = GaussianMarginal(1.0, 2.0, bkd)
>>> joint = IndependentJoint([marginal1, marginal2], bkd)

Transform to standard normal:

>>> transform = IndependentGaussianTransform([marginal1, marginal2], bkd)
>>> samples = np.array([[0.0, 0.5], [1.0, 2.0]])
>>> canonical = transform.map_to_canonical(samples)
"""

# Protocols
# Covariance operators
from .covariance import (
    DenseCholeskyCovarianceOperator,
    DiagonalCovarianceOperator,
    OperatorBasedCovarianceOperator,
)

# Density estimation
from .density import (
    DensityBasisProtocol,
    DensityFitterProtocol,
    ISEOptimizingFitter,
    KDEFitter,
    KernelDensityBasis,
    LinearDensityFitter,
    PiecewiseDensityBasis,
    ProjectionDensityFitter,
    PushforwardDensity,
)

# Multivariate Gaussian distributions
from .gaussian import (
    DenseCholeskyMultivariateGaussian,
    DiagonalMultivariateGaussian,
    GaussianCanonicalForm,
    GaussianLogPDFCore,
    OperatorBasedMultivariateGaussian,
    compute_normalization,
    plot_gaussian_2d_contour,
)

# Joint distributions
from .joint import (
    IndependentJoint,
)

# Likelihood functions
from .likelihood import (
    DiagonalGaussianLogLikelihood,
    GaussianLogLikelihood,
    ModelBasedLogLikelihood,
)
from .protocols import (
    CovarianceOperatorProtocol,
    DiagonalCovarianceOperatorProtocol,
    DistributionProtocol,
    GaussianLogLikelihoodProtocol,
    InvertibleTransformProtocol,
    JointDistributionProtocol,
    LogLikelihoodProtocol,
    MarginalProtocol,
    MarginalWithJacobianProtocol,
    SqrtCovarianceOperatorProtocol,
    TransformProtocol,
    TransformWithJacobianProtocol,
    VectorizedLogLikelihoodProtocol,
)

# Risk measures
from .risk import (
    GaussianAnalyticalRiskMeasures,
)

# Transforms
from .transforms import (
    GaussianTransform,
    IndependentGaussianTransform,
    NatafTransform,
    RosenblattTransform,
)

# Univariate distributions
from .univariate import (
    BetaMarginal,
    CustomDiscreteMarginal,
    DiscreteChebyshevMarginal,
    GammaMarginal,
    GaussianMarginal,
    ScipyContinuousMarginal,
    ScipyDiscreteMarginal,
    UniformMarginal,
)

__all__ = [
    # Protocols
    "DistributionProtocol",
    "MarginalProtocol",
    "MarginalWithJacobianProtocol",
    "JointDistributionProtocol",
    "SqrtCovarianceOperatorProtocol",
    "CovarianceOperatorProtocol",
    "DiagonalCovarianceOperatorProtocol",
    "TransformProtocol",
    "InvertibleTransformProtocol",
    "TransformWithJacobianProtocol",
    "LogLikelihoodProtocol",
    "GaussianLogLikelihoodProtocol",
    "VectorizedLogLikelihoodProtocol",
    # Covariance operators
    "DenseCholeskyCovarianceOperator",
    "DiagonalCovarianceOperator",
    "OperatorBasedCovarianceOperator",
    # Univariate distributions
    "BetaMarginal",
    "CustomDiscreteMarginal",
    "DiscreteChebyshevMarginal",
    "GammaMarginal",
    "GaussianMarginal",
    "UniformMarginal",
    "ScipyContinuousMarginal",
    "ScipyDiscreteMarginal",
    # Multivariate Gaussian
    "GaussianLogPDFCore",
    "DenseCholeskyMultivariateGaussian",
    "DiagonalMultivariateGaussian",
    "OperatorBasedMultivariateGaussian",
    "GaussianCanonicalForm",
    "compute_normalization",
    "plot_gaussian_2d_contour",
    # Joint distributions
    "IndependentJoint",
    # Transforms
    "GaussianTransform",
    "IndependentGaussianTransform",
    "NatafTransform",
    "RosenblattTransform",
    # Likelihood
    "GaussianLogLikelihood",
    "DiagonalGaussianLogLikelihood",
    "ModelBasedLogLikelihood",
    # Risk measures
    "GaussianAnalyticalRiskMeasures",
    # Density estimation
    "DensityBasisProtocol",
    "DensityFitterProtocol",
    "PiecewiseDensityBasis",
    "KernelDensityBasis",
    "LinearDensityFitter",
    "KDEFitter",
    "ProjectionDensityFitter",
    "ISEOptimizingFitter",
    "PushforwardDensity",
]
