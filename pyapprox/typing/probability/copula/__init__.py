"""
Copula module for modeling dependence structures.

Provides copula protocols, implementations, and utilities for
constructing joint distributions from marginals and copulas.
"""

from pyapprox.typing.probability.copula.protocols import (
    CopulaProtocol,
    CopulaWithKLProtocol,
)
from pyapprox.typing.probability.copula.correlation.protocols import (
    CorrelationParameterizationProtocol,
)
from pyapprox.typing.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.typing.probability.copula.gaussian import GaussianCopula
from pyapprox.typing.probability.copula.distribution import CopulaDistribution
from pyapprox.typing.probability.copula.kl_divergence import (
    gaussian_copula_kl_divergence,
)
from pyapprox.typing.probability.copula.bivariate import (
    BivariateCopulaProtocol,
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    create_bivariate_copula,
    register_bivariate_copula,
    list_bivariate_copulas,
)

__all__ = [
    "CopulaProtocol",
    "CopulaWithKLProtocol",
    "CorrelationParameterizationProtocol",
    "CholeskyCorrelationParameterization",
    "GaussianCopula",
    "CopulaDistribution",
    "gaussian_copula_kl_divergence",
    "BivariateCopulaProtocol",
    "BivariateGaussianCopula",
    "ClaytonCopula",
    "FrankCopula",
    "GumbelCopula",
    "create_bivariate_copula",
    "register_bivariate_copula",
    "list_bivariate_copulas",
]
