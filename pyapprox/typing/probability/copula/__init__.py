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

__all__ = [
    "CopulaProtocol",
    "CopulaWithKLProtocol",
    "CorrelationParameterizationProtocol",
    "CholeskyCorrelationParameterization",
    "GaussianCopula",
    "CopulaDistribution",
    "gaussian_copula_kl_divergence",
]
