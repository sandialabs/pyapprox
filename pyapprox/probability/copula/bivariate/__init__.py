"""
Bivariate copula families.

Provides bivariate copula implementations with h-functions for vine
copula construction: Gaussian, Clayton, Frank, and Gumbel.
"""

from pyapprox.probability.copula.bivariate.protocols import (
    BivariateCopulaProtocol,
)
from pyapprox.probability.copula.bivariate.gaussian import (
    BivariateGaussianCopula,
)
from pyapprox.probability.copula.bivariate.clayton import (
    ClaytonCopula,
)
from pyapprox.probability.copula.bivariate.frank import (
    FrankCopula,
)
from pyapprox.probability.copula.bivariate.gumbel import (
    GumbelCopula,
)
from pyapprox.probability.copula.bivariate.registry import (
    register_bivariate_copula,
    create_bivariate_copula,
    list_bivariate_copulas,
)

__all__ = [
    "BivariateCopulaProtocol",
    "BivariateGaussianCopula",
    "ClaytonCopula",
    "FrankCopula",
    "GumbelCopula",
    "register_bivariate_copula",
    "create_bivariate_copula",
    "list_bivariate_copulas",
]
