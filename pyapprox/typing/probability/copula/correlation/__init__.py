"""Correlation matrix parameterization strategies for copulas."""

from pyapprox.typing.probability.copula.correlation.protocols import (
    CorrelationParameterizationProtocol,
)
from pyapprox.typing.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)

__all__ = [
    "CorrelationParameterizationProtocol",
    "CholeskyCorrelationParameterization",
]
