"""
Protocols for inverse problems module.

This package defines the core interfaces (protocols) for:
- Conjugate posterior distributions
- Forward/observation operators
- Laplace approximation
- Log unnormalized posterior
- Gaussian pushforward

All protocols are @runtime_checkable for duck typing support.
"""

from .conjugate import (
    ConjugatePosteriorProtocol,
    GaussianConjugatePosteriorProtocol,
)
from .forward_model import (
    ObservationOperatorProtocol,
    ObservationOperatorWithHessianProtocol,
    ObservationOperatorWithJacobianProtocol,
)
from .laplace import (
    HessianMatVecOperatorProtocol,
    LaplacePosteriorProtocol,
)
from .posterior import (
    LogUnNormalizedPosteriorProtocol,
)
from .pushforward import (
    GaussianPushforwardProtocol,
)

__all__ = [
    # Conjugate
    "ConjugatePosteriorProtocol",
    "GaussianConjugatePosteriorProtocol",
    # Forward model
    "ObservationOperatorProtocol",
    "ObservationOperatorWithJacobianProtocol",
    "ObservationOperatorWithHessianProtocol",
    # Laplace
    "LaplacePosteriorProtocol",
    "HessianMatVecOperatorProtocol",
    # Posterior
    "LogUnNormalizedPosteriorProtocol",
    # Pushforward
    "GaussianPushforwardProtocol",
]
