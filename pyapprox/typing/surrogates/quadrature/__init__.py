"""Quadrature rules for numerical integration.

This module provides quadrature rules for numerical integration including:
- Tensor product quadrature rules
- Stroud cubature rules for hypercubes
- Factory functions for creating quadrature rules

Key classes:
- TensorProductQuadratureRule: Tensor product of 1D rules
- ParameterizedTensorProductQuadratureRule: Level-parameterized tensor product
- StroudCdD2, StroudCdD3, StroudCdD5: Stroud cubature rules

Protocols:
- UnivariateQuadratureRuleProtocol: 1D quadrature
- MultivariateQuadratureRuleProtocol: Fixed multivariate quadrature
- ParameterizedQuadratureRuleProtocol: Level-parameterized quadrature
- AdaptiveQuadratureRuleProtocol: Adaptive quadrature
"""

from .protocols import (
    UnivariateQuadratureRuleProtocol,
    MultivariateQuadratureRuleProtocol,
    ParameterizedQuadratureRuleProtocol,
    AdaptiveQuadratureRuleProtocol,
)

from .tensor_product import (
    TensorProductQuadratureRule,
    ParameterizedTensorProductQuadratureRule,
)

from .cubature import (
    StroudCdD2,
    StroudCdD3,
    StroudCdD5,
)

__all__ = [
    # Protocols
    "UnivariateQuadratureRuleProtocol",
    "MultivariateQuadratureRuleProtocol",
    "ParameterizedQuadratureRuleProtocol",
    "AdaptiveQuadratureRuleProtocol",
    # Tensor product
    "TensorProductQuadratureRule",
    "ParameterizedTensorProductQuadratureRule",
    # Cubature
    "StroudCdD2",
    "StroudCdD3",
    "StroudCdD5",
]
