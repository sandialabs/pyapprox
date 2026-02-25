"""Quadrature rules for numerical integration.

This module provides quadrature rules for numerical integration including:
- Tensor product quadrature rules
- Stroud cubature rules for hypercubes
- Helper functions for creating quadrature rules from marginal distributions

Key classes:
- TensorProductQuadratureRule: Tensor product of 1D rules
- ParameterizedTensorProductQuadratureRule: Level-parameterized tensor product
- StroudCdD2, StroudCdD3, StroudCdD5: Stroud cubature rules

Key functions:
- gauss_quadrature_rule: Create a 1D Gauss quadrature rule from a marginal

Protocols:
- UnivariateQuadratureRuleProtocol: 1D quadrature
- MultivariateQuadratureRuleProtocol: Fixed multivariate quadrature
- ParameterizedQuadratureRuleProtocol: Level-parameterized quadrature
- AdaptiveQuadratureRuleProtocol: Adaptive quadrature
"""

from typing import Tuple

from pyapprox.probability.protocols import MarginalProtocol
from pyapprox.util.backends.protocols import Array, Backend

from .cubature import (
    StroudCdD2,
    StroudCdD3,
    StroudCdD5,
)
from .probability_measure_factory import (
    ProbabilityMeasureQuadratureFactory,
)
from .protocols import (
    AdaptiveQuadratureRuleProtocol,
    MultivariateQuadratureRuleProtocol,
    ParameterizedQuadratureRuleProtocol,
    UnivariateQuadratureRuleProtocol,
)
from .tensor_product import (
    ParameterizedTensorProductQuadratureRule,
    TensorProductQuadratureRule,
)


def gauss_quadrature_rule(
    marginal: MarginalProtocol[Array],
    npoints: int,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """Create a Gauss quadrature rule for a marginal distribution.

    Returns quadrature points in the marginal's physical domain and
    probability-measure weights (summing to 1). This is the recommended
    way to create univariate quadrature rules for computing expectations
    E[f(X)] w.r.t. probability distributions.

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    npoints : int
        Number of quadrature points.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    points : Array
        Quadrature points in physical domain. Shape: (1, npoints)
    weights : Array
        Probability-measure weights summing to 1. Shape: (npoints, 1)
    """
    from pyapprox.surrogates.affine.univariate.factory import (
        create_basis_1d,
    )

    basis = create_basis_1d(marginal, bkd)
    basis.set_nterms(npoints)
    return basis.gauss_quadrature_rule(npoints)


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
    # Factories
    "ProbabilityMeasureQuadratureFactory",
    # Helpers
    "gauss_quadrature_rule",
]
