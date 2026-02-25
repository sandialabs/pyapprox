"""Univariate basis functions for affine surrogates.

This module provides orthonormal polynomial bases for various
probability distributions:

Continuous Distributions
------------------------
- Jacobi family: Legendre (uniform), Chebyshev (arcsine)
- Hermite: Gaussian distribution
- Laguerre: Gamma distribution

Discrete Distributions
----------------------
- Krawtchouk: Binomial distribution
- Hahn: Hypergeometric distribution
- Charlier: Poisson distribution
- DiscreteChebyshev: Uniform discrete distribution

Arbitrary Measures
------------------
- DiscreteNumericOrthonormalPolynomial1D: From weighted samples
"""

# Global orthonormal polynomials (re-exported from globalpoly submodule)
from pyapprox.surrogates.affine.univariate.bspline import (
    BSpline1D,
    HierarchicalBSpline1D,
)

# Factory for creating physical-domain bases from marginals
from pyapprox.surrogates.affine.univariate.factory import (
    create_bases_1d,
    create_basis_1d,
)
from pyapprox.surrogates.affine.univariate.globalpoly import (
    BoundedContinuousNumericOrthonormalPolynomial1D,
    CharlierPolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    ContinuousNumericOrthonormalPolynomial1D,
    DiscreteChebyshevPolynomial1D,
    DiscreteNumericOrthonormalPolynomial1D,
    GaussLobattoQuadratureRule,
    GaussQuadratureRule,
    HahnPolynomial1D,
    HermitePolynomial1D,
    JacobiPolynomial1D,
    KrawtchoukPolynomial1D,
    LaguerrePolynomial1D,
    LegendrePolynomial1D,
    OrthonormalPolynomial1D,
    UnboundedContinuousNumericOrthonormalPolynomial1D,
    WeightedSamplePolynomial1D,
    charlier_recurrence,
    discrete_chebyshev_recurrence,
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
    hahn_recurrence,
    hermite_recurrence,
    jacobi_recurrence,
    krawtchouk_recurrence,
    laguerre_recurrence,
    lanczos_recursion,
)
from pyapprox.surrogates.affine.univariate.lagrange import (
    LagrangeBasis1D,
    univariate_lagrange_first_derivative,
    univariate_lagrange_polynomial,
    univariate_lagrange_second_derivative,
)
from pyapprox.surrogates.affine.univariate.monomial import (
    MonomialBasis1D,
)
from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseConstantLeft,
    PiecewiseConstantMidpoint,
    PiecewiseConstantRight,
    PiecewiseCubic,
    PiecewiseLinear,
    PiecewisePolynomialProtocol,
    PiecewiseQuadratic,
)

# Physical-domain basis wrappers
from pyapprox.surrogates.affine.univariate.transformed import (
    NativeBasis1D,
    TransformedBasis1D,
)

# Transform utilities
from pyapprox.surrogates.affine.univariate.transforms import (
    BoundedAffineTransform1D,
    IdentityTransform1D,
    UnboundedAffineTransform1D,
    Univariate1DTransformProtocol,
    get_transform_from_marginal,
)

__all__ = [
    # Base class and functions
    "OrthonormalPolynomial1D",
    "evaluate_orthonormal_polynomial_1d",
    "evaluate_orthonormal_polynomial_derivatives_1d",
    # Jacobi family (continuous)
    "JacobiPolynomial1D",
    "LegendrePolynomial1D",
    "Chebyshev1stKindPolynomial1D",
    "Chebyshev2ndKindPolynomial1D",
    "jacobi_recurrence",
    # Hermite family (continuous)
    "HermitePolynomial1D",
    "hermite_recurrence",
    # Laguerre family (continuous)
    "LaguerrePolynomial1D",
    "laguerre_recurrence",
    # Discrete polynomials
    "KrawtchoukPolynomial1D",
    "krawtchouk_recurrence",
    "HahnPolynomial1D",
    "hahn_recurrence",
    "CharlierPolynomial1D",
    "charlier_recurrence",
    "DiscreteChebyshevPolynomial1D",
    "discrete_chebyshev_recurrence",
    # Numeric/arbitrary polynomials
    "DiscreteNumericOrthonormalPolynomial1D",
    "WeightedSamplePolynomial1D",
    "lanczos_recursion",
    # Continuous numeric polynomials
    "BoundedContinuousNumericOrthonormalPolynomial1D",
    "UnboundedContinuousNumericOrthonormalPolynomial1D",
    "ContinuousNumericOrthonormalPolynomial1D",
    # Quadrature
    "GaussQuadratureRule",
    "GaussLobattoQuadratureRule",
    # Physical-domain basis wrappers
    "TransformedBasis1D",
    "NativeBasis1D",
    # Factory functions
    "create_basis_1d",
    "create_bases_1d",
    "get_transform_from_marginal",
    # B-splines
    "BSpline1D",
    "HierarchicalBSpline1D",
    # Lagrange basis
    "LagrangeBasis1D",
    "univariate_lagrange_polynomial",
    "univariate_lagrange_first_derivative",
    "univariate_lagrange_second_derivative",
    # Monomial basis
    "MonomialBasis1D",
    # Domain transforms
    "Univariate1DTransformProtocol",
    "IdentityTransform1D",
    "BoundedAffineTransform1D",
    "UnboundedAffineTransform1D",
    # Piecewise polynomials
    "PiecewiseLinear",
    "PiecewiseQuadratic",
    "PiecewiseCubic",
    "PiecewiseConstantLeft",
    "PiecewiseConstantRight",
    "PiecewiseConstantMidpoint",
    "PiecewisePolynomialProtocol",
]
