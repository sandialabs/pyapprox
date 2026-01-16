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
from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
    OrthonormalPolynomial1D,
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
    JacobiPolynomial1D,
    LegendrePolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    jacobi_recurrence,
    HermitePolynomial1D,
    hermite_recurrence,
    LaguerrePolynomial1D,
    laguerre_recurrence,
    KrawtchoukPolynomial1D,
    krawtchouk_recurrence,
    HahnPolynomial1D,
    hahn_recurrence,
    CharlierPolynomial1D,
    charlier_recurrence,
    DiscreteChebyshevPolynomial1D,
    discrete_chebyshev_recurrence,
    DiscreteNumericOrthonormalPolynomial1D,
    WeightedSamplePolynomial1D,
    lanczos_recursion,
    ContinuousNumericOrthonormalPolynomial1D,
    GaussQuadratureRule,
    GaussLobattoQuadratureRule,
)

from pyapprox.typing.surrogates.affine.univariate.bspline import (
    BSpline1D,
    HierarchicalBSpline1D,
)

from pyapprox.typing.surrogates.affine.univariate.lagrange import (
    LagrangeBasis1D,
    univariate_lagrange_polynomial,
    univariate_lagrange_first_derivative,
    univariate_lagrange_second_derivative,
)

from pyapprox.typing.surrogates.affine.univariate.monomial import (
    MonomialBasis1D,
)

from pyapprox.typing.surrogates.affine.univariate.transforms import (
    Univariate1DTransformProtocol,
    IdentityTransform1D,
    BoundedAffineTransform1D,
    UnboundedAffineTransform1D,
)

from pyapprox.typing.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseLinear,
    PiecewiseQuadratic,
    PiecewiseCubic,
    PiecewiseConstantLeft,
    PiecewiseConstantRight,
    PiecewiseConstantMidpoint,
    PiecewisePolynomialProtocol,
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
    "ContinuousNumericOrthonormalPolynomial1D",
    # Quadrature
    "GaussQuadratureRule",
    "GaussLobattoQuadratureRule",
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
