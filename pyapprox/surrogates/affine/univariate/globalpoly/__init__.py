"""Global orthonormal polynomial basis functions.

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

from pyapprox.surrogates.affine.univariate.globalpoly.continuous_numeric import (
    BoundedContinuousNumericOrthonormalPolynomial1D,
    ContinuousNumericOrthonormalPolynomial1D,
    GaussLegendreIntegrator,
    PredictorCorrector,
    UnboundedContinuousNumericOrthonormalPolynomial1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.discrete import (
    CharlierPolynomial1D,
    DiscreteChebyshevPolynomial1D,
    HahnPolynomial1D,
    KrawtchoukPolynomial1D,
    charlier_recurrence,
    discrete_chebyshev_recurrence,
    hahn_recurrence,
    krawtchouk_recurrence,
)
from pyapprox.surrogates.affine.univariate.globalpoly.hermite import (
    HermitePolynomial1D,
    hermite_recurrence,
)
from pyapprox.surrogates.affine.univariate.globalpoly.jacobi import (
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    JacobiPolynomial1D,
    LegendrePolynomial1D,
    jacobi_recurrence,
)
from pyapprox.surrogates.affine.univariate.globalpoly.laguerre import (
    LaguerrePolynomial1D,
    laguerre_recurrence,
)
from pyapprox.surrogates.affine.univariate.globalpoly.monomial_conversion import (
    convert_orthonormal_to_monomials_1d,
)
from pyapprox.surrogates.affine.univariate.globalpoly.numeric import (
    DiscreteNumericOrthonormalPolynomial1D,
    WeightedSamplePolynomial1D,
    lanczos_recursion,
)
from pyapprox.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    OrthonormalPolynomial1D,
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    ClenshawCurtisQuadratureRule,
    GaussLobattoQuadratureRule,
    GaussQuadratureRule,
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
    "PredictorCorrector",
    "GaussLegendreIntegrator",
    # Quadrature
    "GaussQuadratureRule",
    "GaussLobattoQuadratureRule",
    "ClenshawCurtisQuadratureRule",
    # Monomial conversion
    "convert_orthonormal_to_monomials_1d",
]
