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

from pyapprox.typing.surrogates.affine.univariate.orthopoly_base import (
    OrthonormalPolynomial1D,
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
)

from pyapprox.typing.surrogates.affine.univariate.jacobi import (
    JacobiPolynomial1D,
    LegendrePolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    jacobi_recurrence,
)

from pyapprox.typing.surrogates.affine.univariate.hermite import (
    HermitePolynomial1D,
    hermite_recurrence,
)

from pyapprox.typing.surrogates.affine.univariate.laguerre import (
    LaguerrePolynomial1D,
    laguerre_recurrence,
)

from pyapprox.typing.surrogates.affine.univariate.discrete import (
    KrawtchoukPolynomial1D,
    krawtchouk_recurrence,
    HahnPolynomial1D,
    hahn_recurrence,
    CharlierPolynomial1D,
    charlier_recurrence,
    DiscreteChebyshevPolynomial1D,
    discrete_chebyshev_recurrence,
)

from pyapprox.typing.surrogates.affine.univariate.numeric import (
    DiscreteNumericOrthonormalPolynomial1D,
    WeightedSamplePolynomial1D,
    lanczos_recursion,
)

from pyapprox.typing.surrogates.affine.univariate.quadrature import (
    GaussQuadratureRule,
    GaussLobattoQuadratureRule,
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
    # Quadrature
    "GaussQuadratureRule",
    "GaussLobattoQuadratureRule",
]
