"""Univariate basis functions for affine surrogates."""

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

from pyapprox.typing.surrogates.affine.univariate.quadrature import (
    GaussQuadratureRule,
    GaussLobattoQuadratureRule,
)

__all__ = [
    # Base class and functions
    "OrthonormalPolynomial1D",
    "evaluate_orthonormal_polynomial_1d",
    "evaluate_orthonormal_polynomial_derivatives_1d",
    # Jacobi family
    "JacobiPolynomial1D",
    "LegendrePolynomial1D",
    "Chebyshev1stKindPolynomial1D",
    "Chebyshev2ndKindPolynomial1D",
    "jacobi_recurrence",
    # Hermite family
    "HermitePolynomial1D",
    "hermite_recurrence",
    # Quadrature
    "GaussQuadratureRule",
    "GaussLobattoQuadratureRule",
]
