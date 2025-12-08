"""Affine surrogates module.

This module provides protocol-based implementations of affine surrogates,
which are approximations that depend linearly on their hyperparameters.

Submodules
----------
protocols
    Protocol definitions for basis functions, expansions, indices, etc.
univariate
    Univariate (1D) basis functions including orthonormal polynomials.

Phase 1 (Current):
- Univariate basis protocols and implementations
- Jacobi, Legendre, Chebyshev, Hermite polynomials
- Gaussian quadrature rules
"""

from pyapprox.typing.surrogates.affine.protocols import (
    # Univariate protocols
    Basis1DProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DHasHessianProtocol,
    Basis1DHasDerivativesProtocol,
    Basis1DWithJacobianProtocol,
    Basis1DWithJacobianAndHessianProtocol,
    OrthonormalPolynomial1DProtocol,
    Basis1DHasQuadratureProtocol,
    # Multivariate protocols
    BasisProtocol,
    BasisHasJacobianProtocol,
    BasisHasHessianProtocol,
    BasisWithJacobianProtocol,
    BasisWithJacobianAndHessianProtocol,
    MultiIndexBasisProtocol,
    TensorProductBasisProtocol,
    MultiIndexBasisWithJacobianProtocol,
    MultiIndexBasisWithJacobianAndHessianProtocol,
)

from pyapprox.typing.surrogates.affine.univariate import (
    # Base class
    OrthonormalPolynomial1D,
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
    # Jacobi family
    JacobiPolynomial1D,
    LegendrePolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    jacobi_recurrence,
    # Hermite
    HermitePolynomial1D,
    hermite_recurrence,
    # Quadrature
    GaussQuadratureRule,
    GaussLobattoQuadratureRule,
)

__all__ = [
    # Univariate protocols
    "Basis1DProtocol",
    "Basis1DHasJacobianProtocol",
    "Basis1DHasHessianProtocol",
    "Basis1DHasDerivativesProtocol",
    "Basis1DWithJacobianProtocol",
    "Basis1DWithJacobianAndHessianProtocol",
    "OrthonormalPolynomial1DProtocol",
    "Basis1DHasQuadratureProtocol",
    # Multivariate protocols
    "BasisProtocol",
    "BasisHasJacobianProtocol",
    "BasisHasHessianProtocol",
    "BasisWithJacobianProtocol",
    "BasisWithJacobianAndHessianProtocol",
    "MultiIndexBasisProtocol",
    "TensorProductBasisProtocol",
    "MultiIndexBasisWithJacobianProtocol",
    "MultiIndexBasisWithJacobianAndHessianProtocol",
    # Univariate implementations
    "OrthonormalPolynomial1D",
    "evaluate_orthonormal_polynomial_1d",
    "evaluate_orthonormal_polynomial_derivatives_1d",
    "JacobiPolynomial1D",
    "LegendrePolynomial1D",
    "Chebyshev1stKindPolynomial1D",
    "Chebyshev2ndKindPolynomial1D",
    "jacobi_recurrence",
    "HermitePolynomial1D",
    "hermite_recurrence",
    "GaussQuadratureRule",
    "GaussLobattoQuadratureRule",
]
