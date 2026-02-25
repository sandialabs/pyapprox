"""Multivariate basis functions for affine surrogates.

This module provides multivariate basis functions constructed via
tensor products of univariate bases, with full Jacobian and Hessian support.
"""

from pyapprox.surrogates.affine.basis.multiindex import (
    MultiIndexBasis,
)

from pyapprox.surrogates.affine.basis.orthonormal_poly import (
    OrthonormalPolynomialBasis,
)

from pyapprox.surrogates.affine.basis.quadrature_rules import (
    QuadratureRule,
    TensorProductQuadratureRule,
    FixedTensorProductQuadratureRule,
)

from pyapprox.surrogates.affine.basis.kernel_basis import (
    KernelBasis,
)

__all__ = [
    # Core basis classes
    "MultiIndexBasis",
    "OrthonormalPolynomialBasis",
    "KernelBasis",
    # Quadrature rules
    "QuadratureRule",
    "TensorProductQuadratureRule",
    "FixedTensorProductQuadratureRule",
]
