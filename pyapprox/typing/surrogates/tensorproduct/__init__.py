"""Tensor product interpolation and quadrature module.

This module provides classes for tensor product interpolation and quadrature,
which are fundamental building blocks for sparse grids and multivariate
approximation.

Classes
-------
TensorProductInterpolant
    Tensor product interpolant using 1D interpolation bases.
TensorProductQuadratureRule
    Tensor product quadrature rule from 1D rules.
ParameterizedTensorProductQuadratureRule
    Level-parameterized tensor product quadrature.

Protocols
---------
InterpolationBasis1DProtocol
    Protocol for 1D bases suitable for tensor product interpolation.
    Re-exported from affine.protocols.
"""

from pyapprox.typing.surrogates.affine.protocols import (
    InterpolationBasis1DProtocol,
)
from pyapprox.typing.surrogates.tensorproduct.interpolant import (
    TensorProductInterpolant,
)

__all__ = [
    "InterpolationBasis1DProtocol",
    "TensorProductInterpolant",
]
