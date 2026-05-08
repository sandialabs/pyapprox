"""Polynomial basis module for spectral collocation methods."""

from pyapprox.pde.collocation.basis.chebyshev import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
    ChebyshevDerivativeMatrix1D,
    ChebyshevGaussLobattoNodes1D,
)
from pyapprox.pde.collocation.basis.tensor_product import (
    TensorProductBasis,
)

__all__ = [
    # Generic
    "TensorProductBasis",
    # Chebyshev
    "ChebyshevGaussLobattoNodes1D",
    "ChebyshevDerivativeMatrix1D",
    "ChebyshevBasis1D",
    "ChebyshevBasis2D",
    "ChebyshevBasis3D",
]
