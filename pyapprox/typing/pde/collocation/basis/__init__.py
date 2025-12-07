"""Polynomial basis module for spectral collocation methods."""

from pyapprox.typing.pde.collocation.basis.tensor_product import (
    TensorProductBasis,
)
from pyapprox.typing.pde.collocation.basis.chebyshev import (
    ChebyshevGaussLobattoNodes1D,
    ChebyshevDerivativeMatrix1D,
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
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
