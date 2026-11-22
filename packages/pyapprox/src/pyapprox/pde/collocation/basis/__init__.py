"""Polynomial basis module for spectral collocation methods."""

from pyapprox.pde.collocation.basis.chebyshev import (
    ChebyshevBasis,
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
    ChebyshevGauss1D,
    ChebyshevGaussLobatto1D,
)
from pyapprox.pde.collocation.basis.tensor_product import (
    TensorProductBasis,
)

__all__ = [
    # Generic
    "TensorProductBasis",
    # Chebyshev
    "ChebyshevGaussLobatto1D",
    "ChebyshevGauss1D",
    "ChebyshevBasis",
    "ChebyshevBasis1D",
    "ChebyshevBasis2D",
    "ChebyshevBasis3D",
]
