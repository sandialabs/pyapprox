"""Chebyshev polynomial basis for spectral collocation."""

from pyapprox.pde.collocation.basis.chebyshev.basis import ChebyshevBasis
from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.basis.chebyshev.basis_2d import (
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.basis.chebyshev.basis_3d import (
    ChebyshevBasis3D,
)
from pyapprox.pde.collocation.basis.chebyshev.nodal import (
    ChebyshevGauss1D,
    ChebyshevGaussLobatto1D,
)

__all__ = [
    # 1D bases
    "ChebyshevGaussLobatto1D",
    "ChebyshevGauss1D",
    # Bases
    "ChebyshevBasis",
    "ChebyshevBasis1D",
    "ChebyshevBasis2D",
    "ChebyshevBasis3D",
]
