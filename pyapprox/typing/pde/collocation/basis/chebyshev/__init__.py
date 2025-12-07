"""Chebyshev polynomial basis for spectral collocation."""

from pyapprox.typing.pde.collocation.basis.chebyshev.nodes import (
    ChebyshevGaussLobattoNodes1D,
)
from pyapprox.typing.pde.collocation.basis.chebyshev.derivative import (
    ChebyshevDerivativeMatrix1D,
)
from pyapprox.typing.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.typing.pde.collocation.basis.chebyshev.basis_2d import (
    ChebyshevBasis2D,
)
from pyapprox.typing.pde.collocation.basis.chebyshev.basis_3d import (
    ChebyshevBasis3D,
)

__all__ = [
    # 1D Components
    "ChebyshevGaussLobattoNodes1D",
    "ChebyshevDerivativeMatrix1D",
    # Convenience Wrappers
    "ChebyshevBasis1D",
    "ChebyshevBasis2D",
    "ChebyshevBasis3D",
]
