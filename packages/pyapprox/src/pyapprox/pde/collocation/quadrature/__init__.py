"""Quadrature module for spectral collocation methods."""

from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    boundary_quadrature_weights,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature_2d import (
    CollocationQuadrature2D,
)

__all__ = [
    "boundary_quadrature_weights",
    "CollocationQuadrature1D",
    "CollocationQuadrature2D",
]
