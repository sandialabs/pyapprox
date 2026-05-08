"""Quadrature module for spectral collocation methods."""

from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature_2d import (
    CollocationQuadrature2D,
)

__all__ = [
    "CollocationQuadrature1D",
    "CollocationQuadrature2D",
]
