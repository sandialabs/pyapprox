"""Interface representation for domain decomposition.

Provides:
- LegendreInterfaceBasis1D: Legendre polynomial basis for 1D interface functions
- LegendreInterfaceBasis2D: Legendre polynomial basis for 2D interface functions
- Interface1D: Interface for 1D problems (single point)
- Interface: Interface with 1D basis for 2D problems
- Interface2D: Interface with 2D basis for 3D problems
- InterpolationOperator: Lagrange interpolation between point sets
"""

from pyapprox.pde.decomposition.interface.basis import (
    LegendreInterfaceBasis1D,
    LegendreInterfaceBasis2D,
)
from pyapprox.pde.decomposition.interface.interface import (
    Interface,
    Interface1D,
    Interface2D,
)
from pyapprox.pde.decomposition.interface.interpolation import (
    InterpolationOperator,
    RestrictionOperator,
    lagrange_interpolation_matrix,
)

__all__ = [
    "LegendreInterfaceBasis1D",
    "LegendreInterfaceBasis2D",
    "Interface1D",
    "Interface",
    "Interface2D",
    "InterpolationOperator",
    "RestrictionOperator",
    "lagrange_interpolation_matrix",
]
