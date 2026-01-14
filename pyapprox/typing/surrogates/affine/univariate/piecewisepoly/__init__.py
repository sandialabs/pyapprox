"""Piecewise polynomial basis functions for time integration.

This module provides piecewise polynomial basis functions and quadrature rules
for time integration schemes.

Classes
-------
PiecewiseLinear
    Linear basis functions with trapezoidal quadrature.
PiecewiseQuadratic
    Quadratic basis functions with Simpson's rule quadrature.
PiecewiseCubic
    Cubic basis functions with 3/8 Simpson's rule quadrature.
PiecewiseConstantLeft
    Left-constant basis functions with left Riemann sum quadrature.
PiecewiseConstantRight
    Right-constant basis functions with right Riemann sum quadrature.
PiecewiseConstantMidpoint
    Midpoint-constant basis functions with midpoint quadrature.
NodeGenerator
    Abstract base for generating interpolation nodes dynamically.
EquidistantNodeGenerator
    Generate equidistant nodes on an interval.
DynamicPiecewiseBasis
    Wrapper providing set_nterms() for piecewise polynomial bases.
"""

from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.linear import (
    PiecewiseLinear,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.quadratic import (
    PiecewiseQuadratic,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.cubic import (
    PiecewiseCubic,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.left_constant import (
    PiecewiseConstantLeft,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.right_constant import (
    PiecewiseConstantRight,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.mid_constant import (
    PiecewiseConstantMidpoint,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.protocols import (
    PiecewisePolynomialProtocol,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.dynamic import (
    NodeGenerator,
    EquidistantNodeGenerator,
    DynamicPiecewiseBasis,
)

__all__ = [
    "PiecewiseLinear",
    "PiecewiseQuadratic",
    "PiecewiseCubic",
    "PiecewiseConstantLeft",
    "PiecewiseConstantRight",
    "PiecewiseConstantMidpoint",
    "PiecewisePolynomialProtocol",
    "NodeGenerator",
    "EquidistantNodeGenerator",
    "DynamicPiecewiseBasis",
]
