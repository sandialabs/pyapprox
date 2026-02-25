"""Re-exports from new location for backwards compatibility.

The canonical location is now:
    pyapprox.surrogates.affine.univariate.piecewisepoly
"""

from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseConstantLeft,
    PiecewiseConstantMidpoint,
    PiecewiseConstantRight,
    PiecewiseCubic,
    PiecewiseLinear,
    PiecewisePolynomialProtocol,
    PiecewiseQuadratic,
)

__all__ = [
    "PiecewiseLinear",
    "PiecewiseQuadratic",
    "PiecewiseCubic",
    "PiecewiseConstantLeft",
    "PiecewiseConstantRight",
    "PiecewiseConstantMidpoint",
    "PiecewisePolynomialProtocol",
]
