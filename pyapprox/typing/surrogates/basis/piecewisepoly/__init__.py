"""Re-exports from new location for backwards compatibility.

The canonical location is now:
    pyapprox.typing.surrogates.affine.univariate.piecewisepoly
"""

from pyapprox.typing.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseLinear,
    PiecewiseQuadratic,
    PiecewiseCubic,
    PiecewiseConstantLeft,
    PiecewiseConstantRight,
    PiecewiseConstantMidpoint,
    PiecewisePolynomialProtocol,
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
