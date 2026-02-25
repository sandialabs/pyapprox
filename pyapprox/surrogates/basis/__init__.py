"""Basis functions module.

This module now primarily contains piecewise polynomial functions used by
the PDE module. For other basis functions (polynomials, B-splines, etc.),
use pyapprox.surrogates.affine instead.
"""

from pyapprox.surrogates.basis import piecewisepoly

__all__ = ["piecewisepoly"]
