"""Scalar and vector field definitions for spectral collocation.

This module provides classes for defining scalar fields (e.g., bed elevation,
surface height) via SymPy expressions.
"""

from pyapprox.pde.collocation.fields.sympy_field import (
    SympyField2D,
    create_quadratic_bed,
    create_polynomial_surface,
    create_beta_surface,
    create_shallow_wave_bed,
)

__all__ = [
    "SympyField2D",
    "create_quadratic_bed",
    "create_polynomial_surface",
    "create_beta_surface",
    "create_shallow_wave_bed",
]
