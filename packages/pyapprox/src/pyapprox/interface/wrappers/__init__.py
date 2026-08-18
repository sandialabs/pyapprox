"""Wrappers for adding functionality to functions.

This module provides wrappers for:
- Finite difference derivatives

Evaluation counts and wall times are recorded by ``FunctionTimer`` and
``TimedFunction`` in :mod:`pyapprox.interface.functions.timing`.
"""

from pyapprox.interface.wrappers.finite_difference import (
    FiniteDifferenceWrapper,
)

__all__ = [
    "FiniteDifferenceWrapper",
]
