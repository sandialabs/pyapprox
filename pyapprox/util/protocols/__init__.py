"""
Base protocol definitions for PyApprox typing module.

This module provides foundational protocols that establish common interfaces
across the PyApprox typing system.
"""

from .base import (
    ComputationalObject,
    CallableObject,
    DimensionalObject,
    ParameterizedObject,
)

__all__ = [
    "ComputationalObject",
    "CallableObject",
    "DimensionalObject",
    "ParameterizedObject",
]
