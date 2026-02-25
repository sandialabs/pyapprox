"""
OED data generation and management.

This module provides utilities for generating quadrature data for OED
and managing (saving/loading/subsetting) datasets of model evaluations.
"""

from .generator import OEDDataGenerator
from .manager import OEDDataManager

__all__ = [
    "OEDDataGenerator",
    "OEDDataManager",
]
