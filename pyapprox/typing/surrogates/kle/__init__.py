"""
Karhunen-Loève Expansion (KLE) implementations.

This module provides KLE protocols and implementations for representing
random fields as truncated eigenfunction expansions.

Key Protocols
-------------
- KLEProtocol: Base protocol for KLE implementations
- ReducibleKLEProtocol: Protocol for KLEs with reduce/expand operations

Key Classes
-----------
- MeshKLE: Kernel-based KLE computed from mesh coordinates and a kernel
- DataDrivenKLE: SVD-based KLE computed from field samples
- PrincipalComponentAnalysis: PCA for dimensionality reduction

Utilities
---------
- adjust_sign_eig: Ensure sign consistency of eigenvectors
- sort_eigenpairs: Sort eigenpairs by descending eigenvalue

Analytical
----------
- AnalyticalExponentialKLE1D: Analytical KLE for 1D exponential kernel
"""

from .protocols import (
    KLEProtocol,
    ReducibleKLEProtocol,
)
from .utils import (
    adjust_sign_eig,
    sort_eigenpairs,
)
from .mesh_kle import MeshKLE
from .data_driven_kle import DataDrivenKLE
from .pca import PrincipalComponentAnalysis
from .analytical import AnalyticalExponentialKLE1D
from .periodic_random_field import PeriodicReiszGaussianRandomField

__all__ = [
    # Protocols
    "KLEProtocol",
    "ReducibleKLEProtocol",
    # Core
    "MeshKLE",
    "DataDrivenKLE",
    "PrincipalComponentAnalysis",
    # Periodic random fields
    "PeriodicReiszGaussianRandomField",
    # Utilities
    "adjust_sign_eig",
    "sort_eigenpairs",
    # Analytical
    "AnalyticalExponentialKLE1D",
]
