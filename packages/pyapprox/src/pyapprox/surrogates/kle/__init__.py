"""
Karhunen-Loève Expansion (KLE) implementations.

This module provides KLE protocols and implementations for representing
random fields as truncated eigenfunction expansions.

Key Protocols
-------------
- KLEProtocol: Base protocol for KLE implementations
- ReducibleKLEProtocol: Protocol for KLEs with reduce/expand operations
- KLEEigenSolverProtocol: How MeshKLE computes its eigenpairs

Key Classes
-----------
- MeshKLE: Kernel-based KLE computed from mesh coordinates and a kernel
- GalerkinKLE: KLE via Galerkin projection (generalized eigenproblem)
- SPDEMaternKLE: SPDE-based KLE for Matern fields (sparse, O(N) memory)
- DataDrivenKLE: SVD-based KLE computed from field samples
- NystromKLE: KLE evaluable away from its collocation points
- PrecomputedKLE: KLE built from an already-computed basis
- PrincipalComponentAnalysis: PCA for dimensionality reduction

Persistence
-----------
- save_kle / load_kle: store a basis and reload it without re-solving
- save_nystrom_kle / load_nystrom_kle: the same for a Nystrom basis,
  keeping the extension matrix so the reload still evaluates at new
  points rather than only where it was stored

Eigensolvers
------------
Injected into MeshKLE to control how the eigenproblem is solved.

- DenseEigenSolver: assembles the kernel matrix; O(N^2) memory
- PivotedCholeskyEigenSolver: matrix-free, low-rank factorization
- RandomizedEigenSolver: matrix-free, randomized subspace iteration
- finalize_eigenpairs: the convention every solver must return

Utilities
---------
- adjust_sign_eig: Ensure sign consistency of eigenvectors
- sort_eigenpairs: Sort eigenpairs by descending eigenvalue

Analytical
----------
- AnalyticalExponentialKLE1D: Analytical KLE for 1D exponential kernel
"""

from .analytical import AnalyticalExponentialKLE1D
from .data_driven_kle import DataDrivenKLE
from .eigensolvers import (
    DenseEigenSolver,
    KLEEigenSolverProtocol,
    PivotedCholeskyEigenSolver,
    RandomizedEigenSolver,
    finalize_eigenpairs,
)
from .galerkin_kle import GalerkinKLE
from .io import load_kle, load_nystrom_kle, save_kle, save_nystrom_kle
from .mesh_kle import MeshKLE
from .nystrom_kle import NystromKLE, create_nystrom_kle
from .pca import PrincipalComponentAnalysis
from .periodic_random_field import PeriodicReiszGaussianRandomField
from .precomputed_kle import PrecomputedKLE
from .protocols import (
    KLEProtocol,
    ReducibleKLEProtocol,
)
from .spde_kle import SPDEMaternKLE
from .utils import (
    adjust_sign_eig,
    sort_eigenpairs,
)

__all__ = [
    # Protocols
    "KLEProtocol",
    "ReducibleKLEProtocol",
    "KLEEigenSolverProtocol",
    # Core
    "MeshKLE",
    "GalerkinKLE",
    "SPDEMaternKLE",
    "DataDrivenKLE",
    "NystromKLE",
    "create_nystrom_kle",
    "PrecomputedKLE",
    "PrincipalComponentAnalysis",
    # Persistence
    "save_kle",
    "load_kle",
    "save_nystrom_kle",
    "load_nystrom_kle",
    # Eigensolvers
    "DenseEigenSolver",
    "PivotedCholeskyEigenSolver",
    "RandomizedEigenSolver",
    "finalize_eigenpairs",
    # Periodic random fields
    "PeriodicReiszGaussianRandomField",
    # Utilities
    "adjust_sign_eig",
    "sort_eigenpairs",
    # Analytical
    "AnalyticalExponentialKLE1D",
]
