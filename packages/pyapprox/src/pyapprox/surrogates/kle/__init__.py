"""
Karhunen-Loève Expansion (KLE) implementations.

This module provides KLE protocols and implementations for representing
random fields as truncated eigenfunction expansions.

Key Protocols
-------------
- KLEProtocol: Base protocol for KLE implementations
- KLEEigenSolverProtocol: How MeshKLE computes its eigenpairs

Key Classes
-----------
- MeshKLE: Kernel-based KLE computed from mesh coordinates and a kernel
- GalerkinKLE: KLE via Galerkin projection (generalized eigenproblem)
- SPDEMaternKLE: SPDE-based KLE for Matern fields (sparse, O(N) memory)
- DataDrivenKLE: SVD-based KLE computed from field samples
- NystromKLE: KLE evaluable away from its collocation points
- PrecomputedKLE: KLE built from an already-computed basis

Persistence
-----------
- save_kle / load_kle: store a basis and reload it without re-solving
- save_nystrom_kle / load_nystrom_kle: the same for a Nystrom basis,
  keeping the extension matrix so the reload still evaluates at new
  points rather than only where it was stored

Reduction
---------
- KLEEncoder: the same basis read as a reduction rather than as an
  expansion, so a KLE can be used where an encoder is wanted. Composed
  rather than folded into the KLE classes, because a lognormal
  expansion has no linear inverse and so simply has no encoder.
- fit_kle_encoder: build one from snapshots in a single call, with the
  metric, eigensolver and truncation policy all still injectable.

Multifidelity
-------------
- nystrom_kle_on_mesh / nystrom_kles_on_meshes: extend one Nystrom basis
  to other meshes, so every fidelity expands the same field from the same
  coefficients instead of each solving its own eigenproblem

Eigensolvers
------------
Injected into MeshKLE to control how the eigenproblem is solved.

- DenseEigenSolver: assembles the kernel matrix; O(N^2) memory
- PivotedCholeskyEigenSolver: matrix-free, low-rank factorization
- RandomizedEigenSolver: matrix-free, randomized subspace iteration
- finalize_eigenpairs: the convention every solver must return

Snapshot eigensolvers
---------------------
The same seam for DataDrivenKLE, where the covariance is a snapshot
matrix rather than a kernel. Which one applies is decided by the metric,
not by the caller.

- SVDSnapshotSolver: thin SVD; needs a diagonal metric, more accurate
- MethodOfSnapshotsSolver: Gram eigendecomposition; any SPD metric,
  never factorizes it
- default_snapshot_eigensolver: picks between them

Truncation
----------
How many modes of a spectrum to keep. Shared with the encoders, so a
variance fraction means the same thing at every entry point.

- by_count / by_variance_fraction / by_numerical_rank
- by_eigenvalue_floor: a caller-set floor, stricter than machine epsilon
- resolve_nterms: the exactly-one-of rule for the two caller-facing ones

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
from .encoder import KLEEncoder, fit_kle_encoder
from .galerkin_kle import GalerkinKLE
from .io import load_kle, load_nystrom_kle, save_kle, save_nystrom_kle
from .mesh_kle import MeshKLE
from .multifidelity import nystrom_kle_on_mesh, nystrom_kles_on_meshes
from .nystrom_kle import NystromKLE, create_nystrom_kle
from .periodic_random_field import PeriodicReiszGaussianRandomField
from .precomputed_kle import PrecomputedKLE
from .protocols import KLEProtocol
from .snapshot_eigensolvers import (
    MethodOfSnapshotsSolver,
    SnapshotEigenSolverProtocol,
    SVDSnapshotSolver,
    default_snapshot_eigensolver,
)
from .spde_kle import SPDEMaternKLE
from .truncation import (
    by_count,
    by_eigenvalue_floor,
    by_numerical_rank,
    by_variance_fraction,
    resolve_nterms,
)
from .utils import (
    adjust_sign_eig,
    sort_eigenpairs,
)

__all__ = [
    # Protocols
    "KLEProtocol",
    "KLEEigenSolverProtocol",
    # Core
    "MeshKLE",
    "GalerkinKLE",
    "SPDEMaternKLE",
    "DataDrivenKLE",
    "NystromKLE",
    "create_nystrom_kle",
    "PrecomputedKLE",
    # Reduction
    "KLEEncoder",
    "fit_kle_encoder",
    # Persistence
    "save_kle",
    "load_kle",
    "save_nystrom_kle",
    "load_nystrom_kle",
    # Multifidelity
    "nystrom_kle_on_mesh",
    "nystrom_kles_on_meshes",
    # Eigensolvers
    "DenseEigenSolver",
    "PivotedCholeskyEigenSolver",
    "RandomizedEigenSolver",
    "finalize_eigenpairs",
    # Snapshot eigensolvers
    "SnapshotEigenSolverProtocol",
    "SVDSnapshotSolver",
    "MethodOfSnapshotsSolver",
    "default_snapshot_eigensolver",
    # Truncation
    "by_count",
    "by_variance_fraction",
    "by_numerical_rank",
    "by_eigenvalue_floor",
    "resolve_nterms",
    # Periodic random fields
    "PeriodicReiszGaussianRandomField",
    # Utilities
    "adjust_sign_eig",
    "sort_eigenpairs",
    # Analytical
    "AnalyticalExponentialKLE1D",
]
