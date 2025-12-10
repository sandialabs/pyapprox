"""Protocols for sparse grid surrogates."""

from .sparse_grid import (
    SubspaceProtocol,
    SubspaceWithDerivativesProtocol,
    SparseGridProtocol,
    SparseGridWithDerivativesProtocol,
    AdaptiveSparseGridProtocol,
)
from .local_refinement import (
    LocalIndexGeneratorProtocol,
    LocalRefinementCriteriaProtocol,
)

__all__ = [
    # Base protocols
    "SubspaceProtocol",
    "SparseGridProtocol",
    "AdaptiveSparseGridProtocol",
    # Derivative protocols
    "SubspaceWithDerivativesProtocol",
    "SparseGridWithDerivativesProtocol",
    # Local refinement
    "LocalIndexGeneratorProtocol",
    "LocalRefinementCriteriaProtocol",
]
