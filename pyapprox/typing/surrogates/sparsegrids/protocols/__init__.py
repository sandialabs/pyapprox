"""Protocols for sparse grid surrogates."""

from .sparse_grid import (
    SubspaceProtocol,
    SparseGridProtocol,
    AdaptiveSparseGridProtocol,
)
from .local_refinement import (
    LocalIndexGeneratorProtocol,
    LocalRefinementCriteriaProtocol,
)

__all__ = [
    "SubspaceProtocol",
    "SparseGridProtocol",
    "AdaptiveSparseGridProtocol",
    "LocalIndexGeneratorProtocol",
    "LocalRefinementCriteriaProtocol",
]
