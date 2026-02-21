"""Protocols for sparse grid surrogates."""

from .sparse_grid import (
    SubspaceProtocol,
    SubspaceWithDerivativesProtocol,
    SparseGridProtocol,
    SparseGridWithDerivativesProtocol,
    AdaptiveSparseGridProtocol,
)

__all__ = [
    # Base protocols
    "SubspaceProtocol",
    "SparseGridProtocol",
    "AdaptiveSparseGridProtocol",
    # Derivative protocols
    "SubspaceWithDerivativesProtocol",
    "SparseGridWithDerivativesProtocol",
]
