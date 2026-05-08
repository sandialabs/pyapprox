"""Protocols for tensor product module.

This module re-exports the InterpolationBasis1DProtocol from affine.protocols
and defines any additional protocols specific to tensor product operations.
"""

from pyapprox.surrogates.affine.protocols import (
    Basis1DHasHessianProtocol,
    Basis1DHasJacobianProtocol,
    InterpolationBasis1DProtocol,
)

__all__ = [
    "InterpolationBasis1DProtocol",
    "Basis1DHasJacobianProtocol",
    "Basis1DHasHessianProtocol",
]
