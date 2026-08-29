"""Least-squares operator learning between function spaces."""

from pyapprox.surrogates.operatorlearning.encoders import (
    GramProjectionEncoder,
    IdentityFieldEncoder,
    ProductFieldEncoder,
    orthonormalize_basis,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
    IsometricEncoderProtocol,
)

__all__ = [
    "FieldEncoderProtocol",
    "GramProjectionEncoder",
    "IdentityFieldEncoder",
    "IsometricEncoderProtocol",
    "ProductFieldEncoder",
    "orthonormalize_basis",
]
