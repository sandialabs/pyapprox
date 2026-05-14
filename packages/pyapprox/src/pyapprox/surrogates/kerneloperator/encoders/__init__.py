"""Function encoders for kernel operator learning."""

from pyapprox.surrogates.kerneloperator.encoders.identity import (
    IdentityFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.encoders.pca import (
    PCAFunctionEncoder,
)

__all__ = [
    "IdentityFunctionEncoder",
    "PCAFunctionEncoder",
]
