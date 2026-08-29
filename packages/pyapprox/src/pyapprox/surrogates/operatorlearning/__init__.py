"""Least-squares operator learning between function spaces."""

from pyapprox.surrogates.operatorlearning.basis import (
    SeparableOperatorBasis,
)
from pyapprox.surrogates.operatorlearning.diagnostics import (
    bochner_error,
    check_orthonormality,
    christoffel_integral,
    gram_condition_number,
    sample_complexity,
    weighted_gram,
)
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
    "SeparableOperatorBasis",
    "bochner_error",
    "check_orthonormality",
    "christoffel_integral",
    "gram_condition_number",
    "orthonormalize_basis",
    "sample_complexity",
    "weighted_gram",
]
