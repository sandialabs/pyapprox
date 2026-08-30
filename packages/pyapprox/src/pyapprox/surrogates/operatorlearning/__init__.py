"""Least-squares operator learning between function spaces."""

from pyapprox.surrogates.operatorlearning.basis import (
    SeparableOperatorBasis,
)
from pyapprox.surrogates.operatorlearning.diagnostics import (
    bochner_error,
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
from pyapprox.surrogates.operatorlearning.fitters import (
    OperatorFitResult,
    WeightedLeastSquaresOperatorFitter,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
    IsometricEncoderProtocol,
)
from pyapprox.surrogates.operatorlearning.surrogate import OperatorSurrogate

__all__ = [
    "FieldEncoderProtocol",
    "GramProjectionEncoder",
    "IdentityFieldEncoder",
    "IsometricEncoderProtocol",
    "OperatorFitResult",
    "OperatorSurrogate",
    "ProductFieldEncoder",
    "SeparableOperatorBasis",
    "WeightedLeastSquaresOperatorFitter",
    "bochner_error",
    "christoffel_integral",
    "gram_condition_number",
    "orthonormalize_basis",
    "sample_complexity",
    "weighted_gram",
]
