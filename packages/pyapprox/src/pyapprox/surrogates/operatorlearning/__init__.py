"""Least-squares operator learning between function spaces.

``MLPLatentMap`` is deliberately **not** re-exported here. It is an
``nn.Module``, so importing it eagerly would put torch's 2.1s import cost
on every numpy-only caller who touches this package. Import it from
``pyapprox.surrogates.operatorlearning.latent_maps``, as the flow-matching
velocity fields are imported from their own module for the same reason.
"""

from pyapprox.surrogates.operatorlearning.basis import (
    SeparableOperatorBasis,
)
from pyapprox.surrogates.operatorlearning.diagnostics import (
    bochner_error,
    christoffel_integral,
    coefficient_error,
    field_error,
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
    LatentMapProtocol,
    LinearInParamsLatentMapProtocol,
    MultiIndexLatentMapProtocol,
)
from pyapprox.surrogates.operatorlearning.surrogate import OperatorSurrogate

__all__ = [
    "FieldEncoderProtocol",
    "GramProjectionEncoder",
    "IdentityFieldEncoder",
    "IsometricEncoderProtocol",
    "LatentMapProtocol",
    "LinearInParamsLatentMapProtocol",
    "MultiIndexLatentMapProtocol",
    "OperatorFitResult",
    "OperatorSurrogate",
    "ProductFieldEncoder",
    "SeparableOperatorBasis",
    "WeightedLeastSquaresOperatorFitter",
    "bochner_error",
    "christoffel_integral",
    "coefficient_error",
    "field_error",
    "gram_condition_number",
    "orthonormalize_basis",
    "sample_complexity",
    "weighted_gram",
]
