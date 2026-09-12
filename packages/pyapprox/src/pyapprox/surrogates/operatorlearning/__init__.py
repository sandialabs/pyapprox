"""Least-squares operator learning between function spaces.

The torch-dependent pieces are deliberately **not** re-exported here:
``MLPLatentMap``, ``TorchGradientLatentMapFitter`` and the optimizer specs
that configure it. ``MLPLatentMap`` is an ``nn.Module``, so importing any
of them eagerly pulls torch in and raises this package's import cost from
2.4s to 4.9s -- paid by every numpy-only caller who touches it. Import
them from ``pyapprox.surrogates.operatorlearning.latent_maps``, as the
flow-matching velocity fields are imported from their own module for the
same reason. The omission is deliberate; adding them here would be a
regression rather than a fix.

``IterativeOperatorFitResult`` and ``OptimizerStageResult`` *are* exported,
despite describing a gradient fit, because neither imports torch -- they
carry iteration counts and objective values, which any iterative optimizer
produces.
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
    IterativeOperatorFitResult,
    OperatorFitResult,
    OptimizerStageResult,
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
    "IterativeOperatorFitResult",
    "LatentMapProtocol",
    "LinearInParamsLatentMapProtocol",
    "MultiIndexLatentMapProtocol",
    "OperatorFitResult",
    "OperatorSurrogate",
    "OptimizerStageResult",
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
