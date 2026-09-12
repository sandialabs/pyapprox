"""Domains a fixed-basis operator surrogate can be built over."""

from pyapprox.surrogates.operatorlearning.domains.implementations import (
    FixedSampleDomain,
    UniformGridDomain,
)
from pyapprox.surrogates.operatorlearning.domains.protocols import (
    MetricSpaceProtocol,
    OffGridEvaluatorProtocol,
)

__all__ = [
    "FixedSampleDomain",
    "MetricSpaceProtocol",
    "OffGridEvaluatorProtocol",
    "UniformGridDomain",
]
