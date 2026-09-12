"""Domains a fixed-basis operator surrogate can be built over."""

from pyapprox.surrogates.operatorlearning.domains.implementations import (
    UniformGridDomain,
)
from pyapprox.surrogates.operatorlearning.domains.protocols import (
    MetricSpaceProtocol,
    OffGridEvaluatorProtocol,
)

__all__ = [
    "MetricSpaceProtocol",
    "OffGridEvaluatorProtocol",
    "UniformGridDomain",
]
