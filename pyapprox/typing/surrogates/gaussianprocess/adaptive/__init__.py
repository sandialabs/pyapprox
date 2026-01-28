"""Adaptive Gaussian process module.

Provides adaptive sampling strategies and the builder for iteratively
constructing Gaussian process surrogates.
"""

from pyapprox.typing.surrogates.gaussianprocess.adaptive.adaptive_gp_builder import (
    AdaptiveGPBuilder,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.candidate_generator import (
    HybridSobolRandomCandidateGenerator,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.cholesky_sampler import (
    CholeskySampler,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.ivar_sampler import (
    IVARSampler,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.protocols import (
    AdaptiveSamplerProtocol,
    CandidateGeneratorProtocol,
    SamplingScheduleProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.sampling_schedule import (
    ConstantSamplingSchedule,
    ListSamplingSchedule,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.sobol_sampler import (
    SobolAdaptiveSampler,
)

__all__ = [
    "AdaptiveGPBuilder",
    "AdaptiveSamplerProtocol",
    "CandidateGeneratorProtocol",
    "CholeskySampler",
    "ConstantSamplingSchedule",
    "HybridSobolRandomCandidateGenerator",
    "IVARSampler",
    "ListSamplingSchedule",
    "SamplingScheduleProtocol",
    "SobolAdaptiveSampler",
]
