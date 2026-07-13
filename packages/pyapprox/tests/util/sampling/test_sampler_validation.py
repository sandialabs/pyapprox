import pytest
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.sampling import (
    HaltonSampler,
    LatinHypercubeSampler,
    SobolSampler,
)

SAMPLER_CLASSES = [HaltonSampler, LatinHypercubeSampler, SobolSampler]


class _UniformOnIntervalDistribution:
    """Minimal DistributionWithInvCDF mapping [0, 1] to [lb, ub]."""

    def __init__(self, lb: float, ub: float, nvars: int, bkd: Backend) -> None:
        self._lb = lb
        self._ub = ub
        self._nvars = nvars
        self._bkd = bkd

    def nvars(self) -> int:
        return self._nvars

    def invcdf(self, probs):
        return self._lb + (self._ub - self._lb) * probs


class TestSamplerDistributionValidation:

    @pytest.mark.parametrize("sampler_cls", SAMPLER_CLASSES)
    def test_distribution_nvars_mismatch_raises(self, bkd, sampler_cls) -> None:
        dist = _UniformOnIntervalDistribution(-1.0, 1.0, 2, bkd)
        with pytest.raises(ValueError, match="does not match nvars"):
            sampler_cls(3, bkd, distribution=dist)

    @pytest.mark.parametrize("sampler_cls", SAMPLER_CLASSES)
    def test_invalid_distribution_raises(self, bkd, sampler_cls) -> None:
        with pytest.raises(TypeError, match="DistributionWithInvCDF"):
            sampler_cls(3, bkd, distribution="not-a-distribution")

    @pytest.mark.parametrize("sampler_cls", SAMPLER_CLASSES)
    def test_matching_distribution_accepted(self, bkd, sampler_cls) -> None:
        nvars, nsamples = 2, 8
        dist = _UniformOnIntervalDistribution(-1.0, 1.0, nvars, bkd)
        sampler = sampler_cls(nvars, bkd, distribution=dist, seed=0)
        samples, _ = sampler.sample(nsamples)
        assert samples.shape == (nvars, nsamples)
