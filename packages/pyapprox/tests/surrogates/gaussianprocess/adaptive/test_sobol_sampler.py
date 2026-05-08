"""Tests for Sobol adaptive sampler."""

from pyapprox.surrogates.gaussianprocess.adaptive.protocols import (
    AdaptiveSamplerProtocol,
)
from pyapprox.surrogates.gaussianprocess.adaptive.sobol_sampler import (
    SobolAdaptiveSampler,
)


class TestSobolSampler:
    """Base tests for SobolAdaptiveSampler."""

    def test_protocol_compliance(self, bkd) -> None:
        sampler = SobolAdaptiveSampler(2, bkd)
        assert isinstance(sampler, AdaptiveSamplerProtocol)

    def test_shape(self, bkd) -> None:
        nvars, nsamples = 3, 10
        sampler = SobolAdaptiveSampler(nvars, bkd)
        samples = sampler.select_samples(nsamples)
        assert samples.shape == (nvars, nsamples)

    def test_default_bounds(self, bkd) -> None:
        sampler = SobolAdaptiveSampler(2, bkd)
        samples = sampler.select_samples(50)
        assert bkd.all_bool(samples >= 0.0)
        assert bkd.all_bool(samples <= 1.0)

    def test_custom_bounds(self, bkd) -> None:
        bounds = bkd.asarray([[-1.0, 1.0], [0.0, 2.0]])
        sampler = SobolAdaptiveSampler(2, bkd, scaled_bounds=bounds)
        samples = sampler.select_samples(50)
        assert bkd.all_bool(samples[0, :] >= -1.0)
        assert bkd.all_bool(samples[0, :] <= 1.0)
        assert bkd.all_bool(samples[1, :] >= 0.0)
        assert bkd.all_bool(samples[1, :] <= 2.0)

    def test_sequential_calls(self, bkd) -> None:
        """Successive calls return different points (Sobol advances)."""
        sampler = SobolAdaptiveSampler(2, bkd)
        s1 = sampler.select_samples(10)
        s2 = sampler.select_samples(10)
        # Verify points differ (Sobol advances between calls)
        diff = bkd.sum((s1 - s2) ** 2)
        assert float(bkd.to_numpy(bkd.reshape(diff, (1,)))[0]) > 0.0

    def test_set_kernel_noop(self, bkd) -> None:
        sampler = SobolAdaptiveSampler(2, bkd)
        sampler.set_kernel(None)  # type: ignore[arg-type]

    def test_add_training_samples_noop(self, bkd) -> None:
        sampler = SobolAdaptiveSampler(2, bkd)
        dummy = bkd.zeros((2, 5))
        sampler.add_additional_training_samples(dummy)
