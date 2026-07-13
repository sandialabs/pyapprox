import numpy as np
import pytest
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.sampling.latin_hypercube import LatinHypercubeSampler


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


def _assert_latin_hypercube(samples_np: np.ndarray) -> None:
    """Assert each 1D projection has exactly one point per stratum."""
    nvars, nsamples = samples_np.shape
    for dd in range(nvars):
        strata = np.floor(samples_np[dd] * nsamples).astype(int)
        # guard against points exactly at 1.0
        strata = np.clip(strata, 0, nsamples - 1)
        assert np.array_equal(np.sort(strata), np.arange(nsamples))


class TestLatinHypercubeSampler:

    def test_shapes_and_weights(self, bkd) -> None:
        nvars, nsamples = 3, 10
        sampler = LatinHypercubeSampler(nvars, bkd, seed=42)
        assert sampler.nvars() == nvars
        assert sampler.bkd() is bkd
        samples, weights = sampler.sample(nsamples)
        assert samples.shape == (nvars, nsamples)
        assert weights.shape == (nsamples,)
        bkd.assert_allclose(
            bkd.asarray(np.full((nsamples,), 1.0 / nsamples)), weights
        )

    def test_latin_hypercube_property(self, bkd) -> None:
        nvars, nsamples = 4, 20
        sampler = LatinHypercubeSampler(nvars, bkd, seed=0)
        samples, _ = sampler.sample(nsamples)
        samples_np = bkd.to_numpy(samples)
        assert np.all(samples_np >= 0.0) and np.all(samples_np <= 1.0)
        _assert_latin_hypercube(samples_np)

    def test_centered_when_not_scrambled(self, bkd) -> None:
        nvars, nsamples = 2, 8
        sampler = LatinHypercubeSampler(nvars, bkd, scramble=False, seed=0)
        samples, _ = sampler.sample(nsamples)
        samples_np = bkd.to_numpy(samples)
        _assert_latin_hypercube(samples_np)
        # points must sit at stratum midpoints (k + 0.5) / nsamples
        midpoints = (np.floor(samples_np * nsamples) + 0.5) / nsamples
        np.testing.assert_allclose(samples_np, midpoints)

    def test_single_shot_raises_and_reset_reproduces(self, bkd) -> None:
        nvars, nsamples = 3, 10
        sampler = LatinHypercubeSampler(nvars, bkd, seed=42)
        samples1, _ = sampler.sample(nsamples)
        with pytest.raises(RuntimeError, match="not extensible"):
            sampler.sample(nsamples)
        sampler.reset()
        samples2, _ = sampler.sample(nsamples)
        # same seed and nsamples -> same design after reset
        bkd.assert_allclose(samples1, samples2)

    def test_seed_reproducibility(self, bkd) -> None:
        nvars, nsamples = 3, 10
        samples1, _ = LatinHypercubeSampler(nvars, bkd, seed=7).sample(nsamples)
        samples2, _ = LatinHypercubeSampler(nvars, bkd, seed=7).sample(nsamples)
        samples3, _ = LatinHypercubeSampler(nvars, bkd, seed=8).sample(nsamples)
        bkd.assert_allclose(samples1, samples2)
        assert not np.allclose(bkd.to_numpy(samples1), bkd.to_numpy(samples3))

    def test_distribution_transform(self, bkd) -> None:
        nvars, nsamples = 2, 16
        lb, ub = -1.0, 1.0
        dist = _UniformOnIntervalDistribution(lb, ub, nvars, bkd)
        sampler = LatinHypercubeSampler(nvars, bkd, distribution=dist, seed=0)
        samples, _ = sampler.sample(nsamples)
        samples_np = bkd.to_numpy(samples)
        assert samples_np.shape == (nvars, nsamples)
        assert np.all(samples_np >= lb) and np.all(samples_np <= ub)
        # mapping back to [0, 1] must recover the LHS property
        _assert_latin_hypercube((samples_np - lb) / (ub - lb))

    def test_transform_to_normal(self, bkd) -> None:
        nvars, nsamples = 2, 1000
        sampler = LatinHypercubeSampler(
            nvars, bkd, transform_to_normal=True, seed=0
        )
        samples, _ = sampler.sample(nsamples)
        samples_np = bkd.to_numpy(samples)
        assert np.all(np.isfinite(samples_np))
        # stratified uniforms through the normal inverse CDF give tight
        # estimates of the standard normal moments
        np.testing.assert_allclose(samples_np.mean(axis=1), 0.0, atol=1e-2)
        np.testing.assert_allclose(samples_np.std(axis=1), 1.0, atol=1e-2)

    def test_strength_two(self, bkd) -> None:
        # strength=2 requires nsamples = p**2 (p prime) and nvars <= p + 1
        nvars, nsamples = 3, 25
        sampler = LatinHypercubeSampler(nvars, bkd, strength=2, seed=0)
        samples, _ = sampler.sample(nsamples)
        samples_np = bkd.to_numpy(samples)
        _assert_latin_hypercube(samples_np)

    def test_optimization_random_cd(self, bkd) -> None:
        nvars, nsamples = 2, 10
        sampler = LatinHypercubeSampler(
            nvars, bkd, optimization="random-cd", seed=0
        )
        samples, _ = sampler.sample(nsamples)
        _assert_latin_hypercube(bkd.to_numpy(samples))

    def test_integration_accuracy(self, bkd) -> None:
        # LHS variance for sum of 1D functions is much lower than MC:
        # integrate f(x) = sum_d x_d**2 over [0, 1]^d, exact = d / 3
        nvars, nsamples = 4, 100
        sampler = LatinHypercubeSampler(nvars, bkd, seed=3)
        samples, weights = sampler.sample(nsamples)
        samples_np = bkd.to_numpy(samples)
        weights_np = bkd.to_numpy(weights)
        estimate = float((samples_np**2).sum(axis=0) @ weights_np)
        assert abs(estimate - nvars / 3.0) < 1e-2
