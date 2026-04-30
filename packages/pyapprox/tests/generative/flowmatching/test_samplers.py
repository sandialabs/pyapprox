"""Tests for flow matching samplers."""

import pytest

torch = pytest.importorskip("torch")

from pyapprox.generative.flowmatching.samplers import (
    GaussianSourceSampler,
    WeightedEmpiricalSampler,
)


class TestGaussianSourceSampler:
    def test_shape(self):
        sampler = GaussianSourceSampler(d=3, seed=0)
        x0 = sampler.sample_x0(100)
        assert x0.shape == (3, 100)

    def test_dtype(self):
        sampler = GaussianSourceSampler(d=1, dtype=torch.float32, seed=0)
        x0 = sampler.sample_x0(10)
        assert x0.dtype == torch.float32

    def test_seed_reproducibility(self):
        s1 = GaussianSourceSampler(d=2, seed=42)
        s2 = GaussianSourceSampler(d=2, seed=42)
        x1 = s1.sample_x0(50)
        x2 = s2.sample_x0(50)
        torch.testing.assert_close(x1, x2)

    def test_statistics(self):
        sampler = GaussianSourceSampler(d=1, seed=0)
        x0 = sampler.sample_x0(10000)
        assert abs(x0.mean().item()) < 0.1
        assert abs(x0.std().item() - 1.0) < 0.1


class TestWeightedEmpiricalSampler:
    def test_shape(self):
        pool = torch.randn(2, 50)
        w = torch.ones(50)
        sampler = WeightedEmpiricalSampler(pool, w, seed=0)
        x1, weights = sampler.sample_x1(30)
        assert x1.shape == (2, 30)
        assert weights.shape == (30,)

    def test_uniform_weights(self):
        pool = torch.randn(1, 100)
        w = torch.ones(100)
        sampler = WeightedEmpiricalSampler(pool, w, seed=0)
        _, weights = sampler.sample_x1(20)
        expected = torch.full((20,), 1.0 / 20, dtype=torch.float64)
        torch.testing.assert_close(weights, expected)

    def test_seed_reproducibility(self):
        pool = torch.randn(1, 50)
        w = torch.ones(50)
        s1 = WeightedEmpiricalSampler(pool, w, seed=42)
        s2 = WeightedEmpiricalSampler(pool, w, seed=42)
        x1_a, _ = s1.sample_x1(30)
        x1_b, _ = s2.sample_x1(30)
        torch.testing.assert_close(x1_a, x1_b)

    def test_weighted_sampling_bias(self):
        pool = torch.tensor([[0.0, 1.0, 2.0]])
        w = torch.tensor([0.0, 0.0, 1.0])
        sampler = WeightedEmpiricalSampler(pool, w, seed=0)
        x1, _ = sampler.sample_x1(100)
        torch.testing.assert_close(
            x1, torch.full((1, 100), 2.0, dtype=torch.float64),
        )

    def test_samples_from_pool(self):
        pool = torch.tensor([[10.0, 20.0, 30.0]])
        w = torch.ones(3)
        sampler = WeightedEmpiricalSampler(pool, w, seed=0)
        x1, _ = sampler.sample_x1(50)
        for val in x1[0]:
            if val.item() not in (10.0, 20.0, 30.0):
                raise AssertionError(
                    f"Sample {val.item()} not in pool"
                )
