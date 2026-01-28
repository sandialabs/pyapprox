"""Tests for Sobol adaptive sampler."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.surrogates.gaussianprocess.adaptive.protocols import (
    AdaptiveSamplerProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.sobol_sampler import (
    SobolAdaptiveSampler,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestSobolSampler(Generic[Array], unittest.TestCase):
    """Base tests for SobolAdaptiveSampler."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        sampler = SobolAdaptiveSampler(2, self._bkd)
        self.assertIsInstance(sampler, AdaptiveSamplerProtocol)

    def test_shape(self) -> None:
        nvars, nsamples = 3, 10
        sampler = SobolAdaptiveSampler(nvars, self._bkd)
        samples = sampler.select_samples(nsamples)
        self.assertEqual(samples.shape, (nvars, nsamples))

    def test_default_bounds(self) -> None:
        sampler = SobolAdaptiveSampler(2, self._bkd)
        samples = sampler.select_samples(50)
        bkd = self._bkd
        self.assertTrue(bkd.all_bool(samples >= 0.0))
        self.assertTrue(bkd.all_bool(samples <= 1.0))

    def test_custom_bounds(self) -> None:
        bkd = self._bkd
        bounds = bkd.asarray([[-1.0, 1.0], [0.0, 2.0]])
        sampler = SobolAdaptiveSampler(2, bkd, scaled_bounds=bounds)
        samples = sampler.select_samples(50)
        self.assertTrue(bkd.all_bool(samples[0, :] >= -1.0))
        self.assertTrue(bkd.all_bool(samples[0, :] <= 1.0))
        self.assertTrue(bkd.all_bool(samples[1, :] >= 0.0))
        self.assertTrue(bkd.all_bool(samples[1, :] <= 2.0))

    def test_sequential_calls(self) -> None:
        """Successive calls return different points (Sobol advances)."""
        sampler = SobolAdaptiveSampler(2, self._bkd)
        s1 = sampler.select_samples(10)
        s2 = sampler.select_samples(10)
        self.assertFalse(self._bkd.allclose(s1, s2))

    def test_set_kernel_noop(self) -> None:
        sampler = SobolAdaptiveSampler(2, self._bkd)
        sampler.set_kernel(None)  # type: ignore[arg-type]

    def test_add_training_samples_noop(self) -> None:
        sampler = SobolAdaptiveSampler(2, self._bkd)
        dummy = self._bkd.zeros((2, 5))
        sampler.add_additional_training_samples(dummy)


class TestSobolSamplerNumpy(TestSobolSampler[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSobolSamplerTorch(TestSobolSampler[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
