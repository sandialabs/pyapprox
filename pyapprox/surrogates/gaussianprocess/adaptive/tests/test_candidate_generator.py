"""Tests for candidate generator."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.surrogates.gaussianprocess.adaptive.candidate_generator import (
    HybridSobolRandomCandidateGenerator,
)
from pyapprox.surrogates.gaussianprocess.adaptive.protocols import (
    CandidateGeneratorProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestCandidateGenerator(Generic[Array], unittest.TestCase):
    """Base tests for HybridSobolRandomCandidateGenerator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        gen = HybridSobolRandomCandidateGenerator(2, self._bkd)
        self.assertIsInstance(gen, CandidateGeneratorProtocol)

    def test_shape(self) -> None:
        nvars, ncandidates = 3, 100
        gen = HybridSobolRandomCandidateGenerator(nvars, self._bkd)
        candidates = gen.generate(ncandidates)
        self.assertEqual(candidates.shape, (nvars, ncandidates))

    def test_default_bounds(self) -> None:
        gen = HybridSobolRandomCandidateGenerator(2, self._bkd)
        candidates = gen.generate(200)
        bkd = self._bkd
        self.assertTrue(bkd.all_bool(candidates >= 0.0))
        self.assertTrue(bkd.all_bool(candidates <= 1.0))

    def test_custom_bounds(self) -> None:
        bkd = self._bkd
        bounds = bkd.asarray([[-1.0, 1.0], [0.0, 2.0]])
        gen = HybridSobolRandomCandidateGenerator(2, bkd, scaled_bounds=bounds)
        candidates = gen.generate(200)
        self.assertTrue(bkd.all_bool(candidates[0, :] >= -1.0))
        self.assertTrue(bkd.all_bool(candidates[0, :] <= 1.0))
        self.assertTrue(bkd.all_bool(candidates[1, :] >= 0.0))
        self.assertTrue(bkd.all_bool(candidates[1, :] <= 2.0))

    def test_reproducibility(self) -> None:
        gen1 = HybridSobolRandomCandidateGenerator(2, self._bkd, seed=7)
        gen2 = HybridSobolRandomCandidateGenerator(2, self._bkd, seed=7)
        c1 = gen1.generate(50)
        c2 = gen2.generate(50)
        self._bkd.assert_allclose(c1, c2)

    def test_invalid_bounds_shape(self) -> None:
        bounds = self._bkd.asarray([[0.0, 1.0]])
        with self.assertRaises(ValueError):
            HybridSobolRandomCandidateGenerator(2, self._bkd, scaled_bounds=bounds)


class TestCandidateGeneratorNumpy(TestCandidateGenerator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCandidateGeneratorTorch(TestCandidateGenerator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
