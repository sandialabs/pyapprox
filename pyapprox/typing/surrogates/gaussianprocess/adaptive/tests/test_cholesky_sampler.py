"""Tests for Cholesky adaptive sampler."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.surrogates.gaussianprocess.adaptive.cholesky_sampler import (
    CholeskySampler,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.protocols import (
    AdaptiveSamplerProtocol,
)
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestCholeskySampler(Generic[Array], unittest.TestCase):
    """Base tests for CholeskySampler."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _make_kernel(self) -> KernelProtocol[Array]:
        return SquaredExponentialKernel(
            [0.5], (0.01, 10.0), 1, self._bkd
        )

    def test_protocol_compliance(self) -> None:
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 20))
        sampler = CholeskySampler(candidates, bkd)
        self.assertIsInstance(sampler, AdaptiveSamplerProtocol)

    def test_correct_shape(self) -> None:
        bkd = self._bkd
        nvars, ncandidates, nsamples = 2, 50, 5
        candidates = bkd.asarray(np.random.rand(nvars, ncandidates))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(self._make_kernel())
        samples = sampler.select_samples(nsamples)
        self.assertEqual(samples.shape, (nvars, nsamples))

    def test_samples_from_candidates(self) -> None:
        """Selected samples are a subset of candidates."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 30))
        sampler = CholeskySampler(candidates, bkd)
        kernel = SquaredExponentialKernel([0.3], (0.01, 10.0), 1, bkd)
        sampler.set_kernel(kernel)
        samples = sampler.select_samples(5)
        # Each selected sample must appear in candidates
        cand_np = bkd.to_numpy(candidates)
        samp_np = bkd.to_numpy(samples)
        for j in range(samp_np.shape[1]):
            found = False
            for k in range(cand_np.shape[1]):
                if np.allclose(samp_np[:, j], cand_np[:, k]):
                    found = True
                    break
            self.assertTrue(found, f"Sample {j} not found in candidates")

    def test_sequential_selection(self) -> None:
        """Multiple calls accumulate without repeats."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 30))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(self._make_kernel())
        s1 = sampler.select_samples(3)
        s2 = sampler.select_samples(3)
        # No overlap
        s1_np = bkd.to_numpy(s1)
        s2_np = bkd.to_numpy(s2)
        for j in range(s1_np.shape[1]):
            for k in range(s2_np.shape[1]):
                self.assertFalse(
                    np.allclose(s1_np[:, j], s2_np[:, k]),
                    "Duplicate sample selected",
                )

    def test_warm_start_after_kernel_change(self) -> None:
        """After set_kernel, existing pivots are used as init_pivots."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 30))
        sampler = CholeskySampler(candidates, bkd)
        kernel1 = SquaredExponentialKernel([0.3], (0.01, 10.0), 1, bkd)
        sampler.set_kernel(kernel1)
        sampler.select_samples(3)

        # Change kernel — should warm-start with existing 3 pivots
        kernel2 = SquaredExponentialKernel([0.8], (0.01, 10.0), 1, bkd)
        sampler.set_kernel(kernel2)
        # Should be able to select more samples
        s2 = sampler.select_samples(3)
        self.assertEqual(s2.shape, (1, 3))

    def test_partial_then_continue_matches_full(self) -> None:
        """Partial selection + continue matches full selection.

        Replicates legacy test_cholesky_sampling_update: selecting
        nsamples in one call gives the same result as selecting
        nsamples//2 then the rest.
        """
        bkd = self._bkd
        np.random.seed(1)
        ncandidates = 50
        candidates = bkd.asarray(np.random.rand(1, ncandidates))
        nsamples = 10

        # Full selection
        kernel = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)
        sampler1 = CholeskySampler(candidates, bkd)
        sampler1.set_kernel(kernel)
        full_samples = sampler1.select_samples(nsamples)

        # Partial then continue
        sampler2 = CholeskySampler(candidates, bkd)
        sampler2.set_kernel(kernel)
        first_half = sampler2.select_samples(nsamples // 2)
        second_half = sampler2.select_samples(nsamples - nsamples // 2)
        partial_samples = bkd.hstack([first_half, second_half])

        bkd.assert_allclose(partial_samples, full_samples)

    def test_kernel_change_preserves_first_half(self) -> None:
        """After kernel change, first-half samples are preserved.

        Replicates legacy test_cholesky_sampler_update_with_changed_kernel:
        first nsamples//2 match, but after kernel change the remaining
        samples differ.
        """
        bkd = self._bkd
        np.random.seed(1)
        ncandidates = 50
        candidates = bkd.asarray(np.random.rand(1, ncandidates))
        nsamples = 10

        # Full with kernel1
        kernel1 = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)
        sampler1 = CholeskySampler(candidates, bkd)
        sampler1.set_kernel(kernel1)
        full_samples = sampler1.select_samples(nsamples)

        # Partial with kernel1, then switch to kernel2
        kernel2 = SquaredExponentialKernel([0.3], (0.01, 10.0), 1, bkd)
        sampler2 = CholeskySampler(candidates, bkd)
        sampler2.set_kernel(kernel1)
        first_half = sampler2.select_samples(nsamples // 2)
        sampler2.set_kernel(kernel2)
        second_half = sampler2.select_samples(nsamples - nsamples // 2)
        changed_samples = bkd.hstack([first_half, second_half])

        # First half matches
        bkd.assert_allclose(
            changed_samples[:, :nsamples // 2],
            full_samples[:, :nsamples // 2],
        )
        # Full result differs due to kernel change
        diff = bkd.sum((changed_samples - full_samples) ** 2)
        self.assertGreater(
            float(bkd.to_numpy(bkd.reshape(diff, (1,)))[0]), 0.0
        )

    def test_candidate_exhaustion(self) -> None:
        """Selecting more than available raises ValueError."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 5))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(self._make_kernel())
        sampler.select_samples(5)
        with self.assertRaises(ValueError):
            sampler.select_samples(1)

    def test_no_kernel_raises(self) -> None:
        """select_samples before set_kernel raises ValueError."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 10))
        sampler = CholeskySampler(candidates, bkd)
        with self.assertRaises(ValueError):
            sampler.select_samples(3)


class TestCholeskySamplerNumpy(TestCholeskySampler[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCholeskySamplerTorch(TestCholeskySampler[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
