"""Tests for Cholesky adaptive sampler."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.surrogates.gaussianprocess.adaptive.cholesky_sampler import (
    CholeskySampler,
)
from pyapprox.surrogates.gaussianprocess.adaptive.protocols import (
    AdaptiveSamplerProtocol,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


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

    def _make_weight_function(
        self, fun: Any, nvars: int,
    ) -> FunctionFromCallable[Array]:
        """Wrap a callable as FunctionProtocol with nqoi=1."""
        return FunctionFromCallable(1, nvars, fun, self._bkd)

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

    # --- Weight function tests ---

    def test_weight_function_change_triggers_refactorization(self) -> None:
        """Changing weight function causes samples to diverge.

        Replicates legacy test_cholesky_sampler_update_with_changed_weight_function.
        Two samplers start with uniform weights. Sampler 1 selects 10
        samples with uniform weights. Sampler 2 selects 5 with uniform,
        then changes to x^2 weighting, then selects 5 more. First half
        matches, full result diverges.
        """
        bkd = self._bkd
        np.random.seed(1)
        ncandidates = 50
        candidates = bkd.asarray(np.random.rand(1, ncandidates))
        kernel = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)
        nsamples = 10

        uniform_wf = self._make_weight_function(
            lambda x: bkd.ones((1, x.shape[1])), 1,
        )
        quadratic_wf = self._make_weight_function(
            lambda x: bkd.reshape(x[0, :] ** 2, (1, -1)), 1,
        )

        # Sampler 1: uniform weights throughout
        sampler1 = CholeskySampler(candidates, bkd)
        sampler1.set_weight_function(uniform_wf)
        sampler1.set_kernel(kernel)
        full_samples = sampler1.select_samples(nsamples)

        # Sampler 2: uniform for first 5, then switch to quadratic
        sampler2 = CholeskySampler(candidates, bkd)
        sampler2.set_weight_function(uniform_wf)
        sampler2.set_kernel(kernel)
        first_half = sampler2.select_samples(nsamples // 2)
        sampler2.set_weight_function(quadratic_wf)
        second_half = sampler2.select_samples(nsamples - nsamples // 2)
        changed_samples = bkd.hstack([first_half, second_half])

        # First half should match
        bkd.assert_allclose(
            changed_samples[:, :nsamples // 2],
            full_samples[:, :nsamples // 2],
        )
        # Full result should differ after weight change
        diff = bkd.sum((changed_samples - full_samples) ** 2)
        self.assertGreater(
            float(bkd.to_numpy(bkd.reshape(diff, (1,)))[0]), 0.0,
        )

    def test_weight_function_biases_selection(self) -> None:
        """Weight function concentrates samples in high-weight region."""
        bkd = self._bkd
        np.random.seed(42)
        ncandidates = 100
        candidates = bkd.asarray(
            np.linspace(0.0, 1.0, ncandidates).reshape(1, -1)
        )
        kernel = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)
        nsamples = 10

        # Unweighted sampler
        sampler1 = CholeskySampler(candidates, bkd)
        sampler1.set_kernel(kernel)
        s_unweighted = sampler1.select_samples(nsamples)

        # Weighted: strongly prefer right half (x > 0.5)
        right_bias_wf = self._make_weight_function(
            lambda x: bkd.reshape(
                bkd.where(
                    x[0] > 0.5,
                    100.0 * bkd.ones(x.shape[1]),
                    bkd.ones(x.shape[1]),
                ),
                (1, -1),
            ),
            1,
        )
        sampler2 = CholeskySampler(candidates, bkd)
        sampler2.set_weight_function(right_bias_wf)
        sampler2.set_kernel(kernel)
        s_weighted = sampler2.select_samples(nsamples)

        # Count samples in right half
        s_unweighted_np = bkd.to_numpy(s_unweighted)
        s_weighted_np = bkd.to_numpy(s_weighted)
        n_right_unweighted = int(np.sum(s_unweighted_np[0] > 0.5))
        n_right_weighted = int(np.sum(s_weighted_np[0] > 0.5))
        self.assertGreater(n_right_weighted, n_right_unweighted)

    def test_weight_function_none_reverts_to_uniform(self) -> None:
        """Setting weight_function=None reverts to unweighted selection."""
        bkd = self._bkd
        np.random.seed(1)
        candidates = bkd.asarray(np.random.rand(1, 50))
        kernel = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)

        # Unweighted
        sampler1 = CholeskySampler(candidates, bkd)
        sampler1.set_kernel(kernel)
        s1 = sampler1.select_samples(5)

        # Uniform weight function (equivalent to no weights)
        uniform_wf = self._make_weight_function(
            lambda x: bkd.ones((1, x.shape[1])), 1,
        )
        sampler2 = CholeskySampler(candidates, bkd)
        sampler2.set_weight_function(uniform_wf)
        sampler2.set_kernel(kernel)
        s2 = sampler2.select_samples(5)

        bkd.assert_allclose(s1, s2)

    def test_weight_function_invalid_shape_raises(self) -> None:
        """Weight function returning wrong shape raises ValueError."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 20))
        kernel = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)

        # Returns (ncandidates,) instead of (1, ncandidates)
        bad_wf = self._make_weight_function(
            lambda x: bkd.ones(x.shape[1]), 1,
        )
        sampler = CholeskySampler(candidates, bkd)
        with self.assertRaises(ValueError):
            sampler.set_weight_function(bad_wf)

    def test_weight_function_set_before_kernel(self) -> None:
        """Weight function can be set before set_kernel."""
        bkd = self._bkd
        np.random.seed(1)
        candidates = bkd.asarray(np.random.rand(1, 30))
        kernel = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)

        wf = self._make_weight_function(
            lambda x: bkd.reshape(x[0, :] ** 2, (1, -1)), 1,
        )
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_weight_function(wf)
        sampler.set_kernel(kernel)
        samples = sampler.select_samples(5)
        self.assertEqual(samples.shape, (1, 5))

    def test_partial_then_weight_change_then_continue(self) -> None:
        """Select some, change weights, select more — no duplicates."""
        bkd = self._bkd
        np.random.seed(42)
        candidates = bkd.asarray(np.random.rand(1, 50))
        kernel = SquaredExponentialKernel([0.1], (0.01, 10.0), 1, bkd)

        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(kernel)

        # Select 3 unweighted
        s1 = sampler.select_samples(3)
        self.assertEqual(s1.shape, (1, 3))

        # Change to weighted
        wf = self._make_weight_function(
            lambda x: bkd.reshape(x[0, :] ** 2, (1, -1)), 1,
        )
        sampler.set_weight_function(wf)
        # Select 3 more — uses weighted selection, preserves first 3
        s2 = sampler.select_samples(3)
        self.assertEqual(s2.shape, (1, 3))

        # No overlap between batches
        s1_np = bkd.to_numpy(s1)
        s2_np = bkd.to_numpy(s2)
        for j in range(s1_np.shape[1]):
            for k in range(s2_np.shape[1]):
                self.assertFalse(
                    np.allclose(s1_np[:, j], s2_np[:, k]),
                    "Duplicate sample selected after weight change",
                )


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
