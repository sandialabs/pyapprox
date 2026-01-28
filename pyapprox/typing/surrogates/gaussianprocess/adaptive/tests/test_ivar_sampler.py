"""Tests for IVAR adaptive sampler."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.surrogates.gaussianprocess.adaptive.ivar_sampler import (
    IVARSampler,
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


def _compute_P_monte_carlo(
    kernel: KernelProtocol[Array],
    candidates: Array,
    bkd: Backend[Array],
    nquad: int = 200,
) -> Array:
    """Compute P matrix via Monte Carlo for testing.

    P_ij = integral K(z, x_i) K(z, x_j) rho(z) dz
    Approximated with uniform samples on [0, 1]^nvars.
    """
    nvars = candidates.shape[0]
    np.random.seed(123)
    quad_pts = bkd.asarray(np.random.rand(nvars, nquad))
    # K(quad, candidates): (nquad, ncandidates)
    K_qc = kernel(quad_pts, candidates)
    # P_ij = (1/nquad) sum_m K(z_m, x_i) K(z_m, x_j)
    P = (K_qc.T @ K_qc) / nquad
    return P


class TestIVARSampler(Generic[Array], unittest.TestCase):
    """Base tests for IVARSampler."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _make_kernel(self) -> KernelProtocol[Array]:
        return SquaredExponentialKernel(
            [0.3], (0.01, 10.0), 1, self._bkd
        )

    def test_protocol_compliance(self) -> None:
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 20))
        sampler = IVARSampler(candidates, bkd)
        self.assertIsInstance(sampler, AdaptiveSamplerProtocol)

    def test_correct_shape(self) -> None:
        bkd = self._bkd
        nvars, ncandidates, nsamples = 1, 30, 5
        candidates = bkd.asarray(np.random.rand(nvars, ncandidates))
        kernel = self._make_kernel()
        sampler = IVARSampler(candidates, bkd)
        P = _compute_P_monte_carlo(kernel, candidates, bkd)
        sampler.set_P(P)
        sampler.set_kernel(kernel)
        samples = sampler.select_samples(nsamples)
        self.assertEqual(samples.shape, (nvars, nsamples))

    def test_samples_from_candidates(self) -> None:
        """Selected samples are a subset of candidates."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 30))
        kernel = self._make_kernel()
        sampler = IVARSampler(candidates, bkd)
        P = _compute_P_monte_carlo(kernel, candidates, bkd)
        sampler.set_P(P)
        sampler.set_kernel(kernel)
        samples = sampler.select_samples(5)
        cand_np = bkd.to_numpy(candidates)
        samp_np = bkd.to_numpy(samples)
        for j in range(samp_np.shape[1]):
            found = False
            for k in range(cand_np.shape[1]):
                if np.allclose(samp_np[:, j], cand_np[:, k]):
                    found = True
                    break
            self.assertTrue(found, f"Sample {j} not found in candidates")

    def test_no_repeat_selection(self) -> None:
        """Sequential calls do not select the same candidate twice."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 30))
        kernel = self._make_kernel()
        sampler = IVARSampler(candidates, bkd)
        P = _compute_P_monte_carlo(kernel, candidates, bkd)
        sampler.set_P(P)
        sampler.set_kernel(kernel)
        s1 = sampler.select_samples(3)
        s2 = sampler.select_samples(3)
        s1_np = bkd.to_numpy(s1)
        s2_np = bkd.to_numpy(s2)
        for j in range(s1_np.shape[1]):
            for k in range(s2_np.shape[1]):
                self.assertFalse(
                    np.allclose(s1_np[:, j], s2_np[:, k]),
                    "Duplicate sample selected",
                )

    def test_greedy_matches_brute_force(self) -> None:
        """First greedy selection matches brute-force on small problem."""
        bkd = self._bkd
        ncandidates = 8
        candidates = bkd.asarray(np.linspace(0, 1, ncandidates).reshape(1, -1))
        kernel = self._make_kernel()
        K = kernel(candidates, candidates)
        P = _compute_P_monte_carlo(kernel, candidates, bkd, nquad=500)

        # Greedy first point
        sampler = IVARSampler(candidates, bkd)
        sampler.set_P(P)
        sampler.set_kernel(kernel)
        greedy_sample = sampler.select_samples(1)
        greedy_idx = -1
        cand_np = bkd.to_numpy(candidates)
        greedy_np = bkd.to_numpy(greedy_sample)
        for k in range(ncandidates):
            if np.allclose(greedy_np[:, 0], cand_np[:, k]):
                greedy_idx = k
                break

        # Brute force: first point minimizes -P_ii / K_ii
        P_diag = bkd.to_numpy(bkd.diag(P))
        K_diag = bkd.to_numpy(bkd.diag(K))
        brute_idx = int(np.argmin(-P_diag / K_diag))

        self.assertEqual(greedy_idx, brute_idx)

    def test_partial_then_continue_matches_full(self) -> None:
        """Partial IVAR selection + continue matches full selection.

        Replicates legacy _check_greedy_ivar_sampling_update: selecting
        nsamples in one call gives the same sorted result as selecting
        nsamples//2 then the rest.
        """
        bkd = self._bkd
        np.random.seed(1)
        ncandidates = 20
        candidates = bkd.asarray(
            np.linspace(0, 1, ncandidates).reshape(1, -1)
        )
        kernel = self._make_kernel()
        P = _compute_P_monte_carlo(kernel, candidates, bkd, nquad=500)
        nsamples = 6

        # Full selection
        sampler1 = IVARSampler(candidates, bkd)
        sampler1.set_P(P)
        sampler1.set_kernel(kernel)
        full_samples = sampler1.select_samples(nsamples)

        # Partial then continue
        sampler2 = IVARSampler(candidates, bkd)
        sampler2.set_P(P)
        sampler2.set_kernel(kernel)
        first_half = sampler2.select_samples(nsamples // 2)
        second_half = sampler2.select_samples(nsamples - nsamples // 2)
        partial_samples = bkd.hstack([first_half, second_half])

        # Sort before comparing — legacy does this because near-equal
        # priorities can swap symmetric points
        bkd.assert_allclose(
            bkd.sort(bkd.flatten(partial_samples)),
            bkd.sort(bkd.flatten(full_samples)),
        )

    def test_candidate_exhaustion(self) -> None:
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 5))
        kernel = self._make_kernel()
        sampler = IVARSampler(candidates, bkd)
        P = _compute_P_monte_carlo(kernel, candidates, bkd)
        sampler.set_P(P)
        sampler.set_kernel(kernel)
        sampler.select_samples(5)
        with self.assertRaises(ValueError):
            sampler.select_samples(1)

    def test_no_kernel_raises(self) -> None:
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 10))
        sampler = IVARSampler(candidates, bkd)
        with self.assertRaises(ValueError):
            sampler.select_samples(3)

    def test_no_P_raises(self) -> None:
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 10))
        sampler = IVARSampler(candidates, bkd)
        sampler.set_kernel(self._make_kernel())
        with self.assertRaises(ValueError):
            sampler.select_samples(1)

    def test_warm_start_after_kernel_change(self) -> None:
        """After set_kernel, existing pivots are re-added."""
        bkd = self._bkd
        candidates = bkd.asarray(np.random.rand(1, 20))
        kernel1 = SquaredExponentialKernel([0.3], (0.01, 10.0), 1, bkd)
        sampler = IVARSampler(candidates, bkd)
        P = _compute_P_monte_carlo(kernel1, candidates, bkd)
        sampler.set_P(P)
        sampler.set_kernel(kernel1)
        sampler.select_samples(3)

        # Change kernel
        kernel2 = SquaredExponentialKernel([0.8], (0.01, 10.0), 1, bkd)
        P2 = _compute_P_monte_carlo(kernel2, candidates, bkd)
        sampler.set_P(P2)
        sampler.set_kernel(kernel2)
        s2 = sampler.select_samples(3)
        self.assertEqual(s2.shape, (1, 3))


class TestIVARSamplerNumpy(TestIVARSampler[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIVARSamplerTorch(TestIVARSampler[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
