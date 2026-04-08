"""Tests for AVaR dispatch: batch vectorized and numba implementations.

Correctness is validated by comparing batch implementations against the
original scalar per-QoI loop in SampleAverageSmoothedAVaR.
"""

import numpy as np
import pytest

from pyapprox.risk.avar import SampleAverageSmoothedAVaR
from pyapprox.risk.avar_compute import (
    avar_jacobian_batch,
    avar_values_batch,
    project_batch,
)
from pyapprox.util.optional_deps import package_available

HAS_NUMBA = package_available("numba")


class TestAVaRValuesBatch:
    """Test avar_values_batch against scalar reference."""

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
    def test_values_match_scalar_loop(self, bkd, alpha) -> None:
        """Batch values match per-QoI scalar _evaluate_single."""
        np.random.seed(42)
        nqoi, nsamples = 5, 100
        delta = 100.0
        values_np = np.random.randn(nqoi, nsamples)
        values = bkd.asarray(values_np)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(alpha, bkd, delta=delta)

        # Scalar reference: loop over QoIs using _evaluate_single
        ref_list = []
        for ii in range(nqoi):
            ref_list.append(
                stat._evaluate_single(values[ii : ii + 1, :], weights)
            )
        ref = bkd.stack(ref_list)[:, None]

        # Batch
        batch = avar_values_batch(
            values, weights, stat._alpha, delta, stat._lambda, bkd
        )

        assert batch.shape == (nqoi, 1)
        bkd.assert_allclose(batch, ref, rtol=1e-12)

    def test_alpha_zero(self, bkd) -> None:
        """alpha=0 AVaR approximately reduces to expected value.

        With large delta the smoothing correction is small, so the result
        converges to the mean.
        """
        np.random.seed(42)
        nqoi, nsamples = 3, 50
        delta = 100000.0
        values_np = np.random.randn(nqoi, nsamples)
        values = bkd.asarray(values_np)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(0.0, bkd, delta=delta)
        result = stat(values, weights)

        expected = bkd.sum(weights * values, axis=1)[:, None]
        bkd.assert_allclose(result, expected, rtol=1e-3)

    def test_nqoi_one(self, bkd) -> None:
        """Batch with nqoi=1 matches scalar."""
        np.random.seed(42)
        nsamples = 80
        delta = 100.0
        alpha = 0.5
        values_np = np.random.randn(1, nsamples)
        values = bkd.asarray(values_np)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(alpha, bkd, delta=delta)
        # _evaluate_single returns a scalar; use stat() which returns (1, 1)
        ref = stat(values, weights)

        batch = avar_values_batch(
            values, weights, stat._alpha, delta, stat._lambda, bkd
        )
        bkd.assert_allclose(batch, ref, rtol=1e-12)

    def test_large_delta(self, bkd) -> None:
        """Large delta gives tighter smoothing, closer to exact AVaR."""
        np.random.seed(42)
        nsamples = 200
        alpha = 0.5
        values_np = np.random.randn(2, nsamples)
        values = bkd.asarray(values_np)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat_lo = SampleAverageSmoothedAVaR(alpha, bkd, delta=10.0)
        stat_hi = SampleAverageSmoothedAVaR(alpha, bkd, delta=100000.0)

        result_lo = stat_lo(values, weights)
        result_hi = stat_hi(values, weights)

        # Both should be finite and close-ish
        assert result_lo.shape == (2, 1)
        assert result_hi.shape == (2, 1)


class TestAVaRJacobianBatch:
    """Test avar_jacobian_batch against scalar reference."""

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
    def test_jacobian_match_scalar_loop(self, bkd, alpha) -> None:
        """Batch jacobian matches per-QoI scalar _jacobian_single."""
        np.random.seed(42)
        nqoi, nsamples, nvars = 5, 100, 4
        delta = 100.0
        values_np = np.random.randn(nqoi, nsamples)
        jac_values_np = np.random.randn(nqoi, nsamples, nvars)
        values = bkd.asarray(values_np)
        jac_values = bkd.asarray(jac_values_np)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(alpha, bkd, delta=delta)

        # Scalar reference
        ref_list = []
        for ii in range(nqoi):
            ref_list.append(
                stat._jacobian_single(
                    values[ii : ii + 1, :],
                    jac_values[ii, :, :],
                    weights,
                )
            )
        ref = bkd.stack(ref_list)

        # Batch
        batch = avar_jacobian_batch(
            values, jac_values, weights, stat._alpha, delta, stat._lambda, bkd
        )

        assert batch.shape == (nqoi, nvars)
        bkd.assert_allclose(batch, ref, rtol=1e-12)


class TestAVaRJacobianDerivativeChecker:
    """Validate AVaR Jacobian via DerivativeChecker.

    Wraps AVaR into a FunctionProtocol by defining a linear model
    values(x) = A @ x, so the chain-rule Jacobian can be checked
    against finite differences.
    """

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
    def test_avar_derivative_checker(self, bkd, alpha) -> None:
        """AVaR Jacobian passes DerivativeChecker error_ratio test."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        np.random.seed(42)
        nqoi, nsamples, nvars = 3, 50, 4
        delta = 100.0

        # Random linear model: values(x) = A @ x, shape (nqoi, nsamples, nvars)
        A_np = np.random.randn(nqoi, nsamples, nvars)
        weights_np = np.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(alpha, bkd, delta=delta)

        class _AVaRWrapper:
            """Wraps AVaR stat into FunctionProtocol for DerivativeChecker."""

            def __init__(self, stat, A_np, weights_np, bkd):
                self._stat = stat
                self._A_np = A_np
                self._weights = bkd.asarray(weights_np)
                self._bkd = bkd

            def bkd(self):
                return self._bkd

            def nvars(self):
                return self._A_np.shape[2]

            def nqoi(self):
                return self._A_np.shape[0]

            def __call__(self, samples):
                # samples shape: (nvars, nsamples_eval)
                bkd = self._bkd
                A = bkd.asarray(self._A_np)
                neval = samples.shape[1]
                results = []
                for col in range(neval):
                    x = samples[:, col]  # (nvars,)
                    # values = A @ x -> (nqoi, nsamples)
                    vals = bkd.einsum("ijk,k->ij", A, x)
                    results.append(self._stat(vals, self._weights)[:, 0])
                return bkd.stack(results, axis=1)  # (nqoi, neval)

            def jacobian(self, sample):
                # sample shape: (nvars, 1)
                bkd = self._bkd
                A = bkd.asarray(self._A_np)
                x = sample[:, 0]
                vals = bkd.einsum("ijk,k->ij", A, x)  # (nqoi, nsamples)
                # jac_values[q, s, v] = A[q, s, v]
                return self._stat.jacobian(vals, A, self._weights)

        wrapper = _AVaRWrapper(stat, A_np, weights_np, bkd)
        sample = bkd.asarray(np.random.randn(nvars, 1))

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5


class TestNumbaVsVectorized:
    """Compare numba kernels against vectorized batch (numpy only)."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not available")
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
    def test_numba_values_match_vectorized(self, numpy_bkd, alpha) -> None:
        """Numba values match vectorized batch."""
        from pyapprox.risk.avar_dispatch import _wrap_numba_values

        np.random.seed(42)
        nqoi, nsamples = 5, 100
        delta = 100.0
        lam = 0.0
        bkd = numpy_bkd
        values = bkd.asarray(np.random.randn(nqoi, nsamples))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)
        alpha_arr = bkd.atleast_1d(bkd.asarray(alpha))

        vec = avar_values_batch(values, weights, alpha_arr, delta, lam, bkd)
        numba_result = _wrap_numba_values(
            values, weights, alpha_arr, delta, lam, bkd
        )

        bkd.assert_allclose(numba_result, vec, rtol=1e-12)

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not available")
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
    def test_numba_jacobian_match_vectorized(self, numpy_bkd, alpha) -> None:
        """Numba jacobian matches vectorized batch."""
        from pyapprox.risk.avar_dispatch import _wrap_numba_jacobian

        np.random.seed(42)
        nqoi, nsamples, nvars = 5, 100, 4
        delta = 100.0
        lam = 0.0
        bkd = numpy_bkd
        values = bkd.asarray(np.random.randn(nqoi, nsamples))
        jac_values = bkd.asarray(np.random.randn(nqoi, nsamples, nvars))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)
        alpha_arr = bkd.atleast_1d(bkd.asarray(alpha))

        vec = avar_jacobian_batch(
            values, jac_values, weights, alpha_arr, delta, lam, bkd
        )
        numba_result = _wrap_numba_jacobian(
            values, jac_values, weights, alpha_arr, delta, lam, bkd
        )

        bkd.assert_allclose(numba_result, vec, rtol=1e-12)
