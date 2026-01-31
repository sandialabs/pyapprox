"""Tests for covariance structure computations.

Replicates legacy tests from pyapprox/multifidelity/tests/test_stats.py:
- test_covariance_einsum
- test_variance_double_loop
- test_pilot_covariances
- test_mean_variance_covariances
"""

import unittest
from functools import partial
from typing import Any, Generic, List

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests, slower_test  # noqa: F401
from pyapprox.typing.stats.statistics import (
    MultiOutputMeanAndVariance,
    compute_W_from_pilot,
    compute_B_from_pilot,
    compute_V_from_covariance,
    covariance_of_variance_estimator,
    extract_nqoisq_nqoisq_subproblem,
    extract_nqoi_nqoisq_subproblem,
)
from pyapprox.benchmarks.multifidelity_benchmarks import (
    MultiOutputModelEnsembleBenchmark,
)


def _single_qoi(qoi, fun, xx):
    """Extract single QoI from function output."""
    return fun(xx)[:, qoi : qoi + 1]


def _two_qoi(ii, jj, fun, xx):
    """Extract two QoIs from function output."""
    return fun(xx)[:, [ii, jj]]


def _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd):
    """Set up benchmark and extract subproblem data."""
    # Use legacy benchmark (NumpyMixin-based)
    from pyapprox.util.backends.numpy import NumpyMixin
    benchmark = MultiOutputModelEnsembleBenchmark(backend=NumpyMixin)
    cov = benchmark.covariance()
    funs = [benchmark.models()[ii] for ii in model_idx]
    if len(qoi_idx) == 1:
        funs = [partial(_single_qoi, qoi_idx[0], f) for f in funs]
    elif len(qoi_idx) == 2:
        funs = [partial(_two_qoi, *qoi_idx, f) for f in funs]
    idx = np.arange(9).reshape(3, 3)[np.ix_(model_idx, qoi_idx)].flatten()
    cov = cov[np.ix_(idx, idx)]
    costs = benchmark.costs()[model_idx]
    means = benchmark.mean()[np.ix_(model_idx, qoi_idx)]
    return funs, cov, costs, benchmark, means


class TestCovarianceStructure(Generic[Array], unittest.TestCase):
    """Tests for covariance structure computations."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(1)
        if hasattr(torch, 'set_default_dtype'):
            torch.set_default_dtype(torch.float64)

    def test_covariance_einsum(self):
        """Test MC covariance computation via einsum matches benchmark.

        Replicates legacy test_covariance_einsum.
        """
        bkd = self._bkd
        from pyapprox.util.backends.numpy import NumpyMixin
        benchmark = MultiOutputModelEnsembleBenchmark(backend=NumpyMixin)
        npilot_samples = int(1e5)
        np.random.seed(42)
        pilot_samples = benchmark.prior().rvs(npilot_samples)
        pilot_values_np = np.hstack(
            [f(pilot_samples) for f in benchmark.models()]
        )
        pilot_values = bkd.asarray(pilot_values_np)

        means = bkd.sum(pilot_values, axis=0) / npilot_samples
        centered_values = pilot_values - means
        centered_values_sq = bkd.einsum(
            "nk,nl->nkl", centered_values, centered_values
        ).reshape(npilot_samples, -1)
        nqoi = pilot_values.shape[1]
        mc_cov = (
            bkd.sum(centered_values_sq, axis=0) / (npilot_samples - 1)
        ).reshape(nqoi, nqoi)

        benchmark_cov = bkd.asarray(benchmark.covariance())
        # MC estimation has variance - use rtol=2e-2 for reliability
        bkd.assert_allclose(mc_cov, benchmark_cov, rtol=2e-2)

    def test_variance_double_loop(self):
        """Test variance double loop formula equals standard formula.

        Replicates legacy test_variance_double_loop.
        The identity: sum((x_i - x_j)^2) / (2*N*(N-1)) = sum((x - mean)^2) / (N-1)
        """
        bkd = self._bkd
        NN = 5
        from pyapprox.util.backends.numpy import NumpyMixin
        benchmark = MultiOutputModelEnsembleBenchmark(backend=NumpyMixin)
        np.random.seed(42)
        samples_np = benchmark.prior().rvs(NN)
        samples = bkd.asarray(samples_np)

        # Double loop variance
        variance = bkd.zeros(samples.shape[0])
        for ii in range(samples.shape[1]):
            for jj in range(samples.shape[1]):
                variance = variance + (samples[:, ii] - samples[:, jj]) ** 2
        variance = variance / (2 * NN * (NN - 1))

        # Standard centered variance
        mean = bkd.sum(samples, axis=1) / NN
        variance1 = bkd.sum((samples.T - mean).T ** 2, axis=1) / (NN - 1)

        # Backend var
        variance2 = bkd.var(samples, axis=1, ddof=1)

        bkd.assert_allclose(variance, variance1)
        bkd.assert_allclose(variance, variance2)

    def _check_pilot_covariances(self, model_idx, qoi_idx):
        """Check pilot covariances against benchmark exact values."""
        bkd = self._bkd
        funs, cov_exact, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )
        # atol is needed for terms close to zero
        rtol, atol = 1e-2, 5.5e-4
        npilot_samples = int(1e6)
        np.random.seed(123)
        pilot_samples = benchmark.prior().rvs(npilot_samples)

        # Get pilot values
        pilot_values_np = [f(pilot_samples) for f in funs]
        pilot_values = [bkd.asarray(pv) for pv in pilot_values_np]

        # Use the new method
        stat = MultiOutputMeanAndVariance(len(qoi_idx), bkd)
        cov, W, B = stat.compute_pilot_quantities_with_covariance_matrices(
            pilot_values
        )

        # Check covariance
        cov_exact_arr = bkd.asarray(cov_exact)
        bkd.assert_allclose(cov, cov_exact_arr, atol=atol, rtol=rtol)

        # Check W matrix
        from pyapprox.util.backends.numpy import NumpyMixin
        W_exact_np = benchmark.covariance_of_centered_values_kronker_product()
        W_exact_sub = extract_nqoisq_nqoisq_subproblem(
            bkd.asarray(W_exact_np),
            benchmark.nmodels,
            benchmark.nqoi(),
            model_idx,
            qoi_idx,
            bkd,
        )
        bkd.assert_allclose(W, W_exact_sub, atol=atol, rtol=rtol)

        # Check B matrix
        B_exact_np = benchmark.covariance_of_mean_and_variance_estimators()
        B_exact_sub = extract_nqoi_nqoisq_subproblem(
            bkd.asarray(B_exact_np),
            benchmark.nmodels,
            benchmark.nqoi(),
            model_idx,
            qoi_idx,
            bkd,
        )
        bkd.assert_allclose(B, B_exact_sub, atol=atol, rtol=rtol)

    @slower_test
    def test_pilot_covariances(self):
        """Test pilot covariances for various model/QoI subsets."""
        test_cases = [
            [[0], [0]],
            [[1], [0, 1]],
            [[1], [0, 2]],
            [[1, 2], [0]],
            [[0, 1], [0, 2]],
            [[0, 1], [0, 1, 2]],
            [[0, 1, 2], [0]],
            [[0, 1, 2], [0, 2]],
            [[0, 1, 2], [0, 1, 2]],
        ]
        for model_idx, qoi_idx in test_cases:
            with self.subTest(model_idx=model_idx, qoi_idx=qoi_idx):
                np.random.seed(123)
                self._check_pilot_covariances(model_idx, qoi_idx)

    def _mean_variance_realizations(self, funs, variable, nsamples, ntrials):
        """Generate empirical mean and covariance realizations."""
        bkd = self._bkd
        nmodels = len(funs)
        means = []
        covariances = []
        for ii in range(ntrials):
            samples = variable.rvs(nsamples)
            vals_np = np.hstack([f(samples) for f in funs])
            vals = bkd.asarray(vals_np)
            nqoi = vals.shape[1] // nmodels
            means.append(bkd.sum(vals, axis=0) / nsamples)

            # Compute covariance for each model
            cov_parts = []
            for jj in range(nmodels):
                model_vals = vals[:, jj * nqoi : (jj + 1) * nqoi]
                mean_jj = bkd.sum(model_vals, axis=0) / nsamples
                centered = model_vals - mean_jj
                cov_jj = centered.T @ centered / (nsamples - 1)
                cov_parts.append(cov_jj.flatten())
            covariances.append(bkd.concatenate(cov_parts))

        means = bkd.stack(means).T
        covariances = bkd.stack(covariances).T
        return means, covariances

    def _check_mean_variance_covariances(self, model_idx, qoi_idx):
        """Check mean/variance estimator covariances via MC."""
        bkd = self._bkd
        nsamples, ntrials = 20, int(1e5)
        funs, cov, costs, benchmark, means_exact = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )
        np.random.seed(123)
        means, covariances = self._mean_variance_realizations(
            funs, benchmark.prior(), nsamples, ntrials
        )
        nmodels = len(funs)
        nqoi = cov.shape[0] // nmodels

        # atol is needed for terms close to zero
        rtol, atol = 1e-2, 1e-4

        # Check B matrix (cross-covariance of mean and variance estimators)
        from pyapprox.util.backends.numpy import NumpyMixin
        B_exact_np = benchmark.covariance_of_mean_and_variance_estimators()
        B_exact = extract_nqoi_nqoisq_subproblem(
            bkd.asarray(B_exact_np),
            benchmark.nmodels,
            benchmark.nqoi(),
            model_idx,
            qoi_idx,
            bkd,
        )

        # MC cross-covariance
        stacked = bkd.vstack([means, covariances])
        stacked_mean = bkd.sum(stacked, axis=1) / ntrials
        stacked_centered = (stacked.T - stacked_mean).T
        mc_mean_cov_var = stacked_centered @ stacked_centered.T / (ntrials - 1)
        B_mc = mc_mean_cov_var[: nqoi * nmodels, nqoi * nmodels :]

        bkd.assert_allclose(B_mc, B_exact / nsamples, atol=atol, rtol=rtol)

        # Check variance of variance estimator: W/n + V/(n*(n-1))
        cov_arr = bkd.asarray(cov)
        V_exact = compute_V_from_covariance(cov_arr, nmodels, bkd)
        W_exact_np = benchmark.covariance_of_centered_values_kronker_product()
        W_exact = extract_nqoisq_nqoisq_subproblem(
            bkd.asarray(W_exact_np),
            benchmark.nmodels,
            benchmark.nqoi(),
            model_idx,
            qoi_idx,
            bkd,
        )
        cov_var_exact = covariance_of_variance_estimator(W_exact, V_exact, nsamples)
        mc_cov_var = mc_mean_cov_var[nqoi * nmodels :, nqoi * nmodels :]

        bkd.assert_allclose(cov_var_exact, mc_cov_var, atol=atol, rtol=rtol)

    @slower_test
    def test_mean_variance_covariances(self):
        """Test mean/variance estimator covariance formulas."""
        test_cases = [
            [[0], [0]],
            [[1], [0, 1]],
            [[1], [0, 2]],
            [[1, 2], [0]],
            [[0, 1], [0, 2]],
            [[0, 1], [0, 1, 2]],
            [[0, 1, 2], [0]],
            [[0, 1, 2], [0, 1]],
            [[0, 1, 2], [0, 1, 2]],
        ]
        for model_idx, qoi_idx in test_cases:
            with self.subTest(model_idx=model_idx, qoi_idx=qoi_idx):
                np.random.seed(123)
                self._check_mean_variance_covariances(model_idx, qoi_idx)


class TestCovarianceStructureNumpy(TestCovarianceStructure[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCovarianceStructureTorch(TestCovarianceStructure[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
