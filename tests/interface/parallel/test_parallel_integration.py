"""Integration tests for parallel execution module.

Tests parallel execution with real functions and verifies
consistency between parallel and sequential execution.
"""

import numpy as np
import pytest

from pyapprox.util.optional_deps import package_available
from pyapprox_benchmarks.functions.algebraic.quadratic import QuadraticFunction
from pyapprox_benchmarks.functions.algebraic.rosenbrock import RosenbrockFunction
from tests._helpers.markers import slow_test

HAS_JOBLIB = package_available("joblib")
HAS_MPIRE = package_available("mpire")


class TestIntegration:
    """Integration tests for parallel execution."""

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_factory_jacobian_matches_sequential(self, bkd):
        """Test factory wrapper jacobians match sequential evaluation."""
        from pyapprox.interface.parallel import make_parallel

        func = QuadraticFunction(bkd, nvars=3)
        seq_func = make_parallel(func, backend="sequential")
        par_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )

        seq_jac = seq_func.jacobian_batch(samples)
        par_jac = par_func.jacobian_batch(samples)

        assert seq_jac.shape == par_jac.shape
        assert bkd.allclose(seq_jac, par_jac, rtol=1e-12)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_factory_hessian_matches_sequential(self, bkd):
        """Test factory wrapper hessians match sequential evaluation."""
        from pyapprox.interface.parallel import make_parallel

        func = QuadraticFunction(bkd, nvars=2)
        seq_func = make_parallel(func, backend="sequential")
        par_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        seq_hess = seq_func.hessian_batch(samples)
        par_hess = par_func.hessian_batch(samples)

        assert seq_hess.shape == par_hess.shape
        assert bkd.allclose(seq_hess, par_hess, rtol=1e-12)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_factory_hvp_matches_sequential(self, bkd):
        """Test factory wrapper HVPs match sequential evaluation."""
        from pyapprox.interface.parallel import make_parallel

        func = QuadraticFunction(bkd, nvars=2)
        seq_func = make_parallel(func, backend="sequential")
        par_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        vecs = bkd.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])

        seq_hvp = seq_func.hvp_batch(samples, vecs)
        par_hvp = par_func.hvp_batch(samples, vecs)

        assert seq_hvp.shape == par_hvp.shape
        assert bkd.allclose(seq_hvp, par_hvp, rtol=1e-12)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_rosenbrock_jacobian_correctness(self, bkd):
        """Test jacobian correctness using numerical finite differences."""
        from pyapprox.interface.parallel import make_parallel

        func = RosenbrockFunction(bkd)
        parallel_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray([[0.5, 1.0], [0.5, 1.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Numerical gradient check
        eps = 1e-6
        for i in range(2):  # nsamples
            sample_np = np.array(bkd.to_numpy(samples[:, i : i + 1]))
            for j in range(2):  # nvars
                sample_plus_np = sample_np.copy()
                sample_plus_np[j, 0] += eps
                sample_plus = bkd.asarray(sample_plus_np)

                sample_minus_np = sample_np.copy()
                sample_minus_np[j, 0] -= eps
                sample_minus = bkd.asarray(sample_minus_np)

                f_plus = float(func(sample_plus)[0, 0])
                f_minus = float(func(sample_minus)[0, 0])
                numerical_grad = (f_plus - f_minus) / (2 * eps)

                analytic_grad = float(jacobians[i, 0, j])
                assert numerical_grad == pytest.approx(analytic_grad, abs=1e-5)

    @slow_test
    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_backend_switching(self, bkd):
        """Test that different backends produce same results."""
        from pyapprox.interface.parallel import make_parallel

        func = QuadraticFunction(bkd, nvars=2)

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])

        joblib_func = make_parallel(func, backend="joblib_processes", n_jobs=2)
        joblib_jac = joblib_func.jacobian_batch(samples)

        futures_func = make_parallel(func, backend="futures", n_jobs=2)
        futures_jac = futures_func.jacobian_batch(samples)

        assert bkd.allclose(joblib_jac, futures_jac, rtol=1e-12)

        if HAS_MPIRE:
            mpire_func = make_parallel(func, backend="mpire", n_jobs=2)
            mpire_jac = mpire_func.jacobian_batch(samples)

            assert bkd.allclose(joblib_jac, mpire_jac, rtol=1e-12)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_parallel_config_from_module(self, bkd):
        """Test ParallelConfig can be imported and used."""
        from pyapprox.interface.parallel import (
            ParallelConfig,
            ParallelFunctionWrapper,
        )

        func = QuadraticFunction(bkd, nvars=2)
        config = ParallelConfig(backend="joblib_processes", n_jobs=2)
        wrapper = ParallelFunctionWrapper(func, config)

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        jacobians = wrapper.jacobian_batch(samples)

        assert jacobians.shape == (2, 1, 2)

    def test_batch_splitter_round_trip(self, bkd):
        """Test BatchSplitter correctly splits and recombines."""
        from pyapprox.interface.parallel import BatchSplitter

        splitter = BatchSplitter(bkd)
        original = bkd.asarray([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        chunks = splitter.split_samples(original, n_chunks=3)
        # Process each chunk (identity)
        recombined = splitter.combine_outputs(chunks, axis=1)

        assert bkd.allclose(original, recombined)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_parallel_call_matches_unwrapped(self, bkd):
        """Test parallel __call__ matches unwrapped function."""
        from pyapprox.interface.parallel import make_parallel

        func = QuadraticFunction(bkd, nvars=3)
        par_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )

        expected = func(samples)
        par_result = par_func(samples)

        assert expected.shape == par_result.shape
        bkd.assert_allclose(par_result, expected, rtol=1e-12)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_parallel_call_rosenbrock(self, bkd):
        """Test parallel __call__ for Rosenbrock function."""
        from pyapprox.interface.parallel import make_parallel

        func = RosenbrockFunction(bkd)
        par_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray([[0.5, 1.0, -0.5], [0.5, 1.0, -0.5]])
        par_result = par_func(samples)
        seq_result = func(samples)

        bkd.assert_allclose(par_result, seq_result, rtol=1e-12)
