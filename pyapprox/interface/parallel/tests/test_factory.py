"""Tests for parallel function factory."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class MockFunction(Generic[Array]):
    """Mock function for testing parallel wrapper."""

    def __init__(self, bkd: Backend[Array], nvars: int = 2, nqoi: int = 1):
        self._bkd = bkd
        self._nvars = nvars
        self._nqoi = nqoi

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        # Simple quadratic: sum of squares
        return self._bkd.sum(samples**2, axis=0, keepdims=True)

    def jacobian(self, sample: Array) -> Array:
        # Jacobian of sum of squares: 2*x
        return 2 * sample.T  # (nqoi, nvars) = (1, nvars)

    def hessian(self, sample: Array) -> Array:
        # Hessian of sum of squares: 2*I
        return 2 * self._bkd.eye(self._nvars)

    def hvp(self, sample: Array, vec: Array) -> Array:
        # HVP of sum of squares: 2*v
        return 2 * vec

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        # Weighted HVP: weights * 2 * v
        return weights[0, 0] * 2 * vec


class MockMultiOutputFunction(Generic[Array]):
    """Mock multi-output function for testing."""

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 2

    def __call__(self, samples: Array) -> Array:
        x = samples[0:1, :]
        y = samples[1:2, :]
        return self._bkd.vstack([x**2, y**2])

    def jacobian(self, sample: Array) -> Array:
        x = sample[0, 0]
        y = sample[1, 0]
        jac = self._bkd.asarray([[2 * x, 0.0], [0.0, 2 * y]])
        return jac

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        # Each output has hessian [[2, 0], [0, 0]] or [[0, 0], [0, 2]]
        w0 = weights[0, 0]
        w1 = weights[1, 0]
        # weighted hessian = w0 * [[2,0],[0,0]] + w1 * [[0,0],[0,2]]
        #                  = [[2*w0, 0], [0, 2*w1]]
        result = self._bkd.asarray([[2 * w0 * vec[0, 0]], [2 * w1 * vec[1, 0]]])
        return result


class TestParallelFunctionWrapper:
    """Tests for ParallelFunctionWrapper."""

    def test_basic_call(self, bkd):
        """Test that wrapped function still evaluates correctly."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        result = parallel_func(samples)

        expected = func(samples)
        assert bkd.allclose(result, expected)

    def test_jacobian_batch(self, bkd):
        """Test parallel jacobian batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Shape should be (nsamples, nqoi, nvars)
        assert jacobians.shape == (3, 1, 2)

        # Verify each jacobian
        for i in range(3):
            sample = samples[:, i : i + 1]
            expected = func.jacobian(sample)
            assert bkd.allclose(jacobians[i], expected, rtol=1e-10)

    def test_hessian_batch(self, bkd):
        """Test parallel hessian batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        hessians = parallel_func.hessian_batch(samples)

        # Shape should be (nsamples, nvars, nvars)
        assert hessians.shape == (2, 2, 2)

        # Each hessian should be 2*I
        expected = 2 * bkd.eye(2)
        for i in range(2):
            assert bkd.allclose(hessians[i], expected)

    def test_hvp_batch(self, bkd):
        """Test parallel HVP batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        vecs = bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        hvps = parallel_func.hvp_batch(samples, vecs)

        # Shape should be (nsamples, nvars)
        assert hvps.shape == (2, 2)

        # HVP is 2*v
        expected = bkd.asarray([[2.0, 0.0], [0.0, 2.0]])
        assert bkd.allclose(hvps, expected)

    def test_whvp_batch(self, bkd):
        """Test parallel weighted HVP batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        vecs = bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        weights = bkd.asarray([[0.5]])

        whvps = parallel_func.whvp_batch(samples, vecs, weights)

        # Shape should be (nsamples, nvars)
        assert whvps.shape == (2, 2)

        # WHVP is weights * 2 * v = 0.5 * 2 * v = v
        expected = bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        assert bkd.allclose(whvps, expected)

    def test_multi_output_jacobian_batch(self, bkd):
        """Test jacobian batch for multi-output function."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockMultiOutputFunction(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Shape should be (nsamples, nqoi, nvars)
        assert jacobians.shape == (2, 2, 2)

    def test_hasattr_detection(self, bkd):
        """Test that only available methods are wrapped."""
        from pyapprox.interface.parallel.factory import make_parallel

        # Create function without hvp
        class FuncWithoutHVP(Generic[Array]):
            def __init__(self, bkd):
                self._bkd = bkd

            def bkd(self):
                return self._bkd

            def nvars(self):
                return 2

            def nqoi(self):
                return 1

            def __call__(self, samples):
                return self._bkd.sum(samples**2, axis=0, keepdims=True)

            def jacobian(self, sample):
                return 2 * sample.T

        func = FuncWithoutHVP(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        assert hasattr(parallel_func, "jacobian_batch")
        assert not hasattr(parallel_func, "hvp_batch")
        assert not hasattr(parallel_func, "whvp_batch")

    def test_backend_info(self, bkd):
        """Test backend information methods."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)

        seq_func = make_parallel(func, backend="sequential")
        assert seq_func.parallel_backend() == "sequential"

        joblib_func = make_parallel(func, backend="joblib_processes", n_jobs=4)
        assert "joblib" in joblib_func.parallel_backend()
        assert joblib_func.n_workers() == 4

    def test_parallel_execution_joblib(self, bkd):
        """Test actual parallel execution with joblib."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Verify results match sequential
        seq_func = make_parallel(func, backend="sequential")
        seq_jacobians = seq_func.jacobian_batch(samples)

        assert bkd.allclose(jacobians, seq_jacobians)

    def test_parallel_execution_mpire(self, bkd):
        """Test actual parallel execution with mpire."""
        try:
            import mpire  # noqa: F401
        except ImportError:
            pytest.skip("mpire not installed")

        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="mpire", n_jobs=2)

        samples = bkd.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Verify results match sequential
        seq_func = make_parallel(func, backend="sequential")
        seq_jacobians = seq_func.jacobian_batch(samples)

        assert bkd.allclose(jacobians, seq_jacobians)

    def test_parallel_call_matches_sequential(self, bkd):
        """Test that parallel __call__ matches sequential __call__."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        )

        par_result = parallel_func(samples)
        seq_result = func(samples)

        assert par_result.shape == seq_result.shape
        bkd.assert_allclose(par_result, seq_result, rtol=1e-12)

    def test_parallel_call_single_sample(self, bkd):
        """Test that single-sample call skips parallelism."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="joblib_processes", n_jobs=4)

        sample = bkd.asarray([[1.0], [2.0]])
        result = parallel_func(sample)
        expected = func(sample)

        bkd.assert_allclose(result, expected)

    def test_parallel_call_sequential_backend(self, bkd):
        """Test __call__ with sequential backend delegates directly."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = parallel_func(samples)
        expected = func(samples)

        bkd.assert_allclose(result, expected)

    def test_parallel_call_multi_output(self, bkd):
        """Test parallel __call__ with multi-output function."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockMultiOutputFunction(bkd)
        parallel_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = bkd.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        par_result = parallel_func(samples)
        seq_result = func(samples)

        assert par_result.shape == (2, 4)
        bkd.assert_allclose(par_result, seq_result, rtol=1e-12)
