"""Tests for parallel function factory."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests


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


class TestParallelFunctionWrapper(Generic[Array], unittest.TestCase):
    """Tests for ParallelFunctionWrapper - NOT run directly."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_basic_call(self):
        """Test that wrapped function still evaluates correctly."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        result = parallel_func(samples)

        expected = func(samples)
        self.assertTrue(self._bkd.allclose(result, expected))

    def test_jacobian_batch(self):
        """Test parallel jacobian batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Shape should be (nsamples, nqoi, nvars)
        self.assertEqual(jacobians.shape, (3, 1, 2))

        # Verify each jacobian
        for i in range(3):
            sample = samples[:, i : i + 1]
            expected = func.jacobian(sample)
            self.assertTrue(
                self._bkd.allclose(jacobians[i], expected, rtol=1e-10)
            )

    def test_hessian_batch(self):
        """Test parallel hessian batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        hessians = parallel_func.hessian_batch(samples)

        # Shape should be (nsamples, nvars, nvars)
        self.assertEqual(hessians.shape, (2, 2, 2))

        # Each hessian should be 2*I
        expected = 2 * self._bkd.eye(2)
        for i in range(2):
            self.assertTrue(self._bkd.allclose(hessians[i], expected))

    def test_hvp_batch(self):
        """Test parallel HVP batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        vecs = self._bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        hvps = parallel_func.hvp_batch(samples, vecs)

        # Shape should be (nsamples, nvars)
        self.assertEqual(hvps.shape, (2, 2))

        # HVP is 2*v
        expected = self._bkd.asarray([[2.0, 0.0], [0.0, 2.0]])
        self.assertTrue(self._bkd.allclose(hvps, expected))

    def test_whvp_batch(self):
        """Test parallel weighted HVP batch computation."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        vecs = self._bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        weights = self._bkd.asarray([[0.5]])

        whvps = parallel_func.whvp_batch(samples, vecs, weights)

        # Shape should be (nsamples, nvars)
        self.assertEqual(whvps.shape, (2, 2))

        # WHVP is weights * 2 * v = 0.5 * 2 * v = v
        expected = self._bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(self._bkd.allclose(whvps, expected))

    def test_multi_output_jacobian_batch(self):
        """Test jacobian batch for multi-output function."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockMultiOutputFunction(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Shape should be (nsamples, nqoi, nvars)
        self.assertEqual(jacobians.shape, (2, 2, 2))

    def test_hasattr_detection(self):
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

        func = FuncWithoutHVP(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        self.assertTrue(hasattr(parallel_func, "jacobian_batch"))
        self.assertFalse(hasattr(parallel_func, "hvp_batch"))
        self.assertFalse(hasattr(parallel_func, "whvp_batch"))

    def test_backend_info(self):
        """Test backend information methods."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)

        seq_func = make_parallel(func, backend="sequential")
        self.assertEqual(seq_func.parallel_backend(), "sequential")

        joblib_func = make_parallel(func, backend="joblib_processes", n_jobs=4)
        self.assertIn("joblib", joblib_func.parallel_backend())
        self.assertEqual(joblib_func.n_workers(), 4)

    def test_parallel_execution_joblib(self):
        """Test actual parallel execution with joblib."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="joblib_processes", n_jobs=2)

        samples = self._bkd.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Verify results match sequential
        seq_func = make_parallel(func, backend="sequential")
        seq_jacobians = seq_func.jacobian_batch(samples)

        self.assertTrue(self._bkd.allclose(jacobians, seq_jacobians))

    def test_parallel_execution_mpire(self):
        """Test actual parallel execution with mpire."""
        try:
            import mpire
        except ImportError:
            self.skipTest("mpire not installed")

        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="mpire", n_jobs=2)

        samples = self._bkd.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        jacobians = parallel_func.jacobian_batch(samples)

        # Verify results match sequential
        seq_func = make_parallel(func, backend="sequential")
        seq_jacobians = seq_func.jacobian_batch(samples)

        self.assertTrue(self._bkd.allclose(jacobians, seq_jacobians))

    def test_parallel_call_matches_sequential(self):
        """Test that parallel __call__ matches sequential __call__."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(
            func, backend="joblib_processes", n_jobs=2
        )

        samples = self._bkd.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        )

        par_result = parallel_func(samples)
        seq_result = func(samples)

        self.assertEqual(par_result.shape, seq_result.shape)
        self._bkd.assert_allclose(par_result, seq_result, rtol=1e-12)

    def test_parallel_call_single_sample(self):
        """Test that single-sample call skips parallelism."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(
            func, backend="joblib_processes", n_jobs=4
        )

        sample = self._bkd.asarray([[1.0], [2.0]])
        result = parallel_func(sample)
        expected = func(sample)

        self._bkd.assert_allclose(result, expected)

    def test_parallel_call_sequential_backend(self):
        """Test __call__ with sequential backend delegates directly."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockFunction(self._bkd)
        parallel_func = make_parallel(func, backend="sequential")

        samples = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = parallel_func(samples)
        expected = func(samples)

        self._bkd.assert_allclose(result, expected)

    def test_parallel_call_multi_output(self):
        """Test parallel __call__ with multi-output function."""
        from pyapprox.interface.parallel.factory import make_parallel

        func = MockMultiOutputFunction(self._bkd)
        parallel_func = make_parallel(
            func, backend="joblib_processes", n_jobs=2
        )

        samples = self._bkd.asarray(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        )

        par_result = parallel_func(samples)
        seq_result = func(samples)

        self.assertEqual(par_result.shape, (2, 4))
        self._bkd.assert_allclose(par_result, seq_result, rtol=1e-12)


class TestParallelFunctionWrapperNumpy(TestParallelFunctionWrapper[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestParallelFunctionWrapperTorch(TestParallelFunctionWrapper[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
