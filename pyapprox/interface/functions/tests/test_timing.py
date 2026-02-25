"""Tests for function timing wrappers."""

import time
import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
)
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.timing import (
    FunctionTimer,
    MethodTimer,
    TimedFunction,
    TimedFunctionWithJacobian,
    TimedFunctionWithJacobianAndHVP,
    timed,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SleepFunction(Generic[Array]):
    """Test function with configurable per-method sleep times.

    Satisfies FunctionWithJacobianAndHVPProtocol.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        call_time: float = 0.05,
        jac_time: float = 0.07,
        hvp_time: float = 0.09,
    ) -> None:
        self._bkd = bkd
        self._call_time = call_time
        self._jac_time = jac_time
        self._hvp_time = hvp_time

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        time.sleep(self._call_time)
        return self._bkd.zeros((1, samples.shape[1]))

    def jacobian(self, sample: Array) -> Array:
        time.sleep(self._jac_time)
        return self._bkd.zeros((1, 2))

    def hvp(self, sample: Array, vec: Array) -> Array:
        time.sleep(self._hvp_time)
        return self._bkd.zeros((2, 1))


class SleepFunctionWithBatch(SleepFunction[Array]):
    """Adds a batch jacobian that sleeps once for the whole batch."""

    def __init__(
        self,
        bkd: Backend[Array],
        call_time: float = 0.05,
        jac_time: float = 0.07,
        hvp_time: float = 0.09,
        batch_time: float = 0.1,
    ) -> None:
        super().__init__(bkd, call_time, jac_time, hvp_time)
        self._batch_time = batch_time

    def jacobian_batch(self, samples: Array) -> Array:
        time.sleep(self._batch_time)
        n = samples.shape[1]
        return self._bkd.zeros((n, 1, 2))


class VariableSleepFunction(Generic[Array]):
    """Jacobian alternates between two sleep times.

    Satisfies FunctionWithJacobianProtocol.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        time_a: float = 0.05,
        time_b: float = 0.10,
    ) -> None:
        self._bkd = bkd
        self._time_a = time_a
        self._time_b = time_b
        self._call_count = 0

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return self._bkd.zeros((1, samples.shape[1]))

    def jacobian(self, sample: Array) -> Array:
        if self._call_count % 2 == 0:
            time.sleep(self._time_a)
        else:
            time.sleep(self._time_b)
        self._call_count += 1
        return self._bkd.zeros((1, 2))


class SleepFunctionWithInternalCall(SleepFunction[Array]):
    """Jacobian internally calls self(sample)."""

    def jacobian(self, sample: Array) -> Array:
        self(sample)
        time.sleep(self._jac_time)
        return self._bkd.zeros((1, 2))


# ---------------------------------------------------------------------------
# Pure-Python tests (no backend dependency)
# ---------------------------------------------------------------------------


class TestMethodTimer(unittest.TestCase):
    """Tests for MethodTimer (pure Python, no backend)."""

    def test_empty_raises(self) -> None:
        t = MethodTimer()
        self.assertEqual(t.call_count(), 0)
        self.assertEqual(t.total_evals(), 0)
        self.assertAlmostEqual(t.total_time(), 0.0)
        with self.assertRaises(ValueError):
            t.median()

    def test_single_eval_median(self) -> None:
        t = MethodTimer()
        t.record(0.3, 1)
        t.record(0.1, 1)
        t.record(0.2, 1)
        self.assertEqual(t.call_count(), 3)
        self.assertEqual(t.total_evals(), 3)
        self.assertAlmostEqual(t.median(), 0.2)

    def test_batch_eval_median(self) -> None:
        t = MethodTimer()
        t.record(0.1, 20)
        t.record(0.1, 80)
        self.assertEqual(t.call_count(), 2)
        self.assertEqual(t.total_evals(), 100)
        self.assertAlmostEqual(t.median(), 0.2 / 100)

    def test_reset(self) -> None:
        t = MethodTimer()
        t.record(0.5, 1)
        t.record(0.3, 1)
        self.assertEqual(t.call_count(), 2)
        t.reset()
        self.assertEqual(t.call_count(), 0)
        self.assertEqual(t.total_evals(), 0)
        with self.assertRaises(ValueError):
            t.median()


class TestFunctionTimer(unittest.TestCase):
    """Tests for FunctionTimer (pure Python, no backend)."""

    def test_auto_create(self) -> None:
        ft = FunctionTimer()
        t1 = ft.get("foo")
        t2 = ft.get("foo")
        self.assertIs(t1, t2)
        t3 = ft.get("bar")
        self.assertIsNot(t1, t3)

    def test_reset(self) -> None:
        ft = FunctionTimer()
        ft.get("foo").record(0.1)
        ft.get("bar").record(0.2)
        self.assertEqual(ft.get("foo").call_count(), 1)
        self.assertEqual(ft.get("bar").call_count(), 1)
        ft.reset()
        self.assertEqual(ft.get("foo").call_count(), 0)
        self.assertEqual(ft.get("bar").call_count(), 0)

    def test_summary(self) -> None:
        ft = FunctionTimer()
        ft.get("foo").record(0.1)
        ft.get("foo").record(0.3)
        s = ft.summary()
        self.assertIn("foo", s)
        self.assertAlmostEqual(s["foo"]["total_time"], 0.4)
        self.assertAlmostEqual(s["foo"]["call_count"], 2.0)


# ---------------------------------------------------------------------------
# Dual-backend timing tests
# ---------------------------------------------------------------------------

RTOL = 0.15


class TestTimedWrapper(Generic[Array], unittest.TestCase):
    """Dual-backend tests for timed wrapper classes."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_call_median(self) -> None:
        """SleepFunction(call_time=0.05), call 3x with 1 sample each."""
        fn = SleepFunction(self._bkd, call_time=0.05)
        w = timed(fn)
        sample = self._bkd.zeros((2, 1))
        for _ in range(3):
            w(sample)
        t = w.timer().get("__call__")
        self.assertEqual(t.call_count(), 3)
        self.assertEqual(t.total_evals(), 3)
        self._bkd.assert_allclose(
            self._bkd.asarray([t.median()]),
            self._bkd.asarray([0.05]),
            rtol=RTOL,
        )

    def test_jacobian_median(self) -> None:
        """SleepFunction(jac_time=0.07), call jacobian 5x."""
        fn = SleepFunction(self._bkd, jac_time=0.07)
        w = timed(fn)
        sample = self._bkd.zeros((2, 1))
        for _ in range(5):
            w.jacobian(sample)  # type: ignore[union-attr]
        t = w.timer().get("jacobian")
        self.assertEqual(t.call_count(), 5)
        self.assertEqual(t.total_evals(), 5)
        self._bkd.assert_allclose(
            self._bkd.asarray([t.median()]),
            self._bkd.asarray([0.07]),
            rtol=RTOL,
        )

    def test_hvp_median(self) -> None:
        """SleepFunction(hvp_time=0.09), call hvp 3x."""
        fn = SleepFunction(self._bkd, hvp_time=0.09)
        w = timed(fn)
        sample = self._bkd.zeros((2, 1))
        vec = self._bkd.ones((2, 1))
        for _ in range(3):
            w.hvp(sample, vec)  # type: ignore[union-attr]
        t = w.timer().get("hvp")
        self.assertEqual(t.call_count(), 3)
        self._bkd.assert_allclose(
            self._bkd.asarray([t.median()]),
            self._bkd.asarray([0.09]),
            rtol=RTOL,
        )

    def test_different_methods_different_times(self) -> None:
        """Each method has distinct sleep time; verify ordering."""
        fn = SleepFunction(self._bkd, call_time=0.05, jac_time=0.08, hvp_time=0.06)
        w = timed(fn)
        sample = self._bkd.zeros((2, 1))
        vec = self._bkd.ones((2, 1))
        for _ in range(3):
            w(sample)
            w.jacobian(sample)  # type: ignore[union-attr]
            w.hvp(sample, vec)  # type: ignore[union-attr]
        call_med = w.timer().get("__call__").median()
        jac_med = w.timer().get("jacobian").median()
        hvp_med = w.timer().get("hvp").median()
        self.assertGreater(jac_med, hvp_med)
        self.assertGreater(hvp_med, call_med)
        self._bkd.assert_allclose(
            self._bkd.asarray([call_med]),
            self._bkd.asarray([0.05]),
            rtol=RTOL,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([jac_med]),
            self._bkd.asarray([0.08]),
            rtol=RTOL,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([hvp_med]),
            self._bkd.asarray([0.06]),
            rtol=RTOL,
        )

    def test_median_true_median_for_individual(self) -> None:
        """VariableSleepFunction alternating [0.05, 0.10]. Median=0.05."""
        fn = VariableSleepFunction(self._bkd, time_a=0.05, time_b=0.10)
        w = timed(fn)
        sample = self._bkd.zeros((2, 1))
        for _ in range(5):
            w.jacobian(sample)  # type: ignore[union-attr]
        med = w.timer().get("jacobian").median()
        self._bkd.assert_allclose(
            self._bkd.asarray([med]),
            self._bkd.asarray([0.05]),
            rtol=RTOL,
        )
        # Ensure it's NOT the mean (0.07)
        self.assertLess(med, 0.06)

    def test_batch_median_is_weighted_mean(self) -> None:
        """SleepFunctionWithBatch: verify weighted-mean median."""
        fn = SleepFunctionWithBatch(self._bkd, batch_time=0.1)
        w = timed(fn)
        samples_20 = self._bkd.zeros((2, 20))
        samples_80 = self._bkd.zeros((2, 80))
        w.jacobian_batch(samples_20)  # type: ignore[union-attr]
        w.jacobian_batch(samples_80)  # type: ignore[union-attr]
        t = w.timer().get("jacobian_batch")
        self.assertEqual(t.call_count(), 2)
        self.assertEqual(t.total_evals(), 100)
        expected = t.total_time() / 100.0
        self._bkd.assert_allclose(
            self._bkd.asarray([t.median()]),
            self._bkd.asarray([expected]),
        )

    def test_call_nevals_is_nsamples(self) -> None:
        """__call__ with 10 then 20 samples."""
        fn = SleepFunction(self._bkd, call_time=0.05)
        w = timed(fn)
        w(self._bkd.zeros((2, 10)))
        w(self._bkd.zeros((2, 20)))
        t = w.timer().get("__call__")
        self.assertEqual(t.total_evals(), 30)
        self.assertEqual(t.call_count(), 2)

    def test_protocol_preservation(self) -> None:
        """timed() returns appropriate wrapper type."""
        fn = SleepFunction(self._bkd)
        w = timed(fn)
        self.assertIsInstance(w, TimedFunctionWithJacobianAndHVP)
        self.assertTrue(isinstance(w, TimedFunctionWithJacobian))
        self.assertTrue(isinstance(w, TimedFunction))

        # Plain FunctionProtocol should NOT get jacobian
        plain = FunctionFromCallable(
            nqoi=1,
            nvars=2,
            fun=lambda s: self._bkd.zeros((1, s.shape[1])),
            bkd=self._bkd,
        )
        w_plain = timed(plain)
        self.assertIsInstance(w_plain, TimedFunction)
        self.assertNotIsInstance(w_plain, TimedFunctionWithJacobian)
        self.assertFalse(hasattr(w_plain, "jacobian"))

    def test_values_unchanged(self) -> None:
        """IshigamiFunction: wrapped and unwrapped give identical results."""
        fn = IshigamiFunction(self._bkd, a=7.0, b=0.1)
        w = timed(fn)
        samples = self._bkd.asarray(
            [[0.5, 1.0, -0.3], [0.2, -1.5, 0.8], [1.0, 0.5, -1.0]]
        )
        self._bkd.assert_allclose(w(samples), fn(samples))

        sample = samples[:, 0:1]
        self._bkd.assert_allclose(
            w.jacobian(sample),  # type: ignore[union-attr]
            fn.jacobian(sample),
        )
        vec = self._bkd.ones((3, 1))
        self._bkd.assert_allclose(
            w.hvp(sample, vec),  # type: ignore[union-attr]
            fn.hvp(sample, vec),
        )

    def test_no_double_counting(self) -> None:
        """Internal self(sample) in jacobian goes through inner, not wrapper."""
        fn = SleepFunctionWithInternalCall(self._bkd, jac_time=0.05)
        w = timed(fn)
        sample = self._bkd.zeros((2, 1))
        w.jacobian(sample)  # type: ignore[union-attr]
        self.assertEqual(w.timer().get("jacobian").call_count(), 1)
        # __call__ timer should have 0 calls — the internal self(sample)
        # goes through the inner object, not the wrapper.
        self.assertEqual(w.timer().get("__call__").call_count(), 0)

    def test_shared_timer(self) -> None:
        """Two functions sharing one FunctionTimer."""
        shared = FunctionTimer()
        fn1 = SleepFunction(self._bkd, jac_time=0.05)
        fn2 = SleepFunction(self._bkd, jac_time=0.08)
        w1 = timed(fn1, timer=shared)
        w2 = timed(fn2, timer=shared)
        sample = self._bkd.zeros((2, 1))
        w1.jacobian(sample)  # type: ignore[union-attr]
        w2.jacobian(sample)  # type: ignore[union-attr]
        self.assertEqual(shared.get("jacobian").call_count(), 2)

    def test_reset(self) -> None:
        """Reset clears all timing data."""
        fn = SleepFunction(self._bkd, jac_time=0.05)
        w = timed(fn)
        sample = self._bkd.zeros((2, 1))
        for _ in range(3):
            w.jacobian(sample)  # type: ignore[union-attr]
        self.assertEqual(w.timer().get("jacobian").call_count(), 3)
        w.timer().reset()
        self.assertEqual(w.timer().get("jacobian").call_count(), 0)
        self.assertEqual(w.timer().get("jacobian").total_evals(), 0)
        with self.assertRaises(ValueError):
            w.timer().get("jacobian").median()
        w.jacobian(sample)  # type: ignore[union-attr]
        self.assertEqual(w.timer().get("jacobian").call_count(), 1)


# ---------------------------------------------------------------------------
# Concrete backend test classes
# ---------------------------------------------------------------------------


class TestTimedWrapperNumpy(TestTimedWrapper[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestTimedWrapperTorch(TestTimedWrapper[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
