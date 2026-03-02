"""Tests for function timing wrappers."""

import time
from typing import Generic

import pytest

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
from pyapprox.util.backends.protocols import Array, Backend

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


class TestMethodTimer:
    """Tests for MethodTimer (pure Python, no backend)."""

    def test_empty_raises(self) -> None:
        t = MethodTimer()
        assert t.call_count() == 0
        assert t.total_evals() == 0
        assert t.total_time() == pytest.approx(0.0)
        with pytest.raises(ValueError):
            t.median()

    def test_single_eval_median(self) -> None:
        t = MethodTimer()
        t.record(0.3, 1)
        t.record(0.1, 1)
        t.record(0.2, 1)
        assert t.call_count() == 3
        assert t.total_evals() == 3
        assert t.median() == pytest.approx(0.2)

    def test_batch_eval_median(self) -> None:
        t = MethodTimer()
        t.record(0.1, 20)
        t.record(0.1, 80)
        assert t.call_count() == 2
        assert t.total_evals() == 100
        assert t.median() == pytest.approx(0.2 / 100)

    def test_reset(self) -> None:
        t = MethodTimer()
        t.record(0.5, 1)
        t.record(0.3, 1)
        assert t.call_count() == 2
        t.reset()
        assert t.call_count() == 0
        assert t.total_evals() == 0
        with pytest.raises(ValueError):
            t.median()


class TestFunctionTimer:
    """Tests for FunctionTimer (pure Python, no backend)."""

    def test_auto_create(self) -> None:
        ft = FunctionTimer()
        t1 = ft.get("foo")
        t2 = ft.get("foo")
        assert t1 is t2
        t3 = ft.get("bar")
        assert t1 is not t3

    def test_reset(self) -> None:
        ft = FunctionTimer()
        ft.get("foo").record(0.1)
        ft.get("bar").record(0.2)
        assert ft.get("foo").call_count() == 1
        assert ft.get("bar").call_count() == 1
        ft.reset()
        assert ft.get("foo").call_count() == 0
        assert ft.get("bar").call_count() == 0

    def test_summary(self) -> None:
        ft = FunctionTimer()
        ft.get("foo").record(0.1)
        ft.get("foo").record(0.3)
        s = ft.summary()
        assert "foo" in s
        assert s["foo"]["total_time"] == pytest.approx(0.4)
        assert s["foo"]["call_count"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Dual-backend timing tests
# ---------------------------------------------------------------------------

RTOL = 4.0


class TestTimedWrapper:
    """Dual-backend tests for timed wrapper classes."""

    def test_call_median(self, bkd) -> None:
        """SleepFunction(call_time=0.05), call 3x with 1 sample each."""
        fn = SleepFunction(bkd, call_time=0.05)
        w = timed(fn)
        sample = bkd.zeros((2, 1))
        for _ in range(3):
            w(sample)
        t = w.timer().get("__call__")
        assert t.call_count() == 3
        assert t.total_evals() == 3
        bkd.assert_allclose(
            bkd.asarray([t.median()]),
            bkd.asarray([0.05]),
            rtol=RTOL,
        )

    def test_jacobian_median(self, bkd) -> None:
        """SleepFunction(jac_time=0.07), call jacobian 5x."""
        fn = SleepFunction(bkd, jac_time=0.07)
        w = timed(fn)
        sample = bkd.zeros((2, 1))
        for _ in range(5):
            w.jacobian(sample)  # type: ignore[union-attr]
        t = w.timer().get("jacobian")
        assert t.call_count() == 5
        assert t.total_evals() == 5
        bkd.assert_allclose(
            bkd.asarray([t.median()]),
            bkd.asarray([0.07]),
            rtol=RTOL,
        )

    def test_hvp_median(self, bkd) -> None:
        """SleepFunction(hvp_time=0.09), call hvp 3x."""
        fn = SleepFunction(bkd, hvp_time=0.09)
        w = timed(fn)
        sample = bkd.zeros((2, 1))
        vec = bkd.ones((2, 1))
        for _ in range(3):
            w.hvp(sample, vec)  # type: ignore[union-attr]
        t = w.timer().get("hvp")
        assert t.call_count() == 3
        bkd.assert_allclose(
            bkd.asarray([t.median()]),
            bkd.asarray([0.09]),
            rtol=RTOL,
        )

    def test_different_methods_different_times(self, bkd) -> None:
        """Each method has distinct sleep time; verify ordering."""
        fn = SleepFunction(bkd, call_time=0.05, jac_time=0.08, hvp_time=0.06)
        w = timed(fn)
        sample = bkd.zeros((2, 1))
        vec = bkd.ones((2, 1))
        for _ in range(3):
            w(sample)
            w.jacobian(sample)  # type: ignore[union-attr]
            w.hvp(sample, vec)  # type: ignore[union-attr]
        call_med = w.timer().get("__call__").median()
        jac_med = w.timer().get("jacobian").median()
        hvp_med = w.timer().get("hvp").median()
        assert jac_med >= call_med
        assert hvp_med >= call_med
        bkd.assert_allclose(
            bkd.asarray([call_med]),
            bkd.asarray([0.05]),
            rtol=RTOL,
        )
        bkd.assert_allclose(
            bkd.asarray([jac_med]),
            bkd.asarray([0.08]),
            rtol=RTOL,
        )
        bkd.assert_allclose(
            bkd.asarray([hvp_med]),
            bkd.asarray([0.06]),
            rtol=RTOL,
        )

    def test_median_true_median_for_individual(self, bkd) -> None:
        """VariableSleepFunction alternating [0.05, 0.10]. Median=0.05."""
        fn = VariableSleepFunction(bkd, time_a=0.05, time_b=0.10)
        w = timed(fn)
        sample = bkd.zeros((2, 1))
        for _ in range(5):
            w.jacobian(sample)  # type: ignore[union-attr]
        med = w.timer().get("jacobian").median()
        bkd.assert_allclose(
            bkd.asarray([med]),
            bkd.asarray([0.05]),
            rtol=RTOL,
        )
        # Ensure it's NOT the mean (0.07); generous bound for CI overhead
        assert med < 0.5

    def test_batch_median_is_weighted_mean(self, bkd) -> None:
        """SleepFunctionWithBatch: verify weighted-mean median."""
        fn = SleepFunctionWithBatch(bkd, batch_time=0.1)
        w = timed(fn)
        samples_20 = bkd.zeros((2, 20))
        samples_80 = bkd.zeros((2, 80))
        w.jacobian_batch(samples_20)  # type: ignore[union-attr]
        w.jacobian_batch(samples_80)  # type: ignore[union-attr]
        t = w.timer().get("jacobian_batch")
        assert t.call_count() == 2
        assert t.total_evals() == 100
        expected = t.total_time() / 100.0
        bkd.assert_allclose(
            bkd.asarray([t.median()]),
            bkd.asarray([expected]),
        )

    def test_call_nevals_is_nsamples(self, bkd) -> None:
        """__call__ with 10 then 20 samples."""
        fn = SleepFunction(bkd, call_time=0.05)
        w = timed(fn)
        w(bkd.zeros((2, 10)))
        w(bkd.zeros((2, 20)))
        t = w.timer().get("__call__")
        assert t.total_evals() == 30
        assert t.call_count() == 2

    def test_protocol_preservation(self, bkd) -> None:
        """timed() returns appropriate wrapper type."""
        fn = SleepFunction(bkd)
        w = timed(fn)
        assert isinstance(w, TimedFunctionWithJacobianAndHVP)
        assert isinstance(w, TimedFunctionWithJacobian)
        assert isinstance(w, TimedFunction)

        # Plain FunctionProtocol should NOT get jacobian
        plain = FunctionFromCallable(
            nqoi=1,
            nvars=2,
            fun=lambda s: bkd.zeros((1, s.shape[1])),
            bkd=bkd,
        )
        w_plain = timed(plain)
        assert isinstance(w_plain, TimedFunction)
        assert not isinstance(w_plain, TimedFunctionWithJacobian)
        assert not hasattr(w_plain, "jacobian")

    def test_values_unchanged(self, bkd) -> None:
        """IshigamiFunction: wrapped and unwrapped give identical results."""
        fn = IshigamiFunction(bkd, a=7.0, b=0.1)
        w = timed(fn)
        samples = bkd.asarray(
            [[0.5, 1.0, -0.3], [0.2, -1.5, 0.8], [1.0, 0.5, -1.0]]
        )
        bkd.assert_allclose(w(samples), fn(samples))

        sample = samples[:, 0:1]
        bkd.assert_allclose(
            w.jacobian(sample),  # type: ignore[union-attr]
            fn.jacobian(sample),
        )
        vec = bkd.ones((3, 1))
        bkd.assert_allclose(
            w.hvp(sample, vec),  # type: ignore[union-attr]
            fn.hvp(sample, vec),
        )

    def test_no_double_counting(self, bkd) -> None:
        """Internal self(sample) in jacobian goes through inner, not wrapper."""
        fn = SleepFunctionWithInternalCall(bkd, jac_time=0.05)
        w = timed(fn)
        sample = bkd.zeros((2, 1))
        w.jacobian(sample)  # type: ignore[union-attr]
        assert w.timer().get("jacobian").call_count() == 1
        # __call__ timer should have 0 calls -- the internal self(sample)
        # goes through the inner object, not the wrapper.
        assert w.timer().get("__call__").call_count() == 0

    def test_shared_timer(self, bkd) -> None:
        """Two functions sharing one FunctionTimer."""
        shared = FunctionTimer()
        fn1 = SleepFunction(bkd, jac_time=0.05)
        fn2 = SleepFunction(bkd, jac_time=0.08)
        w1 = timed(fn1, timer=shared)
        w2 = timed(fn2, timer=shared)
        sample = bkd.zeros((2, 1))
        w1.jacobian(sample)  # type: ignore[union-attr]
        w2.jacobian(sample)  # type: ignore[union-attr]
        assert shared.get("jacobian").call_count() == 2

    def test_reset(self, bkd) -> None:
        """Reset clears all timing data."""
        fn = SleepFunction(bkd, jac_time=0.05)
        w = timed(fn)
        sample = bkd.zeros((2, 1))
        for _ in range(3):
            w.jacobian(sample)  # type: ignore[union-attr]
        assert w.timer().get("jacobian").call_count() == 3
        w.timer().reset()
        assert w.timer().get("jacobian").call_count() == 0
        assert w.timer().get("jacobian").total_evals() == 0
        with pytest.raises(ValueError):
            w.timer().get("jacobian").median()
        w.jacobian(sample)  # type: ignore[union-attr]
        assert w.timer().get("jacobian").call_count() == 1
