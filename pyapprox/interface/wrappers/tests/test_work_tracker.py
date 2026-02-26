"""Tests for WorkTracker and TrackedModel."""

import pytest

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.wrappers.work_tracker import (
    TrackedModel,
    WorkTracker,
)


class TestWorkTracker:
    """Base tests for WorkTracker."""

    def _setup(self, bkd):
        self.tracker = WorkTracker(bkd)

    def test_initial_state(self, bkd) -> None:
        """Test that tracker starts empty."""
        self._setup(bkd)
        assert self.tracker.num_evaluations("values") == 0
        assert self.tracker.total_time("values") == 0.0
        assert self.tracker.mean_time("values") == 0.0

    def test_record_single(self, bkd) -> None:
        """Test recording a single evaluation."""
        self._setup(bkd)
        self.tracker.record("values", 0.5)
        assert self.tracker.num_evaluations("values") == 1
        bkd.assert_allclose(
            bkd.asarray([self.tracker.total_time("values")]),
            bkd.asarray([0.5]),
        )
        bkd.assert_allclose(
            bkd.asarray([self.tracker.mean_time("values")]),
            bkd.asarray([0.5]),
        )

    def test_record_multiple(self, bkd) -> None:
        """Test recording multiple evaluations."""
        self._setup(bkd)
        self.tracker.record("values", 0.5)
        self.tracker.record("values", 0.3)
        self.tracker.record("values", 0.2)
        assert self.tracker.num_evaluations("values") == 3
        bkd.assert_allclose(
            bkd.asarray([self.tracker.total_time("values")]),
            bkd.asarray([1.0]),
        )
        bkd.assert_allclose(
            bkd.asarray([self.tracker.mean_time("values")]),
            bkd.asarray([1.0 / 3]),
        )

    def test_record_different_types(self, bkd) -> None:
        """Test recording different evaluation types."""
        self._setup(bkd)
        self.tracker.record("values", 0.5)
        self.tracker.record("jacobian", 0.3)
        self.tracker.record("hessian", 0.2)

        assert self.tracker.num_evaluations("values") == 1
        assert self.tracker.num_evaluations("jacobian") == 1
        assert self.tracker.num_evaluations("hessian") == 1
        assert self.tracker.num_evaluations("hvp") == 0

    def test_record_invalid_type(self, bkd) -> None:
        """Test that invalid eval types raise error."""
        self._setup(bkd)
        with pytest.raises(ValueError):
            self.tracker.record("invalid", 0.5)

    def test_reset_single_type(self, bkd) -> None:
        """Test resetting a single evaluation type."""
        self._setup(bkd)
        self.tracker.record("values", 0.5)
        self.tracker.record("jacobian", 0.3)
        self.tracker.reset("values")

        assert self.tracker.num_evaluations("values") == 0
        assert self.tracker.num_evaluations("jacobian") == 1

    def test_reset_all(self, bkd) -> None:
        """Test resetting all evaluation types."""
        self._setup(bkd)
        self.tracker.record("values", 0.5)
        self.tracker.record("jacobian", 0.3)
        self.tracker.reset()

        assert self.tracker.num_evaluations("values") == 0
        assert self.tracker.num_evaluations("jacobian") == 0

    def test_wall_times_array(self, bkd) -> None:
        """Test getting wall times as array."""
        self._setup(bkd)
        self.tracker.record("values", 0.5)
        self.tracker.record("values", 0.3)
        wall_times = self.tracker.wall_times("values")
        bkd.assert_allclose(wall_times, bkd.asarray([0.5, 0.3]))

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        self._setup(bkd)
        assert "empty" in repr(self.tracker)
        self.tracker.record("values", 0.5)
        assert "values" in repr(self.tracker)


class TestTrackedModel:
    """Base tests for TrackedModel."""

    def _setup(self, bkd):
        # Create a simple test function
        def fun(samples):
            return samples[0:1, :] ** 2

        def jac(sample):
            return 2 * sample.T

        self.model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=bkd)
        self.model_with_jac = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=2, fun=fun, jacobian=jac, bkd=bkd
        )
        self.tracker = WorkTracker(bkd)
        self.tracked = TrackedModel(self.model, self.tracker)
        self.tracked_with_jac = TrackedModel(self.model_with_jac, self.tracker)

    def test_passthrough_nvars_nqoi(self, bkd) -> None:
        """Test that nvars and nqoi pass through."""
        self._setup(bkd)
        assert self.tracked.nvars() == 2
        assert self.tracked.nqoi() == 1

    def test_passthrough_bkd(self, bkd) -> None:
        """Test that bkd passes through."""
        self._setup(bkd)
        assert self.tracked.bkd() is bkd

    def test_call_tracks_time(self, bkd) -> None:
        """Test that __call__ tracks time."""
        self._setup(bkd)
        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        values = self.tracked(samples)

        # Check values are correct
        expected = bkd.asarray([[1.0, 4.0]])
        bkd.assert_allclose(values, expected)

        # Check tracking
        assert self.tracker.num_evaluations("values") == 1
        assert self.tracker.total_time("values") > 0.0

    def test_jacobian_tracks_time(self, bkd) -> None:
        """Test that jacobian tracks time."""
        self._setup(bkd)
        sample = bkd.asarray([[1.0], [2.0]])
        jacobian = self.tracked_with_jac.jacobian(sample)

        # Check jacobian shape
        assert jacobian.shape == (1, 2)

        # Check tracking
        assert self.tracker.num_evaluations("jacobian") == 1
        assert self.tracker.total_time("jacobian") > 0.0

    def test_wrapped_model_access(self, bkd) -> None:
        """Test access to wrapped model."""
        self._setup(bkd)
        assert self.tracked.wrapped() is self.model

    def test_tracker_access(self, bkd) -> None:
        """Test access to tracker."""
        self._setup(bkd)
        assert self.tracked.tracker() is self.tracker

    def test_dynamic_method_binding(self, bkd) -> None:
        """Test that jacobian method only exists if model has it."""
        self._setup(bkd)
        # Model without jacobian
        assert not hasattr(self.tracked, "jacobian")

        # Model with jacobian
        assert hasattr(self.tracked_with_jac, "jacobian")
