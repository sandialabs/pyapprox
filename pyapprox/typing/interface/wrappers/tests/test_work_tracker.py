"""Tests for WorkTracker and TrackedModel."""

import time
import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.interface.wrappers.work_tracker import (
    WorkTracker,
    TrackedModel,
)
from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests


class TestWorkTracker(Generic[Array], unittest.TestCase):
    """Base tests for WorkTracker."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.tracker = WorkTracker(self._bkd)

    def test_initial_state(self) -> None:
        """Test that tracker starts empty."""
        self.assertEqual(self.tracker.num_evaluations("values"), 0)
        self.assertEqual(self.tracker.total_time("values"), 0.0)
        self.assertEqual(self.tracker.mean_time("values"), 0.0)

    def test_record_single(self) -> None:
        """Test recording a single evaluation."""
        self.tracker.record("values", 0.5)
        self.assertEqual(self.tracker.num_evaluations("values"), 1)
        self._bkd.assert_allclose(
            self._bkd.asarray([self.tracker.total_time("values")]),
            self._bkd.asarray([0.5]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([self.tracker.mean_time("values")]),
            self._bkd.asarray([0.5]),
        )

    def test_record_multiple(self) -> None:
        """Test recording multiple evaluations."""
        self.tracker.record("values", 0.5)
        self.tracker.record("values", 0.3)
        self.tracker.record("values", 0.2)
        self.assertEqual(self.tracker.num_evaluations("values"), 3)
        self._bkd.assert_allclose(
            self._bkd.asarray([self.tracker.total_time("values")]),
            self._bkd.asarray([1.0]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([self.tracker.mean_time("values")]),
            self._bkd.asarray([1.0 / 3]),
        )

    def test_record_different_types(self) -> None:
        """Test recording different evaluation types."""
        self.tracker.record("values", 0.5)
        self.tracker.record("jacobian", 0.3)
        self.tracker.record("hessian", 0.2)

        self.assertEqual(self.tracker.num_evaluations("values"), 1)
        self.assertEqual(self.tracker.num_evaluations("jacobian"), 1)
        self.assertEqual(self.tracker.num_evaluations("hessian"), 1)
        self.assertEqual(self.tracker.num_evaluations("hvp"), 0)

    def test_record_invalid_type(self) -> None:
        """Test that invalid eval types raise error."""
        with self.assertRaises(ValueError):
            self.tracker.record("invalid", 0.5)

    def test_reset_single_type(self) -> None:
        """Test resetting a single evaluation type."""
        self.tracker.record("values", 0.5)
        self.tracker.record("jacobian", 0.3)
        self.tracker.reset("values")

        self.assertEqual(self.tracker.num_evaluations("values"), 0)
        self.assertEqual(self.tracker.num_evaluations("jacobian"), 1)

    def test_reset_all(self) -> None:
        """Test resetting all evaluation types."""
        self.tracker.record("values", 0.5)
        self.tracker.record("jacobian", 0.3)
        self.tracker.reset()

        self.assertEqual(self.tracker.num_evaluations("values"), 0)
        self.assertEqual(self.tracker.num_evaluations("jacobian"), 0)

    def test_wall_times_array(self) -> None:
        """Test getting wall times as array."""
        self.tracker.record("values", 0.5)
        self.tracker.record("values", 0.3)
        wall_times = self.tracker.wall_times("values")
        self._bkd.assert_allclose(
            wall_times, self._bkd.asarray([0.5, 0.3])
        )

    def test_repr(self) -> None:
        """Test string representation."""
        self.assertIn("empty", repr(self.tracker))
        self.tracker.record("values", 0.5)
        self.assertIn("values", repr(self.tracker))


class TestTrackedModel(Generic[Array], unittest.TestCase):
    """Base tests for TrackedModel."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

        # Create a simple test function
        def fun(samples: Array) -> Array:
            return samples[0:1, :] ** 2

        def jac(sample: Array) -> Array:
            return 2 * sample.T

        self.model = FunctionFromCallable(
            nqoi=1, nvars=2, fun=fun, bkd=self._bkd
        )
        self.model_with_jac = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=2, fun=fun, jacobian=jac, bkd=self._bkd
        )
        self.tracker = WorkTracker(self._bkd)
        self.tracked = TrackedModel(self.model, self.tracker)
        self.tracked_with_jac = TrackedModel(
            self.model_with_jac, self.tracker
        )

    def test_passthrough_nvars_nqoi(self) -> None:
        """Test that nvars and nqoi pass through."""
        self.assertEqual(self.tracked.nvars(), 2)
        self.assertEqual(self.tracked.nqoi(), 1)

    def test_passthrough_bkd(self) -> None:
        """Test that bkd passes through."""
        self.assertIs(self.tracked.bkd(), self._bkd)

    def test_call_tracks_time(self) -> None:
        """Test that __call__ tracks time."""
        samples = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        values = self.tracked(samples)

        # Check values are correct
        expected = self._bkd.asarray([[1.0, 4.0]])
        self._bkd.assert_allclose(values, expected)

        # Check tracking
        self.assertEqual(self.tracker.num_evaluations("values"), 1)
        self.assertGreater(self.tracker.total_time("values"), 0.0)

    def test_jacobian_tracks_time(self) -> None:
        """Test that jacobian tracks time."""
        sample = self._bkd.asarray([[1.0], [2.0]])
        jacobian = self.tracked_with_jac.jacobian(sample)

        # Check jacobian shape
        self.assertEqual(jacobian.shape, (1, 2))

        # Check tracking
        self.assertEqual(self.tracker.num_evaluations("jacobian"), 1)
        self.assertGreater(self.tracker.total_time("jacobian"), 0.0)

    def test_wrapped_model_access(self) -> None:
        """Test access to wrapped model."""
        self.assertIs(self.tracked.wrapped(), self.model)

    def test_tracker_access(self) -> None:
        """Test access to tracker."""
        self.assertIs(self.tracked.tracker(), self.tracker)

    def test_dynamic_method_binding(self) -> None:
        """Test that jacobian method only exists if model has it."""
        # Model without jacobian
        self.assertFalse(hasattr(self.tracked, "jacobian"))

        # Model with jacobian
        self.assertTrue(hasattr(self.tracked_with_jac, "jacobian"))


class TestWorkTrackerNumpy(TestWorkTracker[NDArray[Any]]):
    """NumPy backend tests for WorkTracker."""

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestTrackedModelNumpy(TestTrackedModel[NDArray[Any]]):
    """NumPy backend tests for TrackedModel."""

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestWorkTrackerTorch(TestWorkTracker[torch.Tensor]):
    """PyTorch backend tests for WorkTracker."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestTrackedModelTorch(TestTrackedModel[torch.Tensor]):
    """PyTorch backend tests for TrackedModel."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
