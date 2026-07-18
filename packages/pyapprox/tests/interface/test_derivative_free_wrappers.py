"""Generic wrappers must accept derivative-free ObjectiveProtocol objects.

The capability-mirroring wrappers (timed, make_parallel, TrackedModel,
ActiveSetFunction) require ObjectiveProtocol — evaluation plus a
``derivatives()`` bundle. A derivative-free function participates by
returning ``Derivatives.none()``; it must wrap cleanly and evaluate
unchanged, with every bundle field absent. These tests guard against the
wrappers over-restricting to functions with populated bundles (a
regression that previously broke tutorials wrapping plain forward
models).
"""

import numpy as np
import pytest
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.marginalize import ActiveSetFunction
from pyapprox.interface.functions.timing import timed
from pyapprox.interface.parallel import make_parallel
from pyapprox.interface.wrappers.work_tracker import (
    TrackedModel,
    WorkTracker,
)

BUNDLE_FIELDS = (
    "jacobian",
    "jacobian_batch",
    "jvp",
    "hvp",
    "hvp_batch",
    "whvp",
    "whvp_batch",
    "hessian",
    "hessian_batch",
    "inexact",
)


def _square_sum(samples):
    return (samples**2).sum(axis=0)[None, :]


def _make_function(bkd):
    """A derivative-free function: bundle is Derivatives.none()."""
    return FunctionFromCallable(nqoi=1, nvars=2, fun=_square_sum, bkd=bkd)


def _assert_empty_bundle(bkd, wrapper):
    for field in BUNDLE_FIELDS:
        assert getattr(wrapper.derivatives(), field) is None, field


class _NoDerivatives:
    """FunctionProtocol shape but no derivatives() — must be rejected."""

    def __init__(self, bkd):
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 2

    def nqoi(self):
        return 1

    def __call__(self, samples):
        return _square_sum(samples)


class TestDerivativeFreeWrapping:
    def _samples(self, bkd):
        np.random.seed(0)
        return bkd.asarray(np.random.uniform(-1.0, 1.0, (2, 5)))

    def test_timed_accepts_derivative_free(self, bkd):
        function = _make_function(bkd)
        samples = self._samples(bkd)
        wrapper = timed(function)
        bkd.assert_allclose(wrapper(samples), function(samples), rtol=1e-15)
        _assert_empty_bundle(bkd, wrapper)

    def test_make_parallel_accepts_derivative_free(self, bkd):
        function = _make_function(bkd)
        samples = self._samples(bkd)
        wrapper = make_parallel(function, backend="sequential")
        bkd.assert_allclose(wrapper(samples), function(samples), rtol=1e-15)
        _assert_empty_bundle(bkd, wrapper)

    def test_tracked_model_accepts_derivative_free(self, bkd):
        function = _make_function(bkd)
        samples = self._samples(bkd)
        wrapper = TrackedModel(function, WorkTracker(bkd))
        bkd.assert_allclose(wrapper(samples), function(samples), rtol=1e-15)
        _assert_empty_bundle(bkd, wrapper)

    def test_active_set_accepts_derivative_free(self, bkd):
        function = _make_function(bkd)
        nominal = bkd.asarray(np.array([0.5, -0.25]))
        wrapper = ActiveSetFunction(function, nominal, [0], bkd)
        reduced_samples = bkd.asarray(np.array([[0.1, 0.2, 0.3]]))
        full = bkd.asarray(
            np.array(
                [
                    [0.1, 0.2, 0.3],
                    [-0.25, -0.25, -0.25],
                ]
            )
        )
        bkd.assert_allclose(
            wrapper(reduced_samples), function(full), rtol=1e-15
        )
        _assert_empty_bundle(bkd, wrapper)

    def test_composition_timed_of_parallel(self, bkd):
        """The documented composition order timed(make_parallel(fn))."""
        function = _make_function(bkd)
        samples = self._samples(bkd)
        wrapper = timed(make_parallel(function, backend="sequential"))
        bkd.assert_allclose(wrapper(samples), function(samples), rtol=1e-15)
        _assert_empty_bundle(bkd, wrapper)


class TestMissingDerivativesRejected:
    """Objects without derivatives() fail fast with the remedy named."""

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda fn, bkd: timed(fn),
            lambda fn, bkd: make_parallel(fn, backend="sequential"),
            lambda fn, bkd: TrackedModel(fn, WorkTracker(bkd)),
            lambda fn, bkd: ActiveSetFunction(
                fn, bkd.asarray(np.array([0.0, 0.0])), [0], bkd
            ),
        ],
        ids=["timed", "make_parallel", "tracked_model", "active_set"],
    )
    def test_rejected_with_remedy(self, numpy_bkd, wrap):
        with pytest.raises(TypeError, match="Derivatives.none"):
            wrap(_NoDerivatives(numpy_bkd), numpy_bkd)
