"""Benchmark forward models must wrap in the generic interface wrappers.

Mirrors the tutorial usage that previously broke at docs-build time:
``build_cantilever_beam_1d(...)`` produces a derivative-free forward
model that tutorials wrap in ``timed()`` and ``make_parallel()``. This
exercises the same path with a tiny mesh so the guarantee is enforced by
the fast test suite instead of a 70-minute docs build.
"""

import numpy as np
import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.interface.functions.timing import timed
from pyapprox.interface.parallel import make_parallel
from pyapprox_benchmarks.pde.cantilever_beam import (
    build_cantilever_beam_1d,
)


class TestWrapBenchmarkModels:
    def _make_model(self, bkd):
        problem = build_cantilever_beam_1d(
            bkd, nx=10, num_kle_terms=2
        )
        return problem.function()

    def _samples(self, bkd, model):
        np.random.seed(0)
        return bkd.asarray(
            np.random.normal(0.0, 1.0, (model.nvars(), 3))
        )

    def test_timed_wraps_forward_model(self, bkd):
        model = self._make_model(bkd)
        samples = self._samples(bkd, model)
        wrapper = timed(model)
        bkd.assert_allclose(
            wrapper(samples), model(samples), rtol=1e-14
        )

    def test_make_parallel_wraps_forward_model(self, bkd):
        model = self._make_model(bkd)
        samples = self._samples(bkd, model)
        wrapper = make_parallel(model, backend="sequential")
        bkd.assert_allclose(
            wrapper(samples), model(samples), rtol=1e-14
        )

    def test_timed_of_parallel_composition(self, bkd):
        """The tutorial composition order: timed(make_parallel(fn))."""
        model = self._make_model(bkd)
        samples = self._samples(bkd, model)
        wrapper = timed(make_parallel(model, backend="sequential"))
        bkd.assert_allclose(
            wrapper(samples), model(samples), rtol=1e-14
        )
