"""Tests for PairPlotter."""

import unittest
from typing import Any, Generic

import matplotlib

matplotlib.use("Agg")

import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.marginalize import (
    CrossSectionReducer,
    FunctionMarginalizer,
    ReducedFunction,
)
from pyapprox.interface.functions.plot.pair_plot import PairPlotter
from pyapprox.surrogates.quadrature.tensor_product_factory import (
    TensorProductQuadratureFactory,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _make_polynomial_3d(bkd: Backend[Array]):
    """f(x, y, z) = x*y*z on [0,1]^3, nqoi=1."""

    class _Poly3D:
        def bkd(self):
            return bkd

        def nvars(self):
            return 3

        def nqoi(self):
            return 1

        def __call__(self, samples):
            return bkd.reshape(samples[0] * samples[1] * samples[2], (1, -1))

    return _Poly3D()


def _make_factory(domain, bkd, npoints=5):
    nvars = domain.shape[0]
    return TensorProductQuadratureFactory([npoints] * nvars, domain, bkd)


class TestPairPlotter(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_upper_triangle_off(self) -> None:
        """Upper-triangle axes should be invisible."""
        bkd = self._bkd
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        n = 3
        for i in range(n):
            for j in range(i + 1, n):
                self.assertFalse(axes[i, j].axison)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_diagonal_has_line(self) -> None:
        """Diagonal axes should contain line plots."""
        bkd = self._bkd
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        for i in range(3):
            self.assertTrue(len(axes[i, i].lines) > 0)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_lower_triangle_has_contours(self) -> None:
        """Lower-triangle axes should contain filled contours."""
        bkd = self._bkd
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        for i in range(1, 3):
            for j in range(i):
                self.assertTrue(len(axes[i, j].collections) > 0)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_from_functions(self) -> None:
        """PairPlotter.from_functions produces correct layout."""
        bkd = self._bkd

        def make_1d(val):
            def fn(samples):
                return bkd.ones((1, samples.shape[1])) * val

            return ReducedFunction(1, 1, fn, bkd)

        def make_2d(val):
            def fn(samples):
                return bkd.ones((1, samples.shape[1])) * val

            return ReducedFunction(2, 1, fn, bkd)

        functions_1d = [make_1d(1.0), make_1d(2.0), make_1d(3.0)]
        functions_2d = {
            (1, 0): make_2d(1.0),
            (2, 0): make_2d(2.0),
            (2, 1): make_2d(3.0),
        }
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        plotter = PairPlotter.from_functions(
            functions_1d,
            functions_2d,
            domain,
            bkd,
            variable_names=["a", "b", "c"],
        )
        fig, axes = plotter.plot(npts_1d=5)
        # Check labels
        self.assertEqual(axes[2, 0].get_xlabel(), "a")
        self.assertEqual(axes[2, 1].get_xlabel(), "b")
        self.assertEqual(axes[2, 2].get_xlabel(), "c")
        self.assertEqual(axes[0, 0].get_ylabel(), "a")
        self.assertEqual(axes[1, 0].get_ylabel(), "b")
        self.assertEqual(axes[2, 0].get_ylabel(), "c")
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_cross_section_reducer(self) -> None:
        """PairPlotter with CrossSectionReducer produces plots."""
        bkd = self._bkd
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        nominal = bkd.asarray([0.5, 0.5, 0.5])
        reducer = CrossSectionReducer(func, nominal, bkd)
        plotter = PairPlotter(reducer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        # Diagonal should have lines
        for i in range(3):
            self.assertTrue(len(axes[i, i].lines) > 0)
        # Lower triangle should have contours
        for i in range(1, 3):
            for j in range(i):
                self.assertTrue(len(axes[i, j].collections) > 0)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_nvars(self) -> None:
        """nvars() returns correct value."""
        bkd = self._bkd
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        self.assertEqual(plotter.nvars(), 3)


class TestPairPlotterNumpy(TestPairPlotter[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPairPlotterTorch(TestPairPlotter[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
