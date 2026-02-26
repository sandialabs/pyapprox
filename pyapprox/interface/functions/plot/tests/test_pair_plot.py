"""Tests for PairPlotter."""

import matplotlib

matplotlib.use("Agg")

from pyapprox.interface.functions.marginalize import (
    CrossSectionReducer,
    FunctionMarginalizer,
    ReducedFunction,
)
from pyapprox.interface.functions.plot.pair_plot import PairPlotter
from pyapprox.surrogates.quadrature.tensor_product_factory import (
    TensorProductQuadratureFactory,
)
from pyapprox.util.backends.protocols import Array, Backend


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


class TestPairPlotter:
    def test_upper_triangle_off(self, bkd) -> None:
        """Upper-triangle axes should be invisible."""
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        n = 3
        for i in range(n):
            for j in range(i + 1, n):
                assert not axes[i, j].axison
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_diagonal_has_line(self, bkd) -> None:
        """Diagonal axes should contain line plots."""
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        for i in range(3):
            assert len(axes[i, i].lines) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_lower_triangle_has_contours(self, bkd) -> None:
        """Lower-triangle axes should contain filled contours."""
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        for i in range(1, 3):
            for j in range(i):
                assert len(axes[i, j].collections) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_from_functions(self, bkd) -> None:
        """PairPlotter.from_functions produces correct layout."""

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
        assert axes[2, 0].get_xlabel() == "a"
        assert axes[2, 1].get_xlabel() == "b"
        assert axes[2, 2].get_xlabel() == "c"
        assert axes[0, 0].get_ylabel() == "a"
        assert axes[1, 0].get_ylabel() == "b"
        assert axes[2, 0].get_ylabel() == "c"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_cross_section_reducer(self, bkd) -> None:
        """PairPlotter with CrossSectionReducer produces plots."""
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        nominal = bkd.asarray([0.5, 0.5, 0.5])
        reducer = CrossSectionReducer(func, nominal, bkd)
        plotter = PairPlotter(reducer, domain, bkd)
        fig, axes = plotter.plot(npts_1d=5)
        # Diagonal should have lines
        for i in range(3):
            assert len(axes[i, i].lines) > 0
        # Lower triangle should have contours
        for i in range(1, 3):
            for j in range(i):
                assert len(axes[i, j].collections) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_nvars(self, bkd) -> None:
        """nvars() returns correct value."""
        func = _make_polynomial_3d(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _make_factory(domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)
        plotter = PairPlotter(marginalizer, domain, bkd)
        assert plotter.nvars() == 3
