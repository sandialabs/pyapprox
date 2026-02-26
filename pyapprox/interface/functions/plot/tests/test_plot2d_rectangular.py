import matplotlib.pyplot as plt

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain,
    meshgrid_samples,
)


class TestPlotter2DRectangularDomain:
    def _setup(self, bkd):
        nqoi = 1
        nvars = 2

        def example_function(samples):
            x, y = samples[0], samples[1]
            return bkd.stack([bkd.sin(x) * bkd.cos(y)], axis=0)

        self.function = FunctionFromCallable(
            nqoi=nqoi,
            nvars=nvars,
            fun=example_function,
            bkd=bkd,
        )
        self.plot_limits = [0, 10, 0, 10]
        self.plotter = Plotter2DRectangularDomain(self.function, self.plot_limits)

    def test_meshgrid_samples(self, bkd) -> None:
        self._setup(bkd)
        X, Y, pts = meshgrid_samples(50, self.plotter._plot_limits, bkd)
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert pts.shape == (2, 2500)  # Flattened meshgrid samples
        bkd.assert_allclose(X[0, :], bkd.linspace(0, 10, 50))
        bkd.assert_allclose(Y[:, 0], bkd.linspace(0, 10, 50))

    def test_plot_surface(self, bkd) -> None:
        self._setup(bkd)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self.plotter.plot_surface(ax, qoi=0, npts_1d=50)

    def test_plot_contours(self, bkd) -> None:
        self._setup(bkd)
        fig, ax = plt.subplots()
        self.plotter.plot_contours(ax, qoi=0, npts_1d=50)

    def test_plot(self, bkd) -> None:
        self._setup(bkd)
        # Test 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self.plotter.plot(ax, qoi=0, npts_1d=50)

        # Test 2D plot
        fig, ax = plt.subplots()
        self.plotter.plot(ax, qoi=0, npts_1d=50)
