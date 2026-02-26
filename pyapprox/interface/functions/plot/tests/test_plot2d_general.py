import matplotlib.pyplot as plt

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.plot.plot2d_general import (
    Plotter2DGeneralDomain,
)
from pyapprox.interface.functions.plot.plot2d_rectangular import (
    meshgrid_samples,
)


class TestPlotter2DGeneralDomain:
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
        self.plotter = Plotter2DGeneralDomain(self.function)

        # Generate scattered points in the domain
        X, Y, pts = meshgrid_samples(50, [-1, 1, -1, 1], bkd)
        self.points = pts

    def test_plot_trisurf(self, bkd) -> None:
        self._setup(bkd)
        # just check plot runs
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self.plotter.plot_trisurf(ax, self.points, qoi=0)

    def test_plot_tricontour(self, bkd) -> None:
        self._setup(bkd)
        # just check plot runs
        fig, ax = plt.subplots()
        self.plotter.plot_tricontour(ax, self.points, qoi=0)
