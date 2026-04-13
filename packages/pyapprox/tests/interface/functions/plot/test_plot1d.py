import matplotlib.pyplot as plt

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.plot.plot1d import Plotter1D


class TestPlotter1D:
    def _setup(self, bkd):
        nqoi = 1
        nvars = 1

        def example_function(samples):
            return bkd.sin(samples)

        self.function = FunctionFromCallable(
            nqoi=nqoi, nvars=nvars, fun=example_function, bkd=bkd
        )
        self.plot_limits = [0, 10]
        self.plotter = Plotter1D(self.function, plot_limits=self.plot_limits)

    def test_plot_valid(self, bkd) -> None:
        self._setup(bkd)
        fig, ax = plt.subplots()
        result = self.plotter.plot(ax, npts_1d=100)
        assert isinstance(result, list)
        assert len(result) > 0
