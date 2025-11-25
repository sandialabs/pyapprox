# plotter1d.py

from matplotlib.axes import Axes
from typing import Any, List, Sequence
from pyapprox.typing.interface.functions.function import (
    FunctionProtocol,
    validate_function,
)


class Plotter1D:
    """
    A plotter for 1D functions.

    Parameters
    ----------
    function : FunctionProtocol
        A function object implementing the FunctionProtocol interface.
    plot_limits : Sequence[int]
        The limits of the plot (x_min, x_max).
    """

    def __init__(self, function: FunctionProtocol, plot_limits: Sequence[Any]):
        validate_function(function)
        self._bkd = function._bkd
        self._function = function

        if len(plot_limits) != 2:
            raise ValueError(
                "plot_limits must have exactly 2 entries: [x_min, x_max]."
            )
        self._plot_limits = plot_limits

    def plot(
        self,
        ax: Axes,
        qoi: int = 0,
        npts_1d: int = 51,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Plot a 1D function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the surface.
        qoi : int, optional
            The quantity of interest to plot. Defaults to 0.
        npts_1d : int, optional
            The number of points to use for the plot. Defaults to 51.
        **kwargs : dict
            Additional arguments passed to the plot function.

        Returns
        -------
        List[Any]
            The result of the plotting operation.
        """
        plot_xx = self._bkd.linspace(
            self._plot_limits[0], self._plot_limits[1], npts_1d
        )[None, :]
        return ax.plot(
            self._bkd.to_numpy(plot_xx[0]),
            self._bkd.to_numpy(self._function(plot_xx))[qoi],
            **kwargs,
        )
