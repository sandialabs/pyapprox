from typing import Any, Generic, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.validation import (
    validate_function,
)
from pyapprox.util.backends.protocols import Array


class Plotter1D(Generic[Array]):
    """
    A plotter for 1D functions.

    Parameters
    ----------
    function : FunctionProtocol
        A function object implementing the FunctionProtocol interface.
    plot_limits : Sequence[int]
        The limits of the plot (x_min, x_max).
    """

    def __init__(
        self,
        function: FunctionProtocol[Array],
        plot_limits: Union[Sequence[Any], Array],
    ):
        validate_function(function)
        if function.nvars() != 1:
            raise ValueError("Can only plot functions with nvars() == 1")
        self._bkd = function.bkd()
        self._function = function
        self._validate_plot_limits(plot_limits)
        self._plot_limits = plot_limits

    def _validate_plot_limits(
        self, plot_limits: Union[Sequence[float], Array],
    ) -> None:
        if len(plot_limits) != 2:
            raise ValueError("plot_limits must have exactly 2 entries: [x_min, x_max].")
        if self._bkd.any_bool(~self._bkd.isfinite(self._bkd.asarray(plot_limits))):
            raise ValueError(f"plot limits {plot_limits} must be finite")

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

    def __repr__(self) -> str:
        return "{0}({1})".format(self.__class__.__name__, self._plot_limits)

    def figure(self) -> Tuple[Figure, Axes]:
        return plt.subplots(1, 1, figsize=(8, 6))
