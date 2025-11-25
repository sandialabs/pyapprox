# plotter2d_rectangular.py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.contour import QuadContourSet
from matplotlib.axes import Axes
from typing import Any, Sequence, Union, Tuple
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.function import (
    FunctionProtocol,
    validate_function,
)


def meshgrid_samples(
    num_pts_1d: Union[int, Sequence[int]],
    plot_limits: Sequence[Any],
    bkd: Backend[Array],
    logspace: bool = False,
) -> Tuple[Array, Array, Array]:
    """
    Generate meshgrid samples for 2D plotting.

    Parameters
    ----------
    num_pts_1d : int or Sequence[int]
        The number of points to use for each variable.
        If an integer is provided,
        it will be applied to both dimensions.
    plot_limits : Sequence[int]
        The limits of the plot (x_min, x_max, y_min, y_max).
    logspace : bool, optional
        Whether to use logarithmic spacing. Defaults to False.

    Returns
    -------
    X : Array
        The meshgrid for the first variable.
    Y : Array
        The meshgrid for the second variable.
    pts : Array
        The flattened meshgrid samples.
    """
    if len(plot_limits) != 4:
        raise ValueError(
            "plot_limits must have exactly 4 entries: "
            "[x_min, x_max, y_min, y_max]."
        )
    num_pts_1d = (
        [num_pts_1d] * 2 if isinstance(num_pts_1d, int) else num_pts_1d
    )
    space_fn = bkd.logspace if logspace else bkd.linspace
    x = space_fn(plot_limits[0], plot_limits[1], num_pts_1d[0])
    y = space_fn(plot_limits[2], plot_limits[3], num_pts_1d[1])
    X, Y = bkd.meshgrid(x, y)
    pts = bkd.stack((bkd.flatten(X), bkd.flatten(Y)), axis=0)
    return X, Y, pts


class Plotter2DRectangularDomain:
    """
    A plotter for 2D Function on rectangular domains.

    Parameters
    ----------
    function : FunctionProtocol
        A function object implementing the FunctionProtocol interface.
    plot_limits : Sequence[int]
        The limits of the plot (x_min, x_max, y_min, y_max).

    Attributes
    ----------
    _bkd : Backend[Array]
        Backend for numerical operations.
    _function : FunctionProtocol
        The function to evaluate and plot.
    """

    def __init__(self, function: FunctionProtocol, plot_limits: Sequence[Any]):
        validate_function(function)
        self._bkd = function._bkd

        self._function = function

        if len(plot_limits) != 4:
            raise ValueError(
                "plot_limits must have exactly 4 entries: "
                "[x_min, x_max, y_min, y_max]."
            )
        self._plot_limits = plot_limits

    def plot_surface(
        self,
        ax: Axes3D,
        qoi: int,
        npts_1d: Union[int, Sequence[int]],
        **kwargs: Any,
    ) -> QuadContourSet:
        """
        Plot a 2D surface of the function.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D
            The axis on which to plot the surface (must use 3D projection).
        qoi : int
            The quantity of interest to plot.
        npts_1d : int or Array
            The number of points to use for each variable.
        **kwargs : dict
            Additional arguments passed to the plot_surface function.

        Returns
        -------
        QuadContourSet
            The plotted surface.
        """
        if ax.name != "3d":
            raise ValueError("ax must use 3d projection.")
        X, Y, pts = meshgrid_samples(npts_1d, self._plot_limits, self._bkd)
        vals = self._function(pts)
        Z = self._bkd.reshape(vals[qoi], X.shape)
        return ax.plot_surface(X, Y, Z, **kwargs)

    def plot_contours(
        self,
        ax: Axes,
        qoi: int = 0,
        npts_1d: Union[int, Sequence[int]] = 51,
        **kwargs: Any,
    ) -> QuadContourSet:
        """
        Plot contours of the function for 2D inputs.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the contours.
        plot_limits : Array
            The limits of the plot for each variable.
        qoi : int, optional
            The quantity of interest to plot. Defaults to 0.
        npts_1d : int or Array, optional
            The number of points to use for each variable. Defaults to 51.
        **kwargs : dict
            Additional arguments passed to the contourf function.

        Returns
        -------
        QuadContourSet
            The plotted contours.
        """
        X, Y, pts = meshgrid_samples(npts_1d, self._plot_limits, self._bkd)
        vals = self._function(pts)
        Z = self._bkd.reshape(vals[qoi], X.shape)
        return ax.contourf(X, Y, Z, **kwargs)

    def plot(
        self,
        ax: Union[Axes, Axes3D],
        qoi: int = 0,
        npts_1d: Union[int, Sequence[int]] = 51,
        **kwargs: Any,
    ) -> Any:
        """
        Plot data using the appropriate plotter.

        Parameters
        ----------
        ax : Union[matplotlib.axes.Axes, mpl_toolkits.mplot3d.Axes3D]
            The axis on which to plot the data.
        qoi : int, optional
            The quantity of interest to plot. Defaults to 0.
        npts_1d : int or Array, optional
            The number of points to use for the plot. Defaults to 51.
        **kwargs : dict
            Additional arguments passed to the plotting functions.

        Returns
        -------
        Any
            The result of the plotting operation.
        """
        if isinstance(ax, Axes3D):
            return self.plot_surface(ax, qoi, npts_1d, **kwargs)
        return self.plot_contours(ax, qoi, npts_1d, **kwargs)
