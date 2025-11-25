# plotter2d_rectangular.py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.contour import QuadContourSet
from matplotlib.axes import Axes
from typing import Any, Sequence, Union, Tuple
from pyapprox.typing.util.backend import Array
from pyapprox.typing.interface.functions.function import FunctionProtocol


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
        self._bkd = function._bkd
        self._function = function

        if len(plot_limits) != 4:
            raise ValueError(
                "plot_limits must have exactly 4 entries: "
                "[x_min, x_max, y_min, y_max]."
            )
        self._plot_limits = plot_limits

    def meshgrid_samples(
        self,
        num_pts_1d: Union[int, Sequence[int]],
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
        num_pts_1d = (
            [num_pts_1d] * 2 if isinstance(num_pts_1d, int) else num_pts_1d
        )
        space_fn = self._bkd.logspace if logspace else self._bkd.linspace
        x = space_fn(self._plot_limits[0], self._plot_limits[1], num_pts_1d[0])
        y = space_fn(self._plot_limits[2], self._plot_limits[3], num_pts_1d[1])
        X, Y = self._bkd.meshgrid(x, y)
        pts = self._bkd.stack(
            (self._bkd.flatten(X), self._bkd.flatten(Y)), axis=0
        )
        return X, Y, pts

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
        X, Y, pts = self.meshgrid_samples(npts_1d)
        vals = self._function(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
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
        X, Y, pts = self.meshgrid_samples(npts_1d)
        vals = self._function(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
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
