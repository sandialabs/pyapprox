import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.contour import QuadContourSet
from typing import Any, Callable, List, Tuple, Union, Sequence
from mpl_toolkits.mplot3d import Axes3D

from pyapprox.typing.util.backend import Array
from pyapprox.typing.interface.functions.function import FunctionProtocol


class Plotter:
    def __init__(self, function: FunctionProtocol):
        self._bkd = function._bkd
        self._function = function

    def _plot_surface_1d(
        self,
        ax: Axes,
        qoi: int,
        plot_limits: Array,
        npts_1d: Array,
        **kwargs: Any,
    ) -> List[Any]:
        if plot_limits.shape != (2,):
            raise ValueError("")
        plot_xx = self._bkd.linspace(
            plot_limits[0], plot_limits[1], npts_1d[0]
        )[None, :]
        return ax.plot(
            self._bkd.to_numpy(plot_xx[0]),
            self._bkd.to_numpy(self._function(plot_xx)),
            **kwargs,
        )

    def get_meshgrid_samples(
        self,
        plot_limits: Array,
        num_pts_1d: Union[int, Sequence[int]],
        logspace: bool = False,
    ) -> Tuple[Array, Array, Array]:
        """
        Generate meshgrid samples.

        Parameters
        ----------
        plot_limits : Array
            The limits of the plot for each variable. Should be of shape (4,).
        num_pts_1d : int or Sequence[int]
            The number of points to use for each variable. If an integer is provided,
            it will be applied to both dimensions.
        bkd : Backend[Array]
            Backend for numerical operations.
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
        # Ensure num_pts_1d is a sequence of two integers
        num_pts_1d = (
            [num_pts_1d] * 2 if isinstance(num_pts_1d, int) else num_pts_1d
        )

        # Choose spacing function based on logspace flag
        space_fn = self._bkd.logspace if logspace else self._bkd.linspace

        # Generate x and y points
        x = space_fn(plot_limits[0], plot_limits[1], num_pts_1d[0])
        y = space_fn(plot_limits[2], plot_limits[3], num_pts_1d[1])

        # Create meshgrid
        X, Y = self._bkd.meshgrid(x, y)

        # Flatten meshgrid samples into a 2D array
        pts = self._bkd.stack(
            (self._bkd.flatten(X), self._bkd.flatten(Y)), axis=0
        )
        return X, Y, pts

    def _plot_surface_2d(
        self,
        ax: Axes3D,
        qoi: int,
        plot_limits: Array,
        npts_1d: Array,
        **kwargs: Any,
    ) -> QuadContourSet:
        if ax.name != "3d":
            raise ValueError("ax must use 3d projection")
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        vals = self._function(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
        return ax.plot_surface(X, Y, Z, **kwargs)

    def plot_surface(
        self,
        ax: Axes,
        plot_limits: Array,
        qoi: int = 0,
        npts_1d: Union[int, Sequence, Array] = 51,
        **kwargs: Any,
    ) -> QuadContourSet:
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars >= 3.")

        if isinstance(npts_1d, int):
            npts_1d = [npts_1d] * self.nvars()
        npts_1d = self._bkd.asarray(npts_1d)

        if len(npts_1d) != self.nvars():
            raise ValueError("npts_1d must be a list")

        if self.nvars() == 1:
            return self._plot_surface_1d(
                ax, qoi, plot_limits, npts_1d, **kwargs
            )
        return self._plot_surface_2d(ax, qoi, plot_limits, npts_1d, **kwargs)

    def get_plot_axis(
        self, figsize: Tuple[Any, Any] = (8, 6), surface: bool = False
    ) -> Tuple[Figure, Axes]:
        if self.nvars() < 3 and not surface:
            fig = plt.figure(figsize=figsize)
            return fig, fig.gca()
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(111, projection="3d")

    def plot_contours(
        self,
        ax: Axes,
        plot_limits: Array,
        qoi: int = 0,
        npts_1d: Union[int, Array] = 51,
        **kwargs: Any,
    ) -> QuadContourSet:
        if self.nvars() != 2:
            raise ValueError("Can only plot contours for 2D functions")
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        vals = self.__call__(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
        return ax.contourf(X, Y, Z, **kwargs)

    def plot(
        self,
        ax: Axes,
        plot_limits: Array,
        qoi: int = 0,
        npts_1d: Union[Array, int] = 51,
        **kwargs: Any,
    ) -> QuadContourSet:
        if ax.name == "3d" and self.nvars() != 1:
            return self.plot_surface(ax, plot_limits, qoi, npts_1d, **kwargs)
        return self.plot_contours(ax, plot_limits, qoi, npts_1d, **kwargs)
