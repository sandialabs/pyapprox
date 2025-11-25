import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from typing import Any
from pyapprox.typing.util.backend import Array
from pyapprox.typing.interface.functions.function import (
    FunctionProtocol,
    validate_function,
)


class Plotter2DGeneralDomain:
    """
    A specialized plotter for 2D data on general domains.

    Parameters
    ----------
    function : FunctionProtocol
        A function object implementing the FunctionProtocol interface.

    Attributes
    ----------
    _bkd : Backend[Array]
        Backend for numerical operations.
    _function : FunctionProtocol
        The function to evaluate and plot.
    """

    def __init__(self, function: FunctionProtocol):
        validate_function(function)
        self._bkd = function._bkd
        self._function = function

    def plot_trisurf(
        self,
        ax: Axes3D,
        points: Array,
        qoi: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Plot a 2D surface on a non-rectangular domain using trisurf.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D
            The axis on which to plot the surface (must use 3D projection).
        points : Array
            The scattered points in the domain, of shape (2, nsamples).
        qoi : int, optional
            The quantity of interest to plot. Defaults to 0.
        **kwargs : dict
            Additional arguments passed to the trisurf function.

        Returns
        -------
        Any
            The result of the trisurf plotting operation.
        """
        if not isinstance(ax, Axes3D):
            raise ValueError("ax must use 3d projection.")

        if points.shape[0] != 2:
            raise ValueError("points must have shape (2, npoints).")

        # Evaluate the function at the scattered points
        vals = self._function(points)
        Z = vals[qoi]

        # Create a triangulation object
        triangulation = Triangulation(points[0, :], points[1, :])

        # Plot the trisurf
        return ax.plot_trisurf(
            points[0, :],
            points[1, :],
            Z,
            triangles=triangulation.triangles,
            **kwargs,
        )

    def plot_tricontour(
        self,
        ax: plt.Axes,
        points: Array,
        qoi: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Plot contours on a non-rectangular domain using tricontour.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the contours.
        points : Array
            The scattered points in the domain, of shape (n_points, 2).
        qoi : int, optional
            The quantity of interest to plot. Defaults to 0.
        **kwargs : dict
            Additional arguments passed to the tricontour function.

        Returns
        -------
        Any
            The result of the tricontour plotting operation.
        """
        if points.shape[0] != 2:
            raise ValueError("points must have shape (2, npoints).")

        # Evaluate the function at the scattered points
        vals = self._function(points)
        Z = vals[qoi]

        # Create a triangulation object
        triangulation = Triangulation(points[0, :], points[1, :])

        # Plot the tricontour
        return ax.tricontourf(
            points[0, :],
            points[1, :],
            Z,
            **kwargs,
        )
