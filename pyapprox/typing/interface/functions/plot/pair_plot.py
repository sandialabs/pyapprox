"""Lower-triangular pair plot for multivariate functions.

Displays 1D reduced functions on the diagonal and 2D reduced functions
as contours on the lower triangle.  Reuses Plotter1D and
Plotter2DRectangularDomain.

Two construction paths:

1. ``__init__`` – uses any ``DimensionReducerProtocol`` (e.g.
   ``FunctionMarginalizer`` for integration, ``CrossSectionReducer``
   for cross-sections at nominal values).
2. ``from_functions`` – accepts pre-built 1D / 2D functions directly
   (e.g. independent marginals with no quadrature).
"""

from typing import Any, Dict, Generic, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.marginalize import (
    DimensionReducerProtocol,
    ReducedFunction,
)
from pyapprox.typing.interface.functions.plot.plot1d import Plotter1D
from pyapprox.typing.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain,
)


class PairPlotter(Generic[Array]):
    """Lower-triangular pair plot of 1D and 2D reduced functions.

    Diagonal panels show 1D functions (line plots via Plotter1D).
    Lower-triangle panels show 2D functions (contour plots via
    Plotter2DRectangularDomain).  Upper-triangle panels are hidden.

    Parameters
    ----------
    reducer : DimensionReducerProtocol[Array]
        Dimension reducer wrapping the target function.  Any object
        satisfying ``DimensionReducerProtocol`` works, including
        ``FunctionMarginalizer`` and ``CrossSectionReducer``.
    plot_limits : Array
        Domain bounds, shape ``(nvars, 2)`` with ``[lb, ub]`` per
        variable.
    bkd : Backend[Array]
        Computational backend.
    variable_names : List[str] or None
        Axis labels.  If *None*, variables are labelled
        ``x_0, x_1, ...``.
    """

    def __init__(
        self,
        reducer: DimensionReducerProtocol[Array],
        plot_limits: Array,
        bkd: Backend[Array],
        variable_names: Optional[List[str]] = None,
    ):
        self._reducer = reducer
        self._plot_limits = plot_limits
        self._bkd = bkd
        self._nvars = reducer.nvars()
        self._variable_names = variable_names or [
            f"$x_{{{i}}}$" for i in range(self._nvars)
        ]
        self._functions_1d: Optional[
            List[ReducedFunction[Array]]
        ] = None
        self._functions_2d: Optional[
            Dict[Tuple[int, int], ReducedFunction[Array]]
        ] = None

    @classmethod
    def from_functions(
        cls,
        functions_1d: List[ReducedFunction[Array]],
        functions_2d: Dict[
            Tuple[int, int], ReducedFunction[Array]
        ],
        plot_limits: Array,
        bkd: Backend[Array],
        variable_names: Optional[List[str]] = None,
    ) -> "PairPlotter[Array]":
        """Build from pre-computed 1D and 2D functions.

        This bypasses dimension reduction entirely, e.g. for
        ``IndependentJoint`` where 2D marginals are products of
        1D marginals.

        Parameters
        ----------
        functions_1d : List[ReducedFunction[Array]]
            One 1D function per variable (for diagonal panels).
        functions_2d : Dict[Tuple[int, int], ReducedFunction[Array]]
            2D functions keyed by ``(row, col)`` with ``row > col``
            (for lower-triangle panels).
        plot_limits : Array
            Shape ``(nvars, 2)`` with ``[lb, ub]`` per variable.
        bkd : Backend[Array]
            Computational backend.
        variable_names : List[str] or None
            Axis labels.
        """
        obj = cls.__new__(cls)
        obj._reducer = None  # type: ignore[assignment]
        obj._plot_limits = plot_limits
        obj._bkd = bkd
        obj._nvars = len(functions_1d)
        obj._variable_names = variable_names or [
            f"$x_{{{i}}}$" for i in range(obj._nvars)
        ]
        obj._functions_1d = functions_1d
        obj._functions_2d = functions_2d
        return obj

    def _get_1d_function(self, idx: int) -> ReducedFunction[Array]:
        if self._functions_1d is not None:
            return self._functions_1d[idx]
        return self._reducer.reduce([idx])

    def _get_2d_function(
        self, row: int, col: int
    ) -> ReducedFunction[Array]:
        if self._functions_2d is not None:
            return self._functions_2d[(row, col)]
        return self._reducer.reduce([col, row])

    def _plot_limits_1d(self, idx: int) -> List[float]:
        return [
            float(self._plot_limits[idx, 0]),
            float(self._plot_limits[idx, 1]),
        ]

    def _plot_limits_2d(self, row: int, col: int) -> List[float]:
        return [
            float(self._plot_limits[col, 0]),
            float(self._plot_limits[col, 1]),
            float(self._plot_limits[row, 0]),
            float(self._plot_limits[row, 1]),
        ]

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def plot(
        self,
        fig: Optional[Figure] = None,
        axes: Optional[Any] = None,
        npts_1d: int = 51,
        contour_kwargs: Optional[Dict[str, Any]] = None,
        line_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Figure, Any]:
        """Draw the pair plot.

        Parameters
        ----------
        fig : Figure or None
            Matplotlib figure.  Created if *None*.
        axes : ndarray of Axes or None
            ``(nvars, nvars)`` array of axes.  Created if *None*.
        npts_1d : int
            Points per dimension for plotting.
        contour_kwargs : dict or None
            Extra kwargs forwarded to ``ax.contourf``.
        line_kwargs : dict or None
            Extra kwargs forwarded to ``ax.plot``.

        Returns
        -------
        Tuple[Figure, ndarray of Axes]
        """
        n = self._nvars
        contour_kwargs = contour_kwargs or {}
        line_kwargs = line_kwargs or {}

        if fig is None or axes is None:
            fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))

        for i in range(n):
            for j in range(n):
                ax: Axes = axes[i, j]
                if j > i:
                    ax.axis("off")
                elif i == j:
                    fn = self._get_1d_function(i)
                    plotter = Plotter1D(fn, self._plot_limits_1d(i))
                    plotter.plot(ax, npts_1d=npts_1d, **line_kwargs)
                else:
                    fn = self._get_2d_function(i, j)
                    plotter2d = Plotter2DRectangularDomain(
                        fn, self._plot_limits_2d(i, j)
                    )
                    plotter2d.plot_contours(
                        ax, npts_1d=npts_1d, **contour_kwargs
                    )

        # Axis labels: bottom row gets x-labels, left column gets y-labels
        for j in range(n):
            axes[n - 1, j].set_xlabel(self._variable_names[j])
        for i in range(n):
            axes[i, 0].set_ylabel(self._variable_names[i])

        return fig, axes
