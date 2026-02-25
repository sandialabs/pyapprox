"""
Plotting utilities for kernel matrices.

This module provides functions to visualize kernel matrices as heatmaps.
For multi-output kernels, it draws red boxes around each kernel block.
"""

from typing import Tuple, List, Optional, Union
import numpy as np

from pyapprox.util.backends.protocols import Array


def plot_kernel_matrix_heatmap(
    kernel,
    bounds: Tuple[float, float],
    ax,
    npts: int = 50,
    block_sizes: Optional[List[int]] = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    **imshow_kwargs
):
    """
    Plot the kernel matrix as a heatmap.

    Parameters
    ----------
    kernel : Kernel
        The kernel to evaluate. Can be a single-output or multi-output kernel.
    bounds : Tuple[float, float]
        The bounds (lower, upper) for generating sample points.
    ax : matplotlib.axes.Axes
        The matplotlib axes to plot on.
    npts : int, optional
        Number of sample points per output. Default is 50.
    block_sizes : List[int], optional
        For multi-output kernels, the size of each output block.
        If provided, red boxes will be drawn around each kernel block.
    cmap : str, optional
        The colormap to use. Default is "viridis".
    colorbar : bool, optional
        Whether to add a colorbar. Default is True.
    **imshow_kwargs
        Additional keyword arguments passed to ax.imshow.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The image object from imshow.
    """
    # Generate sample points
    x = np.linspace(bounds[0], bounds[1], npts)

    # Check if kernel is multi-output by checking for noutputs method
    is_multi_output = hasattr(kernel, 'noutputs') and kernel.noutputs() > 1

    if is_multi_output:
        # For multi-output kernels, create list of sample arrays
        noutputs = kernel.noutputs()
        nvars = kernel.nvars()

        # Create sample points for each output (same points for all)
        X_single = kernel._bkd.array(x.reshape(1, -1)) if nvars == 1 else kernel._bkd.array(
            np.tile(x.reshape(1, -1), (nvars, 1))[:, :npts]
        )
        X_list = [X_single for _ in range(noutputs)]

        # Compute full kernel matrix
        K = kernel(X_list)
        K_plot = kernel._bkd.to_numpy(K)

        # Determine block sizes
        if block_sizes is None:
            block_sizes = [npts] * noutputs
    else:
        # Single output kernel
        nvars = kernel.nvars()

        # Create sample array
        if nvars == 1:
            X = kernel._bkd.array(x.reshape(1, -1))
        else:
            # For multi-dimensional, use same values for each dimension
            X = kernel._bkd.array(
                np.tile(x.reshape(1, -1), (nvars, 1))[:, :npts]
            )

        # Compute kernel matrix
        K = kernel(X, X)
        K_plot = kernel._bkd.to_numpy(K)
        block_sizes = None

    # Plot heatmap
    im = ax.imshow(K_plot, cmap=cmap, aspect='auto', **imshow_kwargs)

    # Add colorbar
    if colorbar:
        from matplotlib import pyplot as plt
        plt.colorbar(im, ax=ax)

    # Draw block boundaries for multi-output kernels
    if block_sizes is not None and len(block_sizes) > 1:
        _draw_block_boundaries(ax, block_sizes)

    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')

    return im


def _draw_block_boundaries(ax, block_sizes: List[int], color: str = 'red', linewidth: float = 2.0):
    """
    Draw red boxes around kernel blocks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes to draw on.
    block_sizes : List[int]
        The size of each block.
    color : str, optional
        Color for the boundary lines. Default is 'red'.
    linewidth : float, optional
        Line width for boundaries. Default is 2.0.
    """
    from matplotlib.patches import Rectangle

    cumulative = np.cumsum([0] + block_sizes)
    n_blocks = len(block_sizes)

    for i in range(n_blocks):
        for j in range(n_blocks):
            # Get block boundaries (note: imshow has origin at top-left)
            x_start = cumulative[j] - 0.5
            y_start = cumulative[i] - 0.5
            width = block_sizes[j]
            height = block_sizes[i]

            # Draw rectangle around each block
            rect = Rectangle(
                (x_start, y_start),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=linewidth
            )
            ax.add_patch(rect)


def plot_kernel_matrix_surface(
    kernel,
    bounds: Tuple[float, float],
    ax,
    npts: int = 50,
    **kwargs
):
    """
    Plot the kernel matrix as a 3D surface.

    Note: This function is a placeholder for API compatibility.
    For complex visualization needs, consider using the heatmap function.

    Parameters
    ----------
    kernel : Kernel
        The kernel to evaluate.
    bounds : Tuple[float, float]
        The bounds (lower, upper) for generating sample points.
    ax : matplotlib.axes.Axes3D
        The 3D matplotlib axes to plot on.
    npts : int, optional
        Number of sample points. Default is 50.
    **kwargs
        Additional keyword arguments passed to ax.plot_surface.

    Returns
    -------
    surf : mpl_toolkits.mplot3d.art3d.Poly3DCollection
        The surface plot object.
    """
    # Generate sample points
    x = np.linspace(bounds[0], bounds[1], npts)

    nvars = kernel.nvars()

    # Create sample array
    if nvars == 1:
        X = kernel._bkd.array(x.reshape(1, -1))
    else:
        X = kernel._bkd.array(
            np.tile(x.reshape(1, -1), (nvars, 1))[:, :npts]
        )

    # Compute kernel matrix
    K = kernel(X, X)
    K_plot = kernel._bkd.to_numpy(K)

    # Create meshgrid for 3D plot
    xx, yy = np.meshgrid(np.arange(npts), np.arange(npts))

    # Plot surface
    surf = ax.plot_surface(xx, yy, K_plot, **kwargs)

    ax.set_xlabel('Sample index i')
    ax.set_ylabel('Sample index j')
    ax.set_zlabel('K(x_i, x_j)')

    return surf
