"""Visualization for correlation matrices.

Functions for plotting model correlations.
"""

from typing import Optional, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.axes


def plot_correlation_matrix(
    cov: np.ndarray,
    model_names: Optional[List[str]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> "matplotlib.axes.Axes":
    """Plot correlation matrix as a heatmap.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix. Shape: (nmodels, nmodels)
    model_names : List[str], optional
        Names for each model. If None, uses "M0", "M1", etc.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str
        Colormap to use. Default: "RdBu_r".
    annotate : bool
        Whether to annotate cells with correlation values. Default: True.
    vmin : float
        Minimum value for colormap. Default: -1.0.
    vmax : float
        Maximum value for colormap. Default: 1.0.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> cov = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    >>> ax = plot_correlation_matrix(cov, model_names=["HF", "MF", "LF"])
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    nmodels = cov.shape[0]

    if model_names is None:
        model_names = [f"M{i}" for i in range(nmodels)]

    # Convert covariance to correlation
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
        corr = np.nan_to_num(corr, nan=0.0)

    # Clip to [-1, 1] for numerical stability
    corr = np.clip(corr, -1, 1)

    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(np.arange(nmodels))
    ax.set_xticklabels(model_names)
    ax.set_yticks(np.arange(nmodels))
    ax.set_yticklabels(model_names)

    ax.set_title("Model Correlation Matrix")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Annotate cells
    if annotate:
        for i in range(nmodels):
            for j in range(nmodels):
                color = "white" if abs(corr[i, j]) > 0.5 else "black"
                ax.text(
                    j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=8
                )

    return ax


def plot_covariance_matrix(
    cov: np.ndarray,
    model_names: Optional[List[str]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: str = "viridis",
    annotate: bool = True,
) -> "matplotlib.axes.Axes":
    """Plot covariance matrix as a heatmap.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix. Shape: (nmodels, nmodels)
    model_names : List[str], optional
        Names for each model.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str
        Colormap to use.
    annotate : bool
        Whether to annotate cells with values.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    nmodels = cov.shape[0]

    if model_names is None:
        model_names = [f"M{i}" for i in range(nmodels)]

    im = ax.imshow(cov, cmap=cmap, aspect="equal")

    ax.set_xticks(np.arange(nmodels))
    ax.set_xticklabels(model_names)
    ax.set_yticks(np.arange(nmodels))
    ax.set_yticklabels(model_names)

    ax.set_title("Model Covariance Matrix")

    plt.colorbar(im, ax=ax)

    if annotate:
        vmax = np.max(np.abs(cov))
        for i in range(nmodels):
            for j in range(nmodels):
                color = "white" if abs(cov[i, j]) > 0.5 * vmax else "black"
                ax.text(
                    j, i, f"{cov[i, j]:.2e}",
                    ha="center", va="center", color=color, fontsize=7
                )

    return ax
