"""Visualization for sample allocations.

Functions for plotting sample allocations across models and partitions.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.axes


def plot_allocation(
    nsamples: np.ndarray,
    model_names: Optional[List[str]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot sample allocation as a bar chart.

    Parameters
    ----------
    nsamples : np.ndarray
        Number of samples per model. Shape: (nmodels,)
    model_names : List[str], optional
        Names for each model. If None, uses "M0", "M1", etc.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to plt.bar().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> nsamples = np.array([100, 500, 2000])
    >>> ax = plot_allocation(nsamples, model_names=["HF", "MF", "LF"])
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    nmodels = len(nsamples)
    if model_names is None:
        model_names = [f"M{i}" for i in range(nmodels)]

    x = np.arange(nmodels)

    bars = ax.bar(x, nsamples, **kwargs)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Allocation")

    # Add value labels on bars
    for bar, n in zip(bars, nsamples):
        height = bar.get_height()
        ax.annotate(
            f"{int(n)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return ax


def plot_samples_per_model(
    results: Dict[str, np.ndarray],
    model_names: Optional[List[str]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot sample allocation comparison across estimators.

    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Dictionary mapping estimator names to sample allocations.
    model_names : List[str], optional
        Names for each model.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to plt.bar().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> results = {
    ...     "MFMC": np.array([100, 500, 2000]),
    ...     "MLMC": np.array([150, 300, 1500]),
    ... }
    >>> ax = plot_samples_per_model(results)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    estimator_names = list(results.keys())
    n_estimators = len(estimator_names)

    if n_estimators == 0:
        return ax

    # Get number of models from first result
    first_result = next(iter(results.values()))
    nmodels = len(first_result)

    if model_names is None:
        model_names = [f"M{i}" for i in range(nmodels)]

    x = np.arange(nmodels)
    width = 0.8 / n_estimators

    for i, (name, nsamples) in enumerate(results.items()):
        offset = (i - n_estimators / 2 + 0.5) * width
        ax.bar(x + offset, nsamples, width, label=name, **kwargs)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Allocation Comparison")
    ax.legend()

    return ax


def plot_allocation_matrix(
    allocation_mat: np.ndarray,
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: str = "Blues",
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot allocation matrix as a heatmap.

    Parameters
    ----------
    allocation_mat : np.ndarray
        Allocation matrix. Shape: (nmodels, npartitions)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str
        Colormap to use.
    **kwargs
        Additional arguments passed to plt.imshow().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    nmodels, npartitions = allocation_mat.shape

    im = ax.imshow(allocation_mat, cmap=cmap, aspect="auto", **kwargs)

    ax.set_xticks(np.arange(npartitions))
    ax.set_xticklabels([f"P{i}" for i in range(npartitions)])
    ax.set_yticks(np.arange(nmodels))
    ax.set_yticklabels([f"M{i}" for i in range(nmodels)])

    ax.set_xlabel("Partition")
    ax.set_ylabel("Model")
    ax.set_title("Allocation Matrix")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    return ax
