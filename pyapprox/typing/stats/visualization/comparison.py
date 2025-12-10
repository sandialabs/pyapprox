"""Visualization for estimator comparisons.

Functions for comparing estimator performance.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.axes


def plot_estimator_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = "variance",
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot comparison of estimators by a metric.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results from compare_estimators function.
    metric : str
        Metric to plot. Default: "variance".
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
    >>> from pyapprox.typing.stats import compare_estimators
    >>> # results = compare_estimators(stat, costs, bkd, target_cost=100.0)
    >>> # ax = plot_estimator_comparison(results)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    # Filter valid results
    valid_results = {
        name: info
        for name, info in results.items()
        if "error" not in info and metric in info
    }

    if not valid_results:
        ax.text(0.5, 0.5, "No valid results", ha="center", va="center")
        return ax

    names = list(valid_results.keys())
    values = [valid_results[name][metric] for name in names]

    x = np.arange(len(names))

    bars = ax.bar(x, values, **kwargs)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Estimator")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Estimator Comparison: {metric}")

    plt.tight_layout()

    return ax


def plot_variance_vs_cost(
    costs: List[float],
    variances: Dict[str, List[float]],
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot variance vs computational cost for multiple estimators.

    Parameters
    ----------
    costs : List[float]
        List of target costs.
    variances : Dict[str, List[float]]
        Dictionary mapping estimator names to lists of variances at each cost.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to plt.loglog().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> costs = [10, 100, 1000]
    >>> variances = {"MC": [0.1, 0.01, 0.001], "MFMC": [0.05, 0.005, 0.0005]}
    >>> ax = plot_variance_vs_cost(costs, variances)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    for name, var_list in variances.items():
        ax.loglog(costs, var_list, marker="o", label=name, **kwargs)

    ax.set_xlabel("Computational Cost")
    ax.set_ylabel("Variance")
    ax.set_title("Variance vs Cost")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    return ax


def plot_variance_reduction(
    results: Dict[str, Dict[str, Any]],
    baseline: str = "mc",
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot variance reduction relative to baseline.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results from compare_estimators function.
    baseline : str
        Baseline estimator name. Default: "mc".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to plt.bar().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    if baseline not in results or "variance" not in results[baseline]:
        ax.text(0.5, 0.5, f"Baseline '{baseline}' not found", ha="center", va="center")
        return ax

    baseline_var = results[baseline]["variance"]

    # Compute reductions
    reductions = {}
    for name, info in results.items():
        if "variance" in info and info["variance"] > 0:
            reductions[name] = baseline_var / info["variance"]

    names = list(reductions.keys())
    values = [reductions[name] for name in names]

    x = np.arange(len(names))

    bars = ax.bar(x, values, **kwargs)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Estimator")
    ax.set_ylabel(f"Variance Reduction (vs {baseline.upper()})")
    ax.set_title("Variance Reduction")
    ax.axhline(y=1.0, color="r", linestyle="--", label="Baseline")

    plt.tight_layout()

    return ax
