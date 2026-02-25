"""Sensitivity analysis visualization functions.

This module provides plotting functions for visualizing sensitivity
analysis results including main effects, total effects, and Sobol indices.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyapprox.util.backends.protocols import Array, Backend


def plot_main_effects(
    main_effects: Array,
    ax: Axes,
    bkd: Backend[Array],
    truncation_pct: float = 0.95,
    max_slices: int = 5,
    rv: str = "z",
    qoi: int = 0,
) -> list:
    """Plot main effects as a pie chart.

    Shows the relative contribution of each variable to output variance.

    Parameters
    ----------
    main_effects : Array
        Shape (nvars, nqoi) or (nvars,) - main effect sensitivity indices.
    ax : Axes
        Matplotlib axes for plotting.
    bkd : Backend[Array]
        Backend for array operations.
    truncation_pct : float, optional
        Proportion of sensitivity to show (default 0.95).
    max_slices : int, optional
        Maximum number of pie slices (default 5).
    rv : str, optional
        Variable name for labels (default "z").
    qoi : int, optional
        Index of quantity of interest to plot (default 0).

    Returns
    -------
    list
        Matplotlib pie chart elements.
    """
    main_effects_np = bkd.to_numpy(main_effects)
    if main_effects_np.ndim == 1:
        main_effects_np = main_effects_np[:, None]
    main_effects_np = main_effects_np[:, qoi].copy()

    if main_effects_np.sum() > 1.0 + np.finfo(float).eps:
        raise ValueError("main_effects sum was greater than 1")
    main_effects_sum = float(main_effects_np.sum())

    # Sort main_effects in descending order
    II = np.argsort(main_effects_np)[::-1]
    main_effects_np = main_effects_np[II]

    labels: List[str] = []
    partial_sum = 0.0
    i = 0
    for i in range(II.shape[0]):
        if partial_sum / main_effects_sum < truncation_pct and i < max_slices:
            labels.append(f"${rv}_{{{II[i] + 1}}}$")
            partial_sum += main_effects_np[i]
        else:
            break

    if abs(partial_sum - main_effects_sum) > 0.5:
        main_effects_np = np.resize(main_effects_np, i + 1)
        explode = np.zeros(main_effects_np.shape[0])
        labels.append(r"$\mathrm{other}$")
        main_effects_np[-1] = main_effects_sum - partial_sum
        explode[-1] = 0.1
    else:
        main_effects_np = main_effects_np[:i]
        labels = labels[:i]
        explode = np.zeros(main_effects_np.shape[0])

    p = ax.pie(
        main_effects_np,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        explode=explode,
    )
    return p


def plot_total_effects(
    total_effects: Array,
    ax: Axes,
    bkd: Backend[Array],
    rv: str = "z",
    qoi: int = 0,
) -> list:
    """Plot total effects as a bar chart.

    Shows the total sensitivity index for each variable.

    Parameters
    ----------
    total_effects : Array
        Shape (nvars, nqoi) or (nvars,) - total effect sensitivity indices.
    ax : Axes
        Matplotlib axes for plotting.
    bkd : Backend[Array]
        Backend for array operations.
    rv : str, optional
        Variable name for labels (default "z").
    qoi : int, optional
        Index of quantity of interest to plot (default 0).

    Returns
    -------
    list
        Matplotlib bar chart elements.
    """
    total_effects_np = bkd.to_numpy(total_effects)
    if total_effects_np.ndim == 1:
        total_effects_np = total_effects_np[:, None]
    total_effects_np = total_effects_np[:, qoi]

    width = 0.95
    locations = np.arange(total_effects_np.shape[0])
    p = ax.bar(locations - width / 2, total_effects_np, width, align="edge")
    labels = [f"${rv}_{{{ii + 1}}}$" for ii in range(total_effects_np.shape[0])]
    ax.set_xticks(locations)
    ax.set_xticklabels(labels, rotation=0)
    return p


def plot_interaction_values(
    interaction_values: Array,
    interaction_terms: Array,
    ax: Axes,
    bkd: Backend[Array],
    truncation_pct: float = 0.95,
    max_slices: int = 5,
    rv: str = "z",
    qoi: int = 0,
) -> list:
    """Plot Sobol indices (including interactions) as a pie chart.

    Parameters
    ----------
    interaction_values : Array
        Shape (nterms, nqoi) - Sobol indices for each interaction term.
    interaction_terms : Array
        Shape (nvars, nterms) - binary indicators of active variables.
    ax : Axes
        Matplotlib axes for plotting.
    bkd : Backend[Array]
        Backend for array operations.
    truncation_pct : float, optional
        Proportion of sensitivity to show (default 0.95).
    max_slices : int, optional
        Maximum number of pie slices (default 5).
    rv : str, optional
        Variable name for labels (default "z").
    qoi : int, optional
        Index of quantity of interest to plot (default 0).

    Returns
    -------
    list
        Matplotlib pie chart elements.
    """
    values_np = bkd.to_numpy(interaction_values)
    terms_np = bkd.to_numpy(interaction_terms)
    if values_np.ndim == 1:
        values_np = values_np[:, None]
    values_np = values_np[:, qoi].copy()

    if values_np.sum() > 1.0 + np.finfo(float).eps:
        raise ValueError("interaction_values sum was greater than 1")
    values_sum = float(values_np.sum())

    # Sort by magnitude (descending)
    II = np.argsort(values_np)[::-1]
    values_np = values_np[II]
    terms_sorted = terms_np[:, II]

    labels: List[str] = []
    partial_sum = 0.0
    i = 0
    for i in range(II.shape[0]):
        if partial_sum / values_sum < truncation_pct and i < max_slices:
            # Build label from active variables
            active = np.where(terms_sorted[:, i] > 0)[0]
            if len(active) == 1:
                labels.append(f"${rv}_{{{active[0] + 1}}}$")
            else:
                parts = [f"{rv}_{{{v + 1}}}" for v in active]
                labels.append("$(" + ",".join(parts) + ")$")
            partial_sum += values_np[i]
        else:
            break

    if abs(partial_sum - values_sum) > 0.5:
        values_np = np.resize(values_np, i + 1)
        explode = np.zeros(values_np.shape[0])
        labels.append(r"$\mathrm{other}$")
        values_np[-1] = values_sum - partial_sum
        explode[-1] = 0.1
    else:
        values_np = values_np[:i]
        labels = labels[:i]
        explode = np.zeros(values_np.shape[0])

    p = ax.pie(
        values_np,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        explode=explode,
    )
    return p


def plot_sensitivity_indices_with_confidence_intervals(
    labels: Sequence[str],
    ax: Axes,
    median: np.ndarray,
    q1: np.ndarray,
    q3: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray,
    reference_values: Optional[np.ndarray] = None,
    fliers: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    """Plot sensitivity indices with confidence intervals as boxplots.

    Creates a boxplot visualization with median, quartiles, and whiskers
    for sensitivity indices computed from multiple realizations.

    Parameters
    ----------
    labels : Sequence[str]
        Labels for each index (e.g., variable names).
    ax : Axes
        Matplotlib axes for plotting.
    median : ndarray
        Median values for each index.
    q1 : ndarray
        First quartile (25th percentile) values.
    q3 : ndarray
        Third quartile (75th percentile) values.
    min_val : ndarray
        Minimum values (shown as lower whisker).
    max_val : ndarray
        Maximum values (shown as upper whisker).
    reference_values : ndarray, optional
        Reference/ground truth values to overlay as points.
    fliers : Sequence[ndarray], optional
        Outlier values for each index.

    Returns
    -------
    dict
        Matplotlib boxplot elements.
    """
    nindices = len(median)
    assert len(labels) == nindices
    if reference_values is not None:
        assert len(reference_values) == nindices

    stats = [dict() for _ in range(nindices)]
    for nn in range(nindices):
        if reference_values is not None:
            stats[nn]["mean"] = reference_values[nn]
        stats[nn]["med"] = median[nn]
        stats[nn]["q1"] = q1[nn]
        stats[nn]["q3"] = q3[nn]
        stats[nn]["label"] = labels[nn]
        stats[nn]["whislo"] = min_val[nn]
        stats[nn]["whishi"] = max_val[nn]
        if fliers is not None:
            stats[nn]["fliers"] = fliers[nn]

    showmeans = reference_values is not None
    showfliers = fliers is not None

    bp = ax.bxp(
        stats,
        showfliers=showfliers,
        showmeans=showmeans,
        patch_artist=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="blue",
            markeredgecolor="blue",
            markersize=12,
        ),
        medianprops=dict(color="red"),
    )
    ax.tick_params(axis="x", labelrotation=45)

    colors = ["gray"] * nindices
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    return bp


def plot_morris_screening(
    mu_star: Array,
    sigma: Array,
    ax: Axes,
    bkd: Backend[Array],
    rv: str = "z",
    qoi: int = 0,
    annotate: bool = True,
) -> Tuple[list, list]:
    """Plot Morris screening results (mu_star vs sigma).

    Creates a scatter plot with mu_star (importance) on x-axis and
    sigma (nonlinearity/interactions) on y-axis.

    Parameters
    ----------
    mu_star : Array
        Shape (nvars, nqoi) - mean absolute elementary effects.
    sigma : Array
        Shape (nvars, nqoi) - standard deviation of elementary effects.
    ax : Axes
        Matplotlib axes for plotting.
    bkd : Backend[Array]
        Backend for array operations.
    rv : str, optional
        Variable name for labels (default "z").
    qoi : int, optional
        Index of quantity of interest to plot (default 0).
    annotate : bool, optional
        Whether to annotate points with variable names (default True).

    Returns
    -------
    Tuple[list, list]
        Scatter plot elements and annotation texts.
    """
    mu_star_np = bkd.to_numpy(mu_star)
    sigma_np = bkd.to_numpy(sigma)
    if mu_star_np.ndim == 1:
        mu_star_np = mu_star_np[:, None]
    if sigma_np.ndim == 1:
        sigma_np = sigma_np[:, None]

    mu_star_np = mu_star_np[:, qoi]
    sigma_np = sigma_np[:, qoi]

    scatter = ax.scatter(mu_star_np, sigma_np, s=50, alpha=0.7)

    texts = []
    if annotate:
        for ii in range(len(mu_star_np)):
            label = f"${rv}_{{{ii + 1}}}$"
            txt = ax.annotate(
                label,
                (mu_star_np[ii], sigma_np[ii]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=10,
            )
            texts.append(txt)

    ax.set_xlabel(r"$\mu^*$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("Morris Screening")

    return [scatter], texts


def plot_sensitivity_summary(
    main_effects: Array,
    total_effects: Array,
    bkd: Backend[Array],
    rv: str = "z",
    qoi: int = 0,
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Create a summary plot with main and total effects.

    Parameters
    ----------
    main_effects : Array
        Shape (nvars, nqoi) - main effect sensitivity indices.
    total_effects : Array
        Shape (nvars, nqoi) - total effect sensitivity indices.
    bkd : Backend[Array]
        Backend for array operations.
    rv : str, optional
        Variable name for labels (default "z").
    qoi : int, optional
        Index of quantity of interest to plot (default 0).
    figsize : Tuple[float, float], optional
        Figure size in inches (default (12, 5)).

    Returns
    -------
    Tuple[Figure, Tuple[Axes, Axes]]
        Figure and axes objects.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plot_main_effects(main_effects, ax1, bkd, rv=rv, qoi=qoi)
    ax1.set_title("Main Effects")

    plot_total_effects(total_effects, ax2, bkd, rv=rv, qoi=qoi)
    ax2.set_title("Total Effects")

    plt.tight_layout()
    return fig, (ax1, ax2)
