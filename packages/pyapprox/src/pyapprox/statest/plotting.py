"""Plotting utilities for statistical estimators.

This module provides standalone plotting functions for visualising
allocation matrices, variance reductions, and recursion DAGs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    import networkx

    from pyapprox.statest.acv.base import ACVEstimator


def _plot_partition(
    ii: int,
    jj: int,
    ax: matplotlib.axes.Axes,
    color: str,
    text: Optional[str],
) -> None:
    """Plot a single partition block in an allocation matrix."""
    box = np.array([[ii, jj], [ii + 1, jj], [ii + 1, jj + 1], [ii, jj + 1], [ii, jj]]).T
    ax.plot(*box, color="k")
    ax.fill(*box, color=color)
    if text is not None:
        ax.text(
            *(box[:, 0] + 0.5),
            text,
            verticalalignment="center",
            horizontalalignment="center",
        )


def _plot_allocation_matrix(
    allocation_mat: np.ndarray,
    npartition_samples: Optional[np.ndarray],
    ax: matplotlib.axes.Axes,
) -> None:
    """Plot an allocation matrix.

    Parameters
    ----------
    allocation_mat : array-like, shape (npartitions, 2*nmodels)
        The allocation matrix.
    npartition_samples : array-like or None
        Partition sample counts. If not None, shown inside blocks.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """
    set_symbol = r"\mathcal{Z}"
    allocation_mat = np.asarray(allocation_mat)
    nmodels, nacv_subsets = allocation_mat.shape
    cycle = iter(plt.cm.rainbow(np.linspace(0, 1, nmodels)))
    colors = [c for c in cycle]
    for ii in range(nmodels):
        for jj in range(1, nacv_subsets):
            if allocation_mat[ii, jj] == 1.0:
                if npartition_samples is not None:
                    text = "$%d$" % int(npartition_samples[ii])
                else:
                    text = None
                _plot_partition(jj, ii, ax, colors[ii], text)
    xticks = np.arange(1, nacv_subsets) + 0.5
    ax.set_xticks(xticks)
    labels = [
        (
            r"$%s_{%d}^*$" % (set_symbol, ii // 2)
            if ii % 2 == 0
            else r"$%s_{%d}$" % (set_symbol, ii // 2)
        )
        for ii in range(1, nacv_subsets)
    ]
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(nmodels) + 0.5)
    ax.set_yticklabels([r"$\mathcal{P}_{%d}$" % ii for ii in range(nmodels)])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)


def plot_allocation(
    estimator: ACVEstimator[Array],
    ax: matplotlib.axes.Axes,
    show_partition_sizes: bool = False,
) -> None:
    """Plot the allocation matrix for an ACV estimator.

    Parameters
    ----------
    estimator : ACVEstimator
        An estimator with ``allocation_matrix()`` and
        ``npartition_samples()`` methods.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    show_partition_sizes : bool, optional
        If True, show partition sizes inside each block.
    """
    allocation_mat = estimator.allocation_matrix()
    npartition_samples = (
        estimator.npartition_samples() if show_partition_sizes else None
    )
    _plot_allocation_matrix(allocation_mat, npartition_samples, ax)


def _autolabel(
    ax: matplotlib.axes.Axes,
    rects: list,  # type: ignore[type-arg]
    labels: List[str],
) -> None:
    """Attach a text label inside each bar."""
    for rect, label in zip(rects, labels):
        ax.annotate(
            label,
            xy=(
                rect.get_x() + rect.get_width() / 2,
                rect.get_y() + rect.get_height() / 2,
            ),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
        )


def plot_estimator_variance_reductions(
    optimized_estimators: list,  # type: ignore[type-arg]
    est_labels: List[str],
    ax: plt.Axes,
    ylabel: Optional[str] = None,
    **bar_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot variance reduction relative to single-fidelity MC.

    For each estimator, computes the ratio of MC variance (using the
    same total budget on the HF model only) to the estimator's
    optimized variance.

    Parameters
    ----------
    optimized_estimators : list
        Estimators that have already been allocated (via
        ``allocate_samples``).
    est_labels : list of str
        Label for each estimator bar.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    ylabel : str, optional
        Y-axis label. Defaults to "Estimator variance reduction".
    **bar_kwargs
        Keyword arguments forwarded to ``ax.bar``.

    Returns
    -------
    var_red : ndarray
        Variance reduction ratios (MC / estimator).
    est_covariances : ndarray
        Determinant of each estimator's covariance.
    sf_covariances : ndarray
        Determinant of the single-fidelity MC covariance.
    """
    var_red = []
    est_covs = []
    sf_covs = []
    for est in optimized_estimators:
        est_cov = est.optimized_covariance()
        bkd = est._bkd
        est_det = bkd.to_float(bkd.flatten(est_cov)[0])
        nhf = bkd.to_int(est._rounded_target_cost / est._costs[0])
        sf_cov = est._stat.high_fidelity_estimator_covariance(nhf)
        sf_det = bkd.to_float(bkd.flatten(sf_cov)[0])
        var_red.append(sf_det / est_det if est_det > 0 else 0.0)
        est_covs.append(est_det)
        sf_covs.append(sf_det)

    var_red = np.array(var_red)
    est_covs = np.array(est_covs)
    sf_covs = np.array(sf_covs)

    rects = ax.bar(est_labels, var_red, **bar_kwargs)
    _autolabel(ax, list(rects), ["$%1.2f$" % v for v in var_red])
    if ylabel is None:
        ylabel = "Estimator variance reduction"
    ax.set_ylabel(ylabel)
    return var_red, est_covs, sf_covs


def _hp(
    G: "networkx.Graph",
    root: int,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    xcenter: float = 0.5,
    pos: Optional[Dict[int, Tuple[float, float]]] = None,
    parent: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """Recursive helper for hierarchy_pos."""
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if parent is not None:
        children = [c for c in children if c != parent]
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hp(
                G,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=nextx,
                pos=pos,
                parent=root,
            )
    return pos


def _hierarchy_pos(
    G: "networkx.Graph",
    root: int,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    xcenter: float = 0.5,
) -> Dict[int, Tuple[float, float]]:
    """Compute hierarchical layout positions for a tree graph."""
    return _hp(G, root, width, vert_gap, vert_loc, xcenter)


def plot_recursion_dag(
    estimator: ACVEstimator[Array],
    ax: matplotlib.axes.Axes,
) -> None:
    """Plot the recursion DAG of an ACV estimator.

    Each node represents a model index; edges show the recursion
    structure (which model each low-fidelity model is paired with).

    Parameters
    ----------
    estimator : ACVEstimator
        An estimator with ``_recursion_index`` attribute.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """
    import networkx as nx

    bkd = estimator._bkd
    recursion_index = bkd.to_numpy(estimator._recursion_index).astype(int)
    nmodels = len(recursion_index) + 1

    graph = nx.Graph()
    graph.add_nodes_from(np.arange(nmodels))
    for ii, jj in enumerate(recursion_index):
        graph.add_edge(ii + 1, int(jj))

    pos = _hierarchy_pos(graph, 0, vert_gap=0.1, width=0.1)
    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_size=[2000],
        font_size=24,
    )
