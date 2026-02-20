"""Plotting utilities for statistical estimators.

This module provides standalone plotting functions for visualising
allocation matrices and other estimator data.
"""

import numpy as np
import matplotlib.pyplot as plt


def _plot_partition(ii, jj, ax, color, text):
    """Plot a single partition block in an allocation matrix."""
    box = np.array(
        [[ii, jj], [ii + 1, jj], [ii + 1, jj + 1], [ii, jj + 1], [ii, jj]]
    ).T
    ax.plot(*box, color="k")
    ax.fill(*box, color=color)
    if text is not None:
        ax.text(
            *(box[:, 0] + 0.5),
            text,
            verticalalignment="center",
            horizontalalignment="center",
        )


def _plot_allocation_matrix(allocation_mat, npartition_samples, ax):
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
    ax.set_yticklabels(
        [r"$\mathcal{P}_{%d}$" % ii for ii in range(nmodels)]
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)


def plot_allocation(estimator, ax, show_partition_sizes=False):
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
