"""Plotting utilities for sparse grid sample points.

Provides a reusable scatter-plot function for visualising selected and
candidate sample locations on 1D or 2D sparse grids.

Only numpy/matplotlib are required (no ``Generic[Array]`` or backend
protocol).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_sparse_grid_points(
    ax: Axes,
    selected_samples: NDArray[Any],
    candidate_samples: Optional[NDArray[Any]] = None,
    *,
    selected_marker: str = "o",
    candidate_marker: str = "s",
    selected_color: str = "steelblue",
    candidate_color: str = "#93C5E8",
    selected_size: float = 30,
    candidate_size: float = 25,
    selected_alpha: float = 0.8,
    candidate_alpha: float = 0.5,
    selected_label: str = "Selected",
    candidate_label: str = "Candidate",
    axis_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    equal_aspect: bool = True,
    grid_alpha: float = 0.3,
    pad: float = 0.1,
) -> Dict[str, Any]:
    """Plot sparse grid sample points in 1D or 2D.

    For 1D (nvars==1): scatter along x-axis with y=0.
    For 2D (nvars==2): standard 2D scatter.
    Candidates are drawn first (behind), selected on top.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    selected_samples : NDArray
        Selected samples, shape (nvars, nsamples).
    candidate_samples : Optional[NDArray]
        Candidate samples, shape (nvars, nsamples), or None.
    selected_marker, candidate_marker : str
        Marker styles.
    selected_color, candidate_color : str
        Marker colours.
    selected_size, candidate_size : float
        Marker sizes.
    selected_alpha, candidate_alpha : float
        Marker alpha values.
    selected_label, candidate_label : str
        Legend labels.
    axis_labels : Optional[Sequence[str]]
        Axis labels, e.g. ["$z_1$", "$z_2$"].
    title : Optional[str]
        Axes title.
    equal_aspect : bool
        Whether to set equal aspect ratio.
    grid_alpha : float
        Grid transparency.
    pad : float
        Extra padding around data range for axis limits.

    Returns
    -------
    Dict[str, Any]
        {"selected": PathCollection, "candidate": PathCollection or None}.
    """
    nvars = selected_samples.shape[0]
    if nvars > 2:
        raise ValueError(
            f"plot_sparse_grid_points supports 1D or 2D only, got "
            f"nvars={nvars}"
        )

    result: Dict[str, Any] = {"selected": None, "candidate": None}

    # Collect all points for axis limits
    all_x = [selected_samples[0]]
    if nvars == 2:
        all_y = [selected_samples[1]]
    else:
        all_y = [np.zeros(selected_samples.shape[1])]

    if candidate_samples is not None and candidate_samples.shape[1] > 0:
        all_x.append(candidate_samples[0])
        if nvars == 2:
            all_y.append(candidate_samples[1])
        else:
            all_y.append(np.zeros(candidate_samples.shape[1]))

    all_x_arr = np.concatenate(all_x)
    all_y_arr = np.concatenate(all_y)

    # Draw candidates first (behind)
    if candidate_samples is not None and candidate_samples.shape[1] > 0:
        if nvars == 1:
            cx = candidate_samples[0]
            cy = np.zeros(candidate_samples.shape[1])
        else:
            cx = candidate_samples[0]
            cy = candidate_samples[1]
        result["candidate"] = ax.scatter(
            cx, cy,
            s=candidate_size,
            c=candidate_color,
            marker=candidate_marker,
            alpha=candidate_alpha,
            label=candidate_label,
            zorder=2,
        )

    # Draw selected on top
    if selected_samples.shape[1] > 0:
        if nvars == 1:
            sx = selected_samples[0]
            sy = np.zeros(selected_samples.shape[1])
        else:
            sx = selected_samples[0]
            sy = selected_samples[1]
        result["selected"] = ax.scatter(
            sx, sy,
            s=selected_size,
            c=selected_color,
            marker=selected_marker,
            alpha=selected_alpha,
            label=selected_label,
            zorder=3,
        )

    # Axis limits with padding
    if len(all_x_arr) > 0:
        x_range = float(all_x_arr.max() - all_x_arr.min())
        y_range = float(all_y_arr.max() - all_y_arr.min())
        x_pad = max(pad * x_range, pad)
        y_pad = max(pad * y_range, pad)
        ax.set_xlim(float(all_x_arr.min()) - x_pad,
                    float(all_x_arr.max()) + x_pad)
        ax.set_ylim(float(all_y_arr.min()) - y_pad,
                    float(all_y_arr.max()) + y_pad)

    if equal_aspect:
        ax.set_aspect("equal")

    ax.grid(True, alpha=grid_alpha)

    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        if len(axis_labels) > 1:
            ax.set_ylabel(axis_labels[1])

    if title is not None:
        ax.set_title(title)

    return result
