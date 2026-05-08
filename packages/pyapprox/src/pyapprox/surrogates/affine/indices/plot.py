"""Plotting utilities for multi-index sets.

Provides reusable 2D and 3D visualisation of downward-closed multi-index sets,
with support for per-index colours (e.g. by fidelity level), automatic labels,
and combined selected/candidate views for adaptive sparse grid tutorials.

All functions accept numpy arrays with shape ``(nvars, nindices)`` — the same
column-major convention used throughout *pyapprox*.  Only numpy/matplotlib
are required (no ``Generic[Array]`` or backend protocol).
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ColorMapping = Union[str, Sequence[str], Callable[[NDArray[Any]], str]]
"""Uniform colour string, per-index list, or callable ``index_col -> colour``."""

LabelMapping = Union[None, bool, Sequence[Optional[str]], Callable[[NDArray[Any]], str]]
"""``None`` for no labels, ``True`` for auto ``"(k1,k2)"``, or per-index."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_colors(
    colors: ColorMapping,
    indices: NDArray[Any],
) -> List[str]:
    """Resolve *ColorMapping* to a per-index list of colour strings.

    Parameters
    ----------
    colors : str, list[str], or callable
        Uniform colour, per-index list, or ``f(index_col) -> colour``.
    indices : ndarray, shape ``(nvars, nindices)``
        Multi-index array (columns are individual indices).

    Returns
    -------
    list[str]
        One colour string per index (column).
    """
    nindices = indices.shape[1]
    if callable(colors):
        return [colors(indices[:, j]) for j in range(nindices)]
    if isinstance(colors, str):
        return [colors] * nindices
    colors_seq = list(colors)
    if len(colors_seq) != nindices:
        raise ValueError(f"len(colors)={len(colors_seq)} != nindices={nindices}")
    return colors_seq


def _resolve_labels(
    labels: LabelMapping,
    indices: NDArray[Any],
) -> List[Optional[str]]:
    """Resolve *LabelMapping* to a per-index list of label strings.

    Parameters
    ----------
    labels : None, True, list, or callable
        ``None`` → no labels; ``True`` → auto ``"(k1,k2)"``; list or callable.
    indices : ndarray, shape ``(nvars, nindices)``
        Multi-index array.

    Returns
    -------
    list[str | None]
        One label (or ``None``) per index.
    """
    nindices = indices.shape[1]
    if labels is None:
        return [None] * nindices
    if labels is True:
        return [
            "("
            + ",".join(str(int(indices[d, j])) for d in range(indices.shape[0]))
            + ")"
            for j in range(nindices)
        ]
    if callable(labels):
        return [labels(indices[:, j]) for j in range(nindices)]
    if labels is False:
        return [None] * nindices
    labels_seq: List[Optional[str]] = list(labels)
    if len(labels_seq) != nindices:
        raise ValueError(f"len(labels)={len(labels_seq)} != nindices={nindices}")
    return labels_seq


# ---------------------------------------------------------------------------
# 2D plotting
# ---------------------------------------------------------------------------


def plot_indices_2d(
    ax: Axes,
    indices: NDArray[Any],
    *,
    colors: ColorMapping = "steelblue",
    labels: LabelMapping = None,
    box_width: float = 0.8,
    box_height: float = 0.8,
    alpha: float = 0.7,
    edgecolor: str = "black",
    linewidth: float = 1.5,
    linestyle: str = "-",
    label_fontsize: int = 8,
    label_color: str = "black",
    **rect_kwargs: Any,
) -> Tuple[List[Rectangle], List[Text]]:
    """Draw a 2D multi-index set as coloured rectangles.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes (must be a standard 2D axes).
    indices : ndarray, shape ``(2, nindices)``
        Integer multi-indices, one per column.
    colors : str, list[str], or callable
        Face colour specification (see :data:`ColorMapping`).
    labels : None, True, list, or callable
        Label specification (see :data:`LabelMapping`).
    box_width, box_height : float
        Rectangle dimensions (centred on integer coordinates).
    alpha : float
        Face colour alpha.
    edgecolor, linewidth, linestyle : str / float
        Rectangle border styling.
    label_fontsize : int
        Font size for index labels.
    label_color : str
        Colour of the label text.
    **rect_kwargs
        Extra keyword arguments forwarded to :class:`matplotlib.patches.Rectangle`.

    Returns
    -------
    rectangles : list[Rectangle]
        One patch per index.
    texts : list[Text]
        One text artist per label (empty list when *labels* is ``None``).
    """
    if indices.shape[0] != 2:
        raise ValueError(f"plot_indices_2d requires nvars==2, got {indices.shape[0]}")

    resolved_colors = _resolve_colors(colors, indices)
    resolved_labels = _resolve_labels(labels, indices)

    rectangles: List[Rectangle] = []
    texts: List[Text] = []

    for j in range(indices.shape[1]):
        cx, cy = float(indices[0, j]), float(indices[1, j])
        rect = Rectangle(
            (cx - box_width / 2, cy - box_height / 2),
            box_width,
            box_height,
            facecolor=resolved_colors[j],
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            **rect_kwargs,
        )
        ax.add_patch(rect)
        rectangles.append(rect)

        lbl = resolved_labels[j]
        if lbl is not None:
            t = ax.text(
                cx,
                cy,
                lbl,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=label_color,
            )
            texts.append(t)

    return rectangles, texts


# ---------------------------------------------------------------------------
# 3D plotting
# ---------------------------------------------------------------------------


def plot_indices_3d(
    ax: Any,
    indices: NDArray[Any],
    *,
    colors: ColorMapping = "#2C7FB8CC",
    labels: LabelMapping = None,
    edgecolor: str = "black",
    linewidth: float = 0.5,
    label_fontsize: int = 6,
) -> Tuple[object, List[Text]]:
    """Draw a 3D multi-index set as filled voxels.

    Parameters
    ----------
    ax : Axes3D
        A 3D-projection axes (``fig.add_subplot(projection='3d')``) .
    indices : ndarray, shape ``(3, nindices)``
        Integer multi-indices, one per column.
    colors : str, list[str], or callable
        Face colour specification.
    labels : None, True, list, or callable
        Label specification.
    edgecolor : str
        Voxel edge colour.
    linewidth : float
        Voxel edge width.
    label_fontsize : int
        Font size for labels placed above each voxel.

    Returns
    -------
    voxel_result : object
        The return value of ``ax.voxels()``.
    texts : list[Text]
        One text artist per label (empty list when *labels* is ``None``).
    """
    if indices.shape[0] != 3:
        raise ValueError(f"plot_indices_3d requires nvars==3, got {indices.shape[0]}")

    resolved_colors = _resolve_colors(colors, indices)
    resolved_labels = _resolve_labels(labels, indices)

    # Build boolean voxel grid and face-colour array
    max_vals = indices.max(axis=1).astype(int)
    shape = (max_vals[0] + 1, max_vals[1] + 1, max_vals[2] + 1)
    voxels = np.zeros(shape, dtype=bool)
    facecolors = np.empty(shape, dtype=object)

    for j in range(indices.shape[1]):
        i0 = int(indices[0, j])
        i1 = int(indices[1, j])
        i2 = int(indices[2, j])
        voxels[i0, i1, i2] = True
        facecolors[i0, i1, i2] = resolved_colors[j]

    # Shift coordinate grid by -0.5 so voxels are centred on integer ticks.
    # Without this, ax.voxels() places each cube at [k, k+1] (centre k+0.5).
    # With the shift, each cube spans [k-0.5, k+0.5] (centre k).
    x, y, z = np.meshgrid(
        np.arange(shape[0] + 1) - 0.5,
        np.arange(shape[1] + 1) - 0.5,
        np.arange(shape[2] + 1) - 0.5,
        indexing="ij",
    )

    voxel_result = ax.voxels(
        x,
        y,
        z,
        voxels,
        facecolors=facecolors,
        edgecolors=edgecolor,
        linewidth=linewidth,
    )

    texts: List[Text] = []
    for j in range(indices.shape[1]):
        lbl = resolved_labels[j]
        if lbl is not None:
            i0 = int(indices[0, j])
            i1 = int(indices[1, j])
            i2 = int(indices[2, j])
            t = ax.text(
                i0,
                i1,
                i2 + 0.55,
                lbl,
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
            )
            texts.append(t)

    return voxel_result, texts


# ---------------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------------


def format_index_axes(
    ax: Any,
    indices: NDArray[Any],
    *,
    axis_labels: Optional[Sequence[str]] = None,
    pad: float = 0.5,
    integer_ticks: bool = True,
    equal_aspect: bool = True,
    grid_alpha: float = 0.3,
    view_init: Optional[Tuple[float, float]] = None,
    max_indices: Optional[Sequence[int]] = None,
) -> None:
    """Configure axes limits, ticks, labels and aspect for an index-set plot.

    Detects 3D axes automatically and adjusts behaviour accordingly.

    Parameters
    ----------
    ax : Axes or Axes3D
        Target axes.
    indices : ndarray, shape ``(nvars, nindices)``
        Multi-index array used to determine limits (ignored when
        *max_indices* is provided).
    axis_labels : sequence[str] or None
        Labels for each axis dimension.
    pad : float
        Padding around the data range.
    integer_ticks : bool
        If *True*, place ticks at integer locations.
    equal_aspect : bool
        Set equal aspect ratio (2D only).
    grid_alpha : float
        Grid transparency.
    view_init : tuple(elev, azim) or None
        Camera angle for 3D axes. Defaults to ``(30, 45)`` for 3D.
    max_indices : sequence[int] or None
        Fixed per-dimension maximum index values used for axis limits and
        ticks.  When provided, *indices* is not used for limit computation.
        Useful for animations where the axis range should stay constant
        across frames.
    """
    nvars = indices.shape[0]
    if max_indices is not None:
        max_vals = np.asarray(max_indices, dtype=int)
    else:
        max_vals = indices.max(axis=1).astype(int)

    is_3d = hasattr(ax, "set_zlim")

    # Voxels are shifted so they centre on integer coords (like 2D rectangles),
    # so the same limit formula works for both 2D and 3D.
    ax.set_xlim(-pad, float(max_vals[0]) + pad)
    ax.set_ylim(-pad, float(max_vals[1]) + pad)
    if is_3d and nvars >= 3:
        ax.set_zlim(-pad, float(max_vals[2]) + pad)

    if integer_ticks:
        ax.set_xticks(np.arange(max_vals[0] + 1))
        ax.set_yticks(np.arange(max_vals[1] + 1))
        if is_3d and nvars >= 3:
            ax.set_zticks(np.arange(max_vals[2] + 1))

    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if is_3d and nvars >= 3 and len(axis_labels) >= 3:
            ax.set_zlabel(axis_labels[2])

    if equal_aspect and not is_3d:
        ax.set_aspect("equal")

    ax.grid(True, alpha=grid_alpha)

    if is_3d:
        elev, azim = view_init if view_init is not None else (30, 45)
        ax.view_init(elev, azim)


# ---------------------------------------------------------------------------
# Combined selected + candidate view
# ---------------------------------------------------------------------------


def plot_index_sets(
    ax: Any,
    selected: NDArray[Any],
    candidates: Optional[NDArray[Any]] = None,
    *,
    selected_colors: ColorMapping = "steelblue",
    candidate_colors: ColorMapping = "#93C5E8",
    selected_labels: LabelMapping = None,
    candidate_labels: LabelMapping = None,
    selected_alpha: float = 0.7,
    candidate_alpha: float = 0.4,
    candidate_linestyle: str = "--",
    format_axes: bool = True,
    axis_labels: Optional[Sequence[str]] = None,
    box_width: float = 0.8,
    box_height: float = 0.8,
    max_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Tuple[Any, ...]]:
    """Draw selected (and optionally candidate) index sets.

    Dispatches to :func:`plot_indices_2d` or :func:`plot_indices_3d` based on
    ``nvars``.  Candidates are drawn first so that selected indices appear on
    top.

    Parameters
    ----------
    ax : Axes or Axes3D
        Target axes.
    selected : ndarray, shape ``(nvars, nindices)``
        Selected (active) multi-indices.
    candidates : ndarray or None
        Candidate multi-indices to draw behind the selected set.
    selected_colors, candidate_colors : ColorMapping
        Colour specifications for each set.
    selected_labels, candidate_labels : LabelMapping
        Label specifications for each set.
    selected_alpha, candidate_alpha : float
        Alpha values for each set.
    candidate_linestyle : str
        Border style for candidate boxes (2D only).
    format_axes : bool
        If *True*, call :func:`format_index_axes` after drawing.
    axis_labels : sequence[str] or None
        Forwarded to :func:`format_index_axes`.
    box_width, box_height : float
        Rectangle dimensions (2D only).
    max_indices : sequence[int] or None
        Fixed per-dimension maximum index values for axis limits.
        Forwarded to :func:`format_index_axes`.  Useful for animations
        where the axis range should stay constant across frames.

    Returns
    -------
    dict
        ``{"selected": (artists, texts), "candidates": (artists, texts)}``.
    """
    nvars = selected.shape[0]
    if nvars not in (2, 3):
        raise ValueError(f"plot_index_sets supports nvars in (2, 3), got {nvars}")

    result: Dict[str, Tuple[Any, ...]] = {}
    _draw = _draw_2d if nvars == 2 else _draw_3d

    # Draw candidates first (behind)
    if candidates is not None:
        if candidates.shape[0] != nvars:
            raise ValueError(
                f"candidates nvars={candidates.shape[0]} != selected nvars={nvars}"
            )
        result["candidates"] = _draw(
            ax,
            candidates,
            colors=candidate_colors,
            labels=candidate_labels,
            alpha=candidate_alpha,
            linestyle=candidate_linestyle,
            box_width=box_width,
            box_height=box_height,
        )
    else:
        result["candidates"] = ([], [])

    # Draw selected (on top)
    result["selected"] = _draw(
        ax,
        selected,
        colors=selected_colors,
        labels=selected_labels,
        alpha=selected_alpha,
        linestyle="-",
        box_width=box_width,
        box_height=box_height,
    )

    if format_axes:
        # Combine selected + candidates for limit computation
        all_indices = selected
        if candidates is not None and candidates.shape[1] > 0:
            all_indices = np.hstack([selected, candidates])
        format_index_axes(
            ax,
            all_indices,
            axis_labels=axis_labels,
            max_indices=max_indices,
        )

    return result


# ---------------------------------------------------------------------------
# Internal dispatch helpers
# ---------------------------------------------------------------------------


def _draw_2d(
    ax: Axes,
    indices: NDArray[Any],
    *,
    colors: ColorMapping,
    labels: LabelMapping,
    alpha: float,
    linestyle: str,
    box_width: float,
    box_height: float,
) -> Tuple[List[Rectangle], List[Text]]:
    return plot_indices_2d(
        ax,
        indices,
        colors=colors,
        labels=labels,
        alpha=alpha,
        linestyle=linestyle,
        box_width=box_width,
        box_height=box_height,
    )


def _draw_3d(
    ax: Any,
    indices: NDArray[Any],
    *,
    colors: ColorMapping,
    labels: LabelMapping,
    alpha: float,
    linestyle: str,
    box_width: float,
    box_height: float,
) -> Tuple[object, List[Text]]:
    # 3D voxels do not support linestyle/box_width/box_height — ignore them
    return plot_indices_3d(ax, indices, colors=colors, labels=labels)
