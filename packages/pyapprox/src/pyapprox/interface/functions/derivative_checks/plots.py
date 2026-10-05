"""Step-size-sweep plots for finite-difference derivative checks.

Renders the error-vs-eps curves produced by
:meth:`DerivativeChecker.check_derivatives` /:meth:`JVPChecker.check`.
A correct derivative traces a V: truncation error falling as eps
shrinks, a floor, then rounding error rising again. A wrong derivative
plateaus — no eps makes a wrong derivative agree with a right one.
Pass the same ``fd_eps`` array given to the checker.
"""

from typing import List, Optional, Sequence, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.typing import ColorType

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    VShapeReport,
)
from pyapprox.util.backends.protocols import Array, Backend


def plot_fd_error_sweep(
    fd_eps: Array,
    errors: Union[Array, List[Array], Tuple[Array, ...]],
    bkd: Backend[Array],
    ax: Axes,
    labels: Optional[Sequence[str]] = None,
    slope_guides: Optional[Sequence[int]] = None,
    reports: Optional[Sequence[VShapeReport]] = None,
) -> Axes:
    """Plot finite-difference error sweeps on log-log axes.

    Parameters
    ----------
    fd_eps : Array
        The step sizes given to the checker. Shape: ``(neps,)``.
    errors : Array or sequence of Arrays
        One error curve per entry (e.g. the jacobian and hessian
        entries of ``check_derivatives``, or a correct and a broken
        gradient). Each shape: ``(neps,)``.
    bkd : Backend
        Backend the arrays belong to (used only to convert for
        matplotlib).
    ax : matplotlib.axes.Axes
        Axes to plot on.
    labels : sequence of str, optional
        Legend label per curve.
    slope_guides : sequence of int, optional
        Reference slopes (e.g. ``[1]`` for the one-sided truncation
        branch) drawn as dashed guide lines anchored at the largest
        step of the first curve.
    reports : sequence of VShapeReport, optional
        One ``DerivativeChecker.check_v_shape`` report per curve. Each
        curve then shows, in its color, a triangle at the bottom of its V
        and a shaded band over the steps its order was fitted on; its
        legend entry gives the fitted order and pass or fail. A labelled
        dotted line marks the largest step a bottom may sit at, drawn once
        per distinct value (in black when several curves share it).

    Returns
    -------
    matplotlib.axes.Axes
        The axes, with log-log scaling and axis labels applied.
    """
    eps_np = bkd.to_numpy(fd_eps)
    error_curves: List[Array]
    if isinstance(errors, (list, tuple)):
        error_curves = list(errors)
    else:
        error_curves = [errors]
    if labels is not None and len(labels) != len(error_curves):
        raise ValueError(f"got {len(labels)} labels for {len(error_curves)} curves")
    if reports is not None and len(reports) != len(error_curves):
        raise ValueError(f"got {len(reports)} reports for {len(error_curves)} curves")
    curve_colors: List[ColorType] = []
    for ii, curve in enumerate(error_curves):
        curve_np = bkd.to_numpy(curve)
        if curve_np.shape != eps_np.shape:
            raise ValueError(
                f"error curve {ii} has shape {curve_np.shape} but fd_eps "
                f"has shape {eps_np.shape}"
            )
        label = None if labels is None else labels[ii]
        if reports is not None:
            report = reports[ii]
            verdict = "pass" if report.passed else "fail"
            name = f"curve {ii}" if label is None else label
            label = f"{name}: order {report.order:.2f}, {verdict}"
        (line,) = ax.loglog(eps_np, curve_np, "-o", label=label)
        if reports is not None:
            _draw_v_shape(ax, reports[ii], float(curve_np.min()), line.get_color())
            curve_colors.append(line.get_color())
    if reports is not None:
        _draw_bottom_bounds(ax, reports, curve_colors)
    if slope_guides:
        anchor_eps = float(eps_np.max())
        first_np = bkd.to_numpy(error_curves[0])
        anchor_err = float(first_np[eps_np.argmax()])
        for order in slope_guides:
            guide = anchor_err * (eps_np / anchor_eps) ** order
            ax.loglog(
                eps_np,
                guide,
                "k--",
                alpha=0.5,
                label=rf"$\epsilon^{{{order}}}$",
            )
    ax.set_xlabel(r"step size $\epsilon$")
    ax.set_ylabel("finite-difference error")
    if labels is not None or slope_guides or reports is not None:
        ax.legend()
    return ax


def _draw_v_shape(
    ax: Axes, report: VShapeReport, min_error: float, color: ColorType
) -> None:
    """Mark what ``check_v_shape`` looked at, in the curve's color.

    A triangle at the bottom of the V and a shaded band over the steps the
    order was fitted on (10 to 1000 times the bottom step).
    """
    bottom = report.bottom_step
    ax.loglog([bottom], [min_error], "v", color=color, markersize=10)
    ax.axvspan(10.0 * bottom, 1000.0 * bottom, color=color, alpha=0.12)


def _draw_bottom_bounds(
    ax: Axes, reports: Sequence[VShapeReport], colors: Sequence[ColorType]
) -> None:
    """One dotted, labelled line per distinct largest bottom step.

    Curves usually share the bound (forward and central have one each),
    and drawing it per curve would stack identical lines. A bound shared
    by several curves is drawn once in black; one used by a single curve
    keeps that curve's color.
    """
    users: dict[float, List[ColorType]] = {}
    for report, color in zip(reports, colors):
        users.setdefault(report.max_bottom_step, []).append(color)
    for bound, bound_colors in users.items():
        color = bound_colors[0] if len(bound_colors) == 1 else "k"
        ax.axvline(
            bound,
            color=color,
            linestyle=":",
            alpha=0.8,
            label=f"max bottom step ({bound:.0e})",
        )
