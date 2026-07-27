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

from pyapprox.util.backends.protocols import Array, Backend


def plot_fd_error_sweep(
    fd_eps: Array,
    errors: Union[Array, List[Array], Tuple[Array, ...]],
    bkd: Backend[Array],
    ax: Axes,
    labels: Optional[Sequence[str]] = None,
    slope_guides: Optional[Sequence[int]] = None,
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
        raise ValueError(
            f"got {len(labels)} labels for {len(error_curves)} curves"
        )
    for ii, curve in enumerate(error_curves):
        curve_np = bkd.to_numpy(curve)
        if curve_np.shape != eps_np.shape:
            raise ValueError(
                f"error curve {ii} has shape {curve_np.shape} but fd_eps "
                f"has shape {eps_np.shape}"
            )
        label = None if labels is None else labels[ii]
        ax.loglog(eps_np, curve_np, "-o", label=label)
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
    if labels is not None or slope_guides:
        ax.legend()
    return ax
