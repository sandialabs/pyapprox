"""Tutorial figure helpers for SUPN surrogates.

Covers: supn_surrogates.qmd

Convention B throughout — pure plot functions that accept pre-computed
numpy arrays and only handle plotting.
"""

import numpy as np

from ._style import COLORS, apply_style


def plot_supn_fit_1d(x_np, y_true_np, y_pred_np, ax, title=""):
    """Overlay true function and SUPN fit on a single axis.

    Parameters
    ----------
    x_np : ndarray, shape (n,)
    y_true_np : ndarray, shape (n,)
    y_pred_np : ndarray, shape (n,)
    ax : matplotlib Axes
    title : str
    """
    ax.plot(x_np, y_true_np, "--", color=COLORS["reference"],
            linewidth=1.5, label="Target")
    ax.plot(x_np, y_pred_np, color=COLORS["primary"],
            linewidth=1.5, label="SUPN")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(frameon=False)
    if title:
        ax.set_title(title, fontsize=11)
    apply_style(ax)



def plot_supn_2d_comparison(X_np, Y_np, z_true, z_pred, axes, fig):
    """Three-panel contourf: target | SUPN fit | absolute error.

    Parameters
    ----------
    X_np, Y_np : ndarray, shape (ny, nx)
        Meshgrid coordinate arrays.
    z_true, z_pred : ndarray, shape (ny, nx)
    axes : array of 3 matplotlib Axes
    fig : matplotlib Figure (for colorbars)
    """
    import matplotlib.colors as mcolors

    titles = [r"Target $f$", "SUPN fit", r"$\log_{10}$ absolute error"]
    abs_err = np.abs(z_true - z_pred)
    abs_err_clipped = np.clip(abs_err, 1e-16, None)

    for idx, (ax, ttl) in enumerate(zip(axes, titles)):
        if idx < 2:
            z = [z_true, z_pred][idx]
            cf = ax.contourf(X_np, Y_np, z, levels=30, cmap="RdBu_r")
        else:
            cf = ax.contourf(
                X_np, Y_np, abs_err_clipped, levels=30,
                cmap="YlOrRd",
                norm=mcolors.LogNorm(
                    vmin=abs_err_clipped.min(), vmax=abs_err_clipped.max(),
                ),
            )
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        apply_style(ax)


def plot_supn_heatmap(err_grid, N_values, L_values, ax):
    """Heat-map of relative L2 error over (N, max_level) grid.

    Parameters
    ----------
    err_grid : ndarray, shape (len(N_values), len(L_values))
        err_grid[i, j] = relative L2 error for (N_values[i], L_values[j]).
    N_values : list[int]
    L_values : list[int]
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    log_err = np.log10(np.clip(err_grid, 1e-16, None))
    im = ax.imshow(
        log_err, origin="lower", aspect="auto",
        cmap="YlOrRd_r",
        extent=[
            L_values[0] - 0.5, L_values[-1] + 0.5,
            N_values[0] - 0.5, N_values[-1] + 0.5,
        ],
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}$ relative $L^2$ error")

    # Annotate each cell
    for i, N_val in enumerate(N_values):
        for j, L_val in enumerate(L_values):
            ax.text(
                L_val, N_val,
                f"{err_grid[i, j]:.1e}",
                ha="center", va="center", fontsize=7,
                color="white" if log_err[i, j] < np.median(log_err) else "black",
            )

    ax.set_xlabel("max_level $L$")
    ax.set_ylabel("Width $N$")
    ax.set_xticks(L_values)
    ax.set_yticks(N_values)
    ax.set_title("Relative $L^2$ error", fontsize=11)
    apply_style(ax)
