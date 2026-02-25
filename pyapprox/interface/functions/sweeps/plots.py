"""Plotting functions for parameter sweeps.

Provides visualization utilities for parameter sweep results.
"""

from typing import Any, Dict, Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.interface.functions.sweeps.protocols import (
    ParameterSweeperProtocol,
)


def plot_single_qoi_sweep(
    sweeper: ParameterSweeperProtocol[Array],
    sweep_values: Array,
    sweep_id: int,
    ax: Any,
    bkd: Backend[Array],
    **plot_kwargs: Dict[str, Any],
) -> None:
    """Plot a single QoI sweep.

    Plots the function values along a single parameter sweep direction.

    Parameters
    ----------
    sweeper : ParameterSweeperProtocol[Array]
        The sweeper that generated the samples.
    sweep_values : Array
        Shape (nsamples_total,) - function values at all sweep samples.
    sweep_id : int
        Index of the sweep to plot (0 to nsweeps-1).
    ax : matplotlib.axes.Axes
        Axes on which to plot.
    bkd : Backend[Array]
        Backend for array operations.
    **plot_kwargs
        Additional keyword arguments passed to ax.plot().

    Raises
    ------
    RuntimeError
        If sweeper.rvs() has not been called.
    ValueError
        If sweep_values has incorrect shape.
    """
    canonical_samples = sweeper.canonical_active_samples()
    nsweeps = canonical_samples.shape[0]
    nsamples_per_sweep = sweeper.nsamples_per_sweep()

    expected_shape = (nsamples_per_sweep * nsweeps,)
    if sweep_values.shape != expected_shape:
        raise ValueError(
            f"sweep_values has wrong shape {sweep_values.shape}, "
            f"expected {expected_shape}"
        )

    if sweep_id < 0 or sweep_id >= nsweeps:
        raise ValueError(
            f"sweep_id {sweep_id} out of range [0, {nsweeps - 1}]"
        )

    start = sweep_id * nsamples_per_sweep
    end = (sweep_id + 1) * nsamples_per_sweep

    x_data = bkd.to_numpy(canonical_samples[sweep_id, :])
    y_data = bkd.to_numpy(sweep_values[start:end])

    ax.plot(x_data, y_data, **plot_kwargs)


def plot_all_sweeps(
    sweeper: ParameterSweeperProtocol[Array],
    sweep_values: Array,
    ax: Any,
    bkd: Backend[Array],
    labels: Optional[list[str]] = None,
    **plot_kwargs: Dict[str, Any],
) -> None:
    """Plot all sweeps on the same axes.

    Parameters
    ----------
    sweeper : ParameterSweeperProtocol[Array]
        The sweeper that generated the samples.
    sweep_values : Array
        Shape (nsamples_total,) - function values at all sweep samples.
    ax : matplotlib.axes.Axes
        Axes on which to plot.
    bkd : Backend[Array]
        Backend for array operations.
    labels : Optional[list[str]]
        Labels for each sweep (for legend). If None, uses "Sweep 0", etc.
    **plot_kwargs
        Additional keyword arguments passed to ax.plot().
    """
    canonical_samples = sweeper.canonical_active_samples()
    nsweeps = canonical_samples.shape[0]

    if labels is None:
        labels = [f"Sweep {i}" for i in range(nsweeps)]

    for ii in range(nsweeps):
        plot_single_qoi_sweep(
            sweeper, sweep_values, ii, ax, bkd, label=labels[ii], **plot_kwargs
        )
