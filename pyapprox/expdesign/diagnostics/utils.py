"""Shared diagnostic utilities.

Pure functions for MSE decomposition and convergence rate estimation.
"""

from typing import List, Tuple

import numpy as np


def compute_estimator_mse(
    exact: float,
    estimates: List[float],
) -> Tuple[float, float, float]:
    """Decompose estimator error into bias, variance, and MSE.

    Parameters
    ----------
    exact : float
        Exact analytical value.
    estimates : List[float]
        Independent realizations of the estimator.

    Returns
    -------
    bias : float
        Bias = E[estimate] - exact.
    variance : float
        Variance of the estimator.
    mse : float
        Mean squared error = bias^2 + variance.
    """
    arr = np.array(estimates)
    mean_est = float(np.mean(arr))
    var_est = float(np.var(arr))
    bias = mean_est - exact
    mse = bias**2 + var_est
    return bias, var_est, mse


def compute_convergence_rate(
    sample_counts: List[int],
    values: List[float],
) -> float:
    """Compute convergence rate from log-log linear fit.

    For Monte Carlo estimators, MSE typically decays as O(n^{-r})
    where r is the convergence rate.

    Parameters
    ----------
    sample_counts : List[int]
        Sample counts (x-axis).
    values : List[float]
        Values (e.g., MSE) corresponding to sample counts.

    Returns
    -------
    rate : float
        Convergence rate (negative of log-log slope).
        Rate of 1.0 indicates O(1/n) convergence.
    """
    log_n = np.log(np.array(sample_counts))
    log_vals = np.log(np.array(values))
    slope, _ = np.polyfit(log_n, log_vals, 1)
    return float(-slope)
