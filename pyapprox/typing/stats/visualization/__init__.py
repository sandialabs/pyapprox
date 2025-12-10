"""Visualization utilities for multifidelity estimators.

This module provides plotting functions for visualizing estimator
allocations, variance reduction, and model comparisons.
"""

from pyapprox.typing.stats.visualization.allocation import (
    plot_allocation,
    plot_samples_per_model,
)
from pyapprox.typing.stats.visualization.comparison import (
    plot_estimator_comparison,
    plot_variance_vs_cost,
)
from pyapprox.typing.stats.visualization.correlation import (
    plot_correlation_matrix,
)

__all__ = [
    "plot_allocation",
    "plot_samples_per_model",
    "plot_estimator_comparison",
    "plot_variance_vs_cost",
    "plot_correlation_matrix",
]
