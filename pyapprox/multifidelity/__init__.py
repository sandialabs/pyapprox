"""The :mod:`pyapprox.multifidelity` module implements numerous multi-fidelity
algorithms for quantifying uncertainty and building surrgates from multiple
models of varying cost and fidelity.
"""
from pyapprox.multifidelity.visualize import (
    plot_correlation_matrix, plot_model_costs,
    plot_estimator_variance_reductions,
    plot_estimator_sample_allocation_comparison)
from pyapprox.multifidelity.factory import (
    estimate_model_ensemble_covariance, get_estimator, multioutput_stats)
from pyapprox.util.utilities import get_correlation_from_covariance

__all__ = ["get_estimator",
           "plot_estimator_variance_reductions",
           "plot_estimator_sample_allocation_comparison",
           "plot_correlation_matrix",
           "plot_model_costs",
           "estimate_model_ensemble_covariance",
           "get_correlation_from_covariance",
           "multioutput_stats"]
