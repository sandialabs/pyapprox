"""The :mod:`pyapprox.multifidelity` module implements numerous multi-fidelity
algorithms for quantifying uncertainty and building surrgates from multiple
models of varying cost and fidelity.
"""

from pyapprox.multifidelity.monte_carlo_estimators import (
    get_estimator, estimate_variance, compare_estimator_variances,
    plot_estimator_variances, plot_acv_sample_allocation_comparison
)
from pyapprox.multifidelity.control_variate_monte_carlo import (
    estimate_model_ensemble_covariance)


__all__ = ["get_estimator", "estimate_variance", "compare_estimator_variances",
           "plot_estimator_variances", "plot_acv_sample_allocation_comparison",
           "estimate_model_ensemble_covariance"]
