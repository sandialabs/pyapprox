"""Statistics implementations for multifidelity estimation.

This module provides statistic classes that compute quantities of interest
from model samples and their covariance structure.
"""

from pyapprox.typing.stats.statistics.base import AbstractStatistic
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.statistics.variance import MultiOutputVariance
from pyapprox.typing.stats.statistics.mean_variance import MultiOutputMeanAndVariance
from pyapprox.typing.stats.statistics.covariance_utils import (
    compute_W_entry,
    compute_W_from_pilot,
    compute_B_entry,
    compute_B_from_pilot,
    compute_V_entry,
    compute_V_from_covariance,
    covariance_of_variance_estimator,
    extract_nqoi_nqoi_subproblem,
    extract_nqoisq_nqoisq_subproblem,
    extract_nqoi_nqoisq_subproblem,
    compute_covariance_from_pilot,
)

__all__ = [
    "AbstractStatistic",
    "MultiOutputMean",
    "MultiOutputVariance",
    "MultiOutputMeanAndVariance",
    # Covariance utilities
    "compute_W_entry",
    "compute_W_from_pilot",
    "compute_B_entry",
    "compute_B_from_pilot",
    "compute_V_entry",
    "compute_V_from_covariance",
    "covariance_of_variance_estimator",
    "extract_nqoi_nqoi_subproblem",
    "extract_nqoisq_nqoisq_subproblem",
    "extract_nqoi_nqoisq_subproblem",
    "compute_covariance_from_pilot",
]
