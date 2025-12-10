"""Multifidelity statistics estimation module.

This module provides a protocol-first framework for multifidelity Monte Carlo
estimation. It supports various estimators (MC, CV, MFMC, MLMC, ACV, GroupACV,
MLBLUE) and statistics (mean, variance, mean+variance).

Key Components
--------------
Protocols
    Three-level protocol hierarchies for statistics and estimators:
    - StatisticProtocol → StatisticWithCovarianceProtocol → StatisticWithDiscrepancyProtocol
    - EstimatorProtocol → ControlVariateEstimatorProtocol → ParametricEstimatorProtocol

Statistics
    Implementations for computing quantities of interest:
    - MultiOutputMean: Sample mean estimation

Allocation
    Sample allocation utilities:
    - get_allocation_matrix_from_recursion: Build allocation from recursion index
    - get_nsamples_per_model: Compute samples per model from partitions

Examples
--------
>>> from pyapprox.typing.util.backends.numpy import NumpyBkd
>>> from pyapprox.typing.stats import MultiOutputMean
>>> import numpy as np
>>> bkd = NumpyBkd()
>>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
>>> values = bkd.asarray([[1.0], [2.0], [3.0]])
>>> stat.sample_estimate(values)
array([2.])
"""

from pyapprox.typing.stats.protocols import (
    # Statistic protocols (3-level hierarchy)
    StatisticProtocol,
    StatisticWithCovarianceProtocol,
    StatisticWithDiscrepancyProtocol,
    # Estimator protocols (3-level hierarchy)
    EstimatorProtocol,
    ControlVariateEstimatorProtocol,
    ParametricEstimatorProtocol,
    # Allocator protocol
    AllocatorProtocol,
)

from pyapprox.typing.stats.statistics import (
    AbstractStatistic,
    MultiOutputMean,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
)

from pyapprox.typing.stats.allocation import (
    get_allocation_matrix_from_recursion,
    get_npartitions_from_nmodels,
    get_nsamples_per_model,
    validate_allocation_matrix,
)

from pyapprox.typing.stats.estimators import (
    AbstractEstimator,
    MCEstimator,
    CVEstimator,
    ACVEstimator,
    GMFEstimator,
    GRDEstimator,
    GISEstimator,
    MFMCEstimator,
    MLMCEstimator,
    GroupACVEstimator,
    MLBLUEEstimator,
)

from pyapprox.typing.stats.factory import (
    get_estimator,
    BestEstimator,
    compare_estimators,
)

from pyapprox.typing.stats.aetc import (
    AETCEstimator,
    AETCBLUEEstimator,
)

# Visualization imports (optional - only available if matplotlib is installed)
try:
    from pyapprox.typing.stats.visualization import (
        plot_allocation,
        plot_samples_per_model,
        plot_estimator_comparison,
        plot_variance_vs_cost,
        plot_correlation_matrix,
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

__all__ = [
    # Statistic protocols
    "StatisticProtocol",
    "StatisticWithCovarianceProtocol",
    "StatisticWithDiscrepancyProtocol",
    # Estimator protocols
    "EstimatorProtocol",
    "ControlVariateEstimatorProtocol",
    "ParametricEstimatorProtocol",
    # Allocator protocol
    "AllocatorProtocol",
    # Statistics implementations
    "AbstractStatistic",
    "MultiOutputMean",
    "MultiOutputVariance",
    "MultiOutputMeanAndVariance",
    # Allocation utilities
    "get_allocation_matrix_from_recursion",
    "get_npartitions_from_nmodels",
    "get_nsamples_per_model",
    "validate_allocation_matrix",
    # Estimators
    "AbstractEstimator",
    "MCEstimator",
    "CVEstimator",
    # ACV family
    "ACVEstimator",
    "GMFEstimator",
    "GRDEstimator",
    "GISEstimator",
    "MFMCEstimator",
    "MLMCEstimator",
    # Group ACV family
    "GroupACVEstimator",
    "MLBLUEEstimator",
    # Factory
    "get_estimator",
    "BestEstimator",
    "compare_estimators",
    # AETC
    "AETCEstimator",
    "AETCBLUEEstimator",
    # Visualization (if available)
    "plot_allocation",
    "plot_samples_per_model",
    "plot_estimator_comparison",
    "plot_variance_vs_cost",
    "plot_correlation_matrix",
]
