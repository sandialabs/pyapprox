"""Protocol definitions for multifidelity statistics module.

This module defines the core protocol hierarchies for statistics and estimators.
"""

from pyapprox.typing.stats.protocols.statistic import (
    StatisticProtocol,
    StatisticWithCovarianceProtocol,
    StatisticWithDiscrepancyProtocol,
)
from pyapprox.typing.stats.protocols.estimator import (
    EstimatorProtocol,
    ControlVariateEstimatorProtocol,
    ParametricEstimatorProtocol,
)
from pyapprox.typing.stats.protocols.allocator import (
    AllocatorProtocol,
)

__all__ = [
    # Statistic protocols (3-level hierarchy)
    "StatisticProtocol",
    "StatisticWithCovarianceProtocol",
    "StatisticWithDiscrepancyProtocol",
    # Estimator protocols (3-level hierarchy)
    "EstimatorProtocol",
    "ControlVariateEstimatorProtocol",
    "ParametricEstimatorProtocol",
    # Allocator protocol
    "AllocatorProtocol",
]
