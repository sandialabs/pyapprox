"""Statistical estimators for multi-fidelity Monte Carlo methods.

This module provides implementations of approximate control variate (ACV)
estimators including:
- MCEstimator: Base Monte Carlo estimator
- CVEstimator: Control variate estimator
- ACVEstimator: Approximate control variate estimator base
- GMFEstimator: Generalized multifidelity estimator
- GISEstimator: Generalized integrated sample estimator
- GRDEstimator: Generalized recursive difference estimator
- MFMCEstimator: Multi-fidelity Monte Carlo estimator
- MLMCEstimator: Multi-level Monte Carlo estimator
"""

from pyapprox.typing.statest.protocols import (
    StatisticProtocol,
    EstimatorProtocol,
)
from pyapprox.typing.statest.statistics import (
    MultiOutputStatistic,
    MultiOutputMean,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
)
from pyapprox.typing.statest.strategies import (
    ModelSubsetStrategy,
    AllModelsStrategy,
    FixedSubsetStrategy,
    AllSubsetsStrategy,
    ListSubsetStrategy,
    QoISubsetStrategy,
    AllQoIStrategy,
    FixedQoIStrategy,
    AllQoISubsetsStrategy,
    ListQoIStrategy,
)
from pyapprox.typing.statest.mc_estimator import MCEstimator
from pyapprox.typing.statest.cv_estimator import CVEstimator
from pyapprox.typing.statest.acv import (
    ACVEstimator,
    GMFEstimator,
    GISEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.typing.statest.groupacv import (
    BaseGroupACVEstimator,
    GroupACVEstimatorIS,
    GroupACVEstimatorNested,
    MLBLUEEstimator,
    GroupACVObjective,
    GroupACVTraceObjective,
    GroupACVLogDetObjective,
    MLBLUEObjective,
    GroupACVCostConstraint,
    get_model_subsets,
    default_groupacv_optimizer,
)
from pyapprox.typing.statest.plotting import plot_allocation
from pyapprox.typing.statest.aetc import AETC, AETCBLUE, AETCMC
from pyapprox.typing.statest.search import (
    EstimatorFamily,
    UnifiedSearchResult,
    unified_search,
)

__all__ = [
    "StatisticProtocol",
    "EstimatorProtocol",
    "MultiOutputStatistic",
    "MultiOutputMean",
    "MultiOutputVariance",
    "MultiOutputMeanAndVariance",
    "MCEstimator",
    "CVEstimator",
    "ACVEstimator",
    "GMFEstimator",
    "GISEstimator",
    "GRDEstimator",
    "MFMCEstimator",
    "MLMCEstimator",
    # Shared strategies
    "ModelSubsetStrategy",
    "AllModelsStrategy",
    "FixedSubsetStrategy",
    "AllSubsetsStrategy",
    "ListSubsetStrategy",
    "QoISubsetStrategy",
    "AllQoIStrategy",
    "FixedQoIStrategy",
    "AllQoISubsetsStrategy",
    "ListQoIStrategy",
    # GroupACV exports
    "BaseGroupACVEstimator",
    "GroupACVEstimatorIS",
    "GroupACVEstimatorNested",
    "MLBLUEEstimator",
    "GroupACVObjective",
    "GroupACVTraceObjective",
    "GroupACVLogDetObjective",
    "MLBLUEObjective",
    "GroupACVCostConstraint",
    "get_model_subsets",
    "default_groupacv_optimizer",
    # Plotting
    "plot_allocation",
    # AETC exports
    "AETC",
    "AETCBLUE",
    "AETCMC",
    # Unified search
    "EstimatorFamily",
    "UnifiedSearchResult",
    "unified_search",
]
