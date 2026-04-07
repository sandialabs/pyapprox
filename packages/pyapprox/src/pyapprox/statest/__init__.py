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

from pyapprox.statest.acv import (
    ACVEstimator,
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.statest.aetc import AETC, AETCBLUE, AETCMC
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.groupacv import (
    BaseGroupACVEstimator,
    GroupACVCostConstraint,
    GroupACVEstimatorIS,
    GroupACVEstimatorNested,
    GroupACVLogDetObjective,
    GroupACVObjective,
    GroupACVTraceObjective,
    MLBLUEEstimator,
    MLBLUEObjective,
    default_groupacv_optimizer,
    get_model_subsets,
)
from pyapprox.statest.mc_estimator import MCEstimator
from pyapprox.statest.plotting import (
    plot_allocation,
    plot_estimator_variance_reductions,
    plot_recursion_dag,
)
from pyapprox.statest.protocols import (
    EstimatorProtocol,
    StatisticProtocol,
)
from pyapprox.statest.search import (
    EstimatorFamily,
    UnifiedSearchResult,
    unified_search,
)
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputMeanAndVariance,
    MultiOutputStatistic,
    MultiOutputVariance,
)
from pyapprox.statest.strategies import (
    AllModelsStrategy,
    AllQoIStrategy,
    AllQoISubsetsStrategy,
    AllSubsetsStrategy,
    FixedQoIStrategy,
    FixedSubsetStrategy,
    ListQoIStrategy,
    ListSubsetStrategy,
    ModelSubsetStrategy,
    QoISubsetStrategy,
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
    "plot_estimator_variance_reductions",
    "plot_recursion_dag",
    # AETC exports
    "AETC",
    "AETCBLUE",
    "AETCMC",
    # Unified search
    "EstimatorFamily",
    "UnifiedSearchResult",
    "unified_search",
]
