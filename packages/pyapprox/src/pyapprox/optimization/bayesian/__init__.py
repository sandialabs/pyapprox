"""Bayesian optimization module for pyapprox.

Provides ask/tell/step/run workflows, stateless acquisition functions,
multi-start acquisition optimization, and batch selection strategies.
"""

from pyapprox.optimization.bayesian.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from pyapprox.optimization.bayesian.acquisition_optimizer import (
    AcquisitionOptimizer,
    CandidateGeneratorProtocol,
    SobolCandidateGenerator,
    UniformRandomCandidateGenerator,
)
from pyapprox.optimization.bayesian.batch import KrigingBeliever
from pyapprox.optimization.bayesian.convergence import (
    AcquisitionToleranceCriterion,
    ConvergenceCriterionProtocol,
    DistanceToleranceCriterion,
    ValueToleranceCriterion,
)
from pyapprox.optimization.bayesian.domain import BoxDomain
from pyapprox.optimization.bayesian.fitter_adapter import (
    GPFitterAdapter,
    GPFixedFitterAdapter,
    GPIncrementalFitterAdapter,
)
from pyapprox.optimization.bayesian.hp_schedule import (
    AlwaysOptimizeSchedule,
    EveryKSchedule,
    GeometricSchedule,
    HPRefitScheduleProtocol,
)
from pyapprox.optimization.bayesian.math_utils import normal_cdf, normal_pdf
from pyapprox.optimization.bayesian.optimizer import BayesianOptimizer
from pyapprox.optimization.bayesian.protocols import (
    AcquisitionContext,
    AcquisitionFunctionProtocol,
    AcquisitionOptimizerProtocol,
    BatchStrategyProtocol,
    BODomainProtocol,
    SurrogateFitterProtocol,
    SurrogateProtocol,
)
from pyapprox.optimization.bayesian.state import (
    AskResult,
    BestResult,
    BOState,
    ConvergenceContext,
    RunResult,
    StepResult,
)

__all__ = [
    "AcquisitionContext",
    "AcquisitionFunctionProtocol",
    "AcquisitionOptimizer",
    "AcquisitionOptimizerProtocol",
    "AcquisitionToleranceCriterion",
    "CandidateGeneratorProtocol",
    "AlwaysOptimizeSchedule",
    "AskResult",
    "BatchStrategyProtocol",
    "BayesianOptimizer",
    "BestResult",
    "BODomainProtocol",
    "BOState",
    "BoxDomain",
    "ConvergenceContext",
    "ConvergenceCriterionProtocol",
    "DistanceToleranceCriterion",
    "EveryKSchedule",
    "ExpectedImprovement",
    "GeometricSchedule",
    "GPFitterAdapter",
    "GPFixedFitterAdapter",
    "GPIncrementalFitterAdapter",
    "HPRefitScheduleProtocol",
    "KrigingBeliever",
    "ProbabilityOfImprovement",
    "RunResult",
    "StepResult",
    "SobolCandidateGenerator",
    "SurrogateFitterProtocol",
    "SurrogateProtocol",
    "UniformRandomCandidateGenerator",
    "UpperConfidenceBound",
    "ValueToleranceCriterion",
    "normal_cdf",
    "normal_pdf",
]
