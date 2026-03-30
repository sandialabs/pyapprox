"""
Variational inference module for pyapprox.

This module provides variational inference tools including:
- ELBO objective with joint quadrature design
- Summary statistic protocols and implementations for amortized VI
- Covariance parameterization protocol (for Gaussian copula)
- Convergence diagnostics for early stopping

Variational distributions are conditional distributions from
``pyapprox.probability.conditional`` (e.g., ConditionalGaussian,
ConditionalBeta, ConditionalIndependentJoint) with BasisExpansion
parameter functions.
"""

from pyapprox.inverse.variational.convergence import (
    VIConvergenceMonitor,
    make_rol_convergence_status_test,
    make_scipy_convergence_callback,
)
from pyapprox.inverse.variational.convergence_protocols import (
    ConvergenceCheckProtocol,
    ConvergenceCheckResult,
)
from pyapprox.inverse.variational.elbo import (
    ELBOObjective,
    make_discrete_group_elbo,
    make_single_problem_elbo,
)
from pyapprox.inverse.variational.fitter import (
    VariationalFitter,
    VIFitResult,
)
from pyapprox.inverse.variational.importance_diagnostics import (
    ImportanceWeightedCheck,
    ImportanceWeightedMetrics,
    make_importance_check_from_elbo,
)
from pyapprox.inverse.variational.protocols import (
    CovarianceParameterizationProtocol,
    VariationalDistributionProtocol,
)
from pyapprox.inverse.variational.inexact_elbo import (
    InexactELBOObjective,
    make_inexact_discrete_group_elbo,
    make_inexact_single_problem_elbo,
)
from pyapprox.inverse.variational.summary import (
    Aggregation,
    FlattenAggregation,
    IdentityTransform,
    MaxAggregation,
    MeanAggregation,
    MeanAndVarianceAggregation,
    SummaryStatistic,
    Transform,
    TransformAggregateSummary,
)

__all__ = [
    "ConvergenceCheckProtocol",
    "ConvergenceCheckResult",
    "ELBOObjective",
    "ImportanceWeightedCheck",
    "ImportanceWeightedMetrics",
    "InexactELBOObjective",
    "VIConvergenceMonitor",
    "make_importance_check_from_elbo",
    "make_rol_convergence_status_test",
    "make_scipy_convergence_callback",
    "make_single_problem_elbo",
    "make_discrete_group_elbo",
    "make_inexact_single_problem_elbo",
    "make_inexact_discrete_group_elbo",
    "VIFitResult",
    "VariationalFitter",
    "Aggregation",
    "FlattenAggregation",
    "IdentityTransform",
    "MaxAggregation",
    "MeanAggregation",
    "MeanAndVarianceAggregation",
    "SummaryStatistic",
    "Transform",
    "TransformAggregateSummary",
    "CovarianceParameterizationProtocol",
    "VariationalDistributionProtocol",
]
