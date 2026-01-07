"""
Local OED criteria.

This module provides optimality criteria for experimental design:

- D-optimal: Minimize log det of covariance matrix
- A-optimal: Minimize trace of covariance matrix
- C-optimal: Minimize variance of linear combination
- I-optimal: Minimize integrated prediction variance
- G-optimal: Minimize maximum prediction variance (TODO)
- R-optimal: Minimize risk-based prediction variance (TODO)
"""

from .base import LocalOEDCriterionBase
from .d_optimal import (
    DOptimalCriterion,
    DOptimalLeastSquaresCriterion,
    DOptimalQuantileCriterion,
)
from .c_optimal import (
    COptimalCriterion,
    COptimalLeastSquaresCriterion,
    COptimalQuantileCriterion,
)
from .a_optimal import (
    AOptimalCriterion,
    AOptimalLeastSquaresCriterion,
    AOptimalQuantileCriterion,
)
from .i_optimal import (
    IOptimalCriterion,
    IOptimalLeastSquaresCriterion,
)

__all__ = [
    "LocalOEDCriterionBase",
    # D-optimal
    "DOptimalCriterion",
    "DOptimalLeastSquaresCriterion",
    "DOptimalQuantileCriterion",
    # C-optimal
    "COptimalCriterion",
    "COptimalLeastSquaresCriterion",
    "COptimalQuantileCriterion",
    # A-optimal
    "AOptimalCriterion",
    "AOptimalLeastSquaresCriterion",
    "AOptimalQuantileCriterion",
    # I-optimal
    "IOptimalCriterion",
    "IOptimalLeastSquaresCriterion",
]
