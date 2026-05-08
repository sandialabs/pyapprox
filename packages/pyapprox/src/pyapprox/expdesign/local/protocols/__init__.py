"""
Protocols for local OED components.

This module defines the interfaces for:
- Criteria: Objective functions measuring design quality
- Design matrices: M0, M1 computation for linear regression
- Solvers: Optimization wrappers for finding optimal designs
"""

from .criterion import (
    LocalOEDCriterionProtocol,
    LocalOEDCriterionWithHVPProtocol,
)
from .design_matrices import DesignMatricesProtocol
from .solver import (
    LocalOEDSolverProtocol,
    OptimizerProtocol,
    OptimizerResultProtocol,
)

__all__ = [
    # Criterion protocols
    "LocalOEDCriterionProtocol",
    "LocalOEDCriterionWithHVPProtocol",
    # Design matrices protocol
    "DesignMatricesProtocol",
    # Solver protocols
    "OptimizerResultProtocol",
    "OptimizerProtocol",
    "LocalOEDSolverProtocol",
]
