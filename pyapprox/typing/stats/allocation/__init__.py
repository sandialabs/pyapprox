"""Allocation algorithms for multifidelity estimation.

This module provides utilities for computing allocation matrices and
optimal sample distributions across models.
"""

from pyapprox.typing.stats.allocation.matrices import (
    get_allocation_matrix_from_recursion,
    get_npartitions_from_nmodels,
    get_nsamples_per_model,
    validate_allocation_matrix,
)

__all__ = [
    "get_allocation_matrix_from_recursion",
    "get_npartitions_from_nmodels",
    "get_nsamples_per_model",
    "validate_allocation_matrix",
]
