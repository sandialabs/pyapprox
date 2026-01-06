"""
Gaussian Bayesian Networks for inference.

This module provides tools for inference in linear-Gaussian Bayesian networks:
- GaussianFactor: Factor with variable ID tracking
- GaussianNetwork: Network of linear-Gaussian CPDs
- Variable elimination inference

The implementation reuses GaussianCanonicalForm from the probability module
for efficient factor operations.
"""

from .factor import GaussianFactor
from .scope import (
    get_unique_variable_blocks,
    expand_scope,
    get_partition_indices,
)
from .conversions import convert_cpd_to_canonical
from .network import GaussianNetwork
from .inference import (
    sum_product_eliminate_variable,
    sum_product_variable_elimination,
    cond_prob_variable_elimination,
)

__all__ = [
    # Factor
    "GaussianFactor",
    # Scope utilities
    "get_unique_variable_blocks",
    "expand_scope",
    "get_partition_indices",
    # Conversions
    "convert_cpd_to_canonical",
    # Network
    "GaussianNetwork",
    # Inference
    "sum_product_eliminate_variable",
    "sum_product_variable_elimination",
    "cond_prob_variable_elimination",
]
