"""
Covariance operators for probability module.

This package provides implementations of covariance operators with
square-root (Cholesky) factorization for efficient Gaussian operations.

Classes
-------
DenseCholeskyCovarianceOperator
    Dense covariance with explicit Cholesky factorization.
DiagonalCovarianceOperator
    Diagonal covariance with efficient elementwise operations.
OperatorBasedCovarianceOperator
    Callback-based operator for infinite-dimensional fields.
"""

from .dense import DenseCholeskyCovarianceOperator
from .diagonal import DiagonalCovarianceOperator
from .operator import OperatorBasedCovarianceOperator

__all__ = [
    "DenseCholeskyCovarianceOperator",
    "DiagonalCovarianceOperator",
    "OperatorBasedCovarianceOperator",
]
