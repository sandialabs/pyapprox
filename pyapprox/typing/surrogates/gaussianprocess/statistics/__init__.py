"""
Statistics module for Gaussian Process models.

This module provides tools for computing statistical quantities from fitted
Gaussian Processes, including:

- Expected values of GP predictions: E[f(X)]
- Variance of GP predictions: Var[f(X)]
- Sensitivity indices (Sobol indices)

The key requirement is that the GP uses a **separable (product) kernel**:
    C(x, z) = prod_k C_k(x_k, z_k)

This factorization enables efficient computation of multidimensional integrals
by reducing them to products of 1D integrals.
"""

from pyapprox.typing.surrogates.gaussianprocess.statistics.protocols import (
    KernelIntegralCalculatorProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.validation import (
    validate_separable_kernel,
    validate_zero_mean,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.integrals import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.moments import (
    GaussianProcessStatistics,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.ensemble import (
    GaussianProcessEnsemble,
)

__all__ = [
    "KernelIntegralCalculatorProtocol",
    "validate_separable_kernel",
    "validate_zero_mean",
    "SeparableKernelIntegralCalculator",
    "GaussianProcessStatistics",
    "GaussianProcessSensitivity",
    "GaussianProcessEnsemble",
]
