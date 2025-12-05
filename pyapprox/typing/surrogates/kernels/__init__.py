"""
Kernel implementations for Gaussian processes and surrogate modeling.

This module provides kernel protocols and implementations for use in Gaussian
process regression, radial basis function networks, and other kernel-based methods.

Key Protocols
-------------
- KernelProtocol: Base protocol for kernel implementations
- KernelHasJacobianProtocol: Protocol for kernels with Jacobian support
- KernelHasParameterJacobianProtocol: Protocol for kernels with parameter Jacobian
- KernelWithJacobianProtocol: Composition of kernel and Jacobian protocols
- KernelWithJacobianAndParameterJacobianProtocol: Full derivative support

Key Classes
-----------
- Kernel: Abstract base class for kernel implementations

Examples
--------
>>> from pyapprox.typing.surrogates.kernels import KernelProtocol
>>> def fit_gp(kernel: KernelProtocol, X, y):
...     K = kernel(X, X)
...     # Fit Gaussian process
"""

from .protocols import (
    KernelProtocol,
    KernelHasJacobianProtocol,
    KernelHasParameterJacobianProtocol,
    KernelWithJacobianProtocol,
    KernelWithJacobianAndParameterJacobianProtocol,
    Kernel,
)

__all__ = [
    "KernelProtocol",
    "KernelHasJacobianProtocol",
    "KernelHasParameterJacobianProtocol",
    "KernelWithJacobianProtocol",
    "KernelWithJacobianAndParameterJacobianProtocol",
    "Kernel",
]
