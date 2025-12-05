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

Kernel Implementations
---------------------
- MaternKernel: Matérn kernel for varying levels of smoothness
- ConstantKernel: Constant kernel for scaling or offset
- IIDGaussianNoise: Independent Gaussian noise kernel

Composition Kernels
------------------
- CompositionKernel: Base class for kernel compositions
- ProductKernel: Product of two kernels (element-wise multiplication)
- SumKernel: Sum of two kernels (element-wise addition)

Examples
--------
>>> from pyapprox.typing.surrogates.kernels import MaternKernel, ConstantKernel, IIDGaussianNoise
>>> from pyapprox.typing.util.backends.numpy import NumpyBkd
>>> bkd = NumpyBkd()
>>> # Create individual kernels
>>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
>>> constant = ConstantKernel(2.0, (0.1, 10.0), bkd)
>>> noise = IIDGaussianNoise(0.1, (0.01, 1.0), bkd)
>>> # Compose kernels using operators
>>> gp_kernel = matern * constant + noise
"""

from .protocols import (
    KernelProtocol,
    KernelHasJacobianProtocol,
    KernelHasParameterJacobianProtocol,
    KernelWithJacobianProtocol,
    KernelWithJacobianAndParameterJacobianProtocol,
    Kernel,
)
from .matern import MaternKernel
from .composition import (
    CompositionKernel,
    ProductKernel,
    SumKernel,
)
from .constant import ConstantKernel
from .iid_gaussian_noise import IIDGaussianNoise

__all__ = [
    # Protocols
    "KernelProtocol",
    "KernelHasJacobianProtocol",
    "KernelHasParameterJacobianProtocol",
    "KernelWithJacobianProtocol",
    "KernelWithJacobianAndParameterJacobianProtocol",
    # Base class
    "Kernel",
    # Kernel implementations
    "MaternKernel",
    "ConstantKernel",
    "IIDGaussianNoise",
    # Composition kernels
    "CompositionKernel",
    "ProductKernel",
    "SumKernel",
]
