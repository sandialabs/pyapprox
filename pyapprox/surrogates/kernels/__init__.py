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
- MultiOutputKernelProtocol: Protocol for multi-output kernel implementations

Key Classes
-----------
- Kernel: Abstract base class for kernel implementations

Kernel Implementations
---------------------
- MaternKernel: Matérn kernel for varying levels of smoothness
- IIDGaussianNoise: Independent Gaussian noise kernel

Scaling Functions
-----------------
- PolynomialScaling: Polynomial scaling functions for spatially varying kernels
- ScalingFunctionProtocol: Protocol for scaling functions

Composition Kernels
------------------
- CompositionKernel: Base class for kernel compositions
- ProductKernel: Product of two kernels (element-wise multiplication)
- SumKernel: Sum of two kernels (element-wise addition)

Multi-Output Kernels
-------------------
- IndependentMultiOutputKernel: Independent kernels per output (block-diagonal)
- LinearCoregionalizationKernel: Linear model of coregionalization (LMC)
- MultiLevelKernel: Autoregressive multi-level kernel with spatially varying scalings
- ScalingFunction: Spatially varying scaling functions for multi-level kernels

Examples
--------
Single-output kernel composition:

>>> from pyapprox.surrogates.kernels import MaternKernel, IIDGaussianNoise
>>> from pyapprox.util.backends.numpy import NumpyBkd
>>> bkd = NumpyBkd()
>>> # Create individual kernels
>>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
>>> noise = IIDGaussianNoise(0.1, (0.01, 1.0), bkd)
>>> # Compose kernels using operators
>>> gp_kernel = matern + noise

Multi-output kernel:

>>> from pyapprox.surrogates.kernels.multioutput import IndependentMultiOutputKernel
>>> import numpy as np
>>> # Create independent kernels for each output
>>> kernel1 = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
>>> kernel2 = MaternKernel(1.5, [0.5, 0.5], (0.1, 10.0), 2, bkd)
>>> mo_kernel = IndependentMultiOutputKernel([kernel1, kernel2])
>>> # Evaluate with data for each output
>>> X_list = [bkd.array(np.random.randn(2, 10)), bkd.array(np.random.randn(2, 5))]
>>> K = mo_kernel(X_list)  # Shape: (15, 15)
"""

from .protocols import (
    KernelProtocol,
    KernelHasJacobianProtocol,
    KernelHasParameterJacobianProtocol,
    KernelHasHVPWrtX1Protocol,
    KernelHasHVPWrtParamsProtocol,
    KernelWithJacobianProtocol,
    KernelWithJacobianAndHVPWrtX1Protocol,
    KernelWithParameterJacobianProtocol,
    KernelWithParameterJacobianAndHVPProtocol,
    KernelWithJacobianAndParameterJacobianProtocol,
    KernelWithFullDerivativesProtocol,
    SeparableKernelProtocol,
    Kernel,
)
from .matern import (
    MaternKernel,
    SquaredExponentialKernel,
    Matern52Kernel,
    Matern32Kernel,
    ExponentialKernel,
)
from .composition import (
    CompositionKernel,
    ProductKernel,
    SumKernel,
    SeparableProductKernel,
)
from .iid_gaussian_noise import IIDGaussianNoise
from .scalings import (
    ScalingFunctionProtocol,
    PolynomialScaling,
)
from .multioutput import (
    MultiOutputKernelProtocol,
    IndependentMultiOutputKernel,
    LinearCoregionalizationKernel,
    MultiLevelKernel,
)

__all__ = [
    # Protocols - Base (Has)
    "KernelProtocol",
    "KernelHasJacobianProtocol",
    "KernelHasParameterJacobianProtocol",
    "KernelHasHVPWrtX1Protocol",
    "KernelHasHVPWrtParamsProtocol",
    # Protocols - Composite (With)
    "KernelWithJacobianProtocol",
    "KernelWithJacobianAndHVPWrtX1Protocol",
    "KernelWithParameterJacobianProtocol",
    "KernelWithParameterJacobianAndHVPProtocol",
    "KernelWithJacobianAndParameterJacobianProtocol",
    "KernelWithFullDerivativesProtocol",
    "SeparableKernelProtocol",
    "MultiOutputKernelProtocol",
    "ScalingFunctionProtocol",
    # Base class
    "Kernel",
    # Kernel implementations
    "MaternKernel",
    "SquaredExponentialKernel",
    "Matern52Kernel",
    "Matern32Kernel",
    "ExponentialKernel",
    "IIDGaussianNoise",
    # Scaling functions
    "PolynomialScaling",
    # Composition kernels
    "CompositionKernel",
    "ProductKernel",
    "SumKernel",
    "SeparableProductKernel",
    # Multi-output kernels
    "IndependentMultiOutputKernel",
    "LinearCoregionalizationKernel",
    "MultiLevelKernel",
]
