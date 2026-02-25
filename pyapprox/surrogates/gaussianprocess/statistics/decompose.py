"""
Kernel decomposition utilities for GP statistics.

This module provides utilities to decompose composed kernels into their
base separable kernel and variance scaling factor.
"""

from typing import Tuple, TYPE_CHECKING
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.kernels.protocols import (
    KernelProtocol,
    SeparableKernelProtocol,
)
from pyapprox.surrogates.kernels.composition import (
    ProductKernel,
    SumKernel,
)
from pyapprox.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.surrogates.kernels.iid_gaussian_noise import (
    IIDGaussianNoise,
)


def _decompose_kernel(
    kernel: KernelProtocol[Array],
    bkd: Backend[Array],
) -> Tuple[SeparableKernelProtocol[Array], Array]:
    """
    Decompose a kernel into its separable base kernel and variance scaling.

    The GP statistics code computes integrals using a unit-variance correlation
    kernel C(x,z), then applies the signal variance s² in the statistics formulas.
    This function extracts these components from composed kernels.

    Supported patterns:
    - SeparableKernelProtocol → (kernel, 1.0)
    - PolynomialScaling(degree=0) * SeparableKernelProtocol → (base, s²)
    - SeparableKernelProtocol * PolynomialScaling(degree=0) → (base, s²)

    Parameters
    ----------
    kernel : KernelProtocol[Array]
        The kernel to decompose.
    bkd : Backend[Array]
        Numerical backend.

    Returns
    -------
    base_kernel : SeparableKernelProtocol[Array]
        The separable base kernel with unit variance (C(x,x) = 1).
    kernel_variance : Array
        The kernel variance s² (scalar).

    Raises
    ------
    TypeError
        If the kernel cannot be decomposed into a supported pattern.

    Examples
    --------
    >>> # Unit variance separable kernel
    >>> base, s2 = _decompose_kernel(separable_kernel, bkd)
    >>> # s2 == 1.0

    >>> # Scaled separable kernel
    >>> kernel = PolynomialScaling([2.0], bounds, bkd, nvars) * separable_kernel
    >>> base, s2 = _decompose_kernel(kernel, bkd)
    >>> # s2 == 4.0, base is the original separable_kernel
    """
    # Case 0: SumKernel - strip IIDGaussianNoise and recurse on the signal part
    if isinstance(kernel, SumKernel):
        k1 = getattr(kernel, '_kernel1', None)
        k2 = getattr(kernel, '_kernel2', None)
        if k1 is not None and k2 is not None:
            for signal, noise in [(k1, k2), (k2, k1)]:
                if isinstance(noise, IIDGaussianNoise):
                    return _decompose_kernel(signal, bkd)
        raise TypeError(
            f"SumKernel decomposition requires one component to be "
            f"IIDGaussianNoise, got {type(k1).__name__} + {type(k2).__name__}"
        )

    # Case 1: Already separable with unit variance
    if isinstance(kernel, SeparableKernelProtocol):
        return kernel, bkd.asarray(1.0)

    # Case 2: ProductKernel - check for PolynomialScaling * SeparableKernel
    if isinstance(kernel, ProductKernel):
        # Access components (private API)
        k1 = getattr(kernel, '_kernel1', None)
        k2 = getattr(kernel, '_kernel2', None)

        if k1 is None or k2 is None:
            raise TypeError(
                "Cannot decompose ProductKernel: internal structure not recognized. "
                "Expected _kernel1 and _kernel2 attributes."
            )

        # Check both orderings
        scaling, base = None, None
        for a, b in [(k1, k2), (k2, k1)]:
            if isinstance(a, PolynomialScaling) and a.degree() == 0:
                if isinstance(b, SeparableKernelProtocol):
                    scaling, base = a, b
                    break

        if scaling is not None and base is not None:
            # Extract s from PolynomialScaling coefficients
            s = scaling.hyp_list().get_values()[0]
            s_squared = s * s
            return base, s_squared

        # ProductKernel but not the expected pattern
        if isinstance(k1, PolynomialScaling) or isinstance(k2, PolynomialScaling):
            scaling_comp = k1 if isinstance(k1, PolynomialScaling) else k2
            other_comp = k2 if isinstance(k1, PolynomialScaling) else k1

            if scaling_comp.degree() != 0:
                raise TypeError(
                    f"PolynomialScaling must be constant (degree=0) for separable "
                    f"kernel statistics, but got degree={scaling_comp.degree()}. "
                    f"Non-constant scaling breaks kernel separability."
                )

            raise TypeError(
                f"Expected PolynomialScaling * SeparableKernelProtocol, but the "
                f"base kernel is {type(other_comp).__name__}, which does not "
                f"satisfy SeparableKernelProtocol."
            )

    # Case 3: Unsupported kernel type
    raise TypeError(
        f"Kernel must be either:\n"
        f"  1. A SeparableKernelProtocol (e.g., SeparableProductKernel), or\n"
        f"  2. PolynomialScaling([s], ...) * SeparableKernelProtocol\n"
        f"Got: {type(kernel).__name__}\n\n"
        f"To create a scaled separable kernel, use:\n"
        f"  scaling = PolynomialScaling([s], bounds, bkd, nvars)\n"
        f"  kernel = scaling * SeparableProductKernel(kernels_1d, bkd)"
    )
