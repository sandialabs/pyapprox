"""
Validation functions for GP statistics computations.

These functions validate that the GP and kernel satisfy the requirements
for computing statistics using the separable kernel approach.
"""

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.surrogates.kernels.protocols import (
    KernelProtocol,
    SeparableKernelProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.protocols import (
    GaussianProcessProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.mean_functions import ZeroMean


def validate_separable_kernel(kernel: KernelProtocol[Array]) -> None:
    """
    Validate that a kernel is separable (product structure).

    A separable kernel has the form:
        C(x, z) = prod_k C_k(x_k, z_k)

    where C_k operates only on the k-th dimension. This structure is
    required for efficient computation of multidimensional integrals
    as products of 1D integrals.

    Parameters
    ----------
    kernel : Kernel[Array]
        The kernel to validate.

    Raises
    ------
    TypeError
        If the kernel does not satisfy SeparableKernelProtocol.

    Notes
    -----
    A kernel satisfies SeparableKernelProtocol if it implements:
    - nvars() -> int
    - get_kernel_1d(dim: int) -> KernelProtocol

    The following kernels satisfy this protocol:
    - SeparableProductKernel: Explicitly constructed from 1D kernels
    - SquaredExponentialKernel: exp(-0.5 * sum_d ...) = prod_d exp(...)

    Matern 3/2 and 5/2 kernels are NOT separable because they use
    the combined Euclidean distance inside nonlinear polynomial terms.

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels.matern import SquaredExponentialKernel
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> kernel = SquaredExponentialKernel([1.0, 2.0], (0.1, 10.0), 2, bkd)
    >>> validate_separable_kernel(kernel)  # OK - SE kernel is separable
    """
    # Check if kernel satisfies SeparableKernelProtocol
    if isinstance(kernel, SeparableKernelProtocol):
        return

    # For single-dimensional kernels, they are trivially separable
    if kernel.nvars() == 1:
        return

    # Otherwise, kernel is not separable
    raise TypeError(
        f"Kernel must satisfy SeparableKernelProtocol for GP statistics "
        f"computations, but got {type(kernel).__name__}. "
        f"Separable kernels include: SeparableProductKernel, "
        f"SquaredExponentialKernel (with ARD). "
        f"Note: Matern 3/2 and 5/2 are NOT separable. "
        f"For non-separable kernels, wrap 1D kernels in SeparableProductKernel."
    )


def validate_zero_mean(gp: GaussianProcessProtocol[Array]) -> None:
    """
    Validate that a Gaussian Process uses zero mean function.

    The statistics computations assume a zero-mean GP prior. For GPs
    with non-zero mean functions, the statistics formulas would need
    to be adjusted.

    Parameters
    ----------
    gp : GaussianProcessProtocol[Array]
        The Gaussian Process to validate.

    Raises
    ------
    ValueError
        If the GP does not use a ZeroMean function.

    Notes
    -----
    This function checks if the GP has a `mean` method that returns
    a ZeroMean instance. If the GP doesn't have a mean method, it
    is assumed to use zero mean (which is the default for many GP
    implementations).

    Examples
    --------
    >>> # Assuming gp is a GP with zero mean
    >>> validate_zero_mean(gp)  # OK

    >>> # Assuming gp_nonzero is a GP with constant mean
    >>> validate_zero_mean(gp_nonzero)  # Raises ValueError
    Traceback (most recent call last):
        ...
    ValueError: ...
    """
    # Check if GP has a mean method/attribute
    if not hasattr(gp, 'mean'):
        # No mean attribute - assume zero mean (common default)
        return

    # Get the mean function
    mean_func = gp.mean()

    # Check if it's ZeroMean
    if isinstance(mean_func, ZeroMean):
        return

    # Otherwise, not zero mean
    raise ValueError(
        f"GP must use ZeroMean for statistics computations, but got "
        f"{type(mean_func).__name__}. The statistics formulas assume "
        f"a zero-mean prior. For non-zero mean GPs, subtract the mean "
        f"from the data before fitting."
    )
