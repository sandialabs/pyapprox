"""
Validation functions for GP statistics computations.

These functions validate that the GP and kernel satisfy the requirements
for computing statistics using the separable kernel approach.
"""

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol
from pyapprox.typing.surrogates.kernels.composition import SeparableProductKernel
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
        If the kernel is not a ProductKernel or does not have the
        required separable structure.

    Notes
    -----
    Currently this function checks if the kernel is an instance of
    ProductKernel. More sophisticated checks could verify that the
    component kernels operate on disjoint dimensions.

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels.matern import MaternKernel
    >>> from pyapprox.typing.surrogates.kernels.composition import ProductKernel
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> k1 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    >>> k2 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    >>> prod_kernel = ProductKernel(k1, k2)
    >>> validate_separable_kernel(prod_kernel)  # OK

    >>> single_kernel = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> validate_separable_kernel(single_kernel)  # Raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: ...
    """
    # Check if it's a SeparableProductKernel
    if isinstance(kernel, SeparableProductKernel):
        return

    # For single-dimensional kernels, they are trivially separable
    if kernel.nvars() == 1:
        return

    # Otherwise, kernel is not separable
    raise TypeError(
        f"Kernel must be separable (SeparableProductKernel) for GP statistics "
        f"computations, but got {type(kernel).__name__}. "
        f"For multi-dimensional inputs, construct a separable product kernel: "
        f"kernel = SeparableProductKernel([k1, k2, ..., kd], bkd) where each "
        f"ki is a 1D kernel."
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
