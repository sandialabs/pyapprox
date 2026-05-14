"""Factory function for creating latent regressors."""

from __future__ import annotations

from pyapprox.surrogates.kerneloperator.protocols import (
    LatentRegressorProtocol,
)
from pyapprox.surrogates.kerneloperator.regressors.multioutput_kernel import (
    MultiOutputKernelLatentRegressor,
)
from pyapprox.surrogates.kerneloperator.regressors.scalar_kernel import (
    ScalarKernelLatentRegressor,
)
from pyapprox.surrogates.kernels.multioutput.protocols import (
    MultiOutputKernelProtocol,
)
from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.util.backends.protocols import Array, Backend


def make_latent_regressor(
    kernel: object,
    ncodes_in: int,
    ncodes_out: int,
    bkd: Backend[Array],
    nugget: float = 1e-6,
) -> LatentRegressorProtocol[Array]:
    """Create the appropriate latent regressor for a given kernel.

    Dispatches based on kernel type. This is the ONE place isinstance
    dispatch lives for regressor creation.

    MultiOutputKernelProtocol is checked FIRST because some multi-output
    kernels may also structurally satisfy KernelProtocol.

    Parameters
    ----------
    kernel
        Scalar Kernel or MultiOutputKernelProtocol.
    ncodes_in : int
        Number of input codes.
    ncodes_out : int
        Number of output codes.
    bkd : Backend[Array]
        Computational backend.
    nugget : float
        Nugget for numerical stability.

    Returns
    -------
    LatentRegressorProtocol
        The constructed regressor.
    """
    if isinstance(kernel, MultiOutputKernelProtocol):
        return MultiOutputKernelLatentRegressor(
            kernel, ncodes_in, ncodes_out, bkd, nugget
        )
    if isinstance(kernel, KernelProtocol):
        return ScalarKernelLatentRegressor(
            kernel, ncodes_in, ncodes_out, bkd, nugget
        )
    raise TypeError(
        f"kernel must satisfy KernelProtocol or "
        f"MultiOutputKernelProtocol, got {type(kernel).__name__}"
    )
