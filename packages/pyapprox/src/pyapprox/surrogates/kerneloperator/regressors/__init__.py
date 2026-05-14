"""Latent-space regressors for kernel operator learning."""

from pyapprox.surrogates.kerneloperator.regressors.factory import (
    make_latent_regressor,
)
from pyapprox.surrogates.kerneloperator.regressors.multioutput_kernel import (
    MultiOutputKernelLatentRegressor,
)
from pyapprox.surrogates.kerneloperator.regressors.scalar_kernel import (
    ScalarKernelLatentRegressor,
)

__all__ = [
    "ScalarKernelLatentRegressor",
    "MultiOutputKernelLatentRegressor",
    "make_latent_regressor",
]
