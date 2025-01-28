"""
PyTorch-specific Multi-Output Gaussian Process using autograd for
hyperparameter optimization.

This module provides TorchMultiOutputGP which overrides the loss configuration
to use PyTorch autograd for computing gradients of the negative log marginal
likelihood w.r.t. hyperparameters. This enables optimization with any
differentiable kernel, even those lacking analytical jacobian_wrt_params.
"""

from typing import Union

import torch

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.validation import validate_backends
from pyapprox.typing.surrogates.kernels.multioutput import (
    IndependentMultiOutputKernel,
    LinearCoregionalizationKernel,
)
from pyapprox.typing.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP,
)


class TorchMultiOutputGP(MultiOutputGP[torch.Tensor]):
    """
    PyTorch Multi-Output GP with autograd-based hyperparameter optimization.

    Inherits from MultiOutputGP and overrides only _configure_loss to bind
    autograd-based jacobian on the loss function, enabling gradient-based
    optimization without requiring analytical kernel derivatives.

    Parameters
    ----------
    kernel : IndependentMultiOutputKernel or LinearCoregionalizationKernel
        Multi-output kernel. Must use TorchBkd as its backend.
    nugget : float, optional
        Numerical stability parameter. Default is 1e-6.
    """

    def __init__(
        self,
        kernel: Union[
            IndependentMultiOutputKernel, LinearCoregionalizationKernel
        ],
        nugget: float = 1e-6,
    ):
        bkd = TorchBkd()

        # Validate that kernel uses TorchBkd
        if hasattr(kernel, '_bkd'):
            validate_backends([bkd, kernel._bkd])

        super().__init__(kernel, nugget)

    def _configure_loss(self, loss) -> None:
        """Bind autograd-based jacobian on the loss function."""
        bkd = self._bkd

        def _jacobian_autograd(params: torch.Tensor) -> torch.Tensor:
            if len(params.shape) == 2 and params.shape[1] == 1:
                params = params[:, 0]

            def loss_func(p: torch.Tensor) -> torch.Tensor:
                return loss(p)[0, 0]

            jac = bkd.jacobian(loss_func, params)
            return bkd.reshape(jac, (1, len(params)))

        loss.jacobian = _jacobian_autograd

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TorchMultiOutputGP(kernel={self._kernel.__class__.__name__}, "
            f"nugget={self._nugget}, {fitted_str})"
        )
