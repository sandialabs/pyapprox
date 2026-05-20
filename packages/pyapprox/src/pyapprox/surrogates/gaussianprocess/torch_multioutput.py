"""
PyTorch-specific Multi-Output Gaussian Process using autograd for
hyperparameter optimization.

This module provides TorchMultiOutputGP which overrides the loss configuration
to use PyTorch autograd for computing gradients of the negative log marginal
likelihood w.r.t. hyperparameters. This enables optimization with any
differentiable kernel, even those lacking analytical jacobian_wrt_params.
"""

from __future__ import annotations

from typing import Union

import torch

from pyapprox.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP,
)
from pyapprox.surrogates.kernels.multioutput import (
    DAGMultiOutputKernel,
    IndependentMultiOutputKernel,
    LinearCoregionalizationKernel,
    MultiLevelKernel,
)
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.validation import validate_backends


class TorchMultiOutputGP(MultiOutputGP[torch.Tensor]):
    """
    PyTorch Multi-Output GP with autograd-based hyperparameter optimization.

    Inherits from MultiOutputGP. When used with fitters, the fitter detects
    the PyTorch backend and binds autograd-based jacobian on the loss function,
    enabling gradient-based optimization without analytical kernel derivatives.

    Parameters
    ----------
    kernel : IndependentMultiOutputKernel, LinearCoregionalizationKernel,
             MultiLevelKernel, or DAGMultiOutputKernel
        Multi-output kernel. Must use TorchBkd as its backend.
    nugget : float, optional
        Numerical stability parameter. Default is 1e-6.
    """

    def __init__(
        self,
        kernel: Union[
            IndependentMultiOutputKernel[torch.Tensor],
            LinearCoregionalizationKernel[torch.Tensor],
            MultiLevelKernel[torch.Tensor],
            DAGMultiOutputKernel[torch.Tensor],
        ],
        nugget: float = 1e-6,
    ):
        bkd = TorchBkd()

        # Validate that kernel uses TorchBkd
        if hasattr(kernel, "_bkd"):
            validate_backends([bkd, kernel._bkd])

        super().__init__(kernel, nugget)

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TorchMultiOutputGP(kernel={self._kernel.__class__.__name__}, "
            f"nugget={self._nugget}, {fitted_str})"
        )
