"""
PyTorch-specific Variational Gaussian Process using autograd for derivatives.

This module provides TorchVariationalGaussianProcess which uses PyTorch's
automatic differentiation for ELBO gradient computation during optimization
and for prediction Jacobians.
"""

from __future__ import annotations

from typing import Optional

import torch

from pyapprox.surrogates.gaussianprocess.inducing_samples import (
    InducingSamples,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
)
from pyapprox.surrogates.gaussianprocess.variational import (
    VariationalGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.variational_loss import (
    VariationalGPELBOLoss,
)
from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.validation import validate_backends


class TorchVariationalGaussianProcess(VariationalGaussianProcess[torch.Tensor]):
    """
    PyTorch Variational GP with autograd-based derivatives.

    Inherits from VariationalGaussianProcess and overrides derivative
    methods to use PyTorch autograd. This enables optimization of the
    ELBO without analytical gradient implementations.

    Parameters
    ----------
    kernel : object
        Kernel function using TorchBkd.
    nvars : int
        Number of input variables.
    inducing_samples : InducingSamples[torch.Tensor]
        Inducing point manager.
    mean_function : Optional[MeanFunction], optional
        Mean function. Defaults to ZeroMean.
    nugget : float, optional
        Numerical stability parameter. Default 1e-6.
    """

    def __init__(
        self,
        kernel: Kernel[torch.Tensor],
        nvars: int,
        inducing_samples: InducingSamples[torch.Tensor],
        mean_function: Optional[MeanFunction[torch.Tensor]] = None,
        nugget: float = 1e-6,
    ) -> None:
        bkd = TorchBkd()

        if hasattr(kernel, "_bkd"):
            validate_backends([bkd, kernel._bkd])
        if mean_function is not None and hasattr(mean_function, "_bkd"):
            validate_backends([bkd, mean_function._bkd])

        super().__init__(kernel, nvars, inducing_samples, bkd, mean_function, nugget)

    def _setup_derivative_methods(self) -> None:
        """Bind autograd-based prediction Jacobian methods."""
        self.jacobian = self._jacobian
        self.jacobian_batch = self._jacobian_batch

    def _jacobian(self, sample: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of GP mean w.r.t. inputs (single sample).

        Parameters
        ----------
        sample : torch.Tensor, shape (nvars, 1)
            Single input location.

        Returns
        -------
        torch.Tensor
            Jacobian, shape (nqoi, nvars).
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")
        if sample.shape[1] != 1:
            raise ValueError(
                f"jacobian() expects single sample (nvars, 1), "
                f"got {sample.shape}. Use jacobian_batch() for batches."
            )

        x = sample[:, 0]

        def pred_func(x: torch.Tensor) -> torch.Tensor:
            return self.predict(x[:, None])[:, 0]

        return self._bkd.jacobian(pred_func, x)

    def _jacobian_batch(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of GP mean w.r.t. inputs (batch).

        Parameters
        ----------
        samples : torch.Tensor, shape (nvars, n_samples)
            Input locations.

        Returns
        -------
        torch.Tensor
            Jacobian, shape (n_samples, nqoi, nvars).
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")

        n_samples = samples.shape[1]
        jacs = []
        for i in range(n_samples):
            x_i = samples[:, i]

            def pred_func(x: torch.Tensor) -> torch.Tensor:
                return self.predict(x[:, None])[:, 0]

            jacs.append(self._bkd.jacobian(pred_func, x_i))

        return torch.stack(jacs, dim=0)

    def _configure_loss(self, loss: VariationalGPELBOLoss[torch.Tensor]) -> None:
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
        fitted_str = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"TorchVariationalGaussianProcess("
            f"kernel={self._kernel.__class__.__name__}, "
            f"nvars={self._nvars}, nugget={self._nugget}, {fitted_str})"
        )
