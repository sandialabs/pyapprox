"""
PyTorch-specific Gaussian Process using autograd for derivatives.

This module provides TorchExactGaussianProcess which computes all derivatives
(Jacobians) via PyTorch's automatic differentiation. This enables use
with any differentiable kernel, including TorchMaternKernel with arbitrary nu.

Unlike the backend-agnostic ExactGaussianProcess which requires kernels to
implement analytical jacobian and hvp methods, this class automatically
computes derivatives through the computational graph using bkd.jacobian.
"""

from typing import Optional
import torch

from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.validation import validate_backends
from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
)


class TorchExactGaussianProcess(ExactGaussianProcess[torch.Tensor]):
    """
    PyTorch Gaussian Process with autograd-based derivatives.

    Inherits from ExactGaussianProcess and overrides only the derivative
    methods to use PyTorch autograd instead of analytical kernel derivatives.

    Works with any differentiable kernel, including TorchMaternKernel which
    supports arbitrary smoothness parameter nu.

    Optional Methods
    ----------------
    The following methods are conditionally available:

    - ``jacobian(X)``: Always available. Computes Jacobian of GP mean w.r.t.
      inputs using bkd.jacobian.

    - ``hvp(X, direction)``: NOT available. torch.cdist does not support
      second-order derivatives required for Hessian computation.

    Check availability with ``hasattr(gp, 'jacobian')`` / ``hasattr(gp, 'hvp')``.

    Parameters
    ----------
    kernel : object
        Kernel function. Must be callable as kernel(X1, X2) -> K.
        Should use PyTorch operations for autograd compatibility.
        Must use TorchBkd as its backend.
    nvars : int
        Number of input variables (dimensions).
    mean_function : Optional[MeanFunction], optional
        Mean function. If None, uses ZeroMean. Default is None.
        If provided, must use TorchBkd as its backend.
    nugget : float, optional
        Numerical stability parameter. Default is 1e-6.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels.torch_matern import TorchMaternKernel
    >>> import torch
    >>>
    >>> # Create kernel with nu=2.5
    >>> kernel = TorchMaternKernel(nu=2.5, lenscale=[1.0],
    ...                            lenscale_bounds=(0.1, 10.0), nvars=1)
    >>> gp = TorchExactGaussianProcess(kernel, nvars=1)
    >>>
    >>> # Fit (set params inactive if kernel lacks jacobian_wrt_params)
    >>> gp.hyp_list().set_all_inactive()
    >>> X_train = torch.randn(1, 20, dtype=torch.float64)
    >>> y_train = torch.sin(X_train[0])[None, :]  # Shape: (nqoi, n_train)
    >>> gp.fit(X_train, y_train)
    >>>
    >>> # Predict with gradients
    >>> X_test = torch.randn(1, 5, dtype=torch.float64)
    >>> mean = gp.predict(X_test)
    >>>
    >>> # Compute Jacobian (always available)
    >>> if hasattr(gp, 'jacobian'):
    ...     jac = gp.jacobian(X_test)
    >>>
    >>> # HVP is NOT available for this GP
    >>> hasattr(gp, 'hvp')  # Returns False
    """

    def __init__(
        self,
        kernel,
        nvars: int,
        mean_function: Optional[MeanFunction] = None,
        nugget: float = 1e-6,
    ):
        bkd = TorchBkd()

        # Validate that kernel uses TorchBkd
        if hasattr(kernel, '_bkd'):
            validate_backends([bkd, kernel._bkd])

        # Validate that mean function uses TorchBkd
        if mean_function is not None and hasattr(mean_function, '_bkd'):
            validate_backends([bkd, mean_function._bkd])

        super().__init__(kernel, nvars, bkd, mean_function, nugget)

    def _setup_derivative_methods(self) -> None:
        """
        Bind autograd-based derivative methods.

        Jacobian is always available since torch.cdist supports first-order
        derivatives. HVP is NOT available because torch.cdist does not support
        second-order derivatives required for Hessian computation.
        """
        # Jacobian is always available (first-order derivatives work)
        self.jacobian = self._jacobian
        self.jacobian_batch = self._jacobian_batch

        # HVP is NOT available - torch.cdist doesn't support second-order
        # derivatives needed for Hessian computation
        # self.hvp = self._hvp  # Intentionally not bound

    def _jacobian(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of GP mean w.r.t. inputs (single sample).

        Uses PyTorch autograd through predict() rather than requiring
        analytical kernel derivatives.

        Parameters
        ----------
        sample : torch.Tensor, shape (nvars, 1)
            Single input location.

        Returns
        -------
        jac : torch.Tensor
            Jacobian, shape (nqoi, nvars).
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")

        if sample.shape[1] != 1:
            raise ValueError(
                f"jacobian() expects single sample with shape (nvars, 1), "
                f"got {sample.shape}. Use jacobian_batch() for multiple samples."
            )

        x = sample[:, 0]  # (nvars,)

        # Define prediction function for single sample
        def pred_func(x: torch.Tensor) -> torch.Tensor:
            # x is (nvars,), reshape to (nvars, 1) for predict
            # predict returns (nqoi, 1), flatten to (nqoi,)
            return self.predict(x[:, None])[:, 0]  # (nqoi,)

        # Use bkd.jacobian: returns (nqoi, nvars)
        return self._bkd.jacobian(pred_func, x)

    def _jacobian_batch(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of GP mean w.r.t. inputs (batch).

        Uses PyTorch autograd through predict() rather than requiring
        analytical kernel derivatives.

        Parameters
        ----------
        samples : torch.Tensor, shape (nvars, n_samples)
            Input locations.

        Returns
        -------
        jac : torch.Tensor
            Jacobian, shape (n_samples, nqoi, nvars).
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")

        n_samples = samples.shape[1]

        jacs = []
        for i in range(n_samples):
            x_i = samples[:, i]  # (nvars,)

            # Define prediction function for single sample
            def pred_func(x: torch.Tensor) -> torch.Tensor:
                return self.predict(x[:, None])[:, 0]  # (nqoi,)

            # Use bkd.jacobian: returns (nqoi, nvars)
            jac_i = self._bkd.jacobian(pred_func, x_i)
            jacs.append(jac_i)

        return torch.stack(jacs, dim=0)  # (n_samples, nqoi, nvars)

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
        fitted_str = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"TorchExactGaussianProcess(kernel={self._kernel.__class__.__name__}, "
            f"nvars={self._nvars}, nugget={self._nugget}, {fitted_str})"
        )
