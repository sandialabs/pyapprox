"""
PyTorch-specific Gaussian Process using autograd for derivatives.

This module provides TorchExactGaussianProcess which computes all derivatives
(Jacobians, HVPs) via PyTorch's automatic differentiation. This enables use
with any differentiable kernel, including TorchMaternKernel with arbitrary nu.

Unlike the backend-agnostic ExactGaussianProcess which requires kernels to
implement analytical jacobian and hvp methods, this class automatically
computes derivatives through the computational graph using bkd.jacobian
and bkd.hvp.
"""

from typing import Optional, Callable
import torch

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.validation import validate_backends
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.util.linalg.cholesky_factor import CholeskyFactor
from pyapprox.typing.surrogates.gaussianprocess.data import GPTrainingData
from pyapprox.typing.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ZeroMean
)


class TorchExactGaussianProcess:
    """
    PyTorch Gaussian Process with autograd-based derivatives.

    This class implements exact GP regression using PyTorch, with automatic
    differentiation for computing Jacobian of predictions w.r.t. inputs.

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
    >>> from pyapprox.typing.surrogates.kernels.torch_matern import TorchMaternKernel
    >>> import torch
    >>>
    >>> # Create kernel with nu=2.5
    >>> kernel = TorchMaternKernel(nu=2.5, lenscale=[1.0],
    ...                            lenscale_bounds=(0.1, 10.0), nvars=1)
    >>> gp = TorchExactGaussianProcess(kernel, nvars=1)
    >>>
    >>> # Fit
    >>> X_train = torch.randn(1, 20, dtype=torch.float64)
    >>> y_train = torch.sin(X_train[0])[:, None]
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
        nugget: float = 1e-6
    ):
        self._bkd = TorchBkd()

        # Validate that kernel uses TorchBkd
        if hasattr(kernel, '_bkd'):
            validate_backends([self._bkd, kernel._bkd])

        self._kernel = kernel
        self._nvars = nvars

        if mean_function is None:
            self._mean = ZeroMean(self._bkd)
        else:
            # Validate that mean function uses TorchBkd
            if hasattr(mean_function, '_bkd'):
                validate_backends([self._bkd, mean_function._bkd])
            self._mean = mean_function

        if nugget <= 0:
            raise ValueError(f"nugget must be positive, got {nugget}")
        self._nugget = nugget

        # Training data (set during fit)
        self._data: Optional[GPTrainingData] = None
        self._cholesky: Optional[CholeskyFactor] = None
        self._alpha: Optional[torch.Tensor] = None

        # Setup derivative methods based on kernel capabilities
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally expose derivative methods based on kernel capabilities.

        Jacobian is always available since torch.cdist supports first-order
        derivatives. HVP is NOT available because torch.cdist does not support
        second-order derivatives required for Hessian computation.
        """
        # Jacobian is always available (first-order derivatives work)
        self.jacobian = self._jacobian

        # HVP is NOT available - torch.cdist doesn't support second-order
        # derivatives needed for Hessian computation
        # self.hvp = self._hvp  # Intentionally not bound

    def bkd(self) -> TorchBkd:
        """Return the backend."""
        return self._bkd

    def kernel(self):
        """Return the kernel."""
        return self._kernel

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of output dimensions."""
        if self._data is None:
            return 1
        return self._data.nqoi()

    def is_fitted(self) -> bool:
        """Check if the GP has been fitted."""
        return self._data is not None

    def hyp_list(self) -> HyperParameterList:
        """Return combined hyperparameter list."""
        kernel_hyps = self._kernel.hyp_list()
        mean_hyps = self._mean.hyp_list()
        all_hyps = kernel_hyps.hyperparameters() + mean_hyps.hyperparameters()
        return HyperParameterList(all_hyps)

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Fit the GP to training data.

        Parameters
        ----------
        X_train : torch.Tensor, shape (nvars, n_train)
            Training inputs.
        y_train : torch.Tensor, shape (n_train, nqoi)
            Training outputs.
        """
        self._data = GPTrainingData(X_train, y_train, self._bkd)

        if self._data.nvars() != self._nvars:
            raise ValueError(
                f"X_train has {self._data.nvars()} variables, "
                f"expected {self._nvars}"
            )

        # Compute kernel matrix
        K = self._kernel(X_train, X_train)

        # Add nugget
        K_noisy = K + self._bkd.eye(K.shape[0]) * self._nugget

        # Cholesky factorization
        try:
            L = self._bkd.cholesky(K_noisy)
            self._cholesky = CholeskyFactor(L, self._bkd)
        except Exception as e:
            raise RuntimeError(
                f"Cholesky factorization failed: {e}. "
                "Try increasing nugget."
            )

        # Compute alpha = K^{-1}(y - m)
        mean_pred = self._mean(X_train)
        residual = y_train - mean_pred
        self._alpha = self._cholesky.solve(residual)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict posterior mean.

        Parameters
        ----------
        X : torch.Tensor, shape (nvars, n_test)
            Test inputs.

        Returns
        -------
        mean : torch.Tensor, shape (n_test, nqoi)
            Posterior mean predictions.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before prediction")

        mean_prior = self._mean(X)
        K_star = self._kernel(X, self._data.X())
        return mean_prior + K_star @ self._alpha

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Predict posterior mean (returns shape (nqoi, n_test))."""
        return self.predict(X).T

    def predict_std(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict posterior standard deviation.

        Parameters
        ----------
        X : torch.Tensor, shape (nvars, n_test)
            Test inputs.

        Returns
        -------
        std : torch.Tensor, shape (n_test, nqoi)
            Posterior standard deviations.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before prediction")

        K_star = self._kernel(X, self._data.X())
        K_star_star = self._kernel.diag(X)

        v = self._bkd.solve_triangular(
            self._cholesky.factor(), K_star.T, lower=True
        )

        var = K_star_star - self._bkd.einsum("ij,ij->j", v, v)
        var = var * (var >= 0.0)  # Clamp negative values
        std = self._bkd.sqrt(var)

        nqoi = self._data.nqoi()
        std = self._bkd.reshape(std, (std.shape[0], 1))
        return self._bkd.tile(std, (1, nqoi))

    def _jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of GP mean w.r.t. inputs using bkd.jacobian.

        Parameters
        ----------
        X : torch.Tensor, shape (nvars, n_samples)
            Input locations.

        Returns
        -------
        jac : torch.Tensor
            Jacobian. Shape (nqoi, nvars) for single sample,
            (n_samples, nqoi, nvars) for multiple samples.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")

        n_samples = X.shape[1]
        nqoi = self.nqoi()

        jacs = []
        for i in range(n_samples):
            x_i = X[:, i]  # (nvars,)

            # Define prediction function for single sample
            def pred_func(x: torch.Tensor) -> torch.Tensor:
                # x is (nvars,), reshape to (nvars, 1) for predict
                return self.predict(x[:, None])[:, 0]  # (nqoi,)

            # Use bkd.jacobian: returns (nqoi, nvars)
            jac_i = self._bkd.jacobian(pred_func, x_i)
            jacs.append(jac_i)

        if n_samples == 1:
            return jacs[0]  # (nqoi, nvars)
        else:
            return torch.stack(jacs, dim=0)  # (n_samples, nqoi, nvars)

    def _hvp(self, X: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product using bkd.hvp.

        Note: This method is NOT exposed publicly because torch.cdist does not
        support second-order derivatives required for Hessian computation.

        Computes H(x) @ v where H is the Hessian of the GP mean.

        Parameters
        ----------
        X : torch.Tensor, shape (nvars, n_samples)
            Input locations.
        direction : torch.Tensor, shape (nvars, n_samples)
            Direction vectors.

        Returns
        -------
        hvp : torch.Tensor, shape (nvars, n_samples)
            Hessian-vector products.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing HVP")

        if X.shape != direction.shape:
            raise ValueError(
                f"X and direction must have same shape, "
                f"got {X.shape} and {direction.shape}"
            )

        n_samples = X.shape[1]

        hvps = []
        for i in range(n_samples):
            # Clone and detach to avoid graph issues
            x_i = X[:, i].clone().detach()  # (nvars,)
            v_i = direction[:, i].clone().detach()  # (nvars,)

            # Define scalar-valued prediction function for bkd.hvp
            # (bkd.hvp requires a scalar-valued function)
            def pred_scalar(x: torch.Tensor) -> torch.Tensor:
                # x is (nvars,), reshape to (nvars, 1) for predict
                return self.predict(x[:, None])[0, 0]  # scalar

            # Use bkd.hvp: returns (nvars,)
            hvp_i = self._bkd.hvp(pred_scalar, x_i, v_i)
            hvps.append(hvp_i)

        return torch.stack(hvps, dim=1)  # (nvars, n_samples)

    def neg_log_marginal_likelihood(self) -> float:
        """
        Compute negative log marginal likelihood.

        Returns
        -------
        nlml : float
            Negative log marginal likelihood.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted first")

        import math
        n = self._data.n_samples()

        mean_pred = self._mean(self._data.X())
        residual = self._data.y() - mean_pred
        data_fit = float(self._bkd.sum(residual * self._alpha))

        log_det = self._cholesky.log_determinant()
        constant = n * math.log(2 * math.pi)

        return 0.5 * (data_fit + log_det + constant)

    def loss_value_and_grad(
        self, params: torch.Tensor
    ) -> tuple:
        """
        Compute loss and gradient w.r.t. hyperparameters using autograd.

        Parameters
        ----------
        params : torch.Tensor, shape (nparams,)
            Hyperparameter values (in optimization space).

        Returns
        -------
        loss : float
            Negative log marginal likelihood.
        grad : torch.Tensor, shape (nparams,)
            Gradient of loss w.r.t. parameters.
        """
        # Ensure params requires grad
        if not params.requires_grad:
            params = params.detach().requires_grad_(True)

        # Update hyperparameters
        self.hyp_list().set_active_values(params)

        # Refit
        self.fit(self._data.X(), self._data.y())

        # Compute loss with gradient tracking
        loss = self._neg_log_marginal_likelihood_differentiable()

        # Compute gradient
        grad = torch.autograd.grad(loss, params)[0]

        return float(loss), grad.detach()

    def _neg_log_marginal_likelihood_differentiable(self) -> torch.Tensor:
        """
        Compute NLML in a way that preserves the computational graph.
        """
        import math

        X_train = self._data.X()
        y_train = self._data.y()
        n = X_train.shape[1]

        # Recompute kernel matrix (preserves graph)
        K = self._kernel(X_train, X_train)
        K_noisy = K + torch.eye(n, dtype=K.dtype, device=K.device) * self._nugget

        # Cholesky
        L = torch.linalg.cholesky(K_noisy)

        # Solve for alpha
        mean_pred = self._mean(X_train)
        residual = y_train - mean_pred
        alpha = torch.cholesky_solve(residual, L)

        # Data fit term
        data_fit = (residual * alpha).sum()

        # Log determinant
        log_det = 2.0 * torch.log(torch.diag(L)).sum()

        # Constant
        constant = n * math.log(2 * math.pi)

        return 0.5 * (data_fit + log_det + constant)

    def optimize_hyperparameters(
        self,
        optimizer: Optional[object] = None,
        init_guess: Optional[torch.Tensor] = None,
        maxiter: int = 1000,
        lr: float = 0.01
    ) -> None:
        """
        Optimize hyperparameters using gradient descent with autograd.

        Parameters
        ----------
        optimizer : optional
            PyTorch optimizer class. If None, uses Adam.
        init_guess : optional
            Initial hyperparameter values.
        maxiter : int
            Maximum iterations. Default 1000.
        lr : float
            Learning rate. Default 0.01.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted first")

        if self.hyp_list().nactive_params() == 0:
            return

        # Get initial parameters
        if init_guess is None:
            params = self.hyp_list().get_active_values().clone()
        else:
            params = init_guess.clone()

        params.requires_grad_(True)

        # Create optimizer
        if optimizer is None:
            opt = torch.optim.Adam([params], lr=lr)
        else:
            opt = optimizer([params])

        # Store original data
        X_train = self._data.X().clone()
        y_train = self._data.y().clone()

        best_loss = float('inf')
        best_params = params.clone()

        for i in range(maxiter):
            opt.zero_grad()

            # Update hyperparameters
            self.hyp_list().set_active_values(params.detach())

            # Refit
            self.fit(X_train, y_train)

            # Compute loss with autograd
            loss = self._neg_log_marginal_likelihood_differentiable()

            # Check for improvement
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_params = params.detach().clone()

            # Backward
            loss.backward()

            # Update gradients to params
            if params.grad is None:
                # Manually compute gradient if not set
                with torch.enable_grad():
                    params_with_grad = params.detach().requires_grad_(True)
                    self.hyp_list().set_active_values(params_with_grad)
                    self.fit(X_train, y_train)
                    loss = self._neg_log_marginal_likelihood_differentiable()
                    loss.backward()
                    params.grad = params_with_grad.grad

            opt.step()

        # Set best parameters
        self.hyp_list().set_active_values(best_params)
        self.fit(X_train, y_train)

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"TorchExactGaussianProcess(kernel={self._kernel.__class__.__name__}, "
            f"nvars={self._nvars}, nugget={self._nugget}, {fitted_str})"
        )
