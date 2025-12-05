"""
Loss function for Gaussian Process hyperparameter optimization.

This module provides the NegativeLogMarginalLikelihoodLoss class which
computes the negative log marginal likelihood and its gradients for use
with typing.optimization.minimize optimizers.
"""

import numpy as np
from typing import Generic, Optional
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.typing.util.hyperparameter import HyperParameterList


class NegativeLogMarginalLikelihoodLoss(Generic[Array]):
    """
    Loss function for GP hyperparameter optimization.

    Computes negative log marginal likelihood (NLML) and its gradient
    with respect to hyperparameters for use with optimization algorithms.

    The NLML is:
        -log p(y | X, θ) = 0.5 * [y^T K^{-1} y + log|K| + n log(2π)]
    where K = K(X, X; θ) + σ²I.

    The gradient is:
        ∂(-log p)/∂θ_i = 0.5 * trace[(α α^T - K^{-1}) ∂K/∂θ_i]
    where α = K^{-1}(y - m(X)).

    Parameters
    ----------
    gp : ExactGaussianProcess[Array]
        The Gaussian Process model.
    X_train : Array
        Training input data, shape (nvars, n_train).
    y_train : Array
        Training output data, shape (n_train, nqoi).

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels import MaternKernel
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>>
    >>> bkd = NumpyBkd()
    >>> kernel = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> gp = ExactGaussianProcess(kernel, 2, bkd, noise_variance=0.1)
    >>>
    >>> X_train = bkd.array(np.random.randn(2, 10))
    >>> y_train = bkd.array(np.random.randn(10, 1))
    >>>
    >>> loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)
    >>> params = gp.hyp_list().get_active_values()
    >>> nll = loss(params)
    >>> grad = loss.jacobian(params)
    """

    def __init__(
        self,
        gp: ExactGaussianProcess[Array],
        X_train: Array,
        y_train: Array
    ):
        self._gp = gp
        self._X_train = X_train
        self._y_train = y_train
        self._bkd = gp.bkd()
        self._hyp_list = gp.hyp_list()

    def nvars(self) -> int:
        """
        Number of active hyperparameters (optimization variables).

        Returns
        -------
        int
            Number of active hyperparameters to optimize.
        """
        return self._hyp_list.nactive_params()

    def nqoi(self) -> int:
        """
        Number of quantities of interest (always 1 for scalar loss).

        Returns
        -------
        int
            Always returns 1 (scalar loss function).
        """
        return 1

    def bkd(self) -> Backend[Array]:
        """
        Return the backend.

        Returns
        -------
        Backend[Array]
            The backend instance.
        """
        return self._bkd

    def __call__(self, params: Array) -> Array:
        """
        Compute negative log marginal likelihood.

        Parameters
        ----------
        params : Array
            Active hyperparameters in optimization space.
            Can be 1D array (nactive,) or 2D array (nactive, 1).

        Returns
        -------
        nll : Array
            Negative log likelihood, shape (1, 1).
        """
        # Ensure params is 1D for set_active_values
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        # Update hyperparameters
        self._hyp_list.set_active_values(params)

        # Refit GP with new hyperparameters
        self._gp.fit(self._X_train, self._y_train)

        # Compute NLL
        nll = self._gp.neg_log_marginal_likelihood()

        # Return as (1, 1) array
        return self._bkd.reshape(self._bkd.array([nll]), (1, 1))

    def jacobian(self, params: Array) -> Array:
        """
        Compute gradient of NLML w.r.t. hyperparameters.

        Uses the formula:
            ∂(-log p)/∂θ_i = 0.5 * trace[(α α^T - K^{-1}) ∂K/∂θ_i]

        where α = K^{-1}(y - m(X)).

        Parameters
        ----------
        params : Array
            Active hyperparameters in optimization space.
            Can be 1D array (nactive,) or 2D array (nactive, 1).

        Returns
        -------
        grad : Array
            Gradient, shape (1, nactive).
        """
        # Ensure params is 1D for set_active_values
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        # Update and refit
        self._hyp_list.set_active_values(params)
        self._gp.fit(self._X_train, self._y_train)

        # Get alpha and cholesky from GP
        alpha = self._gp._alpha  # Shape: (n_train, nqoi)
        cholesky = self._gp._cholesky
        n_train = self._X_train.shape[1]

        # Get hyperparameter lists for kernel and mean
        kernel_hyps = self._gp._kernel.hyp_list()
        mean_hyps = self._gp._mean.hyp_list()

        grad_values = []

        # 1. Compute gradients w.r.t. kernel hyperparameters
        kernel = self._gp._kernel
        if kernel_hyps.nparams() > 0:
            if not hasattr(kernel, 'jacobian_wrt_params'):
                raise NotImplementedError(
                    f"Kernel {type(kernel).__name__} does not implement "
                    "jacobian_wrt_params() method required for gradient computation"
                )

            # Get kernel gradients: shape (n_train, n_train, nparams)
            K_grad = kernel.jacobian_wrt_params(self._X_train)
            n_kernel_params = K_grad.shape[2]

            for i in range(n_kernel_params):
                dK = K_grad[:, :, i]  # Shape: (n_train, n_train)

                # First term: α^T @ dK @ α
                # alpha has shape (n_train, nqoi), for single output use first column
                alpha_vec = alpha[:, 0:1]  # Shape: (n_train, 1)
                term1 = self._bkd.sum(alpha_vec * (dK @ alpha_vec))

                # Second term: trace(K^{-1} @ dK)
                # K^{-1} @ dK = cholesky.solve(dK)
                Kinv_dK = cholesky.solve(dK)
                term2 = self._bkd.trace(Kinv_dK)

                # Gradient for this kernel parameter
                # ∂(-log p)/∂θ = 0.5 * (trace(K^{-1} @ dK) - α^T @ dK @ α)
                grad_i = 0.5 * (term2 - term1)
                grad_values.append(grad_i)

        # 2. Compute gradients w.r.t. mean function hyperparameters
        if mean_hyps.nparams() > 0:
            # For mean functions, the gradient is:
            # ∂(-log p)/∂m = -α^T @ ∂m/∂θ

            # For ConstantMean: m(X) = c, so ∂m/∂c = ones vector
            # ∂(-log p)/∂c = -sum(α)

            # We assume the mean function has simple structure
            # For now, handle ConstantMean specifically
            mean_func = self._gp._mean
            if hasattr(mean_func, '_constant'):
                # ConstantMean case
                alpha_vec = alpha[:, 0]  # Shape: (n_train,)
                grad_constant = -self._bkd.sum(alpha_vec)
                grad_values.append(grad_constant)
            else:
                # For other mean functions, would need to implement
                # gradient computation based on mean function structure
                raise NotImplementedError(
                    f"Gradient computation not implemented for "
                    f"mean function {type(mean_func).__name__}"
                )

        # Stack gradients and reshape to (1, nactive)
        grad = self._bkd.array(grad_values)
        grad = self._bkd.reshape(grad, (1, len(grad_values)))

        # Apply chain rule for transformation from user space to optimization space
        # The hyperparameters may be log-transformed, so we need to apply the
        # Jacobian of the transformation
        # For now, assume direct correspondence (can be extended later)

        return grad

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NegativeLogMarginalLikelihoodLoss("
            f"nvars={self.nvars()}, "
            f"n_train={self._X_train.shape[1]})"
        )
