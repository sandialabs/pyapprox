"""
Unified loss function for Gaussian Process hyperparameter optimization.

This module provides a unified loss class that works for both single-output
and multi-output GPs by exploiting the fact that NLL computation is identical
once the kernel matrix is formed.
"""

from typing import Generic, Tuple
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList


class GPNegativeLogMarginalLikelihoodLoss(Generic[Array]):
    """
    Unified loss function for GP hyperparameter optimization.

    Works for both ExactGaussianProcess and MultiOutputGP by using a common
    interface. Once the kernel matrix K is formed, the NLL and gradient
    computations are identical:

    NLL:
        -log p(y | X, θ) = 0.5 * [y^T K^{-1} y + log|K| + n log(2π)]

    Gradient:
        ∂(-log p)/∂θ_i = 0.5 * trace[(α α^T - K^{-1}) ∂K/∂θ_i]
        where α = K^{-1}(y - m(X))

    Parameters
    ----------
    gp : GP object
        The Gaussian Process model (ExactGaussianProcess or MultiOutputGP).
        Must have: bkd(), hyp_list(), fit(), neg_log_marginal_likelihood(),
        and internal _alpha, _cholesky, _kernel attributes.
    fit_args : tuple
        Arguments to pass to gp.fit().
        For single-output: (X_train, y_train)
        For multi-output: (X_train_list, y_train_stacked)

    Examples
    --------
    Single-output GP:
    >>> from pyapprox.typing.surrogates.gaussianprocess.exact import ExactGaussianProcess
    >>> loss = GPNegativeLogMarginalLikelihoodLoss(gp, (X_train, y_train))

    Multi-output GP:
    >>> from pyapprox.typing.surrogates.gaussianprocess.multioutput import MultiOutputGP
    >>> loss = GPNegativeLogMarginalLikelihoodLoss(gp, ([X1, X2], y_stacked))
    """

    def __init__(
        self,
        gp,  # Union[ExactGaussianProcess, MultiOutputGP]
        fit_args: Tuple
    ):
        self._gp = gp
        self._fit_args = fit_args
        self._bkd = gp.bkd()
        self._hyp_list = gp.hyp_list()

    def nvars(self) -> int:
        """Number of active hyperparameters."""
        return self._hyp_list.nactive_params()

    def nqoi(self) -> int:
        """Number of quantities of interest (always 1 for scalar loss)."""
        return 1

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list."""
        return self._hyp_list

    def __call__(self, params: Array) -> Array:
        """
        Compute negative log marginal likelihood.

        Parameters
        ----------
        params : Array
            Active hyperparameters, shape (nactive,) or (nactive, 1).

        Returns
        -------
        nll : Array
            Negative log likelihood, shape (1, 1).
        """
        # Ensure params is 1D
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        # Update hyperparameters
        self._hyp_list.set_active_values(params)

        # Refit GP with new hyperparameters
        self._gp.fit(*self._fit_args)

        # Compute NLL (same formula for both single and multi-output)
        nll = self._gp.neg_log_marginal_likelihood()

        # Return as (1, 1) array
        return self._bkd.reshape(self._bkd.array([nll]), (1, 1))

    def jacobian(self, params: Array) -> Array:
        """
        Compute gradient of NLML w.r.t. hyperparameters.

        Uses unified formula that works for both single and multi-output GPs:
            ∂(-log p)/∂θ_kernel = 0.5 * trace[(α α^T - K^{-1}) ∂K/∂θ]
            ∂(-log p)/∂θ_mean = -α^T ∂m/∂θ  (if mean function exists)

        Parameters
        ----------
        params : Array
            Active hyperparameters, shape (nactive,) or (nactive, 1).

        Returns
        -------
        grad : Array
            Gradient, shape (1, nactive).
        """
        # Ensure params is 1D
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        # Update and refit
        self._hyp_list.set_active_values(params)
        self._gp.fit(*self._fit_args)

        # Get alpha and cholesky (common to all GPs)
        alpha = self._gp._alpha
        cholesky = self._gp._cholesky

        grad_values = []

        # 1. Kernel hyperparameter gradients (same for all GPs)
        kernel = self._gp._kernel
        kernel_hyps = kernel.hyp_list()

        if kernel_hyps.nparams() > 0:
            if not hasattr(kernel, 'jacobian_wrt_params'):
                raise NotImplementedError(
                    f"Kernel {type(kernel).__name__} does not implement "
                    "jacobian_wrt_params() method"
                )

            # Get kernel gradients
            # For single-output: K_grad = kernel.jacobian_wrt_params(X)
            # For multi-output: K_grad = kernel.jacobian_wrt_params(X_list)
            X_data = self._fit_args[0]  # X or X_list
            K_grad = kernel.jacobian_wrt_params(X_data)
            n_kernel_params = K_grad.shape[2]

            for i in range(n_kernel_params):
                dK = K_grad[:, :, i]

                # Gradient formula (same for all GPs):
                # ∂(-log p)/∂θ = 0.5 * (trace(K^{-1} @ dK) - α^T @ dK @ α)
                term1 = self._bkd.sum(alpha * (dK @ alpha))
                Kinv_dK = cholesky.solve(dK)
                term2 = self._bkd.trace(Kinv_dK)

                grad_i = 0.5 * (term2 - term1)
                grad_values.append(grad_i)

        # 2. Mean function hyperparameter gradients (if mean exists)
        if hasattr(self._gp, '_mean'):
            mean_hyps = self._gp._mean.hyp_list()

            if mean_hyps.nparams() > 0:
                # Get training data
                # For single-output GP: X_train is first arg
                # For multi-output GP: X_train_list is first arg
                X_data = self._fit_args[0]

                # Get mean function Jacobian
                mean_jac = self._gp._mean.jacobian_wrt_params(X_data)

                for i in range(mean_hyps.nparams()):
                    dm_dtheta = mean_jac[i, :, :]
                    grad_mean_i = -self._bkd.sum(alpha * dm_dtheta)
                    grad_values.append(grad_mean_i)

        # Stack full gradients (for all parameters, fixed or not)
        if len(grad_values) > 0:
            full_grad = self._bkd.array(grad_values)
        else:
            full_grad = self._bkd.zeros((0,))

        # Extract only active parameter gradients
        active_grad = self._hyp_list.extract_active(full_grad)
        grad = self._bkd.reshape(active_grad, (1, len(active_grad)))

        return grad

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GPNegativeLogMarginalLikelihoodLoss("
            f"nvars={self.nvars()}, "
            f"gp_type={type(self._gp).__name__})"
        )
