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

    Optional Methods
    ----------------
    This class uses dynamic method binding based on kernel capabilities:

    - ``jacobian(params)``: Available if kernel has ``jacobian_wrt_params``.
      Required for gradient-based optimization.
    - ``hvp(params, direction)``: Available if kernel has ``hvp_wrt_params``.
      Enables second-order optimizers (e.g., trust-constr with Hessian-vector products).

    Check availability with ``hasattr(loss, 'jacobian')`` or ``hasattr(loss, 'hvp')``.

    Notes
    -----
    This class follows the dynamic binding pattern for optional methods:

    - Private methods ``_jacobian`` and ``_hvp`` contain the implementation
    - During ``__init__``, public methods are conditionally assigned based on
      kernel capabilities via ``_setup_derivative_methods()``
    - Optimizers should use ``hasattr()`` to check for available methods

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels import MaternKernel
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>>
    >>> bkd = NumpyBkd()
    >>> kernel = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> gp = ExactGaussianProcess(kernel, 2, bkd, nugget=1e-10)
    >>>
    >>> X_train = bkd.array(np.random.randn(2, 10))
    >>> y_train = bkd.array(np.random.randn(10, 1))
    >>>
    >>> loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)
    >>> params = gp.hyp_list().get_active_values()
    >>> nll = loss(params)
    >>>
    >>> # Check if gradient is available
    >>> if hasattr(loss, 'jacobian'):
    ...     grad = loss.jacobian(params)
    >>>
    >>> # Check if HVP is available (for second-order optimizers)
    >>> if hasattr(loss, 'hvp'):
    ...     direction = bkd.ones_like(params)
    ...     hvp_result = loss.hvp(params, direction)
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

        # Validate kernel support and conditionally add methods
        self._setup_derivative_methods()

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

    def _setup_derivative_methods(self) -> None:
        """
        Validate kernel support and conditionally add derivative methods.

        This method checks if the kernel and mean function support the required
        derivative operations (jacobian_wrt_params, hvp_wrt_params) and
        dynamically adds jacobian() and hvp() methods to the loss instance.

        This allows optimizers to check if loss.jacobian or loss.hvp exist
        to determine what derivative information is available.

        Raises
        ------
        ValueError
            If kernel does not support jacobian_wrt_params (required for gradients).
        """
        kernel = self._gp._kernel
        mean = self._gp._mean

        # Check kernel support for Jacobian (required for gradient-based optimization)
        has_kernel_jacobian = hasattr(kernel, 'jacobian_wrt_params')
        has_mean_jacobian = hasattr(mean, 'jacobian_wrt_params')

        if not has_kernel_jacobian:
            raise ValueError(
                f"Kernel {type(kernel).__name__} does not implement "
                "jacobian_wrt_params(). Gradient-based optimization requires "
                "this method. Please implement it or use a gradient-free optimizer."
            )

        # Kernel Jacobian is required, mean Jacobian is optional
        # If mean has no parameters, it doesn't need jacobian_wrt_params

        # Check kernel support for HVP (optional, enables second-order methods)
        has_kernel_hvp = hasattr(kernel, 'hvp_wrt_params')
        has_mean_hvp = hasattr(mean, 'hvp_wrt_params')

        # Dynamically add jacobian method (always available if we get here)
        self.jacobian = self._jacobian

        # Dynamically add hvp method only if kernel supports it
        if has_kernel_hvp:
            self.hvp = self._hvp
            self._supports_hvp = True
        else:
            self._supports_hvp = False

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

    def _jacobian(self, params: Array) -> Array:
        """
        Compute gradient of NLML w.r.t. hyperparameters.

        Uses the formula:
            ∂(-log p)/∂θ_i = 0.5 * trace[(α α^T - K^{-1}) ∂K/∂θ_i]

        where α = K^{-1}(y - m(X)).

        This is a private method. The public jacobian() method is dynamically
        added during __init__ if the kernel supports jacobian_wrt_params().

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

        # Build gradient array directly without lists
        grad_parts = []

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

            # Vectorized computation for all kernel parameters at once
            alpha_vec = alpha[:, 0:1]  # Shape: (n_train, 1)

            # term1[i] = α^T @ K_grad[:,:,i] @ α for all i
            # Using einsum: sum over both n_train dimensions
            term1 = self._bkd.einsum('ji,ijk,jk->k',
                                      alpha_vec.T, K_grad, alpha_vec)

            # term2[i] = trace(K^{-1} @ K_grad[:,:,i]) for all i
            # Batch solve: K^{-1} @ K_grad
            K_grad_flat = self._bkd.reshape(K_grad, (n_train, n_train * n_kernel_params))
            Kinv_K_grad_flat = cholesky.solve(K_grad_flat)
            Kinv_K_grad = self._bkd.reshape(Kinv_K_grad_flat,
                                             (n_train, n_train, n_kernel_params))
            # Compute trace for each parameter: sum of diagonal elements
            term2 = self._bkd.einsum('iik->k', Kinv_K_grad)

            # Gradient for all kernel parameters
            # ∂(-log p)/∂θ = 0.5 * (trace(K^{-1} @ dK) - α^T @ dK @ α)
            grad_kernel = 0.5 * (term2 - term1)
            grad_parts.append(grad_kernel)

        # 2. Compute gradients w.r.t. mean function hyperparameters
        if mean_hyps.nparams() > 0:
            # For mean functions, the gradient is:
            # ∂(-log p)/∂θ_m = -α^T @ ∂m/∂θ_m
            #
            # where α = K^{-1}(y - m(X))

            # Get mean function Jacobian: shape (nparams_mean, n_train, 1)
            mean_jac = self._gp._mean.jacobian_wrt_params(self._X_train)

            # Vectorized gradient for all mean parameters at once
            # grad[i] = -α^T @ mean_jac[i,:,:]
            # Using einsum to sum over n_train and nqoi dimensions
            grad_mean = -self._bkd.einsum('jk,ijk->i', alpha, mean_jac)
            grad_parts.append(grad_mean)

        # Concatenate all gradient parts and reshape to (1, nactive)
        if len(grad_parts) == 0:
            grad = self._bkd.zeros((1, 0))
        elif len(grad_parts) == 1:
            grad = self._bkd.reshape(grad_parts[0], (1, -1))
        else:
            grad = self._bkd.concatenate(grad_parts, axis=0)
            grad = self._bkd.reshape(grad, (1, -1))

        # Note: The kernel's jacobian_wrt_params should already account for
        # parameter transformations (e.g., log-space), so no additional
        # chain rule application is needed here

        return grad

    def _hvp(self, params: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product of NLML w.r.t. hyperparameters using adjoint method.

        Ultra-fast vectorized implementation using the adjoint method that computes
        H·v where H is the Hessian of the negative log marginal likelihood with
        respect to hyperparameters θ, and v is a direction vector.

        **Adjoint Method Formula**:
            For the kernel log-determinant term: φ(θ) = log|K(θ)|
            [∇²φ(θ)·V]_i = -tr(K^{-1} ∂K/∂θ_i K^{-1} D_V)

        where D_V = Σ_j (∂K/∂θ_j)·v_j is the directional derivative of K.

        This avoids computing individual Hessian elements H_ij and instead computes
        the HVP directly in O(pn²) time vs O(p²n²) for naive methods.

        For the full NLML, we include contributions from both log|K| and data fit terms.

        This is a private method. The public hvp() method is dynamically
        added during __init__ if the kernel supports hvp_wrt_params().

        Parameters
        ----------
        params : Array
            Active hyperparameters in optimization space.
            Can be 1D array (nactive,) or 2D array (nactive, 1).
        direction : Array
            Direction vector for HVP.
            Can be 1D array (nactive,) or 2D array (nactive, 1).

        Returns
        -------
        hvp : Array
            Hessian-vector product, shape (1, nactive).

        Notes
        -----
        **Adjoint Method Advantages**:
        - No nested loops over parameter pairs: O(pn²) vs O(p²n²)
        - Single directional derivative D_V computed once
        - Reuses Cholesky factorization for all solves
        - 10-100× faster than element-wise Hessian computation

        **Vectorization Strategy**:
        1. Compute directional derivative D_V = Σ_j (∂K/∂θ_j)·v_j using einsum
        2. Solve: W = K^{-1} D_V (single Cholesky solve)
        3. Compute: W_kinv = W @ K^{-1} = K^{-1} D_V K^{-1} (reuse Cholesky)
        4. For each i: hvp_i = -tr(∂K/∂θ_i @ W_kinv) (vectorized trace)

        Total: O(n³) Cholesky + O(pn²) gradient computation + O(n³) solve + O(pn²) traces
        """
        # Ensure params and direction are 1D
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]
        if len(direction.shape) == 2 and direction.shape[1] == 1:
            direction = direction[:, 0]

        nactive = self._hyp_list.nactive_params()
        if direction.shape[0] != nactive:
            raise ValueError(
                f"Direction vector has {direction.shape[0]} elements, "
                f"expected {nactive}"
            )

        # Update and refit
        self._hyp_list.set_active_values(params)
        self._gp.fit(self._X_train, self._y_train)

        # Get alpha and cholesky from GP
        alpha = self._gp._alpha  # Shape: (n_train, nqoi)
        alpha_vec = alpha[:, 0:1]  # Shape: (n_train, 1) for single output
        cholesky = self._gp._cholesky
        n_train = self._X_train.shape[1]

        # Get hyperparameter lists
        kernel_hyps = self._gp._kernel.hyp_list()
        mean_hyps = self._gp._mean.hyp_list()
        n_kernel_params = kernel_hyps.nparams()
        n_mean_params = mean_hyps.nparams()

        # Build HVP array directly without lists
        hvp_parts = []

        # === PART 1: Kernel hyperparameters ===
        # Use TRUE ADJOINT METHOD for O(1) solves instead of O(p) solves
        if n_kernel_params > 0:
            kernel = self._gp._kernel

            # Get all kernel gradients at once: shape (n_train, n_train, nparams)
            K_grad = kernel.jacobian_wrt_params(self._X_train)

            # Compute directional derivative of K: D_V = Σ_j v_j * ∂K/∂θ_j
            # Using einsum for vectorization: D_V[i,j] = Σ_k K_grad[i,j,k] * direction[k]
            D_V = self._bkd.einsum('ijk,k->ij', K_grad, direction[:n_kernel_params])

            # Solve: K^{-1} D_V (single solve!)
            Kinv_DV = cholesky.solve(D_V)

            # Compute HVP directly without forming full Hessian
            # dDV[i,j,param_i] = Σ_j (∂²K/∂θ_i∂θ_j) * v[j]
            if hasattr(kernel, 'hvp_wrt_params'):
                dDV = kernel.hvp_wrt_params(self._X_train, direction[:n_kernel_params])
            else:
                raise NotImplementedError(
                    f"Kernel {type(kernel).__name__} must implement "
                    "hvp_wrt_params() for HVP computation"
                )
            # Shape: (n_train, n_train, n_kernel_params)

            # Solve: K^{-1} ∂D_V/∂θ_i for ALL i at once (single batched solve!)
            # We need to solve K X[:,:,i] = dDV[:,:,i] for each i
            # Reshape to (n_train, n_train * n_kernel_params), solve, reshape back
            dDV_flat = self._bkd.reshape(dDV, (n_train, n_train * n_kernel_params))
            Kinv_dDV_flat = cholesky.solve(dDV_flat)
            Kinv_dDV = self._bkd.reshape(Kinv_dDV_flat, (n_train, n_train, n_kernel_params))

            # Precompute shared quantities
            Kinv_DV_alpha = Kinv_DV @ alpha_vec  # Shape: (n_train, 1)

            # Solve K X = K_grad for all parameters (batched)
            K_grad_flat = self._bkd.reshape(K_grad, (n_train, n_train * n_kernel_params))
            Kinv_K_grad_flat = cholesky.solve(K_grad_flat)
            Kinv_K_grad = self._bkd.reshape(Kinv_K_grad_flat, (n_train, n_train, n_kernel_params))

            # ADJOINT METHOD: Compute ALL HVP components at once using einsum
            #
            # [H·v]_i = 0.5 * [-tr(K^{-1} ∂K/∂θ_i K^{-1} D_V) + tr(K^{-1} ∂D_V/∂θ_i)]
            #          + 0.5 * [2 α^T K^{-1} ∂K/∂θ_i K^{-1} D_V α - α^T ∂D_V/∂θ_i α]

            # For term1, we need K^{-1} ∂K/∂θ_i K^{-1} for all i
            # We solve K X = (K^{-1} ∂K/∂θ_i)^T for all i using batched solve
            # Trick: Transpose each slice by swapping dimensions, then solve all at once
            # Kinv_K_grad has shape (n, n, p), we need to transpose to (n, n, p) with [i,j] -> [j,i]
            # This is equivalent to swapping first two dimensions
            Kinv_K_grad_T = self._bkd.transpose(Kinv_K_grad, (1, 0, 2))  # Shape: (n, n, p)
            # Flatten to (n, n*p) for batched solve
            Kinv_K_grad_T_flat = self._bkd.reshape(Kinv_K_grad_T, (n_train, n_train * n_kernel_params))
            # Solve K X = Kinv_K_grad^T for all parameters at once
            Kinv_K_grad_Kinv_T_flat = cholesky.solve(Kinv_K_grad_T_flat)
            # Reshape back to (n, n, p)
            Kinv_K_grad_Kinv_T = self._bkd.reshape(Kinv_K_grad_Kinv_T_flat,
                                                    (n_train, n_train, n_kernel_params))
            # Transpose back to get K^{-1} ∂K/∂θ_i K^{-1}
            Kinv_K_grad_Kinv = self._bkd.transpose(Kinv_K_grad_Kinv_T, (1, 0, 2))

            # term1[i] = -0.5 * tr(D_V @ Kinv_K_grad_Kinv[:,:,i])
            #          = -0.5 * sum_{jk} D_V[j,k] * Kinv_K_grad_Kinv[j,k,i]
            term1 = -0.5 * self._bkd.einsum('ij,ijk->k', D_V, Kinv_K_grad_Kinv)

            # term2[i] = 0.5 * tr(Kinv_dDV[:,:,i])
            #          = 0.5 * sum_j Kinv_dDV[j,j,i]
            term2 = 0.5 * self._bkd.einsum('iik->k', Kinv_dDV)

            # term3[i] = α^T ∂K/∂θ_i K^{-1} D_V α
            #          = sum_{jk} alpha[j] * K_grad[j,k,i] * (Kinv_DV_alpha)[k]
            term3 = self._bkd.einsum('j,jki,k->i',
                                      alpha_vec[:, 0], K_grad, Kinv_DV_alpha[:, 0])

            # term4[i] = -0.5 * α^T ∂D_V/∂θ_i α
            #          = -0.5 * sum_{jk} alpha[j] * dDV[j,k,i] * alpha[k]
            term4 = -0.5 * self._bkd.einsum('j,jki,k->i',
                                             alpha_vec[:, 0], dDV, alpha_vec[:, 0])

            # Total HVP for kernel parameters (vectorized!)
            hvp_kernel = term1 + term2 + term3 + term4
            hvp_parts.append(hvp_kernel)

        # === PART 2: Mean function hyperparameters ===
        if n_mean_params > 0:
            # For mean functions, the Hessian is:
            # ∂²L/∂θ_m_i∂θ_m_j = ∂m_i^T K^{-1} ∂m_j
            #
            # HVP using adjoint: [H·v]_i = Σ_j (∂m_i^T K^{-1} ∂m_j)·v_j
            #                              = ∂m_i^T K^{-1} (Σ_j ∂m_j·v_j)
            #                              = ∂m_i^T K^{-1} D_V_mean

            mean_jac = self._gp._mean.jacobian_wrt_params(self._X_train)
            # Shape: (nparams_mean, n_train, 1)

            # Compute directional derivative for mean
            D_V_mean = self._bkd.einsum('ijk,i->jk',
                                         mean_jac,
                                         direction[n_kernel_params:])
            # Shape: (n_train, 1)

            # Solve: K^{-1} D_V_mean
            Kinv_DV_mean = cholesky.solve(D_V_mean)

            # Vectorized: hvp[i] = ∂m_i^T @ Kinv_DV_mean for all i
            # mean_jac shape: (nparams_mean, n_train, 1)
            # Kinv_DV_mean shape: (n_train, 1)
            # Result: sum over n_train and 1 dimensions
            hvp_mean = self._bkd.einsum('ijk,jk->i', mean_jac, Kinv_DV_mean)
            hvp_parts.append(hvp_mean)

        # Concatenate all HVP parts and reshape to (1, nactive)
        if len(hvp_parts) == 0:
            hvp = self._bkd.zeros((1, 0))
        elif len(hvp_parts) == 1:
            hvp = self._bkd.reshape(hvp_parts[0], (1, -1))
        else:
            hvp = self._bkd.concatenate(hvp_parts, axis=0)
            hvp = self._bkd.reshape(hvp, (1, -1))

        # Note: The kernel's hessian_wrt_params should already account for
        # parameter transformations (e.g., log-space), so no additional
        # chain rule application is needed here

        return hvp

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NegativeLogMarginalLikelihoodLoss("
            f"nvars={self.nvars()}, "
            f"n_train={self._X_train.shape[1]})"
        )
