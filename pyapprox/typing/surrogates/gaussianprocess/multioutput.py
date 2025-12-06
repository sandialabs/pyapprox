"""
Multi-output Gaussian Process implementations.

This module provides GP classes that handle multiple outputs using
multi-output kernels (IndependentMultiOutputKernel or LinearCoregionalizationKernel).
"""

from typing import List, Tuple, Union, Generic
import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.linalg.cholesky_factor import CholeskyFactor
from pyapprox.typing.surrogates.kernels.multioutput import (
    IndependentMultiOutputKernel,
    LinearCoregionalizationKernel,
)


class MultiOutputGP(Generic[Array]):
    """
    Gaussian Process for multi-output prediction.

    Uses multi-output kernels (IndependentMultiOutputKernel or
    LinearCoregionalizationKernel) to model multiple outputs with a single GP.

    Mathematical Model
    ------------------
    For multi-output GP with M outputs:
        y_stacked = [y_0; y_1; ...; y_{M-1}]
        K = multi_output_kernel(X_list, X_list)

    Posterior mean:
        μ*(X*) = K(X*, X) @ (K + σ²I)^{-1} @ y

    Posterior variance:
        Var*(X*) = K(X*, X*) - K(X*, X) @ (K + σ²I)^{-1} @ K(X*, X)^T

    Parameters
    ----------
    kernel : IndependentMultiOutputKernel or LinearCoregionalizationKernel
        Multi-output kernel for covariance computation.
    nugget : float, optional
        Small value added to diagonal for numerical stability during
        matrix inversion. Default: 1e-6.

    Attributes
    ----------
    _kernel : MultiOutputKernel
        The multi-output kernel.
    _nugget : float
        Nugget for numerical conditioning.
    _bkd : Backend
        Backend for numerical operations.
    _is_fitted : bool
        Whether the GP has been fitted to data.
    _X_train_list : List[Array]
        Training inputs for each output (set during fit).
    _y_train_stacked : Array
        Stacked training outputs (set during fit).
    _cholesky : CholeskyFactor
        Cholesky factor of K + σ²I (set during fit).
    _alpha : Array
        Precomputed (K + σ²I)^{-1} @ y (set during fit).

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels import MaternKernel
    >>> from pyapprox.typing.surrogates.kernels.multioutput import IndependentMultiOutputKernel
    >>> from pyapprox.typing.surrogates.gaussianprocess.multioutput import MultiOutputGP
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> # Create kernels for 2 outputs
    >>> k1 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    >>> k2 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    >>> mo_kernel = IndependentMultiOutputKernel([k1, k2])
    >>> # Create GP
    >>> gp = MultiOutputGP(mo_kernel, nugget=1e-6)
    >>> # Fit to data
    >>> X_train = bkd.array(np.random.randn(1, 10))
    >>> y1 = bkd.sin(X_train[0, :])
    >>> y2 = bkd.cos(X_train[0, :])
    >>> y_stacked = bkd.reshape(bkd.concatenate([y1, y2]), (-1, 1))
    >>> gp.fit([X_train, X_train], y_stacked)
    >>> # Predict
    >>> X_test = bkd.array(np.random.randn(1, 5))
    >>> mean = gp.predict([X_test, X_test])
    >>> mean, std = gp.predict_with_uncertainty([X_test, X_test])
    """

    def __init__(
        self,
        kernel: Union[IndependentMultiOutputKernel, LinearCoregionalizationKernel],
        nugget: float = 1e-6,
    ):
        """
        Initialize the MultiOutputGP.

        Parameters
        ----------
        kernel : IndependentMultiOutputKernel or LinearCoregionalizationKernel
            Multi-output kernel.
        nugget : float, optional
            Small value added to diagonal for numerical stability. Default: 1e-6.
        """
        self._kernel = kernel
        self._nugget = nugget
        self._bkd = kernel.bkd()
        self._is_fitted = False

    def bkd(self) -> Backend[Array]:
        """
        Return the backend.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical operations.
        """
        return self._bkd

    def kernel(self) -> Union[IndependentMultiOutputKernel, LinearCoregionalizationKernel]:
        """
        Return the multi-output kernel.

        Returns
        -------
        kernel : IndependentMultiOutputKernel or LinearCoregionalizationKernel
            The multi-output kernel.
        """
        return self._kernel

    def hyp_list(self):
        """
        Return the hyperparameter list from the kernel.

        Returns
        -------
        HyperParameterList
            The hyperparameter list from the multi-output kernel.
        """
        return self._kernel.hyp_list()

    def is_fitted(self) -> bool:
        """
        Check if GP has been fitted.

        Returns
        -------
        is_fitted : bool
            True if fit() has been called.
        """
        return self._is_fitted

    def fit(self, X_train_list: List[Array], y_train_stacked: Array) -> None:
        """
        Fit the multi-output GP to training data.

        Parameters
        ----------
        X_train_list : List[Array]
            Training inputs for each output. Each array has shape (nvars, n_i).
            For most cases, all outputs use same X: X_train_list = [X] * noutputs.
        y_train_stacked : Array
            Stacked training outputs, shape (sum(n_i), 1).
            Format: [y_0; y_1; ...; y_{M-1}] where y_i are outputs.

        Raises
        ------
        ValueError
            If lengths don't match or shapes are invalid.
        """
        if len(X_train_list) != self._kernel.noutputs():
            raise ValueError(
                f"X_train_list length ({len(X_train_list)}) must match "
                f"number of outputs ({self._kernel.noutputs()})"
            )

        # Validate stacked output shape
        n_total = sum(X.shape[1] for X in X_train_list)
        if y_train_stacked.shape[0] != n_total:
            raise ValueError(
                f"y_train_stacked has {y_train_stacked.shape[0]} rows, "
                f"expected {n_total} (sum of samples across outputs)"
            )
        if y_train_stacked.shape[1] != 1:
            raise ValueError(
                f"y_train_stacked must have shape (n_total, 1), "
                f"got {y_train_stacked.shape}"
            )

        self._X_train_list = X_train_list
        self._y_train_stacked = y_train_stacked

        # Build kernel matrix
        K = self._kernel(X_train_list, block_format=False)
        K_noisy = K + self._bkd.eye(K.shape[0]) * self._nugget

        # Cholesky factorization
        L = self._bkd.cholesky(K_noisy)
        self._cholesky = CholeskyFactor(L, self._bkd)

        # Precompute alpha = (K + σ²I)^{-1} @ y
        self._alpha = self._cholesky.solve(y_train_stacked)

        self._is_fitted = True

    def neg_log_marginal_likelihood(self) -> float:
        """
        Compute negative log marginal likelihood.

        For multi-output GP with stacked outputs:
            -log p(y|X,θ) = 0.5 * [y^T (K + σ²I)^{-1} y + log|K + σ²I| + n log(2π)]

        where y is the stacked output vector and K is the block kernel matrix.

        Returns
        -------
        nll : float
            Negative log marginal likelihood.

        Raises
        ------
        RuntimeError
            If GP has not been fitted yet.

        Notes
        -----
        This value is used for hyperparameter optimization. Lower values indicate
        better model fit, balancing data fit (y^T K^{-1} y) with model complexity
        (log|K|).
        """
        if not self._is_fitted:
            raise RuntimeError("GP must be fitted before computing NLL")

        # Data fit term: y^T (K + σ²I)^{-1} y
        # We have alpha = (K + σ²I)^{-1} y, so this is y^T alpha
        data_fit = self._bkd.sum(self._y_train_stacked * self._alpha)

        # Complexity penalty: log|K + σ²I|
        log_det = self._cholesky.log_determinant()

        # Constant term: n * log(2π)
        n_total = self._y_train_stacked.shape[0]
        constant = n_total * np.log(2.0 * np.pi)

        # Negative log marginal likelihood
        nll = 0.5 * (data_fit + log_det + constant)

        return float(nll)

    def predict(self, X_test_list: List[Array]) -> Array:
        """
        Predict mean at test points.

        Parameters
        ----------
        X_test_list : List[Array]
            Test inputs for each output. Each has shape (nvars, n_test_i).
            Can be different test points for each output.

        Returns
        -------
        y_pred : Array
            Stacked predictions, shape (sum(n_test_i), 1).

        Raises
        ------
        ValueError
            If GP not fitted or X_test_list length doesn't match noutputs.
        """
        if not self._is_fitted:
            raise ValueError("GP must be fitted before prediction")

        if len(X_test_list) != self._kernel.noutputs():
            raise ValueError(
                f"X_test_list length ({len(X_test_list)}) must match "
                f"number of outputs ({self._kernel.noutputs()})"
            )

        # Cross-covariance: K(X_test, X_train)
        K_star = self._kernel(X_test_list, self._X_train_list, block_format=False)

        # Mean prediction: μ* = K_star @ alpha
        y_pred = K_star @ self._alpha

        return y_pred

    def predict_with_uncertainty(
        self, X_test_list: List[Array]
    ) -> Tuple[Array, Array]:
        """
        Predict mean and standard deviation.

        Parameters
        ----------
        X_test_list : List[Array]
            Test inputs for each output. Each has shape (nvars, n_test_i).

        Returns
        -------
        mean : Array
            Mean predictions, shape (sum(n_test_i), 1).
        std : Array
            Standard deviations, shape (sum(n_test_i), 1).

        Raises
        ------
        ValueError
            If GP not fitted or X_test_list length doesn't match noutputs.
        """
        if not self._is_fitted:
            raise ValueError("GP must be fitted before prediction")

        if len(X_test_list) != self._kernel.noutputs():
            raise ValueError(
                f"X_test_list length ({len(X_test_list)}) must match "
                f"number of outputs ({self._kernel.noutputs()})"
            )

        # Mean prediction
        mean = self.predict(X_test_list)

        # Cross-covariance for variance computation
        K_star = self._kernel(X_test_list, self._X_train_list, block_format=False)

        # Prior variance (diagonal of K(X_test, X_test))
        # Use block format to efficiently get diagonals
        K_star_star_blocks = self._kernel(X_test_list, block_format=True)

        # Extract diagonal from each diagonal block
        diag_blocks = []
        for i in range(len(X_test_list)):
            block_ii = K_star_star_blocks[i][i]
            # Diagonal of this block
            diag_i = self._bkd.diag(block_ii)
            diag_blocks.append(diag_i)

        # Stack all diagonals
        K_star_star_diag = self._bkd.concatenate(diag_blocks, axis=0)

        # Solve: v = L^{-1} @ K_star^T
        v = self._bkd.solve_triangular(
            self._cholesky.factor(), K_star.T, lower=True
        )

        # Posterior variance: σ²* = K** - v^T @ v
        # v^T @ v is computed as sum of v * v along rows
        var = K_star_star_diag - self._bkd.sum(v * v, axis=0)

        # Numerical stability: clip small negative values to zero
        # (arise from numerical errors in Cholesky solve)
        var = var * (var >= 0.0)

        # Standard deviation
        std = self._bkd.sqrt(var)[:, None]

        return mean, std

    def predict_covariance(
        self, X_test_list: List[Array]
    ) -> Tuple[Array, Array]:
        """
        Predict mean and full covariance matrix.

        Parameters
        ----------
        X_test_list : List[Array]
            Test inputs for each output. Each has shape (nvars, n_test_i).

        Returns
        -------
        mean : Array
            Mean predictions, shape (sum(n_test_i), 1).
        cov : Array
            Covariance matrix, shape (sum(n_test_i), sum(n_test_i)).

        Raises
        ------
        ValueError
            If GP not fitted or X_test_list length doesn't match noutputs.

        Notes
        -----
        This computes the full predictive covariance matrix, which can be
        expensive for large test sets. Use predict_with_uncertainty() for
        just the diagonal (variance).
        """
        if not self._is_fitted:
            raise ValueError("GP must be fitted before prediction")

        if len(X_test_list) != self._kernel.noutputs():
            raise ValueError(
                f"X_test_list length ({len(X_test_list)}) must match "
                f"number of outputs ({self._kernel.noutputs()})"
            )

        # Mean prediction
        mean = self.predict(X_test_list)

        # Cross-covariance
        K_star = self._kernel(X_test_list, self._X_train_list, block_format=False)

        # Prior covariance
        K_star_star = self._kernel(X_test_list, block_format=False)

        # Solve: v = L^{-1} @ K_star^T
        v = self._bkd.solve_triangular(
            self._cholesky.factor(), K_star.T, lower=True
        )

        # Posterior covariance: Σ* = K** - v^T @ v
        cov = K_star_star - v.T @ v

        return mean, cov

    def optimize_hyperparameters(self, optimizer=None) -> None:
        """
        Optimize hyperparameters by minimizing negative log marginal likelihood.

        Uses typing.optimization.minimize optimizers with analytical gradients
        from the multi-output kernel's jacobian_wrt_params() method.

        Parameters
        ----------
        optimizer : OptimizerProtocol, optional
            Optimizer from typing.optimization.minimize.
            If None, uses ScipyTrustConstrOptimizer with default settings.

        Raises
        ------
        RuntimeError
            If GP has not been fitted yet.

        Examples
        --------
        >>> from pyapprox.typing.surrogates.kernels.matern import MaternKernel
        >>> from pyapprox.typing.surrogates.kernels.multioutput import IndependentMultiOutputKernel
        >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
        >>> import numpy as np
        >>>
        >>> bkd = NumpyBkd()
        >>> k1 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
        >>> k2 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
        >>> mo_kernel = IndependentMultiOutputKernel([k1, k2])
        >>> gp = MultiOutputGP(mo_kernel)
        >>>
        >>> X_train = bkd.array(np.random.randn(1, 20))
        >>> y_stacked = bkd.array(np.random.randn(40, 1))
        >>> gp.fit([X_train, X_train], y_stacked)
        >>>
        >>> # Optimize hyperparameters
        >>> gp.optimize_hyperparameters()
        """
        if not self._is_fitted:
            raise RuntimeError("GP must be fitted before optimization")

        from pyapprox.typing.surrogates.gaussianprocess.gp_loss import (
            GPNegativeLogMarginalLikelihoodLoss
        )
        from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer
        )

        # Create loss function
        loss = GPNegativeLogMarginalLikelihoodLoss(
            self, (self._X_train_list, self._y_train_stacked)
        )

        # Get bounds for active hyperparameters
        bounds_tuple = self.hyp_list().get_bounds()
        bounds = self._bkd.array([
            [lb, ub] for lb, ub in bounds_tuple
        ])

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = ScipyTrustConstrOptimizer(
                objective=loss,
                bounds=bounds,
                verbosity=0
            )

        # Run optimization
        init_guess = self.hyp_list().get_active_values()

        # Reshape init_guess to (n, 1) if it's 1D
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(init_guess, (len(init_guess), 1))

        result = optimizer.minimize(init_guess)

        # Update hyperparameters with optimal values
        # optima() returns shape (n, 1), flatten to 1D for set_active_values
        optimal_params = result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]

        self.hyp_list().set_active_values(optimal_params)
        self.fit(self._X_train_list, self._y_train_stacked)

    def __repr__(self) -> str:
        """
        String representation.

        Returns
        -------
        repr : str
            String representation.
        """
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"MultiOutputGP(\n"
            f"  kernel={repr(self._kernel)},\n"
            f"  nugget={self._nugget},\n"
            f"  status={fitted_str}\n"
            f")"
        )
