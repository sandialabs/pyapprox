"""
Multi-output Gaussian Process implementations.

This module provides GP classes that handle multiple outputs using
multi-output kernels (IndependentMultiOutputKernel or LinearCoregionalizationKernel).
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Generic, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from pyapprox.surrogates.gaussianprocess.gp_loss import (
        GPNegativeLogMarginalLikelihoodLoss,
    )

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.surrogates.gaussianprocess.multioutput_data import (
    MultiOutputGPTrainingData,
)
from pyapprox.surrogates.kernels.multioutput import (
    DAGMultiOutputKernel,
    IndependentMultiOutputKernel,
    LinearCoregionalizationKernel,
    MultiLevelKernel,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList
from pyapprox.util.linalg.cholesky_factor import CholeskyFactor


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
    >>> from pyapprox.surrogates.kernels import MaternKernel
    >>> from pyapprox.surrogates.kernels.multioutput import IndependentMultiOutputKernel
    >>> from pyapprox.surrogates.gaussianprocess.multioutput import MultiOutputGP
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> # Create kernels for 2 outputs
    >>> k1 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    >>> k2 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    >>> mo_kernel = IndependentMultiOutputKernel([k1, k2])
    >>> # Create GP
    >>> gp = MultiOutputGP(mo_kernel, nugget=1e-6)
    >>> # Fit to data using list format (preferred)
    >>> X_train = bkd.array(np.random.randn(1, 10))
    >>> y1 = bkd.reshape(bkd.sin(X_train[0, :]), (1, -1))  # Shape: (1, 10)
    >>> y2 = bkd.reshape(bkd.cos(X_train[0, :]), (1, -1))  # Shape: (1, 10)
    >>> gp.fit([X_train, X_train], [y1, y2])  # Fits data AND optimizes hyperparameters
    >>> # Predict
    >>> X_test = bkd.array(np.random.randn(1, 5))
    >>> mean_list = gp.predict([X_test, X_test])  # Returns list of (1, n_test) arrays
    >>> mean_list, std_list = gp.predict_with_uncertainty([X_test, X_test])

    Hyperparameter optimization is controlled via:

    1. **Default behavior**: Uses ScipyTrustConstrOptimizer with maxiter=1000
    2. **Custom optimizer**: Call ``set_optimizer(optimizer)`` before ``fit()``
    3. **Skip optimization**: Set all hyperparameters inactive via
       ``gp.hyp_list().set_all_inactive()`` before ``fit()``
    """

    def __init__(
        self,
        kernel: Union[
            IndependentMultiOutputKernel[Array],
            LinearCoregionalizationKernel[Array],
            MultiLevelKernel[Array],
            DAGMultiOutputKernel[Array],
        ],
        nugget: float = 1e-6,
    ):
        """
        Initialize the MultiOutputGP.

        Parameters
        ----------
        kernel : IndependentMultiOutputKernel, LinearCoregionalizationKernel,
                 MultiLevelKernel, or DAGMultiOutputKernel
            Multi-output kernel.
        nugget : float, optional
            Small value added to diagonal for numerical stability. Default: 1e-6.
        """
        self._kernel = kernel
        self._nugget = nugget
        self._bkd = kernel.bkd()
        self._is_fitted = False
        self._data: Optional[MultiOutputGPTrainingData[Array]] = None

        # Optimizer for hyperparameter tuning (None means use default)
        self._optimizer: Optional[BindableOptimizerProtocol[Array]] = None

    def _clone_unfitted(self) -> "MultiOutputGP[Array]":
        """Return a deep copy of this GP with fitted state cleared.

        Returns
        -------
        MultiOutputGP[Array]
            An unfitted copy with the same configuration.
        """
        clone = copy.deepcopy(self)
        clone._data = None
        clone._is_fitted = False
        clone._cholesky = None
        clone._alpha = None
        if hasattr(clone, "_X_train_list"):
            clone._X_train_list = None
        if hasattr(clone, "_y_train_stacked"):
            clone._y_train_stacked = None
        return clone

    def _copy_fitted_state_from(self, other: "MultiOutputGP[Array]") -> None:
        """Copy all fitted state from another GP into self.

        Parameters
        ----------
        other : MultiOutputGP[Array]
            The source GP to copy fitted state from.
        """
        self._data = other._data
        self._is_fitted = other._is_fitted
        self._cholesky = other._cholesky
        self._alpha = other._alpha
        self._X_train_list = other._X_train_list
        self._y_train_stacked = other._y_train_stacked
        # Copy optimized hyperparameters
        self._kernel.hyp_list().set_values(other._kernel.hyp_list().get_values())

    def bkd(self) -> Backend[Array]:
        """
        Return the backend.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical operations.
        """
        return self._bkd

    def kernel(
        self,
    ) -> Union[IndependentMultiOutputKernel[Array], LinearCoregionalizationKernel[Array]]:
        """
        Return the multi-output kernel.

        Returns
        -------
        kernel : IndependentMultiOutputKernel or LinearCoregionalizationKernel
            The multi-output kernel.
        """
        return self._kernel

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return the hyperparameter list from the kernel.

        Returns
        -------
        HyperParameterList
            The hyperparameter list from the multi-output kernel.
        """
        return self._kernel.hyp_list()

    def set_optimizer(self, optimizer: BindableOptimizerProtocol[Array]) -> None:
        """Set the optimizer for hyperparameter optimization during fit().

        .. deprecated::
            Pass optimizer to ``MultiOutputGPMaximumLikelihoodFitter``
            constructor instead.

        Parameters
        ----------
        optimizer : BindableOptimizerProtocol[Array]
            An optimizer configured with options but NOT bound to an objective.
            During fit(), the optimizer will be cloned and bound to the
            negative log marginal likelihood loss function.

        Examples
        --------
        >>> from pyapprox.optimization.minimize.scipy.trust_constr import (
        ...     ScipyTrustConstrOptimizer
        ... )
        >>> optimizer = ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-8)
        >>> gp.set_optimizer(optimizer)
        >>> gp.fit(X_train_list, y_train_list)
        """
        if not isinstance(optimizer, BindableOptimizerProtocol):
            raise TypeError(
                f"optimizer must satisfy BindableOptimizerProtocol, "
                f"got {type(optimizer).__name__}"
            )
        self._optimizer = optimizer

    def optimizer(self) -> Optional[BindableOptimizerProtocol[Array]]:
        """Return the current optimizer (None means use default).

        Returns
        -------
        Optional[BindableOptimizerProtocol[Array]]
            The configured optimizer, or None if using the default
            ScipyTrustConstrOptimizer.
        """
        return self._optimizer

    def is_fitted(self) -> bool:
        """
        Check if GP has been fitted.

        Returns
        -------
        is_fitted : bool
            True if fit() has been called.
        """
        return self._is_fitted

    def cholesky(self) -> CholeskyFactor[Array]:
        """
        Return the Cholesky factor of the kernel matrix.

        Returns
        -------
        CholeskyFactor[Array]
            Cholesky factor of K + nugget*I.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("GP must be fitted before accessing cholesky.")
        return self._cholesky

    def alpha(self) -> Array:
        """
        Return the precomputed weights alpha = (K + nugget*I)^{-1} y.

        Returns
        -------
        Array
            Precomputed weights, shape (n_total, 1).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("GP must be fitted before accessing alpha.")
        return self._alpha

    def data(self) -> MultiOutputGPTrainingData[Array]:
        """
        Return the training data container.

        Returns
        -------
        MultiOutputGPTrainingData[Array]
            Training data (X_list, y_list) used to fit the GP.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        if self._data is None:
            raise RuntimeError("GP must be fitted before accessing data.")
        return self._data

    def fit(
        self, X_train_list: List[Array], y_train: Union[List[Array], Array]
    ) -> None:
        """Fit GP to data and optimize active hyperparameters.

        This is a convenience method that delegates to
        ``MultiOutputGPMaximumLikelihoodFitter``. For cleaner separation of
        concerns, prefer using the fitter directly::

            from pyapprox.surrogates.gaussianprocess.fitters import (
                MultiOutputGPMaximumLikelihoodFitter,
            )
            fitter = MultiOutputGPMaximumLikelihoodFitter(bkd, optimizer=...)
            result = fitter.fit(gp, X_train_list, y_train)
            fitted_gp = result.surrogate()

        .. deprecated::
            Use ``MultiOutputGPMaximumLikelihoodFitter`` or
            ``MultiOutputGPFixedHyperparameterFitter`` directly.

        Parameters
        ----------
        X_train_list : List[Array]
            Training inputs for each output. Each array has shape (nvars, n_i).
            For most cases, all outputs use same X: X_train_list = [X] * noutputs.
        y_train : Union[List[Array], Array]
            Training outputs. Can be provided in two formats:
            - List format (preferred): List of arrays, each with shape (1, n_i).
              This follows the standard convention where values are (nqoi, n_samples).
            - Stacked format (legacy): Single array with shape (sum(n_i), 1).
              Format: [y_0; y_1; ...; y_{M-1}] where y_i are outputs.

        Raises
        ------
        ValueError
            If lengths don't match or shapes are invalid.
        RuntimeError
            If Cholesky factorization fails (matrix not positive definite).

        Examples
        --------
        >>> # Standard fit with optimization
        >>> gp.fit(X_train_list, y_train_list)

        >>> # Custom optimizer
        >>> from pyapprox.optimization.minimize.scipy.trust_constr import (
        ...     ScipyTrustConstrOptimizer
        ... )
        >>> gp.set_optimizer(ScipyTrustConstrOptimizer(maxiter=500))
        >>> gp.fit(X_train_list, y_train_list)

        >>> # Skip optimization (fixed hyperparameters)
        >>> gp.hyp_list().set_all_inactive()
        >>> gp.fit(X_train_list, y_train_list)
        """
        from pyapprox.surrogates.gaussianprocess.fitters.multioutput_fitter import (
            MultiOutputGPMaximumLikelihoodFitter,
        )

        fitter = MultiOutputGPMaximumLikelihoodFitter(
            bkd=self._bkd,
            optimizer=self._optimizer,
        )
        result = fitter.fit(self, X_train_list, y_train)
        self._copy_fitted_state_from(result.surrogate())

    def _configure_loss(self, loss: GPNegativeLogMarginalLikelihoodLoss[Array]) -> None:
        """Configure loss function after creation.

        Override in subclasses to customize gradient computation.
        """
        pass

    def _fit_internal(
        self, X_train_list: List[Array], y_train: Union[List[Array], Array]
    ) -> None:
        """Internal fit: store data, compute Cholesky, compute alpha.

        This is the pure data-fitting method without optimization.

        Parameters
        ----------
        X_train_list : List[Array]
            Training inputs for each output. Each array has shape (nvars, n_i).
        y_train : Union[List[Array], Array]
            Training outputs (list or stacked format).

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

        # Handle both list and stacked formats
        if isinstance(y_train, list):
            # List format: each y has shape (1, n_i)
            self._data = MultiOutputGPTrainingData(X_train_list, y_train, self._bkd)
            y_train_stacked = self._data.y_stacked()
        else:
            # Stacked format (legacy): shape (sum(n_i), 1)
            y_train_stacked = y_train
            # Create data container by unstacking
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
            # Convert stacked to list format for data container
            y_list = []
            offset = 0
            for X in X_train_list:
                n_i = X.shape[1]
                y_i = self._bkd.reshape(
                    y_train_stacked[offset : offset + n_i, 0], (1, n_i)
                )
                y_list.append(y_i)
                offset += n_i
            self._data = MultiOutputGPTrainingData(X_train_list, y_list, self._bkd)

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

    def neg_log_marginal_likelihood(self) -> Array:
        """
        Compute negative log marginal likelihood.

        For multi-output GP with stacked outputs:
            -log p(y|X,θ) = 0.5 * [y^T (K + σ²I)^{-1} y + log|K + σ²I| + n log(2π)]

        where y is the stacked output vector and K is the block kernel matrix.

        Returns
        -------
        nll : Array
            Scalar negative log marginal likelihood.

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
        constant = n_total * math.log(2.0 * math.pi)

        # Negative log marginal likelihood
        nll = 0.5 * (data_fit + log_det + constant)

        return nll

    def _unstack_predictions(
        self, stacked: Array, n_samples_list: List[int]
    ) -> List[Array]:
        """
        Convert stacked predictions to list format.

        Parameters
        ----------
        stacked : Array
            Stacked array, shape (sum(n_i), 1).
        n_samples_list : List[int]
            Number of samples for each output.

        Returns
        -------
        List[Array]
            List of arrays, each with shape (1, n_i).
        """
        result = []
        offset = 0
        for n_i in n_samples_list:
            arr_i = self._bkd.reshape(stacked[offset : offset + n_i, 0], (1, n_i))
            result.append(arr_i)
            offset += n_i
        return result

    def predict(self, X_test_list: List[Array]) -> List[Array]:
        """
        Predict mean at test points.

        Parameters
        ----------
        X_test_list : List[Array]
            Test inputs for each output. Each has shape (nvars, n_test_i).
            Can be different test points for each output.

        Returns
        -------
        y_pred : List[Array]
            Predictions for each output, each with shape (1, n_test_i).

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

        # Mean prediction: μ* = K_star @ alpha (stacked format)
        y_pred_stacked = K_star @ self._alpha

        # Convert to list format
        n_samples_list = [X.shape[1] for X in X_test_list]
        return self._unstack_predictions(y_pred_stacked, n_samples_list)

    def predict_with_uncertainty(
        self, X_test_list: List[Array]
    ) -> Tuple[List[Array], List[Array]]:
        """
        Predict mean and standard deviation.

        Parameters
        ----------
        X_test_list : List[Array]
            Test inputs for each output. Each has shape (nvars, n_test_i).

        Returns
        -------
        mean : List[Array]
            Mean predictions for each output, each with shape (1, n_test_i).
        std : List[Array]
            Standard deviations for each output, each with shape (1, n_test_i).

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

        # Mean prediction (already in list format)
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
        v = self._bkd.solve_triangular(self._cholesky.factor(), K_star.T, lower=True)

        # Posterior variance: σ²* = K** - v^T @ v
        # v^T @ v is computed as sum of v * v along rows
        var = K_star_star_diag - self._bkd.sum(v * v, axis=0)

        # Numerical stability: clip small negative values to zero
        # (arise from numerical errors in Cholesky solve)
        var = var * (var >= 0.0)

        # Standard deviation (stacked format)
        std_stacked = self._bkd.sqrt(var)[:, None]

        # Convert to list format
        n_samples_list = [X.shape[1] for X in X_test_list]
        std = self._unstack_predictions(std_stacked, n_samples_list)

        return mean, std

    def predict_covariance(self, X_test_list: List[Array]) -> Tuple[List[Array], Array]:
        """
        Predict mean and full covariance matrix.

        Parameters
        ----------
        X_test_list : List[Array]
            Test inputs for each output. Each has shape (nvars, n_test_i).

        Returns
        -------
        mean : List[Array]
            Mean predictions for each output, each with shape (1, n_test_i).
        cov : Array
            Covariance matrix, shape (sum(n_test_i), sum(n_test_i)).
            This is the full joint covariance across all outputs.

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

        # Mean prediction (already in list format)
        mean = self.predict(X_test_list)

        # Cross-covariance
        K_star = self._kernel(X_test_list, self._X_train_list, block_format=False)

        # Prior covariance
        K_star_star = self._kernel(X_test_list, block_format=False)

        # Solve: v = L^{-1} @ K_star^T
        v = self._bkd.solve_triangular(self._cholesky.factor(), K_star.T, lower=True)

        # Posterior covariance: Σ* = K** - v^T @ v
        cov = K_star_star - v.T @ v

        return mean, cov

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
