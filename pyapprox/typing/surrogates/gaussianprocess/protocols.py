"""
Protocol definitions for Gaussian Process implementations.

This module defines a hierarchy of protocols that establish the interface
for Gaussian Process regression. All GPs support mean, standard deviation,
and covariance predictions.
"""

from typing import Protocol, Optional, runtime_checkable, Generic
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import Kernel
from pyapprox.typing.util.hyperparameter import HyperParameterList


@runtime_checkable
class GaussianProcessProtocol(Protocol, Generic[Array]):
    """
    Base protocol for all Gaussian Process implementations.

    Defines the fundamental properties that all GPs must have:
    - A backend for numerical operations
    - A covariance kernel function
    - Number of input variables
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical operations.

        Returns
        -------
        Backend[Array]
            The backend instance (e.g., NumpyBkd or TorchBkd).
        """
        ...

    def kernel(self) -> Kernel[Array]:
        """
        Return the covariance kernel function.

        Returns
        -------
        Kernel[Array]
            The kernel instance defining the covariance structure.
        """
        ...

    def nvars(self) -> int:
        """
        Return the number of input variables (dimensionality).

        Returns
        -------
        int
            Number of input dimensions.
        """
        ...


@runtime_checkable
class FittableGPProtocol(GaussianProcessProtocol[Array], Protocol):
    """
    Protocol for GPs that can be fitted to training data.

    Extends GaussianProcessProtocol with the ability to fit the GP
    to observed data and check if fitting has been performed.
    """

    def fit(self, X_train: Array, y_train: Array) -> None:
        """
        Fit the Gaussian Process to training data.

        Parameters
        ----------
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (n_train, nqoi).
        """
        ...

    def is_fitted(self) -> bool:
        """
        Check if the GP has been fitted to data.

        Returns
        -------
        bool
            True if fit() has been called successfully, False otherwise.
        """
        ...


@runtime_checkable
class PredictiveGPProtocol(FittableGPProtocol[Array], Protocol):
    """
    Protocol for GPs that can make predictions with uncertainty.

    Extends FittableGPProtocol with the ability to predict posterior
    mean, standard deviation, and full covariance at new locations.
    All GPs must support all three prediction methods.
    """

    def predict(self, X: Array) -> Array:
        """
        Predict the posterior mean at new input locations.

        Parameters
        ----------
        X : Array
            Input locations for prediction, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior mean predictions, shape (n_test, nqoi).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...

    def predict_std(self, X: Array) -> Array:
        """
        Predict the posterior standard deviation at new input locations.

        Parameters
        ----------
        X : Array
            Input locations for prediction, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior standard deviation, shape (n_test, nqoi).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...

    def predict_covariance(self, X: Array) -> Array:
        """
        Predict full posterior covariance matrix.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior covariance matrix. For single output (nqoi=1):
            shape (n_test, n_test). For multiple outputs: shape
            (n_test*nqoi, n_test*nqoi) with block structure.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...

    def __call__(self, X: Array) -> Array:
        """
        Predict posterior mean at new locations (alias for predict).

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior mean, shape (n_test, nqoi).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...


@runtime_checkable
class TrainableGPProtocol(PredictiveGPProtocol[Array], Protocol):
    """
    Protocol for GPs with trainable hyperparameters.

    Extends PredictiveGPProtocol with methods for hyperparameter
    optimization, including access to the marginal likelihood and
    automated optimization.
    """

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters.

        Returns
        -------
        HyperParameterList
            List containing all kernel and GP hyperparameters.
        """
        ...

    def neg_log_marginal_likelihood(self) -> float:
        """
        Compute the negative log marginal likelihood.

        The negative log marginal likelihood (NLML) is:
            -log p(y | X, θ) = 0.5 * [y^T K^{-1} y + log|K| + n log(2π)]

        where K = K(X, X) + σ²I is the noisy covariance matrix.

        Returns
        -------
        float
            Negative log marginal likelihood value.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...

    def optimize_hyperparameters(
        self,
        optimizer: Optional[object] = None
    ) -> None:
        """
        Optimize hyperparameters by minimizing negative log marginal likelihood.

        Uses an optimizer from pyapprox.typing.optimization.minimize to
        find hyperparameters that maximize the marginal likelihood.

        Parameters
        ----------
        optimizer : Optional[object]
            Optimizer instance from typing.optimization.minimize.
            If None, uses a default optimizer (typically
            ScipyTrustConstrOptimizer).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...
