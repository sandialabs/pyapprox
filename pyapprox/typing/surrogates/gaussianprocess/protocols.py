"""
Protocol definitions for Gaussian Process implementations.

This module defines a hierarchy of protocols that establish the interface
for Gaussian Process regression. All GPs support mean, standard deviation,
and covariance predictions.
"""

from typing import Protocol, Optional, runtime_checkable, Generic

from pyapprox.typing.surrogates.gaussianprocess.output_transform import (
    OutputAffineTransformProtocol,
)
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import Kernel
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.surrogates.gaussianprocess.data import GPTrainingData
from pyapprox.typing.util.linalg.cholesky_factor import CholeskyFactor
from pyapprox.typing.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)


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
            Training output data, shape (nqoi, n_train).
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
            Posterior mean predictions, shape (nqoi, n_test).

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
            Posterior standard deviation, shape (nqoi, n_test).

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
            Posterior mean, shape (nqoi, n_test).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...

    def data(self) -> GPTrainingData[Array]:
        """
        Return the training data container.

        Returns
        -------
        GPTrainingData[Array]
            Training data (X, y) used to fit the GP.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...

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
        ...

    def alpha(self) -> Array:
        """
        Return the precomputed weights alpha = A^{-1}(y - m(X)).

        Returns
        -------
        Array
            Precomputed weights, shape (nqoi, n_train).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted yet.
        """
        ...

    def output_transform(
        self,
    ) -> Optional["OutputAffineTransformProtocol[Array]"]:
        """
        Return the output affine transform, or None if not set.

        Returns
        -------
        Optional[OutputAffineTransformProtocol[Array]]
            The output transform used to map between scaled and
            original output spaces.
        """
        ...


@runtime_checkable
class TrainableGPProtocol(PredictiveGPProtocol[Array], Protocol):
    """
    Protocol for GPs with trainable hyperparameters.

    Extends PredictiveGPProtocol with methods for hyperparameter
    optimization, including access to the marginal likelihood and
    optimizer configuration.

    Hyperparameter optimization is integrated into ``fit()``:
    - By default, ``fit()`` optimizes active hyperparameters
    - Use ``set_optimizer()`` to configure a custom optimizer
    - Set all hyperparameters inactive to skip optimization
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

    def set_optimizer(
        self,
        optimizer: BindableOptimizerProtocol[Array]
    ) -> None:
        """
        Set the optimizer for hyperparameter optimization during fit().

        Parameters
        ----------
        optimizer : BindableOptimizerProtocol[Array]
            An optimizer configured with options but NOT bound to an objective.
            During fit(), the optimizer will be cloned and bound to the
            negative log marginal likelihood loss function.

        Examples
        --------
        >>> from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
        ...     ScipyTrustConstrOptimizer
        ... )
        >>> optimizer = ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-8)
        >>> gp.set_optimizer(optimizer)
        >>> gp.fit(X_train, y_train)
        """
        ...

    def optimizer(self) -> Optional[BindableOptimizerProtocol[Array]]:
        """
        Return the current optimizer (None means use default).

        Returns
        -------
        Optional[BindableOptimizerProtocol[Array]]
            The configured optimizer, or None if using the default
            ScipyTrustConstrOptimizer.
        """
        ...
