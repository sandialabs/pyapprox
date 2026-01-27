"""
Exact Gaussian Process regression implementation.

This module provides the ExactGaussianProcess class which performs
full GP regression using Cholesky factorization for numerical stability.
"""

import numpy as np
from typing import Generic, Optional
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import Kernel
from pyapprox.typing.surrogates.gaussianprocess.data import GPTrainingData
from pyapprox.typing.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ZeroMean
)
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.util.linalg.cholesky_factor import CholeskyFactor
from pyapprox.typing.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)


class ExactGaussianProcess(Generic[Array]):
    """
    Exact Gaussian Process regression using Cholesky factorization.

    This class implements full GP regression for the model:
        Prior: f(x) ~ GP(m(x), k(x, x'))
        Likelihood: y = f(x) + ε, ε ~ N(0, σ²I)

    The posterior predictive distribution is:
        f*|y ~ N(μ*, Σ*)
    where:
        μ* = m(x*) + k(x*, X)[K + σ²I]^{-1}(y - m(X))
        Σ* = k(x*, x*) - k(x*, X)[K + σ²I]^{-1}k(X, x*)

    Parameters
    ----------
    kernel : Kernel[Array]
        Covariance kernel function.
    nvars : int
        Number of input variables (dimensions).
    bkd : Backend[Array]
        Backend for numerical operations.
    mean_function : Optional[MeanFunction[Array]]
        Mean function. If None, uses ZeroMean. Default is None.
    nugget : float
        Numerical stability parameter added to kernel matrix diagonal.
        Not a hyperparameter - fixed to user-provided value.
        Must be positive. Default is 1e-6.
        Note: Observation noise should be modeled via the kernel
        (e.g., IIDGaussianNoise), not via this nugget parameter.

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
    >>> y_train = bkd.array(np.random.randn(1, 10))  # Shape: (nqoi, n_train)
    >>> gp.fit(X_train, y_train)  # Fits data AND optimizes hyperparameters
    >>>
    >>> X_test = bkd.array(np.random.randn(2, 5))
    >>> mean = gp.predict(X_test)
    >>> std = gp.predict_std(X_test)

    Hyperparameter optimization is controlled via:

    1. **Default behavior**: Uses ScipyTrustConstrOptimizer with maxiter=1000
    2. **Custom optimizer**: Call ``set_optimizer(optimizer)`` before ``fit()``
    3. **Skip optimization**: Set all hyperparameters inactive via
       ``gp.hyp_list().set_all_inactive()`` before ``fit()``

    Optional Methods
    ----------------
    This class uses dynamic method binding based on kernel capabilities:

    - ``hvp(sample, direction)``: Available if kernel implements
      ``KernelWithJacobianAndHVPWrtX1Protocol`` (i.e., has ``hvp_wrt_x1`` method).

    Check availability with ``hasattr(gp, 'hvp')``.

    Notes
    -----
    The ``jacobian`` method is always available as it only requires the kernel
    to have a ``jacobian`` method (which all kernels have).

    This class follows the dynamic binding pattern for optional methods.
    See docs/OPTIONAL_METHODS_CONVENTION.md for details.
    """

    def __init__(
        self,
        kernel: Kernel[Array],
        nvars: int,
        bkd: Backend[Array],
        mean_function: Optional[MeanFunction[Array]] = None,
        nugget: float = 1e-6
    ):
        self._kernel = kernel
        self._nvars = nvars
        self._bkd = bkd

        # Set mean function (default to zero mean)
        if mean_function is None:
            self._mean = ZeroMean(bkd)
        else:
            self._mean = mean_function

        # Nugget for numerical stability
        if nugget <= 0:
            raise ValueError(
                f"nugget must be positive, got {nugget}"
            )
        self._nugget = nugget

        # Training data (set during fit)
        self._data: Optional[GPTrainingData[Array]] = None

        # Precomputed quantities (set during fit)
        self._cholesky: Optional[CholeskyFactor[Array]] = None

        # Conditionally add derivative methods based on kernel capabilities
        self._setup_derivative_methods()

        self._alpha: Optional[Array] = None

        # Optimizer for hyperparameter tuning (None means use default)
        self._optimizer: Optional[BindableOptimizerProtocol[Array]] = None

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally add hvp/hvp_batch methods based on kernel capabilities.

        The hvp methods are only exposed if the kernel implements
        KernelWithJacobianAndHVPWrtX1Protocol (i.e., has hvp_wrt_x1 method).
        """
        from pyapprox.typing.surrogates.kernels.protocols import KernelWithJacobianAndHVPWrtX1Protocol
        if isinstance(self._kernel, KernelWithJacobianAndHVPWrtX1Protocol):
            self.hvp = self._hvp
            self.hvp_batch = self._hvp_batch

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def kernel(self) -> Kernel[Array]:
        """Return the covariance kernel."""
        return self._kernel

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (output dimensions).

        Returns
        -------
        int
            Number of output dimensions. Returns 1 if not fitted.
        """
        if self._data is None:
            return 1  # Default to 1 if not fitted
        return self._data.nqoi()

    def is_fitted(self) -> bool:
        """Check if the GP has been fitted."""
        return self._data is not None

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
        if self._data is None:
            raise RuntimeError("GP must be fitted before accessing data.")
        return self._data

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
        if self._cholesky is None:
            raise RuntimeError("GP must be fitted before accessing cholesky.")
        return self._cholesky

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
        if self._alpha is None:
            raise RuntimeError("GP must be fitted before accessing alpha.")
        return self._alpha

    def mean(self) -> MeanFunction[Array]:
        """
        Return the mean function.

        Returns
        -------
        MeanFunction[Array]
            The mean function instance.
        """
        return self._mean

    def hyp_list(self) -> HyperParameterList:
        """
        Return combined hyperparameter list.

        Returns
        -------
        HyperParameterList
            Combined list of kernel and mean function hyperparameters.
        """
        kernel_hyps = self._kernel.hyp_list()
        mean_hyps = self._mean.hyp_list()

        # Combine hyperparameter lists
        all_hyps = kernel_hyps.hyperparameters() + mean_hyps.hyperparameters()
        return HyperParameterList(all_hyps)

    def set_optimizer(
        self, optimizer: BindableOptimizerProtocol[Array]
    ) -> None:
        """Set the optimizer for hyperparameter optimization during fit().

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

    def fit(self, X_train: Array, y_train: Array) -> None:
        """Fit GP to data and optimize active hyperparameters.

        This method:
        1. Computes the Cholesky factorization and precomputes weights
        2. If any hyperparameters are active, optimizes them by minimizing
           the negative log marginal likelihood

        If all hyperparameters are inactive (fixed), only step 1 is performed.

        Parameters
        ----------
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (nqoi, n_train).

        Raises
        ------
        ValueError
            If data shapes are invalid.
        RuntimeError
            If Cholesky factorization fails (matrix not positive definite).

        Examples
        --------
        >>> # Standard fit with optimization
        >>> gp.fit(X_train, y_train)

        >>> # Custom optimizer
        >>> from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
        ...     ScipyTrustConstrOptimizer
        ... )
        >>> gp.set_optimizer(ScipyTrustConstrOptimizer(maxiter=500))
        >>> gp.fit(X_train, y_train)

        >>> # Skip optimization (fixed hyperparameters)
        >>> gp.hyp_list().set_all_inactive()
        >>> gp.fit(X_train, y_train)
        """
        # Initial fit
        self._fit_internal(X_train, y_train)

        # Check if optimization is needed
        if self.hyp_list().nactive_params() == 0:
            return  # All params fixed, nothing to optimize

        # Create loss function
        from pyapprox.typing.surrogates.gaussianprocess.gp_loss import (
            GPNegativeLogMarginalLikelihoodLoss
        )
        loss = GPNegativeLogMarginalLikelihoodLoss(
            self, (self._data.X(), self._data.y())
        )

        # Get bounds for active hyperparameters
        bounds = self.hyp_list().get_active_bounds()

        # Get optimizer (clone if user-provided to avoid shared state)
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer
            )
            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        # Bind optimizer to loss and bounds
        optimizer.bind(loss, bounds)

        # Get initial guess from current hyperparameter values
        init_guess = self.hyp_list().get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(init_guess, (len(init_guess), 1))

        # Run optimization
        result = optimizer.minimize(init_guess)

        # Update hyperparameters with optimal values
        optimal_params = result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        self.hyp_list().set_active_values(optimal_params)

        # Final refit with optimal hyperparameters
        self._fit_internal(X_train, y_train)

    def _fit_internal(self, X_train: Array, y_train: Array) -> None:
        """Internal fit: store data, compute Cholesky, compute alpha.

        This is the pure data-fitting method without optimization.

        Parameters
        ----------
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (nqoi, n_train).

        Raises
        ------
        ValueError
            If data shapes are invalid.
        RuntimeError
            If Cholesky factorization fails (matrix not positive definite).
        """
        # Validate and store training data
        self._data = GPTrainingData(X_train, y_train, self._bkd)

        # Check nvars matches
        if self._data.nvars() != self._nvars:
            raise ValueError(
                f"X_train has {self._data.nvars()} variables, "
                f"expected {self._nvars}"
            )

        # Compute kernel matrix K(X, X)
        K = self._kernel(X_train, X_train)

        # Add nugget for numerical stability: K_noisy = K + nugget*I
        K_noisy = K + self._bkd.eye(K.shape[0]) * self._nugget

        # Compute Cholesky factorization
        try:
            L = self._bkd.cholesky(K_noisy)
            self._cholesky = CholeskyFactor(L, self._bkd)
        except Exception as e:
            raise RuntimeError(
                "Cholesky factorization failed. The kernel matrix K + nugget*I "
                "may not be positive definite. Try increasing nugget. "
                f"Original error: {e}"
            )

        # Precompute α = (K + σ²I)^{-1}(y - m(X))
        # y_train shape: (nqoi, n_train), mean_pred shape: (1, n_train)
        # residual shape: (nqoi, n_train)
        mean_pred = self._mean(X_train)
        residual = y_train - mean_pred
        # Solve for each output: alpha shape becomes (nqoi, n_train)
        # Cholesky solve expects (n_train, k), so transpose, solve, transpose back
        self._alpha = self._cholesky.solve(residual.T).T

    def predict(self, X: Array) -> Array:
        """
        Predict posterior mean at new input locations.

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
            If the GP has not been fitted.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before making predictions")

        # Prior mean: shape (1, n_test)
        mean_prior = self._mean(X)

        # Compute k(X*, X): shape (n_test, n_train)
        K_star = self._kernel(X, self._data.X())

        # Posterior mean: μ* = m(X*) + α @ k(X*, X)^T
        # alpha shape: (nqoi, n_train), K_star shape: (n_test, n_train)
        # Result: (nqoi, n_test)
        mean_posterior = mean_prior + self._alpha @ K_star.T

        return mean_posterior

    def __call__(self, X: Array) -> Array:
        """
        Predict posterior mean (alias for predict).

        Returns predictions in format (nqoi, n_test) for compatibility
        with FunctionProtocol.
        """
        return self.predict(X)

    def predict_std(self, X: Array) -> Array:
        """
        Predict posterior standard deviation at new input locations.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior standard deviation, shape (nqoi, n_test).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before making predictions")

        # Compute k(X*, X)
        K_star = self._kernel(X, self._data.X())

        # Compute k(X*, X*)
        K_star_star = self._kernel.diag(X)

        # Solve L v = k(X, X*)^T for v where L is Cholesky factor
        v = self._bkd.solve_triangular(
            self._cholesky.factor(), K_star.T, lower=True
        )

        # Posterior variance: var* = k(X*, X*) - v^T v
        var_posterior = K_star_star - self._bkd.einsum("ij,ij->j", v, v)

        # Ensure non-negative (numerical stability - clamp negative values to zero)
        # Use array operations: multiply by mask to zero out negative values
        var_posterior = var_posterior * (var_posterior >= 0.0)

        # Standard deviation
        std = self._bkd.sqrt(var_posterior)

        # Reshape to (nqoi, n_test) - tile for each output
        nqoi = self._data.nqoi()
        std = self._bkd.reshape(std, (1, std.shape[0]))
        std = self._bkd.tile(std, (nqoi, 1))

        return std

    def jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the GP mean with respect to inputs (single sample).

        For a GP with mean m(x) and covariance k(x, x'), the posterior mean is:
            μ*(x) = m(x) + α @ k(x, X)^T
        where α = [K + σ²I]^{-1}(y - m(X)).

        The Jacobian is:
            ∂μ*/∂x = ∂m/∂x + α @ ∂k(x, X)^T/∂x

        For ZeroMean and ConstantMean, ∂m/∂x = 0.

        Parameters
        ----------
        sample : Array
            Single input location, shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian of shape (nqoi, nvars).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        ValueError
            If sample is not a single sample.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")

        if sample.shape[1] != 1:
            raise ValueError(
                f"jacobian() expects single sample with shape (nvars, 1), "
                f"got {sample.shape}. Use jacobian_batch() for multiple samples."
            )

        # Kernel Jacobian: ∂k(x, X)/∂x has shape (1, n_train, nvars)
        K_jac = self._kernel.jacobian(sample, self._data.X())

        # Compute: α @ ∂k(x, X)^T/∂x
        # K_jac shape: (1, n_train, nvars), α shape: (nqoi, n_train)
        # Result shape: (1, nqoi, nvars) -> squeeze to (nqoi, nvars)
        jac = self._bkd.einsum("lj,ijk->ilk", self._alpha, K_jac)

        return jac[0, :, :]

    def jacobian_batch(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the GP mean with respect to inputs (batch).

        Parameters
        ----------
        samples : Array
            Input locations, shape (nvars, n_samples).

        Returns
        -------
        Array
            Jacobian of shape (n_samples, nqoi, nvars).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")

        # Kernel Jacobian: ∂k(x, X)/∂x has shape (n_samples, n_train, nvars)
        K_jac = self._kernel.jacobian(samples, self._data.X())

        # For each sample point, compute: α @ ∂k(x, X)^T/∂x
        # K_jac shape: (n_samples, n_train, nvars), α shape: (nqoi, n_train)
        # Result shape: (n_samples, nqoi, nvars)
        jac = self._bkd.einsum("lj,ijk->ilk", self._alpha, K_jac)

        return jac

    def _hvp_using_kernel_hvp(
        self, x_star: Array, V: Array, X_train: Array, alpha: Array
    ) -> Array:
        """
        Compute HVP using kernel's hvp_wrt_x1() method.

        This is the preferred path for kernels that implement
        KernelWithJacobianAndHVPWrtX1Protocol. Works for:
        - Matern kernels with anisotropic length scales
        - Kernel compositions (if they implement hvp_wrt_x1)
        - Any kernel with efficient HVP computation

        Parameters
        ----------
        x_star : Array, shape (nvars,)
            Query point
        V : Array, shape (nvars,)
            Direction vector
        X_train : Array, shape (nvars, n_train)
            Training points
        alpha : Array, shape (n_train,)
            Dual coefficients

        Returns
        -------
        hvp : Array, shape (nvars,)
            Hessian-vector product
        """
        # Call kernel's hvp_wrt_x1 method
        # X1 = x_star (1 point), X2 = X_train (n_train points), direction = V
        # Returns: (1, n_train, nvars)
        kernel_hvp = self._kernel.hvp_wrt_x1(
            x_star[:, None],  # (nvars, 1)
            X_train,          # (nvars, n_train)
            V                 # (nvars,)
        )  # Shape: (1, n_train, nvars)

        # Contract with alpha: Σ_i H[k(x, x_i)]·V · α_i
        # Shape: (1, n_train, nvars) · (n_train,) -> (1, nvars) -> (nvars,)
        hvp = self._bkd.einsum('iqj,q->j', kernel_hvp, alpha)

        return hvp

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """
        Compute Hessian-vector product for GP mean (single sample).

        This is a private method. The public hvp() method is dynamically
        added during __init__ if the kernel supports KernelWithJacobianAndHVPWrtX1Protocol.

        This computes H(x)·v where H is the Hessian of the GP mean prediction
        with respect to inputs x, and v is a direction vector.

        Parameters
        ----------
        sample : Array
            Single input location, shape (nvars, 1).
        vec : Array
            Direction vector, shape (nvars, 1).

        Returns
        -------
        Array
            Hessian-vector product, shape (nvars, 1).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        ValueError
            If shapes don't match or not single sample.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing HVP")

        if sample.shape[1] != 1:
            raise ValueError(
                f"hvp() expects single sample with shape (nvars, 1), "
                f"got {sample.shape}. Use hvp_batch() for multiple samples."
            )

        if sample.shape != vec.shape:
            raise ValueError(
                f"sample and vec must have same shape, "
                f"got {sample.shape} and {vec.shape}"
            )

        nvars = sample.shape[0]
        if nvars != self._nvars:
            raise ValueError(
                f"sample has {nvars} variables, expected {self._nvars}"
            )

        nqoi = self._data.nqoi()
        if nqoi > 1:
            raise NotImplementedError(
                "HVP currently only supports single-output GPs (nqoi=1)"
            )

        # Get training data
        X_train = self._data.X()  # (nvars, n_train)
        n_train = X_train.shape[1]

        # Get α - shape: (nqoi, n_train) = (1, n_train)
        alpha = self._bkd.reshape(self._alpha, (n_train,))  # (n_train,)

        # Reshape inputs
        V = self._bkd.reshape(vec, (nvars,))  # (nvars,)
        x_star = self._bkd.reshape(sample, (nvars,))  # (nvars,)

        hvp = self._hvp_using_kernel_hvp(x_star, V, X_train, alpha)

        # Reshape to (nvars, 1)
        return self._bkd.reshape(hvp, (nvars, 1))

    def _hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """
        Compute Hessian-vector product for GP mean (batch).

        Parameters
        ----------
        samples : Array
            Input locations, shape (nvars, n_samples).
        vecs : Array
            Direction vectors, shape (nvars, n_samples).

        Returns
        -------
        Array
            Hessian-vector products, shape (n_samples, nvars).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        ValueError
            If shapes don't match.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing HVP")

        if samples.shape != vecs.shape:
            raise ValueError(
                f"samples and vecs must have same shape, "
                f"got {samples.shape} and {vecs.shape}"
            )

        nvars = samples.shape[0]
        n_samples = samples.shape[1]

        if nvars != self._nvars:
            raise ValueError(
                f"samples has {nvars} variables, expected {self._nvars}"
            )

        nqoi = self._data.nqoi()
        if nqoi > 1:
            raise NotImplementedError(
                "HVP currently only supports single-output GPs (nqoi=1)"
            )

        # Compute HVP for each sample
        hvps = []
        for i in range(n_samples):
            sample_i = samples[:, i:i+1]
            vec_i = vecs[:, i:i+1]
            hvp_i = self._hvp(sample_i, vec_i)  # (nvars, 1)
            hvps.append(hvp_i[:, 0])  # (nvars,)

        # Stack to (n_samples, nvars)
        return self._bkd.stack(hvps, axis=0)

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
            Posterior covariance matrix, shape (n_test, n_test) for
            single output (nqoi=1), or (n_test*nqoi, n_test*nqoi) for
            multiple outputs with block structure.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before making predictions")

        n_test = X.shape[1]
        nqoi = self._data.nqoi()

        # Compute k(X*, X)
        K_star = self._kernel(X, self._data.X())

        # Compute k(X*, X*)
        K_star_star = self._kernel(X, X)

        # Solve L v = k(X, X*)^T for v
        v = self._bkd.solve_triangular(
            self._cholesky.factor(), K_star.T, lower=True
        )

        # Posterior covariance: Σ* = k(X*, X*) - v^T v
        cov_posterior = K_star_star - v.T @ v

        # Ensure symmetry and positive semi-definiteness
        cov_posterior = 0.5 * (cov_posterior + cov_posterior.T)

        if nqoi == 1:
            return cov_posterior
        else:
            # For multiple outputs, create block diagonal structure
            # Each output has the same covariance
            cov_full = self._bkd.zeros((n_test * nqoi, n_test * nqoi))
            for i in range(nqoi):
                start = i * n_test
                end = (i + 1) * n_test
                cov_full[start:end, start:end] = cov_posterior

            return cov_full

    def neg_log_marginal_likelihood(self) -> float:
        """
        Compute the negative log marginal likelihood.

        The negative log marginal likelihood is:
            -log p(y | X, θ) = 0.5 * [y^T K^{-1} y + log|K| + n log(2π)]

        where K = K(X, X) + σ²I.

        Returns
        -------
        float
            Negative log marginal likelihood value.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        """
        if not self.is_fitted():
            raise RuntimeError(
                "GP must be fitted before computing marginal likelihood"
            )

        n = self._data.n_samples()

        # Data fit term: (y - m)^T (K + σ²I)^{-1} (y - m) = (y - m)^T α
        mean_pred = self._mean(self._data.X())
        residual = self._data.y() - mean_pred
        data_fit = float(self._bkd.sum(residual * self._alpha))

        # Complexity penalty: log|K + σ²I|
        log_det = self._cholesky.log_determinant()

        # Constant term
        constant = n * np.log(2 * np.pi)

        # Total negative log marginal likelihood
        nlml = 0.5 * (data_fit + log_det + constant)

        return nlml

    def __repr__(self) -> str:
        """Return string representation."""
        fitted_str = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"ExactGaussianProcess(kernel={self._kernel.__class__.__name__}, "
            f"nvars={self._nvars}, nugget={self._nugget}, "
            f"mean={self._mean.__class__.__name__}, {fitted_str})"
        )
