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
    >>> y_train = bkd.array(np.random.randn(10, 1))
    >>> gp.fit(X_train, y_train)
    >>>
    >>> X_test = bkd.array(np.random.randn(2, 5))
    >>> mean = gp.predict(X_test)
    >>> std = gp.predict_std(X_test)

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

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally add hvp method based on kernel capabilities.

        The hvp method is only exposed if the kernel implements
        KernelWithJacobianAndHVPWrtX1Protocol (i.e., has hvp_wrt_x1 method).
        """
        from pyapprox.typing.surrogates.kernels.protocols import KernelWithJacobianAndHVPWrtX1Protocol
        if isinstance(self._kernel, KernelWithJacobianAndHVPWrtX1Protocol):
            self.hvp = self._hvp

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
            Precomputed weights, shape (n_train, nqoi).

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

    def fit(self, X_train: Array, y_train: Array) -> None:
        """
        Fit the Gaussian Process to training data.

        This computes the Cholesky factorization of K + σ²I and
        precomputes α = (K + σ²I)^{-1}(y - m(X)) for efficient
        prediction.

        Parameters
        ----------
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (n_train, nqoi).

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
        mean_pred = self._mean(X_train)
        residual = y_train - mean_pred
        self._alpha = self._cholesky.solve(residual)

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
            Posterior mean, shape (n_test, nqoi).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before making predictions")

        # Prior mean
        mean_prior = self._mean(X)

        # Compute k(X*, X)
        K_star = self._kernel(X, self._data.X())

        # Posterior mean: μ* = m(X*) + k(X*, X) α
        mean_posterior = mean_prior + K_star @ self._alpha

        return mean_posterior

    def __call__(self, X: Array) -> Array:
        """
        Predict posterior mean (alias for predict).

        Returns predictions in format (nqoi, n_test) for compatibility
        with FunctionProtocol plotting utilities.
        """
        # predict() returns (n_test, nqoi), transpose to (nqoi, n_test)
        return self.predict(X).T

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
            Posterior standard deviation, shape (n_test, nqoi).

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

        # Reshape to (n_test, nqoi) - tile for each output
        nqoi = self._data.nqoi()
        std = self._bkd.reshape(std, (std.shape[0], 1))
        std = self._bkd.tile(std, (1, nqoi))

        return std

    def jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the GP mean with respect to inputs.

        For a GP with mean m(x) and covariance k(x, x'), the posterior mean is:
            μ*(x) = m(x) + k(x, X) α
        where α = [K + σ²I]^{-1}(y - m(X)).

        The Jacobian is:
            ∂μ*/∂x = ∂m/∂x + ∂k(x, X)/∂x @ α

        For ZeroMean and ConstantMean, ∂m/∂x = 0.

        Parameters
        ----------
        sample : Array
            Input locations, shape (nvars, n_samples).

        Returns
        -------
        Array
            Jacobian of shape (nqoi, nvars) for single sample or
            (n_samples, nqoi, nvars) for multiple samples.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing Jacobian")

        # Kernel Jacobian: ∂k(x, X)/∂x has shape (n_samples, n_train, nvars)
        K_jac = self._kernel.jacobian(sample, self._data.X())

        # For each sample point, compute: ∂k(x, X)/∂x @ α
        # K_jac shape: (n_samples, n_train, nvars)
        # α shape: (n_train, nqoi)
        # Result shape: (n_samples, nqoi, nvars) - using einsum to get right order
        jac = self._bkd.einsum("ijk,jl->ilk", K_jac, self._alpha)

        # If single sample, remove first dimension to get (nqoi, nvars)
        if sample.shape[1] == 1:
            jac = jac[0, :, :]

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

    def _hvp(self, sample: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product for GP mean with respect to inputs.

        This is a private method. The public hvp() method is dynamically
        added during __init__ if the kernel supports KernelWithJacobianAndHVPWrtX1Protocol.

        This computes H(x)·V where H is the Hessian of the GP mean prediction
        with respect to inputs x, and V is a direction vector.

        For a GP with mean m(x) and covariance k(x, x'), the posterior mean is:
            μ*(x) = m(x) + k(x, X) α
        where α = [K + σ²I]^{-1}(y - m(X)).

        Parameters
        ----------
        sample : Array
            Input locations, shape (nvars, n_samples).
        direction : Array
            Direction vector, shape (nvars, n_samples).

        Returns
        -------
        Array
            Hessian-vector product, shape (nvars, n_samples).

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.
        ValueError
            If shapes don't match.

        Notes
        -----
        This uses analytical second derivatives via kernel's hvp_wrt_x1 method.
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before computing HVP")

        if sample.shape != direction.shape:
            raise ValueError(
                f"sample and direction must have same shape, "
                f"got {sample.shape} and {direction.shape}"
            )

        nvars = sample.shape[0]
        n_samples = sample.shape[1]

        if nvars != self._nvars:
            raise ValueError(
                f"sample has {nvars} variables, expected {self._nvars}"
            )

        # For multiple samples, compute HVP for each independently
        if n_samples > 1:
            hvps = []
            for i in range(n_samples):
                sample_i = sample[:, i:i+1]
                direction_i = direction[:, i:i+1]
                hvp_i = self._hvp(sample_i, direction_i)
                hvps.append(hvp_i)
            return self._bkd.concatenate(hvps, axis=1)

        # Single sample case: sample shape (nvars, 1), direction shape (nvars, 1)
        nqoi = self._data.nqoi()
        if nqoi > 1:
            raise NotImplementedError(
                "HVP currently only supports single-output GPs (nqoi=1)"
            )

        # Get training data
        X_train = self._data.X()  # (nvars, n_train)
        n_train = X_train.shape[1]

        # Get α (already computed during fit) - shape: (n_train, 1)
        alpha = self._bkd.reshape(self._alpha, (n_train,))  # (n_train,)

        # Reshape inputs
        V = self._bkd.reshape(direction, (nvars,))  # (nvars,)
        x_star = self._bkd.reshape(sample, (nvars,))  # (nvars,)

        hvp = self._hvp_using_kernel_hvp(x_star, V, X_train, alpha)

        # Reshape to (nvars, 1) to match input shape
        return self._bkd.reshape(hvp, (nvars, 1))

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

    def optimize_hyperparameters(
        self,
        optimizer: Optional[object] = None,
        init_guess: Optional[Array] = None
    ) -> None:
        """
        Optimize hyperparameters by minimizing negative log marginal likelihood.

        Uses typing.optimization.minimize optimizers to find optimal hyperparameters
        by minimizing the negative log marginal likelihood.

        Parameters
        ----------
        optimizer : Optional[object]
            Optimizer instance (e.g., ScipyTrustConstrOptimizer,
            ScipyDifferentialEvolutionOptimizer). If None, uses
            ScipyTrustConstrOptimizer with verbosity=0 and maxiter=1000.
        init_guess : Optional[Array]
            Initial guess for hyperparameters in optimization space.
            If None, uses current hyperparameter values.

        Raises
        ------
        RuntimeError
            If the GP has not been fitted.

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
        >>> X_train = bkd.array(np.random.randn(2, 20))
        >>> y_train = bkd.array(np.random.randn(20, 1))
        >>> gp.fit(X_train, y_train)
        >>>
        >>> # Optimize hyperparameters using default optimizer
        >>> gp.optimize_hyperparameters()
        """
        if not self.is_fitted():
            raise RuntimeError(
                "GP must be fitted before optimizing hyperparameters"
            )

        # Handle edge case: no active parameters to optimize
        if self.hyp_list().nactive_params() == 0:
            # Nothing to optimize - all parameters are fixed
            return

        from pyapprox.typing.surrogates.gaussianprocess.gp_loss import (
            GPNegativeLogMarginalLikelihoodLoss
        )

        # Create loss function
        loss = GPNegativeLogMarginalLikelihoodLoss(
            self, (self._data.X(), self._data.y())
        )

        # Get bounds for active hyperparameters only
        bounds = self.hyp_list().get_active_bounds()

        # Get initial guess
        if init_guess is None:
            init_guess = self.hyp_list().get_active_values()

        # Reshape init_guess to (n, 1) if it's 1D
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(init_guess, (len(init_guess), 1))

        # Create optimizer if not provided
        if optimizer is None:
            from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer
            )
            optimizer = ScipyTrustConstrOptimizer(
                objective=loss,
                bounds=bounds,
                verbosity=0,
                maxiter=1000
            )

        # Run optimization
        result = optimizer.minimize(init_guess)

        # Update hyperparameters with optimal values
        # optima() returns shape (n, 1), flatten to 1D for set_active_values
        optimal_params = result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        self.hyp_list().set_active_values(optimal_params)

        # Refit with optimal hyperparameters
        self.fit(self._data.X(), self._data.y())

    def __repr__(self) -> str:
        """Return string representation."""
        fitted_str = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"ExactGaussianProcess(kernel={self._kernel.__class__.__name__}, "
            f"nvars={self._nvars}, nugget={self._nugget}, "
            f"mean={self._mean.__class__.__name__}, {fitted_str})"
        )
