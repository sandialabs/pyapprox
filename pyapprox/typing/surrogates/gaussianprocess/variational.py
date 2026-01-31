"""
Variational Gaussian Process regression using inducing points.

This module provides the VariationalGaussianProcess class which implements
variational inference for scalable GP regression using the Titsias (2009)
formulation with inducing points. The ELBO (Evidence Lower Bound) is used
as the objective for hyperparameter optimization.

References
----------
.. [Titsias2009] Michalis Titsias. *Variational Learning of Inducing
   Variables in Sparse Gaussian Processes*. AISTATS, 2009.
.. [VanDerWilk2020] Mark van der Wilk et al. *A Framework for Interdomain
   and Multioutput Gaussian Processes*. 2020.
"""

import math
from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import Kernel
from pyapprox.typing.surrogates.gaussianprocess.data import GPTrainingData
from pyapprox.typing.surrogates.gaussianprocess.output_transform import (
    OutputAffineTransformProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.input_transform import (
    InputAffineTransformProtocol,
    IdentityInputTransform,
)
from pyapprox.typing.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ZeroMean,
)
from pyapprox.typing.surrogates.gaussianprocess.inducing_samples import (
    InducingSamples,
)
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.util.linalg.cholesky_factor import CholeskyFactor
from pyapprox.typing.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)


class VariationalGaussianProcess(Generic[Array]):
    """
    Variational Gaussian Process with inducing points (Titsias 2009).

    Uses the Nyström approximation to the covariance matrix and optimizes
    the ELBO (Evidence Lower Bound) for hyperparameter estimation. Noise
    is estimated as part of the variational inference procedure (not via
    the kernel).

    The variational GP exposes effective ``alpha()`` and ``cholesky()``
    quantities that make it compatible with the statistics module
    (moments, sensitivity analysis). These correspond to the effective
    covariance ``K_eff = K_XU K_UU^{-1} K_XU^T + sigma^2 I``.

    Parameters
    ----------
    kernel : Kernel[Array]
        Covariance kernel (must NOT be a SumKernel — noise is managed
        by InducingSamples).
    nvars : int
        Number of input variables.
    inducing_samples : InducingSamples[Array]
        Inducing point manager with noise hyperparameter.
    bkd : Backend[Array]
        Backend for numerical operations.
    mean_function : Optional[MeanFunction[Array]]
        Mean function. Defaults to ZeroMean.
    nugget : float
        Numerical stability parameter for K_UU. Default 1e-6.
    """

    def __init__(
        self,
        kernel: Kernel[Array],
        nvars: int,
        inducing_samples: InducingSamples[Array],
        bkd: Backend[Array],
        mean_function: Optional[MeanFunction[Array]] = None,
        nugget: float = 1e-6,
    ) -> None:
        self._kernel = kernel
        self._nvars = nvars
        self._bkd = bkd
        self._inducing_samples = inducing_samples

        if mean_function is None:
            self._mean = ZeroMean(bkd)
        else:
            self._mean = mean_function

        if nugget <= 0:
            raise ValueError(f"nugget must be positive, got {nugget}")
        self._nugget = nugget

        # Training data (set during fit)
        self._data: Optional[GPTrainingData[Array]] = None

        # Effective quantities for PredictiveGPProtocol / statistics compat
        self._cholesky: Optional[CholeskyFactor[Array]] = None
        self._alpha: Optional[Array] = None

        # Cached ELBO quantities (set in _fit_internal)
        self._neg_elbo: Optional[Array] = None

        # Transforms
        self._output_transform: Optional[
            OutputAffineTransformProtocol[Array]
        ] = None
        self._input_transform: InputAffineTransformProtocol[Array] = (
            IdentityInputTransform(nvars, bkd)
        )

        # Optimizer
        self._optimizer: Optional[BindableOptimizerProtocol[Array]] = None

        # Setup derivative methods (overridden in Torch subclass)
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """Override in subclasses for autograd-based derivatives."""
        pass

    # ---- Accessors (PredictiveGPProtocol) ----

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
        """Return the number of output dimensions."""
        if self._data is None:
            return 1
        return self._data.nqoi()

    def is_fitted(self) -> bool:
        """Check if the GP has been fitted."""
        return self._data is not None

    def data(self) -> GPTrainingData[Array]:
        """Return training data in scaled space."""
        if self._data is None:
            raise RuntimeError("GP must be fitted before accessing data.")
        return self._data

    def cholesky(self) -> CholeskyFactor[Array]:
        """Return effective Cholesky factor for statistics compatibility.

        This is chol(K_eff) where K_eff = K_XU K_UU^{-1} K_XU^T + sigma^2 I.
        """
        if self._cholesky is None:
            raise RuntimeError(
                "GP must be fitted before accessing cholesky."
            )
        return self._cholesky

    def alpha(self) -> Array:
        """Return effective alpha = K_eff^{-1}(y - m(X)) for statistics.

        Shape: (nqoi, n_train).
        """
        if self._alpha is None:
            raise RuntimeError("GP must be fitted before accessing alpha.")
        return self._alpha

    def mean(self) -> MeanFunction[Array]:
        """Return the mean function."""
        return self._mean

    def output_transform(
        self,
    ) -> Optional[OutputAffineTransformProtocol[Array]]:
        """Return the output transform, or None."""
        return self._output_transform

    def input_transform(self) -> InputAffineTransformProtocol[Array]:
        """Return the input transform (never None)."""
        return self._input_transform

    def inducing_samples(self) -> InducingSamples[Array]:
        """Return the inducing samples manager."""
        return self._inducing_samples

    def hyp_list(self) -> HyperParameterList:
        """Return combined hyperparameter list (kernel + mean + inducing)."""
        kernel_hyps = self._kernel.hyp_list().hyperparameters()
        mean_hyps = self._mean.hyp_list().hyperparameters()
        inducing_hyps = self._inducing_samples.hyp_list().hyperparameters()
        return HyperParameterList(kernel_hyps + mean_hyps + inducing_hyps)

    def set_optimizer(
        self, optimizer: BindableOptimizerProtocol[Array]
    ) -> None:
        """Set the optimizer for hyperparameter optimization."""
        self._optimizer = optimizer

    def optimizer(self) -> Optional[BindableOptimizerProtocol[Array]]:
        """Return the current optimizer (None = use default)."""
        return self._optimizer

    def _configure_loss(self, loss) -> None:  # type: ignore[no-untyped-def]
        """Hook for subclasses to customize the loss (e.g., bind autograd)."""
        pass

    # ---- Core fitting ----

    def _fit_internal(self, X_train: Array, y_train: Array) -> None:
        """Compute ELBO quantities and effective alpha/cholesky.

        Parameters
        ----------
        X_train : Array
            Training inputs in scaled space, shape (nvars, n_train).
        y_train : Array
            Training outputs in scaled space, shape (nqoi, n_train).
        """
        bkd = self._bkd
        self._data = GPTrainingData(
            X_train, y_train, bkd,
            output_transform=self._output_transform,
        )
        if self._data.nvars() != self._nvars:
            raise ValueError(
                f"X_train has {self._data.nvars()} variables, "
                f"expected {self._nvars}"
            )

        U = self._inducing_samples.get_samples()
        noise_std = self._inducing_samples.get_noise()  # shape (1,)
        noise_var = noise_std[0] ** 2

        n_train = X_train.shape[1]

        # Kernel matrices
        K_UU = self._kernel(U, U)
        K_UU = K_UU + bkd.eye(K_UU.shape[0]) * self._nugget
        L_UU = bkd.cholesky(K_UU)

        K_XU = self._kernel(X_train, U)

        # --- ELBO computation (Nyström approximation) ---
        # Delta = L_UU^{-1} K_XU^T / noise_std, shape (M, N)
        Delta = bkd.solve_triangular(L_UU, K_XU.T, lower=True) / noise_std[0]

        # Omega = I_M + Delta @ Delta^T, shape (M, M)
        M = Delta.shape[0]
        Omega = bkd.eye(M) + Delta @ Delta.T
        L_Omega = bkd.cholesky(Omega)

        # Mean residual
        mean_pred = self._mean(X_train)
        residual = y_train - mean_pred  # (nqoi, n_train)

        # For nqoi=1 (standard case), squeeze to (n_train, 1)
        values = residual.T  # (n_train, nqoi)

        # gamma = L_Omega^{-1} Delta values, shape (M, nqoi)
        gamma = bkd.solve_triangular(
            L_Omega, Delta @ values, lower=True
        )

        # Log determinant: log|Omega| + 2*N*log(noise_std)
        log_det = (
            2.0 * bkd.sum(bkd.log(bkd.get_diagonal(L_Omega)))
            + 2.0 * n_train * bkd.log(noise_std[0])
        )

        # Log probability
        log_pdf = -0.5 * (
            n_train * math.log(2.0 * math.pi)
            + log_det
            + (bkd.sum(values * values) - bkd.sum(gamma * gamma)) / noise_var
        )

        # Trace regularization: 0.5/sigma^2 * (tr(K_XX) - tr(Q_XX))
        K_XX_diag = self._kernel.diag(X_train)
        # Q_XX_diag = sum of squared columns of L_UU^{-1} K_XU^T
        tmp = bkd.solve_triangular(L_UU, K_XU.T, lower=True)  # (M, N)
        Q_XX_diag = bkd.einsum("ij,ij->j", tmp, tmp)
        K_tilde_trace = bkd.sum(K_XX_diag) - bkd.sum(Q_XX_diag)
        log_pdf = log_pdf - 0.5 / noise_var * K_tilde_trace

        self._neg_elbo = -log_pdf

        # --- Effective alpha and cholesky for statistics compatibility ---
        # K_eff = K_XU K_UU^{-1} K_XU^T + sigma^2 I
        # We form this explicitly and factorize it.
        # For moderate n_train this is fine; for very large n_train
        # a Woodbury approach would be better.
        K_UU_inv_KXU_T = bkd.solve_triangular(
            L_UU.T,
            bkd.solve_triangular(L_UU, K_XU.T, lower=True),
            lower=False,
        )  # (M, N)
        K_eff = K_XU @ K_UU_inv_KXU_T + noise_var * bkd.eye(n_train)

        try:
            L_eff = bkd.cholesky(K_eff)
            self._cholesky = CholeskyFactor(L_eff, bkd)
        except Exception as e:
            raise RuntimeError(
                "Cholesky of effective covariance failed. "
                f"Original error: {e}"
            )

        # alpha = K_eff^{-1} residual^T, then transpose to (nqoi, n_train)
        self._alpha = self._cholesky.solve(residual.T).T

    def neg_log_marginal_likelihood(self) -> Array:
        """Return the negative ELBO (computed in _fit_internal).

        Returns
        -------
        Array
            Scalar negative ELBO value.
        """
        if self._neg_elbo is None:
            raise RuntimeError(
                "GP must be fitted before computing ELBO"
            )
        return self._neg_elbo

    # ---- Prediction ----

    def predict(self, X: Array) -> Array:
        """Predict posterior mean at new input locations.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior mean, shape (nqoi, n_test).
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before making predictions")

        X = self._input_transform.transform(X)
        mean_prior = self._mean(X)
        K_star = self._kernel(X, self._data.X())
        mean_posterior = mean_prior + self._alpha @ K_star.T

        if self._output_transform is not None:
            mean_posterior = self._output_transform.transform(mean_posterior)

        return mean_posterior

    def __call__(self, X: Array) -> Array:
        """Predict posterior mean (alias for predict)."""
        return self.predict(X)

    def predict_std(self, X: Array) -> Array:
        """Predict posterior standard deviation.

        Uses effective cholesky for the computation, same formula
        as ExactGaussianProcess.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior std, shape (nqoi, n_test).
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before making predictions")

        bkd = self._bkd
        X = self._input_transform.transform(X)
        K_star = self._kernel(X, self._data.X())
        K_star_star = self._kernel.diag(X)

        v = bkd.solve_triangular(
            self._cholesky.factor(), K_star.T, lower=True
        )
        var_posterior = K_star_star - bkd.einsum("ij,ij->j", v, v)
        var_posterior = var_posterior * (var_posterior >= 0.0)
        std = bkd.sqrt(var_posterior)

        nqoi = self._data.nqoi()
        std = bkd.reshape(std, (1, std.shape[0]))
        std = bkd.tile(std, (nqoi, 1))

        if self._output_transform is not None:
            scale = self._output_transform.scale()
            std = scale[:, None] * std

        return std

    def predict_covariance(self, X: Array) -> Array:
        """Predict posterior covariance matrix.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior covariance, shape (n_test, n_test).
        """
        if not self.is_fitted():
            raise RuntimeError("GP must be fitted before making predictions")

        bkd = self._bkd
        X = self._input_transform.transform(X)
        K_star = self._kernel(X, self._data.X())
        K_star_star = self._kernel(X, X)

        v = bkd.solve_triangular(
            self._cholesky.factor(), K_star.T, lower=True
        )
        cov = K_star_star - v.T @ v

        if self._output_transform is not None:
            s = self._output_transform.scale()[0]
            cov = s ** 2 * cov

        return cov

    # ---- Fitting with optimization ----

    def fit(
        self,
        X_train: Array,
        y_train: Array,
        output_transform: Optional[
            OutputAffineTransformProtocol[Array]
        ] = None,
        input_transform: Optional[
            InputAffineTransformProtocol[Array]
        ] = None,
    ) -> None:
        """Fit variational GP and optimize active hyperparameters.

        Parameters
        ----------
        X_train : Array
            Training inputs, shape (nvars, n_train).
        y_train : Array
            Training outputs, shape (nqoi, n_train).
        output_transform : Optional[OutputAffineTransformProtocol[Array]]
            If provided, y_train is in original space and will be scaled.
        input_transform : Optional[InputAffineTransformProtocol[Array]]
            If provided, X_train is in original space and will be scaled.
        """
        self._output_transform = output_transform

        if input_transform is not None:
            self._input_transform = input_transform
        else:
            self._input_transform = IdentityInputTransform(
                self._nvars, self._bkd
            )

        X_train = self._input_transform.transform(X_train)

        if output_transform is not None:
            y_train = output_transform.inverse_transform(y_train)

        self._fit_internal(X_train, y_train)

        if self.hyp_list().nactive_params() == 0:
            return

        from pyapprox.typing.surrogates.gaussianprocess.variational_loss import (
            VariationalGPELBOLoss,
        )
        loss = VariationalGPELBOLoss(
            self, (self._data.X(), self._data.y())
        )
        self._configure_loss(loss)

        bounds = self.hyp_list().get_active_bounds()

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )
            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        optimizer.bind(loss, bounds)

        init_guess = self.hyp_list().get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(
                init_guess, (len(init_guess), 1)
            )

        result = optimizer.minimize(init_guess)

        optimal_params = result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        self.hyp_list().set_active_values(optimal_params)

        self._fit_internal(X_train, y_train)

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"VariationalGaussianProcess("
            f"kernel={self._kernel.__class__.__name__}, "
            f"nvars={self._nvars}, nugget={self._nugget}, "
            f"{fitted_str})"
        )
