"""
Log unnormalized posterior for Bayesian inference.

This module provides the log unnormalized posterior:
    log p(theta | data) propto log p(data | theta) + log p(theta)

Used for:
- MAP estimation via optimization
- MCMC sampling
- Laplace approximation (via Hessian at MAP)
"""

from typing import Any, Callable, Generic, Optional

import numpy as np
from scipy import optimize

from pyapprox.probability.protocols import (
    DistributionProtocol,
    LogLikelihoodProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class LogUnNormalizedPosterior(Generic[Array]):
    """
    Log unnormalized posterior combining likelihood and prior.

    Computes:
        log p(theta | data) propto log p(data | theta) + log p(theta)

    where log p(data | theta) is evaluated via a model that maps
    parameters theta to model outputs (observations).

    Parameters
    ----------
    model_fn : Callable[[Array], Array]
        Function that maps parameter samples to model outputs.
        Signature: model_fn(params) -> outputs
        where params has shape (nvars, nsamples) and
        outputs has shape (nobs, nsamples).
    likelihood : LogLikelihoodProtocol[Array]
        Log-likelihood function with observations already set.
    prior : DistributionProtocol[Array]
        Prior distribution for parameters.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.likelihood import GaussianLogLikelihood
    >>> from pyapprox.probability.covariance import (
    ...     DiagonalCovarianceOperator
    ... )
    >>> from pyapprox.probability.gaussian import (
    ...     DenseCholeskyMultivariateGaussian
    ... )
    >>> bkd = NumpyBkd()
    >>> # Define model: y = A @ theta
    >>> A = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> def model_fn(theta):
    ...     return A @ theta
    >>> # Setup likelihood with noise
    >>> noise_var = np.array([0.01, 0.01])
    >>> noise_op = DiagonalCovarianceOperator(noise_var, bkd)
    >>> likelihood = GaussianLogLikelihood(noise_op, bkd)
    >>> obs = np.array([[1.0], [1.5]])
    >>> likelihood.set_observations(obs)
    >>> # Setup prior
    >>> prior_mean = np.zeros((2, 1))
    >>> prior_cov = np.eye(2)
    >>> prior = DenseCholeskyMultivariateGaussian(prior_mean, prior_cov, bkd)
    >>> # Create posterior
    >>> posterior = LogUnNormalizedPosterior(model_fn, likelihood, prior, bkd)
    >>> log_post = posterior(np.array([[0.5], [0.5]]))
    """

    def __init__(
        self,
        model_fn: Callable[[Array], Array],
        likelihood: LogLikelihoodProtocol[Array],
        prior: DistributionProtocol[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._model_fn = model_fn
        self._likelihood = likelihood
        self._prior = prior
        self._nvars = prior.nvars()

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of model parameters."""
        return self._nvars

    def model(self) -> Callable[[Array], Array]:
        """Return the model function."""
        return self._model_fn

    def likelihood(self) -> LogLikelihoodProtocol[Array]:
        """Return the likelihood function."""
        return self._likelihood

    def prior(self) -> DistributionProtocol[Array]:
        """Return the prior distribution."""
        return self._prior

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the log unnormalized posterior.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log posterior values. Shape: (nsamples,)
        """
        if samples.ndim == 1:
            samples = self._bkd.reshape(samples, (self._nvars, 1))

        # Evaluate model
        model_outputs = self._model_fn(samples)

        # Log-likelihood
        loglike = self._likelihood.logpdf(model_outputs)

        # Log-prior
        logprior = self._prior.logpdf(samples)

        # Compute sum and flatten to 1D
        result = loglike + logprior
        return self._bkd.flatten(result)

    # TODO: Pyapprox uses dynamic binding and protocols for derivative methods
    # make sure to do the same here, e.g. self.jacobian = self.jacobian if
    # all components needed are defined. Do not ever call finite difference
    # approximation if components not provided jacobian is not binded.
    # Same for other derivatives like hessian, etc
    def jacobian(
        self,
        sample: Array,
        model_jacobian_fn: Optional[Callable[[Array], Array]] = None,
        likelihood_gradient_fn: Optional[Callable[[Array], Array]] = None,
        prior_logpdf_jacobian_fn: Optional[Callable[[Array], Array]] = None,
    ) -> Array:
        """
        Compute gradient of log posterior at a single sample.

        Uses the chain rule:
            d/dtheta log p(theta | y) = J_model^T @ d/d(output) log p(y | output)
                                        + d/dtheta log p(theta)

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1)
        model_jacobian_fn : Callable, optional
            Function to compute model Jacobian. If None, uses finite differences.
        likelihood_gradient_fn : Callable, optional
            Function to compute likelihood gradient w.r.t. model outputs.
            If None, assumes likelihood has a `gradient` method.
        prior_logpdf_jacobian_fn : Callable, optional
            Function to compute prior logpdf gradient.
            If None, uses finite differences.

        Returns
        -------
        Array
            Gradient. Shape: (nvars, 1)
        """
        if sample.ndim == 1:
            sample = self._bkd.reshape(sample, (self._nvars, 1))

        # Evaluate model
        model_output = self._model_fn(sample)

        # Likelihood gradient w.r.t. model output
        if likelihood_gradient_fn is not None:
            dloglike_doutput = likelihood_gradient_fn(model_output)
        elif hasattr(self._likelihood, "gradient"):
            dloglike_doutput = self._likelihood.gradient(model_output)
        else:
            raise ValueError(
                "Likelihood does not have gradient method. "
                "Provide likelihood_gradient_fn."
            )

        # Model Jacobian
        if model_jacobian_fn is not None:
            J_model = model_jacobian_fn(sample)  # (nobs, nvars)
        else:
            J_model = self._finite_diff_jacobian(self._model_fn, sample)

        # Chain rule: J_model^T @ dloglike_doutput
        dloglike_dtheta = J_model.T @ dloglike_doutput

        # Prior gradient
        if prior_logpdf_jacobian_fn is not None:
            dlogprior_dtheta = prior_logpdf_jacobian_fn(sample)
        elif hasattr(self._prior, "logpdf_jacobian"):
            dlogprior_dtheta = self._prior.logpdf_jacobian(sample)
        else:
            # Use finite differences
            dlogprior_dtheta = self._finite_diff_jacobian(
                lambda s: self._prior.logpdf(s)[:, None], sample
            ).T

        return dloglike_dtheta + dlogprior_dtheta

    def hessian(
        self,
        sample: Array,
        model_jacobian_fn: Optional[Callable[[Array], Array]] = None,
        model_hessian_fn: Optional[Callable[[Array, Array], Array]] = None,
        prior_logpdf_hessian_fn: Optional[Callable[[Array], Array]] = None,
    ) -> Array:
        """
        Compute Hessian of log posterior at a single sample.

        For Gaussian likelihood, the Gauss-Newton approximation is:
            H = J^T @ Cov_noise^{-1} @ J + H_prior

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1)
        model_jacobian_fn : Callable, optional
            Function to compute model Jacobian. If None, uses finite differences.
        model_hessian_fn : Callable, optional
            Function to compute model Hessian-vector products.
        prior_logpdf_hessian_fn : Callable, optional
            Function to compute prior Hessian. If None, uses finite differences.

        Returns
        -------
        Array
            Hessian matrix. Shape: (nvars, nvars)
        """
        if sample.ndim == 1:
            sample = self._bkd.reshape(sample, (self._nvars, 1))

        # Model Jacobian
        if model_jacobian_fn is not None:
            J_model = model_jacobian_fn(sample)  # (nobs, nvars)
        else:
            J_model = self._finite_diff_jacobian(self._model_fn, sample)

        # Gauss-Newton Hessian of likelihood (ignoring second-order model terms)
        # H_like = -J^T @ Cov^{-1} @ J
        noise_op = self._likelihood.noise_covariance_operator()
        weighted_J = noise_op.apply_inv(J_model)
        H_like = -weighted_J.T @ noise_op.apply_inv_transpose(weighted_J)

        # Prior Hessian
        if prior_logpdf_hessian_fn is not None:
            H_prior = prior_logpdf_hessian_fn(sample)
        elif hasattr(self._prior, "logpdf_hessian"):
            H_prior = self._prior.logpdf_hessian(sample)
        else:
            # Use finite differences
            def logprior_fn(s):
                return self._prior.logpdf(s)[:, None]

            def jac_fn(s):
                return self._finite_diff_jacobian(logprior_fn, s).T

            H_prior = self._finite_diff_jacobian(jac_fn, sample)

        return H_like + H_prior

    def apply_hessian(
        self,
        sample: Array,
        vec: Array,
        model_jacobian_fn: Optional[Callable[[Array], Array]] = None,
        prior_logpdf_hessian_fn: Optional[Callable[[Array], Array]] = None,
    ) -> Array:
        """
        Apply Hessian-vector product.

        More efficient than forming full Hessian for large problems.

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1)
        vec : Array
            Vector to multiply. Shape: (nvars, 1)
        model_jacobian_fn : Callable, optional
            Function to compute model Jacobian.
        prior_logpdf_hessian_fn : Callable, optional
            Function to compute prior Hessian.

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nvars, 1)
        """
        H = self.hessian(
            sample,
            model_jacobian_fn=model_jacobian_fn,
            prior_logpdf_hessian_fn=prior_logpdf_hessian_fn,
        )
        return H @ vec

    # TODO: pass in configured bindable optimizer so we can use any optimizer
    def maximum_aposteriori_point(
        self,
        initial_guess: Optional[Array] = None,
        method: str = "L-BFGS-B",
        bounds: Optional[Array] = None,
        tol: float = 1e-8,
        options: Optional[dict[str, Any]] = None,
    ) -> Array:
        """
        Find the maximum a posteriori (MAP) point.

        Uses scipy.optimize.minimize to find the mode of the posterior.

        Parameters
        ----------
        initial_guess : Array, optional
            Initial point for optimization. Shape: (nvars, 1)
            If None, uses prior mean if available, else zeros.
        method : str, default="L-BFGS-B"
            Optimization method for scipy.optimize.minimize.
        bounds : Array, optional
            Parameter bounds. Shape: (nvars, 2)
        tol : float, default=1e-8
            Tolerance for optimization convergence.
        options : dict, optional
            Additional options for scipy.optimize.minimize.

        Returns
        -------
        Array
            MAP estimate. Shape: (nvars, 1)
        """
        if initial_guess is None:
            if hasattr(self._prior, "mean"):
                initial_guess = self._prior.mean()
            else:
                initial_guess = self._bkd.zeros((self._nvars, 1))

        x0 = self._bkd.to_numpy(initial_guess).flatten()

        # Objective: negative log posterior (we minimize)
        def objective(x):
            x_arr = self._bkd.asarray(x.reshape(self._nvars, 1))
            neg_logpost = -self._bkd.to_float(self.__call__(x_arr))
            return neg_logpost

        # Setup bounds if provided
        scipy_bounds = None
        if bounds is not None:
            bounds_np = self._bkd.to_numpy(bounds)
            scipy_bounds = [
                (bounds_np[i, 0], bounds_np[i, 1]) for i in range(self._nvars)
            ]

        # Optimize
        opt_options = options or {}
        result = optimize.minimize(
            objective,
            x0,
            method=method,
            bounds=scipy_bounds,
            tol=tol,
            options=opt_options,
        )

        map_point = self._bkd.asarray(result.x.reshape(self._nvars, 1))
        return map_point

    # TODO: delete
    def _finite_diff_jacobian(
        self,
        fn: Callable[[Array], Array],
        x: Array,
        eps: float = 1e-7,
    ) -> Array:
        """
        Compute Jacobian using finite differences.

        Parameters
        ----------
        fn : Callable
            Function to differentiate.
        x : Array
            Point at which to compute Jacobian. Shape: (nvars, 1)
        eps : float
            Finite difference step size.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (noutput, nvars)
        """
        x_np = self._bkd.to_numpy(x)
        f0 = fn(x)
        noutput = f0.shape[0]

        jac = np.zeros((noutput, self._nvars))
        for i in range(self._nvars):
            x_plus = x_np.copy()
            x_plus[i, 0] += eps
            f_plus = fn(self._bkd.asarray(x_plus))
            jac[:, i] = (
                self._bkd.to_numpy(f_plus) - self._bkd.to_numpy(f0)
            ).flatten() / eps

        return self._bkd.asarray(jac)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"LogUnNormalizedPosterior(nvars={self._nvars})"
