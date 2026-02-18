"""
Model-based log-likelihood.

Composes a forward model with a noise-model likelihood into a single
parameter-to-log-likelihood object.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.probability.protocols.likelihood import (
    LogLikelihoodProtocol,
)


class ModelBasedLogLikelihood(Generic[Array]):
    """
    Compose a forward model with a noise-model likelihood.

    Given a model ``f: params -> model_outputs`` and a noise likelihood
    ``p(obs | model_outputs)``, this class provides a single object that
    evaluates ``log p(obs | f(params))`` and its derivatives via the
    chain rule.

    Parameters
    ----------
    model : FunctionProtocol[Array]
        Forward model mapping parameters to model outputs.
        Must have ``nqoi() == noise_likelihood.nobs()``.
    noise_likelihood : LogLikelihoodProtocol[Array]
        Noise-model log-likelihood evaluating ``log p(obs | model_outputs)``.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.likelihood import (
    ...     DiagonalGaussianLogLikelihood,
    ...     ModelBasedLogLikelihood,
    ... )
    >>> from pyapprox.typing.interface.functions.fromcallable import (
    ...     FunctionFromCallable,
    ... )
    >>> bkd = NumpyBkd()
    >>> model = FunctionFromCallable(
    ...     nqoi=2, nvars=2,
    ...     fun=lambda x: x,
    ...     bkd=bkd,
    ... )
    >>> noise_var = bkd.asarray([0.01, 0.01])
    >>> noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
    >>> composed = ModelBasedLogLikelihood(model, noise_lik, bkd)
    >>> composed.nobs()
    2
    >>> composed.nvars()
    2
    """

    def __init__(
        self,
        model: FunctionProtocol[Array],
        noise_likelihood: LogLikelihoodProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(model, FunctionProtocol):
            raise TypeError(
                "model must satisfy FunctionProtocol, "
                f"got {type(model).__name__}"
            )
        if not isinstance(noise_likelihood, LogLikelihoodProtocol):
            raise TypeError(
                "noise_likelihood must satisfy LogLikelihoodProtocol, "
                f"got {type(noise_likelihood).__name__}"
            )
        if model.nqoi() != noise_likelihood.nobs():
            raise ValueError(
                f"model.nqoi() ({model.nqoi()}) must equal "
                f"noise_likelihood.nobs() ({noise_likelihood.nobs()})"
            )
        self._model = model
        self._noise_likelihood = noise_likelihood
        self._bkd = bkd

        # Dynamically bind optional methods
        if hasattr(noise_likelihood, "rvs"):
            self.rvs = self._rvs
        if hasattr(noise_likelihood, "gradient") and hasattr(model, "jacobian"):
            self.jacobian = self._jacobian
            self.gradient = self._gradient
        if hasattr(noise_likelihood, "logpdf_vectorized"):
            self.logpdf_vectorized = self._logpdf_vectorized
        if hasattr(noise_likelihood, "set_design_weights"):
            self.set_design_weights = self._set_design_weights

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def model(self) -> FunctionProtocol[Array]:
        """Get the wrapped forward model."""
        return self._model

    def noise_likelihood(self) -> LogLikelihoodProtocol[Array]:
        """Get the wrapped noise-model likelihood."""
        return self._noise_likelihood

    def nvars(self) -> int:
        """Return the number of input parameters."""
        return self._model.nvars()

    def nobs(self) -> int:
        """Return the number of observations."""
        return self._noise_likelihood.nobs()

    def set_observations(self, obs: Array) -> None:
        """
        Set the observed data.

        Parameters
        ----------
        obs : Array
            Observed data. Shape depends on the noise likelihood.
        """
        self._noise_likelihood.set_observations(obs)

    def logpdf(self, parameters: Array) -> Array:
        """
        Evaluate the composed log-likelihood.

        Computes ``log p(obs | f(parameters))``.

        Parameters
        ----------
        parameters : Array
            Model parameters. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log-likelihood values. Shape: (1, nsamples)
        """
        model_outputs = self._model(parameters)
        return self._noise_likelihood.logpdf(model_outputs)

    def __call__(self, parameters: Array) -> Array:
        """Alias for logpdf."""
        return self.logpdf(parameters)

    def _rvs(self, parameters: Array, nsamples: int = 1) -> Array:
        """
        Sample from the likelihood given model parameters.

        Parameters
        ----------
        parameters : Array
            Model parameters. Shape: (nvars, nsamples)
        nsamples : int
            Number of noise samples per parameter sample.

        Returns
        -------
        Array
            Noisy observations. Shape: (nobs, nsamples * n_param_samples)
        """
        model_outputs = self._model(parameters)
        return self._noise_likelihood.rvs(model_outputs, nsamples)

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute Jacobian of log-likelihood w.r.t. parameters.

        Uses the chain rule:
            d(logpdf)/d(params) = d(logpdf)/d(shapes)^T @ d(shapes)/d(params)
                                = gradient(f(p))^T @ model.jacobian(p)

        where gradient has shape (nobs, 1) and model.jacobian has shape
        (nobs, nvars), giving result shape (1, nvars).

        Parameters
        ----------
        sample : Array
            Single parameter sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian of log-likelihood. Shape: (1, nvars)
        """
        model_output = self._model(sample)
        # gradient: (nobs, 1)
        grad_wrt_outputs = self._noise_likelihood.gradient(model_output)
        # model jacobian: (nqoi, nvars) = (nobs, nvars)
        J_model = self._model.jacobian(sample)
        # chain rule: (1, nobs) @ (nobs, nvars) = (1, nvars)
        return grad_wrt_outputs.T @ J_model

    def _gradient(self, sample: Array) -> Array:
        """
        Compute gradient of log-likelihood w.r.t. parameters.

        Parameters
        ----------
        sample : Array
            Single parameter sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Gradient. Shape: (nvars, 1)
        """
        return self._jacobian(sample).T

    def _logpdf_vectorized(
        self, parameters: Array, observations: Array
    ) -> Array:
        """
        Batched log-likelihood evaluation.

        Computes ``log p(obs | f(params))`` for all combinations.

        Parameters
        ----------
        parameters : Array
            Model parameters. Shape: (nvars, n_param_samples)
        observations : Array
            Observed data. Shape: (nobs, n_obs_samples)

        Returns
        -------
        Array
            Log-likelihood matrix. Shape: (n_param_samples, n_obs_samples)
        """
        model_outputs = self._model(parameters)
        return self._noise_likelihood.logpdf_vectorized(
            model_outputs, observations
        )

    def _set_design_weights(self, weights: Array) -> None:
        """
        Set weights for experimental design.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs,)
        """
        self._noise_likelihood.set_design_weights(weights)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ModelBasedLogLikelihood("
            f"nvars={self.nvars()}, nobs={self.nobs()}, "
            f"model={type(self._model).__name__}, "
            f"noise={type(self._noise_likelihood).__name__})"
        )
