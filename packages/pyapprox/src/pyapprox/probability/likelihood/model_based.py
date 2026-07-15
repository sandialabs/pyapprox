"""
Model-based log-likelihood.

Composes a forward model with a noise-model likelihood into a single
parameter-to-log-likelihood object.
"""

from typing import Generic, Optional

from pyapprox.interface.functions.derivatives import Derivatives, JacobianFn
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.probability.protocols.likelihood import (
    LogLikelihoodHasDesignWeightsProtocol,
    LogLikelihoodHasRVSProtocol,
    LogLikelihoodProtocol,
    VectorizedLogLikelihoodProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.likelihood import (
    ...     DiagonalGaussianLogLikelihood,
    ...     ModelBasedLogLikelihood,
    ... )
    >>> from pyapprox.interface.functions.fromcallable import (
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
                f"model must satisfy FunctionProtocol, got {type(model).__name__}"
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

        # Capture optional capability once at construction; absence is
        # None, never a missing attribute
        self._noise_jac: Optional[JacobianFn[Array]] = None
        self._model_jac: Optional[JacobianFn[Array]] = None
        if isinstance(noise_likelihood, ObjectiveProtocol) and isinstance(
            model, ObjectiveProtocol
        ):
            noise_jac = noise_likelihood.derivatives().jacobian
            model_jac = model.derivatives().jacobian
            if noise_jac is not None and model_jac is not None:
                self._noise_jac = noise_jac
                self._model_jac = model_jac
        if self._model_jac is not None:
            self._derivs: Derivatives[Array] = Derivatives.first_order(
                jacobian=self._jacobian
            )
        else:
            self._derivs = Derivatives.none()

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle (jacobian w.r.t. parameters)."""
        return self._derivs

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

    def rvs(self, parameters: Array, nsamples: int = 1) -> Array:
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
        noise_likelihood = self._noise_likelihood
        if not isinstance(noise_likelihood, LogLikelihoodHasRVSProtocol):
            raise RuntimeError(
                "rvs is unavailable; the noise likelihood cannot sample"
            )
        model_outputs = self._model(parameters)
        return noise_likelihood.rvs(model_outputs, nsamples)

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
        noise_jac = self._noise_jac
        model_jac = self._model_jac
        if noise_jac is None or model_jac is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        model_output = self._model(sample)
        # likelihood bundle jacobian: (1, nobs)
        like_jac = noise_jac(model_output)
        # model jacobian: (nqoi, nvars) = (nobs, nvars)
        J_model = model_jac(sample)
        # chain rule: (1, nobs) @ (nobs, nvars) = (1, nvars)
        return like_jac @ J_model

    def logpdf_vectorized(self, parameters: Array, observations: Array) -> Array:
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
        noise_likelihood = self._noise_likelihood
        if not isinstance(noise_likelihood, VectorizedLogLikelihoodProtocol):
            raise RuntimeError(
                "logpdf_vectorized is unavailable; the noise likelihood "
                "does not support batched evaluation"
            )
        model_outputs = self._model(parameters)
        return noise_likelihood.logpdf_vectorized(model_outputs, observations)

    def set_design_weights(self, weights: Array) -> None:
        """
        Set weights for experimental design.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs,)
        """
        noise_likelihood = self._noise_likelihood
        if not isinstance(
            noise_likelihood, LogLikelihoodHasDesignWeightsProtocol
        ):
            raise RuntimeError(
                "set_design_weights is unavailable; the noise likelihood "
                "does not support design weights"
            )
        noise_likelihood.set_design_weights(weights)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ModelBasedLogLikelihood("
            f"nvars={self.nvars()}, nobs={self.nobs()}, "
            f"model={type(self._model).__name__}, "
            f"noise={type(self._noise_likelihood).__name__})"
        )
