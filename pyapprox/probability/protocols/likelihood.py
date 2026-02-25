"""
Protocols for likelihood functions.

These protocols define the interface for likelihood functions used in
Bayesian inference and optimal experimental design.

Protocol Hierarchy
------------------
LogLikelihoodProtocol
    Base protocol for log-likelihood evaluation.
GaussianLogLikelihoodProtocol
    Specialized for Gaussian likelihoods with noise covariance.
VectorizedLogLikelihoodProtocol
    Supports batched evaluation for OED.
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.protocols.covariance import (
    SqrtCovarianceOperatorProtocol,
)


@runtime_checkable
class LogLikelihoodProtocol(Protocol, Generic[Array]):
    """
    Base protocol for log-likelihood functions.

    A likelihood function evaluates p(observations | model_outputs).

    Methods
    -------
    bkd()
        Get the computational backend.
    set_observations(obs)
        Set the observed data.
    logpdf(model_outputs)
        Evaluate log-likelihood.
    nobs()
        Number of observations.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def set_observations(self, obs: Array) -> None:
        """
        Set the observed data.

        Parameters
        ----------
        obs : Array
            Observed data. Shape depends on likelihood type.
        """
        ...

    def logpdf(self, model_outputs: Array) -> Array:
        """
        Evaluate the log-likelihood.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, nsamples)

        Returns
        -------
        Array
            Log-likelihood values. Shape: (nsamples,)
        """
        ...

    def nobs(self) -> int:
        """
        Return the number of observations.

        Returns
        -------
        int
            Number of observations.
        """
        ...


@runtime_checkable
class GaussianLogLikelihoodProtocol(Protocol, Generic[Array]):
    """
    Protocol for Gaussian log-likelihood functions.

    For Gaussian noise model:
        obs = model_output + noise, noise ~ N(0, noise_cov)

    Log-likelihood:
        log p(obs | model) = -0.5 * ||L^{-1}(obs - model)||^2 - log|L| - n/2*log(2*pi)

    where noise_cov = L @ L.T.

    Methods
    -------
    noise_covariance_operator()
        Get the noise covariance operator.
    rvs(model_outputs)
        Sample from the likelihood (add noise to model outputs).
    set_design_weights(weights)
        Set weights for experimental design.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def set_observations(self, obs: Array) -> None:
        ...

    def logpdf(self, model_outputs: Array) -> Array:
        ...

    def nobs(self) -> int:
        ...

    def noise_covariance_operator(self) -> SqrtCovarianceOperatorProtocol[Array]:
        """
        Get the noise covariance operator.

        Returns
        -------
        SqrtCovarianceOperatorProtocol
            The sqrt covariance operator for the noise.
        """
        ...

    def rvs(self, model_outputs: Array) -> Array:
        """
        Sample from the likelihood (add noise to model outputs).

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, nsamples)

        Returns
        -------
        Array
            Noisy observations. Shape: (nobs, nsamples)
        """
        ...

    def set_design_weights(self, weights: Array) -> None:
        """
        Set weights for experimental design.

        Weights scale the contribution of each observation to the likelihood.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs,)
        """
        ...


@runtime_checkable
class VectorizedLogLikelihoodProtocol(Protocol, Generic[Array]):
    """
    Protocol for vectorized log-likelihood evaluation.

    Supports batched evaluation for optimal experimental design (OED)
    where we need to evaluate many (observation, model_output) pairs.

    The vectorization can be over:
    - Observations (fixed model outputs, varying observations)
    - Model outputs (fixed observations, varying model outputs)
    - Both (full outer product)

    Methods
    -------
    logpdf_vectorized(model_outputs, observations)
        Batched log-likelihood evaluation.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def logpdf_vectorized(
        self, model_outputs: Array, observations: Array
    ) -> Array:
        """
        Batched log-likelihood evaluation.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, n_model_samples)
        observations : Array
            Observed data. Shape: (nobs, n_obs_samples)

        Returns
        -------
        Array
            Log-likelihood values.
            Shape depends on implementation:
            - Fixed mean: (n_obs_samples,) if model_outputs is single sample
            - Fixed input: (n_model_samples,) if observations is single sample
            - Full: (n_model_samples, n_obs_samples)
        """
        ...
