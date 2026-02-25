"""
Protocols for conjugate posterior distributions.

Conjugate priors allow analytical computation of the posterior distribution.
When the prior and likelihood belong to the same parametric family,
the posterior can be computed exactly without approximation.
"""

from typing import Any, Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ConjugatePosteriorProtocol(Protocol, Generic[Array]):
    """
    Base protocol for conjugate posterior distributions.

    This protocol defines the interface for any conjugate prior solver
    that computes exact posterior distributions given observations.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Get number of prior/posterior variables.
    nobs()
        Get number of observations per experiment.
    compute(obs)
        Compute posterior given observations.
    evidence()
        Return the marginal likelihood (model evidence).
    posterior_variable()
        Return posterior as a distribution object.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the number of prior/posterior variables.

        Returns
        -------
        int
            Number of variables in the prior/posterior.
        """
        ...

    def nobs(self) -> int:
        """
        Return the number of observations per experiment.

        Returns
        -------
        int
            Number of observations.
        """
        ...

    def compute(self, obs: Array) -> None:
        """
        Compute posterior given observations.

        Parameters
        ----------
        obs : Array
            Observations. Shape: (nobs, nexperiments)
        """
        ...

    def evidence(self) -> float:
        """
        Return the marginal likelihood (model evidence).

        The evidence p(obs) is the normalizing constant of the posterior:
            p(obs) = integral p(obs|theta) p(theta) d(theta)

        Returns
        -------
        float
            Model evidence.
        """
        ...

    def posterior_variable(self) -> Any:
        """
        Return posterior as a distribution object.

        Returns
        -------
        Any
            Posterior distribution (type depends on conjugate pair).
        """
        ...


@runtime_checkable
class GaussianConjugatePosteriorProtocol(Protocol, Generic[Array]):
    """
    Extended protocol for Gaussian conjugate posteriors.

    For linear observation models with Gaussian prior and noise,
    the posterior is exactly Gaussian. This protocol adds methods
    specific to Gaussian posteriors.

    This extends ConjugatePosteriorProtocol with:
    - posterior_mean(): Get posterior mean vector
    - posterior_covariance(): Get posterior covariance matrix
    - expected_kl_divergence(): Get E_data[KL(posterior || prior)]
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """Return the number of prior/posterior variables."""
        ...

    def nobs(self) -> int:
        """Return the number of observations per experiment."""
        ...

    def compute(self, obs: Array) -> None:
        """Compute posterior given observations."""
        ...

    def evidence(self) -> float:
        """Return the marginal likelihood (model evidence)."""
        ...

    def posterior_variable(self) -> Any:
        """Return posterior as a Gaussian distribution."""
        ...

    def posterior_mean(self) -> Array:
        """
        Return the posterior mean.

        Returns
        -------
        Array
            Posterior mean. Shape: (nvars, 1)
        """
        ...

    def posterior_covariance(self) -> Array:
        """
        Return the posterior covariance.

        Returns
        -------
        Array
            Posterior covariance matrix. Shape: (nvars, nvars)
        """
        ...

    def expected_kl_divergence(self) -> float:
        """
        Return the expected KL divergence between posterior and prior.

        The expectation is taken with respect to the data distribution.
        This is useful for experimental design (expected information gain).

        Returns
        -------
        float
            E_data[KL(posterior || prior)]
        """
        ...
