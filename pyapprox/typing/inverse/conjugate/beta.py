"""
Beta conjugate posterior for Bernoulli likelihood.

For binary observations with Bernoulli likelihood and Beta prior,
the posterior is exactly Beta.

Prior: p ~ Beta(alpha, beta)
Likelihood: obs_i ~ Bernoulli(p)
Posterior: p | obs ~ Beta(alpha + sum(obs), beta + n - sum(obs))
"""

from typing import Generic, Optional, Any
import math

from scipy.special import gammaln, betaln

from pyapprox.typing.util.backends.protocols import Array, Backend


def _log_beta_function(a: float, b: float) -> float:
    """Compute log of Beta function B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)."""
    return betaln(a, b)


class BetaConjugatePosterior(Generic[Array]):
    """
    Beta conjugate posterior for Bernoulli likelihood.

    For independent binary observations:
        obs_i ~ Bernoulli(p)
        p ~ Beta(alpha, beta)

    The posterior is:
        p | obs ~ Beta(alpha + sum(obs), beta + n - sum(obs))

    Parameters
    ----------
    alpha : float
        First shape parameter of Beta prior.
    beta : float
        Second shape parameter of Beta prior.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> solver = BetaConjugatePosterior(1.0, 1.0, bkd)  # Uniform prior
    >>> obs = np.array([[1, 1, 0, 1, 0]])  # 5 coin flips, 3 heads
    >>> solver.compute(obs)
    >>> # Posterior is Beta(1+3, 1+2) = Beta(4, 3)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        bkd: Backend[Array],
    ):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive")

        self._bkd = bkd
        self._alpha_prior = alpha
        self._beta_prior = beta

        # State
        self._obs: Optional[Array] = None
        self._alpha_post: Optional[float] = None
        self._beta_post: Optional[float] = None
        self._evidence: Optional[float] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (always 1 for Beta)."""
        return 1

    def nobs(self) -> int:
        """Return the number of observations."""
        if self._obs is None:
            return 0
        return self._obs.shape[1] if self._obs.ndim == 2 else self._obs.shape[0]

    def compute(self, obs: Array) -> None:
        """
        Compute the posterior given observations.

        Parameters
        ----------
        obs : Array
            Binary observations (0 or 1). Shape: (1, n) or (n,)
        """
        if obs.ndim == 1:
            obs = self._bkd.reshape(obs, (1, -1))
        if obs.shape[0] != 1:
            raise ValueError(
                f"obs must have shape (1, n) for Beta posterior, got {obs.shape}"
            )
        self._obs = obs

        # Count successes and failures
        n = obs.shape[1]
        successes = float(self._bkd.sum(obs))
        failures = n - successes

        # Posterior parameters
        self._alpha_post = self._alpha_prior + successes
        self._beta_post = self._beta_prior + failures

        # Compute evidence
        self._compute_evidence(n, successes)

    def _compute_evidence(self, n: int, successes: float) -> None:
        """
        Compute the model evidence.

        p(obs) = B(alpha + s, beta + f) / B(alpha, beta)

        where s = successes, f = failures = n - s
        """
        log_evidence = (
            _log_beta_function(self._alpha_post, self._beta_post)
            - _log_beta_function(self._alpha_prior, self._beta_prior)
        )
        self._evidence = math.exp(log_evidence)

    def posterior_alpha(self) -> float:
        """
        Return the posterior alpha parameter.

        Returns
        -------
        float
            Posterior alpha = prior_alpha + sum(obs)
        """
        if self._alpha_post is None:
            raise RuntimeError("Must call compute() first")
        return self._alpha_post

    def posterior_beta(self) -> float:
        """
        Return the posterior beta parameter.

        Returns
        -------
        float
            Posterior beta = prior_beta + n - sum(obs)
        """
        if self._beta_post is None:
            raise RuntimeError("Must call compute() first")
        return self._beta_post

    def posterior_mean(self) -> float:
        """
        Return the posterior mean.

        For Beta(a, b), mean = a / (a + b)

        Returns
        -------
        float
            Posterior mean.
        """
        if self._alpha_post is None:
            raise RuntimeError("Must call compute() first")
        return self._alpha_post / (self._alpha_post + self._beta_post)

    def posterior_variance(self) -> float:
        """
        Return the posterior variance.

        For Beta(a, b), var = ab / ((a+b)^2 (a+b+1))

        Returns
        -------
        float
            Posterior variance.
        """
        if self._alpha_post is None:
            raise RuntimeError("Must call compute() first")
        a, b = self._alpha_post, self._beta_post
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def evidence(self) -> float:
        """
        Return the model evidence.

        Returns
        -------
        float
            Model evidence p(observations).
        """
        if self._evidence is None:
            raise RuntimeError("Must call compute() first")
        return self._evidence

    def posterior_variable(self) -> Any:
        """
        Return the posterior as a distribution.

        Returns scipy.stats.beta distribution object.

        Returns
        -------
        scipy.stats.rv_frozen
            Beta posterior distribution.
        """
        from scipy import stats
        if self._alpha_post is None:
            raise RuntimeError("Must call compute() first")
        return stats.beta(self._alpha_post, self._beta_post)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._alpha_post is None:
            return f"BetaConjugatePosterior(alpha={self._alpha_prior}, beta={self._beta_prior})"
        return (
            f"BetaConjugatePosterior(posterior_alpha={self._alpha_post:.2f}, "
            f"posterior_beta={self._beta_post:.2f})"
        )
