"""
Dirichlet conjugate posterior for multinomial likelihood.

For categorical observations with multinomial likelihood and Dirichlet prior,
the posterior is exactly Dirichlet.

Prior: p ~ Dirichlet(alpha_1, ..., alpha_K)
Likelihood: obs ~ Multinomial(n, p)
Posterior: p | obs ~ Dirichlet(alpha_1 + count_1, ..., alpha_K + count_K)
"""

from typing import Generic, Optional, Any
import math

from scipy.special import gammaln

from pyapprox.typing.util.backends.protocols import Array, Backend


def _log_multivariate_beta(alphas) -> float:
    """
    Compute log of multivariate Beta function.

    B(alpha) = prod_k Gamma(alpha_k) / Gamma(sum_k alpha_k)
    log B(alpha) = sum_k log Gamma(alpha_k) - log Gamma(sum_k alpha_k)
    """
    return float(sum(gammaln(a) for a in alphas) - gammaln(sum(alphas)))


class DirichletConjugatePosterior(Generic[Array]):
    """
    Dirichlet conjugate posterior for multinomial likelihood.

    For categorical observations over K categories:
        obs ~ Multinomial(n, p)
        p ~ Dirichlet(alpha_1, ..., alpha_K)

    The posterior is:
        p | obs ~ Dirichlet(alpha_1 + count_1, ..., alpha_K + count_K)

    where count_k is the number of observations in category k.

    Parameters
    ----------
    alphas : Array
        Shape parameters of Dirichlet prior. Shape: (K,)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # 3 categories with uniform prior
    >>> solver = DirichletConjugatePosterior(np.array([1.0, 1.0, 1.0]), bkd)
    >>> # Observations: category indices (0, 1, 2)
    >>> obs = np.array([[0, 0, 1, 2, 0]])  # 3 in cat 0, 1 in cat 1, 1 in cat 2
    >>> solver.compute(obs)
    >>> # Posterior is Dirichlet(1+3, 1+1, 1+1) = Dirichlet(4, 2, 2)
    """

    def __init__(
        self,
        alphas: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd

        # Ensure alphas is 1D
        alphas_np = bkd.to_numpy(alphas).flatten()
        if any(a <= 0 for a in alphas_np):
            raise ValueError("All alpha values must be positive")

        self._alphas_prior = alphas_np.copy()
        self._K = len(alphas_np)

        # State
        self._obs: Optional[Array] = None
        self._counts: Optional[Array] = None
        self._alphas_post: Optional[Array] = None
        self._evidence: Optional[float] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of categories."""
        return self._K

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
            Category observations (integers 0 to K-1). Shape: (1, n) or (n,)
        """
        if obs.ndim == 1:
            obs = self._bkd.reshape(obs, (1, -1))
        if obs.shape[0] != 1:
            raise ValueError(
                f"obs must have shape (1, n) for Dirichlet posterior, got {obs.shape}"
            )
        self._obs = obs

        # Count observations in each category
        obs_np = self._bkd.to_numpy(obs).flatten().astype(int)
        counts = [0.0] * self._K
        for cat in obs_np:
            if cat < 0 or cat >= self._K:
                raise ValueError(f"Observation {cat} out of range [0, {self._K - 1}]")
            counts[cat] += 1
        self._counts = counts

        # Posterior parameters
        self._alphas_post = [
            self._alphas_prior[k] + counts[k] for k in range(self._K)
        ]

        # Compute evidence
        self._compute_evidence()

    def _compute_evidence(self) -> None:
        """
        Compute the model evidence.

        p(obs) = B(alpha + counts) / B(alpha) * multinomial coefficient

        For the conjugate case:
        log p(obs) = log B(alpha_post) - log B(alpha_prior) + log(n! / prod count_k!)
        """
        n = sum(self._counts)
        log_multinomial = gammaln(n + 1) - sum(gammaln(c + 1) for c in self._counts)

        log_evidence = (
            _log_multivariate_beta(self._alphas_post)
            - _log_multivariate_beta(self._alphas_prior)
            + log_multinomial
        )
        self._evidence = math.exp(log_evidence)

    def posterior_alphas(self) -> Array:
        """
        Return the posterior alpha parameters.

        Returns
        -------
        Array
            Posterior alphas. Shape: (K,)
        """
        if self._alphas_post is None:
            raise RuntimeError("Must call compute() first")
        return self._bkd.asarray(self._alphas_post)

    def posterior_mean(self) -> Array:
        """
        Return the posterior mean.

        For Dirichlet(alpha), mean_k = alpha_k / sum(alpha)

        Returns
        -------
        Array
            Posterior mean. Shape: (K,)
        """
        if self._alphas_post is None:
            raise RuntimeError("Must call compute() first")
        total = sum(self._alphas_post)
        return self._bkd.asarray([a / total for a in self._alphas_post])

    def posterior_variance(self) -> Array:
        """
        Return the posterior marginal variances.

        For Dirichlet(alpha), var_k = alpha_k(S - alpha_k) / (S^2(S+1))
        where S = sum(alpha)

        Returns
        -------
        Array
            Posterior variances. Shape: (K,)
        """
        if self._alphas_post is None:
            raise RuntimeError("Must call compute() first")
        S = sum(self._alphas_post)
        variances = [
            a * (S - a) / (S * S * (S + 1)) for a in self._alphas_post
        ]
        return self._bkd.asarray(variances)

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

        Returns scipy.stats.dirichlet distribution object.

        Returns
        -------
        scipy.stats.rv_frozen
            Dirichlet posterior distribution.
        """
        from scipy import stats
        if self._alphas_post is None:
            raise RuntimeError("Must call compute() first")
        return stats.dirichlet(self._alphas_post)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._alphas_post is None:
            return f"DirichletConjugatePosterior(K={self._K}, alphas_prior={self._alphas_prior})"
        return f"DirichletConjugatePosterior(K={self._K}, alphas_post={self._alphas_post})"
