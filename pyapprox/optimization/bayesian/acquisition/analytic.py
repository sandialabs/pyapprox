"""Analytic acquisition functions for single-objective Bayesian optimization.

All acquisition functions are stateless and operate on a single output
(nqoi=1). They return values where higher is always better.
"""

from typing import Generic

from pyapprox.optimization.bayesian.math_utils import normal_cdf, normal_pdf
from pyapprox.optimization.bayesian.protocols import AcquisitionContext
from pyapprox.util.backends.protocols import Array


def _get_mean_and_std_jacobians(sample, ctx):
    """Compute d_mu/dx and d_sigma/dx for a single sample.

    Parameters
    ----------
    sample : Array
        Single input, shape (nvars, 1).
    ctx : AcquisitionContext
        Context with fitted surrogate.

    Returns
    -------
    d_mu_dx : Array or None
        Mean jacobian, shape (1, nvars). None if not available.
    d_sigma_dx : Array or None
        Std jacobian, shape (1, nvars). None if not available.
    """
    surrogate = ctx.surrogate
    d_mu_dx = None
    d_sigma_dx = None

    if hasattr(surrogate, "jacobian"):
        d_mu_dx = surrogate.jacobian(sample)  # (nqoi, nvars) -> (1, nvars)

    if hasattr(surrogate, "predict_std_jacobian"):
        d_sigma_dx = surrogate.predict_std_jacobian(sample)  # (1, nvars)

    return d_mu_dx, d_sigma_dx


class ExpectedImprovement(Generic[Array]):
    """Expected Improvement acquisition function.

    EI(x) = (best - mu) * Phi(Z) + sigma * phi(Z)
    where Z = (best - mu) / sigma.

    Returns 0 where sigma <= jitter.

    Parameters
    ----------
    jitter : float
        Minimum sigma threshold. Default 1e-6.
    """

    def __init__(self, jitter: float = 1e-6) -> None:
        self._jitter = jitter

    def evaluate(self, X: Array, ctx: AcquisitionContext[Array]) -> Array:
        """Evaluate EI at candidate points.

        Parameters
        ----------
        X : Array
            Candidate points, shape (nvars, n).
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate and best value.

        Returns
        -------
        Array
            EI values, shape (n,). Higher is better.
        """
        bkd = ctx.bkd
        mu = ctx.surrogate.predict(X)  # (nqoi, n)
        sigma = ctx.surrogate.predict_std(X)  # (nqoi, n)

        # Use first output only (nqoi=1)
        mu = mu[0]  # (n,)
        sigma = sigma[0]  # (n,)

        # Standard EI:
        #   minimize: improvement = f_best_min - mu (positive when mu < best)
        #   maximize: improvement = mu - f_best_max (positive when mu > best)
        # best_value stores the raw best: min(y) or max(y) respectively
        if ctx.minimize:
            improvement = ctx.best_value[0] - mu
        else:
            improvement = mu - ctx.best_value[0]

        # Mask where sigma is too small
        mask = sigma > self._jitter

        # Safe Z computation (avoid division by zero)
        safe_sigma = bkd.where(mask, sigma, bkd.ones_like(sigma))
        Z = improvement / safe_sigma

        ei = improvement * normal_cdf(Z, bkd) + safe_sigma * normal_pdf(Z, bkd)
        # Zero out where sigma <= jitter
        ei = bkd.where(mask, ei, bkd.zeros_like(ei))

        return ei

    def jacobian(
        self, sample: Array, ctx: AcquisitionContext[Array]
    ) -> Array:
        """Compute Jacobian of EI at a single point.

        dEI/dx = d_improvement/dx * Phi(Z) + phi(Z) * d_sigma/dx

        Parameters
        ----------
        sample : Array
            Single input, shape (nvars, 1).
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate and best value.

        Returns
        -------
        Array
            Jacobian, shape (1, nvars).
        """
        bkd = ctx.bkd
        mu = ctx.surrogate.predict(sample)[0, 0]  # scalar
        sigma = ctx.surrogate.predict_std(sample)[0, 0]  # scalar

        if sigma <= self._jitter:
            nvars = sample.shape[0]
            return bkd.zeros((1, nvars))

        if ctx.minimize:
            improvement = ctx.best_value[0] - mu
        else:
            improvement = mu - ctx.best_value[0]

        Z = improvement / sigma

        d_mu_dx, d_sigma_dx = _get_mean_and_std_jacobians(sample, ctx)

        # d_improvement/dx
        if ctx.minimize:
            d_imp_dx = -d_mu_dx  # (1, nvars)
        else:
            d_imp_dx = d_mu_dx  # (1, nvars)

        # dEI/dx = d_imp/dx * Phi(Z) + phi(Z) * d_sigma/dx
        phi_Z = normal_pdf(bkd.reshape(Z, (1,)), bkd)[0]
        Phi_Z = normal_cdf(bkd.reshape(Z, (1,)), bkd)[0]

        result = d_imp_dx * Phi_Z
        if d_sigma_dx is not None:
            result = result + d_sigma_dx * phi_Z

        return result


class UpperConfidenceBound(Generic[Array]):
    """Upper Confidence Bound acquisition function.

    UCB(x) = sign * mu + beta * sigma
    where sign = -1 if minimizing, +1 if maximizing.

    Parameters
    ----------
    beta : float
        Exploration-exploitation trade-off parameter. Default 2.0.
    """

    def __init__(self, beta: float = 2.0) -> None:
        self._beta = beta

    def evaluate(self, X: Array, ctx: AcquisitionContext[Array]) -> Array:
        """Evaluate UCB at candidate points.

        Parameters
        ----------
        X : Array
            Candidate points, shape (nvars, n).
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate.

        Returns
        -------
        Array
            UCB values, shape (n,). Higher is better.
        """
        mu = ctx.surrogate.predict(X)[0]  # (n,)
        sigma = ctx.surrogate.predict_std(X)[0]  # (n,)
        sign = -1.0 if ctx.minimize else 1.0
        return sign * mu + self._beta * sigma

    def jacobian(
        self, sample: Array, ctx: AcquisitionContext[Array]
    ) -> Array:
        """Compute Jacobian of UCB at a single point.

        dUCB/dx = sign * d_mu/dx + beta * d_sigma/dx

        Parameters
        ----------
        sample : Array
            Single input, shape (nvars, 1).
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate.

        Returns
        -------
        Array
            Jacobian, shape (1, nvars).
        """
        d_mu_dx, d_sigma_dx = _get_mean_and_std_jacobians(sample, ctx)

        sign = -1.0 if ctx.minimize else 1.0
        result = sign * d_mu_dx  # (1, nvars)
        if d_sigma_dx is not None:
            result = result + self._beta * d_sigma_dx

        return result


class ProbabilityOfImprovement(Generic[Array]):
    """Probability of Improvement acquisition function.

    PI(x) = Phi(Z) where Z = (best - mu) / sigma.

    Returns 0 where sigma <= jitter.

    Parameters
    ----------
    jitter : float
        Minimum sigma threshold. Default 1e-6.
    """

    def __init__(self, jitter: float = 1e-6) -> None:
        self._jitter = jitter

    def evaluate(self, X: Array, ctx: AcquisitionContext[Array]) -> Array:
        """Evaluate PI at candidate points.

        Parameters
        ----------
        X : Array
            Candidate points, shape (nvars, n).
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate and best value.

        Returns
        -------
        Array
            PI values, shape (n,). Higher is better.
        """
        bkd = ctx.bkd
        mu = ctx.surrogate.predict(X)[0]  # (n,)
        sigma = ctx.surrogate.predict_std(X)[0]  # (n,)

        if ctx.minimize:
            improvement = ctx.best_value[0] - mu
        else:
            improvement = mu - ctx.best_value[0]

        mask = sigma > self._jitter
        safe_sigma = bkd.where(mask, sigma, bkd.ones_like(sigma))
        Z = improvement / safe_sigma

        pi = normal_cdf(Z, bkd)
        pi = bkd.where(mask, pi, bkd.zeros_like(pi))

        return pi

    def jacobian(
        self, sample: Array, ctx: AcquisitionContext[Array]
    ) -> Array:
        """Compute Jacobian of PI at a single point.

        dPI/dx = phi(Z) * dZ/dx
        dZ/dx = (d_imp/dx * sigma - improvement * d_sigma/dx) / sigma^2

        Parameters
        ----------
        sample : Array
            Single input, shape (nvars, 1).
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate and best value.

        Returns
        -------
        Array
            Jacobian, shape (1, nvars).
        """
        bkd = ctx.bkd
        mu = ctx.surrogate.predict(sample)[0, 0]  # scalar
        sigma = ctx.surrogate.predict_std(sample)[0, 0]  # scalar

        if sigma <= self._jitter:
            nvars = sample.shape[0]
            return bkd.zeros((1, nvars))

        if ctx.minimize:
            improvement = ctx.best_value[0] - mu
        else:
            improvement = mu - ctx.best_value[0]

        Z = improvement / sigma

        d_mu_dx, d_sigma_dx = _get_mean_and_std_jacobians(sample, ctx)

        if ctx.minimize:
            d_imp_dx = -d_mu_dx  # (1, nvars)
        else:
            d_imp_dx = d_mu_dx  # (1, nvars)

        # dZ/dx = (d_imp/dx * sigma - improvement * d_sigma/dx) / sigma^2
        dZ_dx = d_imp_dx / sigma
        if d_sigma_dx is not None:
            dZ_dx = dZ_dx - (improvement / (sigma * sigma)) * d_sigma_dx

        phi_Z = normal_pdf(bkd.reshape(Z, (1,)), bkd)[0]

        return phi_Z * dZ_dx
