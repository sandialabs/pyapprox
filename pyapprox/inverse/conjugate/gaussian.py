"""
Gaussian conjugate posterior for linear observation models.

For a linear model with Gaussian prior and Gaussian noise,
the posterior is exactly Gaussian (no approximation needed).

This was previously called "Laplace" in pyapprox.inference but
has been renamed to clarify it's an exact conjugate solution,
not the Laplace approximation for nonlinear models.
"""

from typing import Generic, Optional
import math

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator


def _compute_expected_kl_divergence(
    prior_mean: Array,
    prior_cov: Array,
    posterior_covariance: Array,
    nu_vec: Array,
    Cmat: Array,
    bkd: Backend[Array],
    prior_hessian: Optional[Array] = None,
) -> float:
    """
    Compute the expected KL divergence between Gaussian posterior and prior.

    The expectation is taken with respect to the data distribution.
    This is useful for computing expected information gain in experimental design.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nvars, 1)
    prior_cov : Array
        Prior covariance. Shape: (nvars, nvars)
    posterior_covariance : Array
        Posterior covariance. Shape: (nvars, nvars)
    nu_vec : Array
        Expected posterior mean. Shape: (nvars, 1)
    Cmat : Array
        Covariance of posterior mean. Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.
    prior_hessian : Array, optional
        Inverse of prior covariance (for efficiency if pre-computed).

    Returns
    -------
    float
        E_data[KL(posterior || prior)]
    """
    if prior_hessian is None:
        prior_hessian = bkd.inv(prior_cov)
    nvars = posterior_covariance.shape[0]

    # KL = 0.5 * (tr(Sigma_prior^{-1} Sigma_post) - nvars
    #            + log(|Sigma_prior|/|Sigma_post|)
    #            + (mu_prior - mu_post)^T Sigma_prior^{-1} (mu_prior - mu_post))
    # Taking expectation over data adds the variance term
    kl_div = bkd.trace(prior_hessian @ posterior_covariance) - nvars

    # Use slogdet for numerical stability
    _, log_det_prior = bkd.slogdet(prior_cov)
    _, log_det_post = bkd.slogdet(posterior_covariance)
    kl_div = kl_div + float(log_det_prior) - float(log_det_post)

    kl_div = kl_div + bkd.trace(prior_hessian @ Cmat)
    xi = prior_mean - nu_vec
    kl_div = kl_div + float((xi.T @ prior_hessian @ xi)[0, 0])
    kl_div = 0.5 * kl_div
    return float(kl_div)


class DenseGaussianConjugatePosterior(Generic[Array]):
    """
    Gaussian conjugate posterior for linear observation models.

    For the linear model:
        observations = A @ parameters + offset + noise
        noise ~ N(0, noise_covariance)
        parameters ~ N(prior_mean, prior_covariance)

    The posterior is exactly Gaussian:
        parameters | observations ~ N(posterior_mean, posterior_covariance)

    Parameters
    ----------
    observation_matrix : Array
        The matrix A in the linear model. Shape: (nobs, nvars)
    prior_mean : Array
        Mean of the Gaussian prior. Shape: (nvars, 1)
    prior_covariance : Array
        Covariance of the Gaussian prior. Shape: (nvars, nvars)
    noise_covariance : Array
        Covariance of the observation noise. Shape: (nobs, nobs)
    bkd : Backend[Array]
        Computational backend.
    observation_offset : Array, optional
        Offset b in the linear model A @ x + b. Shape: (nobs, 1)

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> A = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> prior_mean = np.zeros((2, 1))
    >>> prior_cov = np.eye(2)
    >>> noise_cov = 0.1 * np.eye(2)
    >>> solver = DenseGaussianConjugatePosterior(
    ...     A, prior_mean, prior_cov, noise_cov, bkd
    ... )
    >>> obs = np.array([[1.0], [1.5]])
    >>> solver.compute(obs)
    >>> posterior_mean = solver.posterior_mean()
    """

    def __init__(
        self,
        observation_matrix: Array,
        prior_mean: Array,
        prior_covariance: Array,
        noise_covariance: Array,
        bkd: Backend[Array],
        observation_offset: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._nobs, self._nvars = observation_matrix.shape
        self._matrix = observation_matrix

        if prior_mean.shape != (self._nvars, 1):
            raise ValueError(
                f"prior_mean has wrong shape {prior_mean.shape}, "
                f"expected ({self._nvars}, 1)"
            )
        self._prior_mean = prior_mean

        if prior_covariance.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"prior_covariance has wrong shape {prior_covariance.shape}, "
                f"expected ({self._nvars}, {self._nvars})"
            )
        self._prior_cov = prior_covariance

        if noise_covariance.shape != (self._nobs, self._nobs):
            raise ValueError(
                f"noise_covariance has wrong shape {noise_covariance.shape}, "
                f"expected ({self._nobs}, {self._nobs})"
            )
        self._noise_cov = noise_covariance

        if observation_offset is None:
            observation_offset = bkd.zeros((self._nobs, 1))
        if observation_offset.shape != (self._nobs, 1):
            raise ValueError(
                f"observation_offset has wrong shape {observation_offset.shape}, "
                f"expected ({self._nobs}, 1)"
            )
        self._offset = observation_offset

        # Pre-compute inverses for efficiency
        self._noise_cov_inv = bkd.inv(self._noise_cov)
        self._prior_hessian = bkd.inv(self._prior_cov)

        # Create prior distribution object
        self._prior = DenseCholeskyMultivariateGaussian(
            self._prior_mean, self._prior_cov, bkd
        )

        # State
        self._obs: Optional[Array] = None
        self._posterior_mean: Optional[Array] = None
        self._posterior_cov: Optional[Array] = None
        self._evidence: Optional[float] = None
        self._kl_div: Optional[float] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nobs(self) -> int:
        """Return the number of observations per experiment."""
        return self._nobs

    def compute(self, obs: Array) -> None:
        """
        Compute the posterior given observations.

        Parameters
        ----------
        obs : Array
            Observations. Shape: (nobs, nexperiments)
        """
        if obs.ndim == 1:
            obs = self._bkd.reshape(obs, (self._nobs, 1))
        if obs.shape[0] != self._nobs:
            raise ValueError(
                f"obs has wrong shape {obs.shape}, expected ({self._nobs}, nexperiments)"
            )
        self._obs = obs
        nexperiments = obs.shape[1]

        # Misfit Hessian: A^T @ Sigma_noise^{-1} @ A
        misfit_hessian = self._matrix.T @ self._noise_cov_inv @ self._matrix

        # Posterior covariance: (n * H_misfit + H_prior)^{-1}
        # where n is number of experiments
        self._posterior_cov = self._bkd.inv(
            nexperiments * misfit_hessian + self._prior_hessian
        )

        # Posterior mean
        obs_sum = self._bkd.sum(obs, axis=1)[:, None]
        residual = (
            obs_sum - nexperiments * self._matrix @ self._prior_mean
            - nexperiments * self._offset
        )
        temp = self._matrix.T @ (self._noise_cov_inv @ residual)
        self._posterior_mean = self._prior_mean + self._posterior_cov @ temp

        # Compute additional quantities
        self._compute_evidence()
        self._compute_expected_posterior_statistics()
        self._compute_expected_kl_divergence()

    def _compute_evidence(self) -> None:
        """
        Compute the model evidence (marginal likelihood).

        References
        ----------
        Friel, N. and Wyse, J. (2012), Estimating the evidence – a review.
        Statistica Neerlandica, 66: 288-308.
        """
        # Evidence = (2*pi)^{n/2} * |Sigma_post|^{1/2} * L(mu_post) * p(mu_post)
        # where L is likelihood and p is prior
        # Use slogdet for numerical stability
        _, log_det_post = self._bkd.slogdet(self._posterior_cov)

        # Log-likelihood at posterior mean
        residual = self._obs - self._matrix @ self._posterior_mean - self._offset
        # For multiple experiments, sum log-likelihoods
        nexperiments = self._obs.shape[1]
        log_like = 0.0
        _, log_det_noise = self._bkd.slogdet(self._noise_cov)
        for i in range(nexperiments):
            r = residual[:, i:i+1]
            quad = float((r.T @ self._noise_cov_inv @ r)[0, 0])
            log_like -= 0.5 * (
                self._nobs * math.log(2 * math.pi) + float(log_det_noise) + quad
            )

        # Log-prior at posterior mean
        diff = self._posterior_mean - self._prior_mean
        _, log_det_prior = self._bkd.slogdet(self._prior_cov)
        log_prior = -0.5 * (
            self._nvars * math.log(2 * math.pi)
            + float(log_det_prior)
            + float((diff.T @ self._prior_hessian @ diff)[0, 0])
        )

        # Evidence (in log form for stability, then exponentiate)
        log_evidence = (
            0.5 * self._nvars * math.log(2 * math.pi)
            + 0.5 * float(log_det_post)
            + log_like
            + log_prior
        )
        self._evidence = math.exp(log_evidence)

    def _compute_expected_posterior_statistics(self) -> None:
        """
        Compute expected posterior statistics.

        The posterior mean is itself a random variable (depends on data).
        This computes its mean and variance with respect to data distribution.
        """
        Rmat = self._posterior_cov @ self._matrix.T @ self._noise_cov_inv
        ROmat = Rmat @ self._matrix
        self._nu_vec = (
            ROmat @ self._prior_mean
            + self._posterior_cov @ self._prior_hessian @ self._prior_mean
        )
        self._Cmat = (
            ROmat @ self._prior_cov @ ROmat.T
            + Rmat @ self._noise_cov @ Rmat.T
        )

    def _compute_expected_kl_divergence(self) -> None:
        """Compute expected KL divergence between posterior and prior."""
        self._kl_div = _compute_expected_kl_divergence(
            self._prior_mean,
            self._prior_cov,
            self._posterior_cov,
            self._nu_vec,
            self._Cmat,
            self._bkd,
            self._prior_hessian,
        )

    def posterior_mean(self) -> Array:
        """
        Return the posterior mean.

        Returns
        -------
        Array
            Posterior mean. Shape: (nvars, 1)
        """
        if self._posterior_mean is None:
            raise RuntimeError("Must call compute() first")
        return self._posterior_mean

    def posterior_covariance(self) -> Array:
        """
        Return the posterior covariance.

        Returns
        -------
        Array
            Posterior covariance. Shape: (nvars, nvars)
        """
        if self._posterior_cov is None:
            raise RuntimeError("Must call compute() first")
        return self._posterior_cov

    def evidence(self) -> float:
        """
        Return the model evidence (marginal likelihood).

        Returns
        -------
        float
            Model evidence p(observations).
        """
        if self._evidence is None:
            raise RuntimeError("Must call compute() first")
        return self._evidence

    def expected_kl_divergence(self) -> float:
        """
        Return the expected KL divergence between posterior and prior.

        Returns
        -------
        float
            E_data[KL(posterior || prior)]
        """
        if self._kl_div is None:
            raise RuntimeError("Must call compute() first")
        return self._kl_div

    def posterior_variable(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """
        Return the posterior as a Gaussian distribution.

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            Posterior Gaussian distribution.
        """
        return DenseCholeskyMultivariateGaussian(
            self.posterior_mean(),
            self.posterior_covariance(),
            self._bkd,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DenseGaussianConjugatePosterior(nvars={self._nvars}, nobs={self._nobs})"
