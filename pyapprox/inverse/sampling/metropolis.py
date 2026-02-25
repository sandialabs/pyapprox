"""
Metropolis-Hastings MCMC samplers.

This module provides:
- MetropolisHastingsSampler: Standard Metropolis-Hastings algorithm
- AdaptiveMetropolisSampler: Adaptive Metropolis with covariance estimation
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Tuple

import numpy as np

from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.util.backends.protocols import Array, Backend


@dataclass
class MCMCResult(Generic[Array]):
    """
    Result of MCMC sampling.

    Attributes
    ----------
    samples : Array
        MCMC samples. Shape: (nvars, nsamples)
    acceptance_rate : float
        Fraction of proposals accepted.
    log_posteriors : Array
        Log posterior values at each sample. Shape: (nsamples,)
    """

    samples: Array
    acceptance_rate: float
    log_posteriors: Array


class MetropolisHastingsSampler(Generic[Array]):
    """
    Standard Metropolis-Hastings MCMC sampler.

    Uses a Gaussian proposal distribution centered at the current state.
    Leverages DenseCholeskyMultivariateGaussian from typing.probability
    for efficient proposal generation.

    Parameters
    ----------
    log_posterior_fn : Callable[[Array], Array]
        Function that evaluates log posterior.
        Signature: log_posterior_fn(samples) -> log_posterior_values
        where samples has shape (nvars, nsamples).
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.
    proposal_cov : Array, optional
        Proposal covariance matrix. Shape: (nvars, nvars).
        If None, uses identity matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Simple Gaussian target
    >>> def log_posterior(samples):
    ...     return -0.5 * np.sum(samples**2, axis=0)
    >>> sampler = MetropolisHastingsSampler(log_posterior, nvars=2, bkd=bkd)
    >>> result = sampler.sample(nsamples=1000, burn=100)
    """

    def __init__(
        self,
        log_posterior_fn: Callable[[Array], Array],
        nvars: int,
        bkd: Backend[Array],
        proposal_cov: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._log_posterior_fn = log_posterior_fn
        self._nvars = nvars

        if proposal_cov is None:
            # Use explicit float64 for backend compatibility
            proposal_cov = bkd.asarray(np.eye(nvars, dtype=np.float64))
        else:
            if proposal_cov.shape != (nvars, nvars):
                raise ValueError(
                    f"proposal_cov has wrong shape {proposal_cov.shape}, "
                    f"expected ({nvars}, {nvars})"
                )

        self._proposal_cov = proposal_cov
        # Create proposal distribution (centered at zero - we add current state)
        self._proposal_dist = self._create_proposal_distribution(proposal_cov)

    def _create_proposal_distribution(
        self, proposal_cov: Array
    ) -> DenseCholeskyMultivariateGaussian[Array]:
        """
        Create the proposal distribution.

        Parameters
        ----------
        proposal_cov : Array
            Proposal covariance. Shape: (nvars, nvars)

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            Proposal distribution centered at zero.
        """
        # Use explicit float64 for backend compatibility
        zero_mean = self._bkd.asarray(np.zeros((self._nvars, 1), dtype=np.float64))
        return DenseCholeskyMultivariateGaussian(zero_mean, proposal_cov, self._bkd)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def set_proposal_covariance(self, proposal_cov: Array) -> None:
        """
        Set the proposal covariance matrix.

        Parameters
        ----------
        proposal_cov : Array
            Proposal covariance matrix. Shape: (nvars, nvars)
        """
        if proposal_cov.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"proposal_cov has wrong shape {proposal_cov.shape}, "
                f"expected ({self._nvars}, {self._nvars})"
            )
        self._proposal_cov = proposal_cov
        self._proposal_dist = self._create_proposal_distribution(proposal_cov)

    def proposal_covariance(self) -> Array:
        """
        Return the proposal covariance matrix.

        Returns
        -------
        Array
            Proposal covariance. Shape: (nvars, nvars)
        """
        return self._proposal_cov

    def _propose(self, current: Array) -> Array:
        """
        Generate a proposal from the current state.

        Uses the Gaussian proposal distribution from typing.probability.

        Parameters
        ----------
        current : Array
            Current state. Shape: (nvars, 1)

        Returns
        -------
        Array
            Proposed state. Shape: (nvars, 1)
        """
        # Sample from zero-centered Gaussian and add current state
        increment = self._proposal_dist.rvs(1)
        return current + increment

    def sample(
        self,
        nsamples: int,
        burn: int = 0,
        initial_state: Optional[Array] = None,
        bounds: Optional[Array] = None,
    ) -> MCMCResult[Array]:
        """
        Run MCMC sampling.

        Parameters
        ----------
        nsamples : int
            Number of samples to return (after burn-in).
        burn : int, default=0
            Number of burn-in samples to discard.
        initial_state : Array, optional
            Initial state for the chain. Shape: (nvars, 1).
            If None, uses zeros.
        bounds : Array, optional
            Parameter bounds. Shape: (nvars, 2).
            Proposals outside bounds are rejected.

        Returns
        -------
        MCMCResult
            MCMC results containing samples, acceptance rate, and log posteriors.
        """
        total_samples = nsamples + burn

        # Initialize
        if initial_state is None:
            current = self._bkd.asarray(np.zeros((self._nvars, 1), dtype=np.float64))
        else:
            if initial_state.shape != (self._nvars, 1):
                raise ValueError(
                    f"initial_state has wrong shape {initial_state.shape}, "
                    f"expected ({self._nvars}, 1)"
                )
            current = initial_state

        current_logpost = float(self._log_posterior_fn(current)[0])

        # Storage
        samples_np = np.zeros((self._nvars, total_samples))
        logposts_np = np.zeros(total_samples)
        n_accepted = 0

        for i in range(total_samples):
            # Propose
            proposal = self._propose(current)

            # Check bounds if provided
            if bounds is not None:
                bounds_np = self._bkd.to_numpy(bounds)
                proposal_np = self._bkd.to_numpy(proposal)
                in_bounds = np.all(proposal_np[:, 0] >= bounds_np[:, 0]) and np.all(
                    proposal_np[:, 0] <= bounds_np[:, 1]
                )
                if not in_bounds:
                    # Reject - stay at current
                    samples_np[:, i] = self._bkd.to_numpy(current)[:, 0]
                    logposts_np[i] = current_logpost
                    continue

            # Evaluate log posterior at proposal
            proposal_logpost = float(self._log_posterior_fn(proposal)[0])

            # Acceptance probability (log scale)
            log_alpha = proposal_logpost - current_logpost

            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                # Accept
                current = proposal
                current_logpost = proposal_logpost
                if i >= burn:
                    n_accepted += 1

            samples_np[:, i] = self._bkd.to_numpy(current)[:, 0]
            logposts_np[i] = current_logpost

        # Remove burn-in
        samples_np = samples_np[:, burn:]
        logposts_np = logposts_np[burn:]

        acceptance_rate = n_accepted / nsamples if nsamples > 0 else 0.0

        return MCMCResult(
            samples=self._bkd.asarray(samples_np),
            acceptance_rate=acceptance_rate,
            log_posteriors=self._bkd.asarray(logposts_np),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MetropolisHastingsSampler(nvars={self._nvars})"


class AdaptiveMetropolisSampler(MetropolisHastingsSampler[Array]):
    """
    Adaptive Metropolis sampler with online covariance estimation.

    Updates the proposal covariance based on the history of accepted samples,
    following Haario et al. (2001). Uses DenseCholeskyMultivariateGaussian
    from typing.probability for efficient proposal generation.

    Parameters
    ----------
    log_posterior_fn : Callable[[Array], Array]
        Function that evaluates log posterior.
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.
    proposal_cov : Array, optional
        Initial proposal covariance matrix.
    adaptation_start : int, default=100
        Start adapting after this many samples.
    adaptation_interval : int, default=50
        Update proposal covariance every this many samples.
    scaling_factor : float, optional
        Scaling factor for proposal covariance.
        Default is 2.4^2 / nvars (optimal for Gaussian targets).

    References
    ----------
    Haario, H., Saksman, E., and Tamminen, J. (2001).
    "An Adaptive Metropolis Algorithm."
    Bernoulli, 7(2), 223-242.
    """

    def __init__(
        self,
        log_posterior_fn: Callable[[Array], Array],
        nvars: int,
        bkd: Backend[Array],
        proposal_cov: Optional[Array] = None,
        adaptation_start: int = 100,
        adaptation_interval: int = 50,
        scaling_factor: Optional[float] = None,
    ):
        super().__init__(log_posterior_fn, nvars, bkd, proposal_cov)
        self._adaptation_start = adaptation_start
        self._adaptation_interval = adaptation_interval

        if scaling_factor is None:
            self._scaling_factor = 2.4**2 / nvars
        else:
            self._scaling_factor = scaling_factor

        self._nugget = 1e-6  # Regularization for covariance

    def sample(
        self,
        nsamples: int,
        burn: int = 0,
        initial_state: Optional[Array] = None,
        bounds: Optional[Array] = None,
    ) -> MCMCResult[Array]:
        """
        Run adaptive MCMC sampling.

        Parameters
        ----------
        nsamples : int
            Number of samples to return (after burn-in).
        burn : int, default=0
            Number of burn-in samples to discard.
        initial_state : Array, optional
            Initial state for the chain. Shape: (nvars, 1).
        bounds : Array, optional
            Parameter bounds. Shape: (nvars, 2).

        Returns
        -------
        MCMCResult
            MCMC results containing samples, acceptance rate, and log posteriors.
        """
        total_samples = nsamples + burn

        # Initialize
        if initial_state is None:
            current = self._bkd.asarray(np.zeros((self._nvars, 1), dtype=np.float64))
        else:
            if initial_state.shape != (self._nvars, 1):
                raise ValueError(
                    f"initial_state has wrong shape {initial_state.shape}, "
                    f"expected ({self._nvars}, 1)"
                )
            current = initial_state

        current_logpost = float(self._log_posterior_fn(current)[0])

        # Storage
        samples_np = np.zeros((self._nvars, total_samples))
        logposts_np = np.zeros(total_samples)
        n_accepted = 0

        # Online mean and covariance estimation
        sample_mean = self._bkd.to_numpy(current).copy()
        sample_cov = np.zeros((self._nvars, self._nvars))

        for i in range(total_samples):
            # Propose
            proposal = self._propose(current)

            # Check bounds if provided
            if bounds is not None:
                bounds_np = self._bkd.to_numpy(bounds)
                proposal_np = self._bkd.to_numpy(proposal)
                in_bounds = np.all(proposal_np[:, 0] >= bounds_np[:, 0]) and np.all(
                    proposal_np[:, 0] <= bounds_np[:, 1]
                )
                if not in_bounds:
                    samples_np[:, i] = self._bkd.to_numpy(current)[:, 0]
                    logposts_np[i] = current_logpost
                    # Update statistics even for rejected samples
                    sample_mean, sample_cov = self._update_statistics(
                        current, sample_mean, sample_cov, i
                    )
                    continue

            # Evaluate log posterior at proposal
            proposal_logpost = float(self._log_posterior_fn(proposal)[0])

            # Acceptance probability
            log_alpha = proposal_logpost - current_logpost

            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                current = proposal
                current_logpost = proposal_logpost
                if i >= burn:
                    n_accepted += 1

            samples_np[:, i] = self._bkd.to_numpy(current)[:, 0]
            logposts_np[i] = current_logpost

            # Update running statistics
            sample_mean, sample_cov = self._update_statistics(
                current, sample_mean, sample_cov, i
            )

            # Adapt proposal covariance
            if (
                i >= self._adaptation_start
                and (i - self._adaptation_start) % self._adaptation_interval == 0
            ):
                self._adapt_proposal(sample_cov)

        # Remove burn-in
        samples_np = samples_np[:, burn:]
        logposts_np = logposts_np[burn:]

        acceptance_rate = n_accepted / nsamples if nsamples > 0 else 0.0

        return MCMCResult(
            samples=self._bkd.asarray(samples_np),
            acceptance_rate=acceptance_rate,
            log_posteriors=self._bkd.asarray(logposts_np),
        )

    def _update_statistics(
        self,
        new_sample: Array,
        mean: np.ndarray,
        cov: np.ndarray,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update running mean and covariance using Welford's algorithm.

        Parameters
        ----------
        new_sample : Array
            New sample. Shape: (nvars, 1)
        mean : np.ndarray
            Current running mean. Shape: (nvars, 1)
        cov : np.ndarray
            Current running covariance. Shape: (nvars, nvars)
        n : int
            Current sample count (0-indexed).

        Returns
        -------
        mean : np.ndarray
            Updated mean.
        cov : np.ndarray
            Updated covariance.
        """
        sample_np = self._bkd.to_numpy(new_sample)

        if n == 0:
            return sample_np.copy(), np.zeros((self._nvars, self._nvars))

        # Update mean
        delta = sample_np - mean
        new_mean = mean + delta / (n + 1)

        # Update covariance (Welford's online algorithm)
        new_cov = (n - 1) / n * cov
        new_cov += self._scaling_factor * self._nugget * np.eye(self._nvars) / n
        new_cov += self._scaling_factor * (delta @ delta.T) / (n + 1)

        return new_mean, new_cov

    def _adapt_proposal(self, sample_cov: np.ndarray) -> None:
        """
        Adapt the proposal covariance based on sample history.

        Uses DenseCholeskyMultivariateGaussian from typing.probability
        to create the new proposal distribution.

        Parameters
        ----------
        sample_cov : np.ndarray
            Empirical covariance of samples. Shape: (nvars, nvars)
        """
        # Add regularization to ensure positive definiteness
        regularized_cov = sample_cov + self._nugget * np.eye(self._nvars)

        try:
            new_proposal_cov = self._bkd.asarray(regularized_cov.astype(np.float64))
            # Create new proposal distribution using typing.probability
            self._proposal_cov = new_proposal_cov
            self._proposal_dist = self._create_proposal_distribution(new_proposal_cov)
        except Exception:
            # If Cholesky fails in the Gaussian constructor, don't update
            pass

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AdaptiveMetropolisSampler(nvars={self._nvars}, "
            f"adaptation_start={self._adaptation_start})"
        )
