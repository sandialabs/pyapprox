"""
Delayed Rejection Adaptive Metropolis (DRAM) sampler.

This module provides DRAM, which combines:
1. Delayed Rejection: If a proposal is rejected, try a second proposal
   with a smaller step size before giving up.
2. Adaptive Metropolis: Update the proposal covariance based on the
   sample history for improved mixing.

References
----------
Haario, H., Laine, M., Mira, A., and Saksman, E. (2006).
"DRAM: Efficient adaptive MCMC."
Statistics and Computing, 16(4), 339-354.
"""

from typing import Callable, Generic, Optional, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.inverse.sampling.metropolis import (
    AdaptiveMetropolisSampler,
    MCMCResult,
)


class DelayedRejectionAdaptiveMetropolis(AdaptiveMetropolisSampler[Array]):
    """
    Delayed Rejection Adaptive Metropolis (DRAM) sampler.

    Extends AdaptiveMetropolisSampler with delayed rejection:
    if the first proposal is rejected, a second proposal with
    a scaled-down covariance is tried.

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
        Default is 2.4^2 / nvars.
    dr_scale : float, default=0.1
        Scale factor for the second (delayed rejection) proposal.
        The second proposal uses covariance = dr_scale * proposal_cov.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> def log_posterior(samples):
    ...     return -0.5 * np.sum(samples**2, axis=0)
    >>> sampler = DelayedRejectionAdaptiveMetropolis(
    ...     log_posterior, nvars=2, bkd=bkd
    ... )
    >>> result = sampler.sample(nsamples=1000, burn=100)
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
        dr_scale: float = 0.1,
    ):
        super().__init__(
            log_posterior_fn,
            nvars,
            bkd,
            proposal_cov,
            adaptation_start,
            adaptation_interval,
            scaling_factor,
        )
        self._dr_scale = dr_scale
        # Create second stage proposal distribution
        self._update_dr_proposal()

    def _update_dr_proposal(self) -> None:
        """Update the delayed rejection proposal distribution."""
        dr_cov = self._dr_scale * self._proposal_cov
        zero_mean = self._bkd.asarray(
            np.zeros((self._nvars, 1), dtype=np.float64)
        )
        try:
            self._dr_proposal_dist = DenseCholeskyMultivariateGaussian(
                zero_mean, dr_cov, self._bkd
            )
        except Exception:
            # If Cholesky fails, just keep the old one
            pass

    def _adapt_proposal(self, sample_cov: np.ndarray) -> None:
        """Adapt the proposal covariance and update DR proposal."""
        super()._adapt_proposal(sample_cov)
        self._update_dr_proposal()

    def _delayed_rejection_step(
        self,
        current: Array,
        current_logpost: float,
        first_proposal: Array,
        first_logpost: float,
        first_log_alpha: float,
    ) -> Tuple[Array, float, bool]:
        """
        Perform delayed rejection step.

        Parameters
        ----------
        current : Array
            Current state. Shape: (nvars, 1)
        current_logpost : float
            Log posterior at current state.
        first_proposal : Array
            First (rejected) proposal. Shape: (nvars, 1)
        first_logpost : float
            Log posterior at first proposal.
        first_log_alpha : float
            Log acceptance probability for first proposal.

        Returns
        -------
        new_state : Array
            New state (either second proposal or current).
        new_logpost : float
            Log posterior at new state.
        accepted : bool
            Whether second proposal was accepted.
        """
        # Generate second proposal from current with scaled covariance
        increment = self._dr_proposal_dist.rvs(1)
        second_proposal = current + increment

        # Evaluate log posterior at second proposal
        second_logpost = float(self._log_posterior_fn(second_proposal)[0])

        # Compute log alpha for first proposal from second proposal
        # (reverse direction)
        first_from_second_log_alpha = first_logpost - second_logpost

        # Compute correction terms for detailed balance
        # See Haario et al. (2006) for derivation
        log_one_minus_alpha1 = np.log(1.0 - min(1.0, np.exp(first_log_alpha)))
        log_one_minus_alpha1_rev = np.log(
            1.0 - min(1.0, np.exp(first_from_second_log_alpha))
        )

        # Second stage acceptance probability
        log_alpha2 = (
            second_logpost
            - current_logpost
            + log_one_minus_alpha1_rev
            - log_one_minus_alpha1
        )

        if np.log(np.random.uniform()) < log_alpha2:
            return second_proposal, second_logpost, True
        else:
            return current, current_logpost, False

    def sample(
        self,
        nsamples: int,
        burn: int = 0,
        initial_state: Optional[Array] = None,
        bounds: Optional[Array] = None,
    ) -> MCMCResult[Array]:
        """
        Run DRAM sampling.

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
            Proposals outside bounds are rejected.

        Returns
        -------
        MCMCResult
            MCMC results containing samples, acceptance rate, and log posteriors.
        """
        total_samples = nsamples + burn

        # Initialize
        if initial_state is None:
            current = self._bkd.asarray(
                np.zeros((self._nvars, 1), dtype=np.float64)
            )
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
        n_dr_accepted = 0  # Track delayed rejection acceptances

        # Online mean and covariance estimation
        sample_mean = self._bkd.to_numpy(current).copy()
        sample_cov = np.zeros((self._nvars, self._nvars))

        for i in range(total_samples):
            # First proposal
            proposal = self._propose(current)

            # Check bounds if provided
            in_bounds = True
            if bounds is not None:
                bounds_np = self._bkd.to_numpy(bounds)
                proposal_np = self._bkd.to_numpy(proposal)
                in_bounds = (
                    np.all(proposal_np[:, 0] >= bounds_np[:, 0])
                    and np.all(proposal_np[:, 0] <= bounds_np[:, 1])
                )

            if not in_bounds:
                # Reject - stay at current
                samples_np[:, i] = self._bkd.to_numpy(current)[:, 0]
                logposts_np[i] = current_logpost
                sample_mean, sample_cov = self._update_statistics(
                    current, sample_mean, sample_cov, i
                )
                continue

            # Evaluate log posterior at proposal
            proposal_logpost = float(self._log_posterior_fn(proposal)[0])

            # First stage acceptance probability
            log_alpha = proposal_logpost - current_logpost

            if np.log(np.random.uniform()) < log_alpha:
                # Accept first proposal
                current = proposal
                current_logpost = proposal_logpost
                if i >= burn:
                    n_accepted += 1
            else:
                # First proposal rejected - try delayed rejection
                new_state, new_logpost, dr_accepted = (
                    self._delayed_rejection_step(
                        current,
                        current_logpost,
                        proposal,
                        proposal_logpost,
                        log_alpha,
                    )
                )

                if dr_accepted:
                    current = new_state
                    current_logpost = new_logpost
                    if i >= burn:
                        n_accepted += 1
                        n_dr_accepted += 1

            samples_np[:, i] = self._bkd.to_numpy(current)[:, 0]
            logposts_np[i] = current_logpost

            # Update running statistics
            sample_mean, sample_cov = self._update_statistics(
                current, sample_mean, sample_cov, i
            )

            # Adapt proposal covariance
            if (
                i >= self._adaptation_start
                and (i - self._adaptation_start) % self._adaptation_interval
                == 0
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

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DelayedRejectionAdaptiveMetropolis(nvars={self._nvars}, "
            f"dr_scale={self._dr_scale})"
        )
