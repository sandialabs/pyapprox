"""
Hamiltonian Monte Carlo (HMC) sampler.

This module provides a basic HMC implementation using leapfrog integration.
HMC uses Hamiltonian dynamics to generate proposals that explore the
target distribution more efficiently than random-walk Metropolis.

References
----------
Neal, R. (2011). "MCMC using Hamiltonian dynamics."
    Handbook of Markov Chain Monte Carlo.
"""

from typing import Callable, Generic, Optional, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.inverse.sampling.metropolis import MCMCResult


class HamiltonianMonteCarlo(Generic[Array]):
    """
    Hamiltonian Monte Carlo sampler.

    Uses leapfrog integration to simulate Hamiltonian dynamics and
    Metropolis accept/reject for detailed balance.

    Parameters
    ----------
    log_posterior_fn : Callable[[Array], Array]
        Function that evaluates log posterior. Signature:
        log_posterior_fn(samples) -> log_posterior_values
        where samples has shape (nvars, nsamples).
    gradient_fn : Callable[[Array], Array]
        Function that evaluates gradient of log posterior. Signature:
        gradient_fn(sample) -> gradient
        where sample has shape (nvars, 1) and gradient has shape (nvars, 1).
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.
    step_size : float, default=0.1
        Leapfrog step size (epsilon).
    num_leapfrog_steps : int, default=10
        Number of leapfrog steps per proposal (L).
    mass_matrix : Array, optional
        Mass matrix for kinetic energy. Default is identity.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Gaussian target
    >>> def log_posterior(samples):
    ...     return -0.5 * np.sum(samples**2, axis=0)
    >>> def gradient(sample):
    ...     return -sample
    >>> sampler = HamiltonianMonteCarlo(
    ...     log_posterior, gradient, nvars=2, bkd=bkd,
    ...     step_size=0.1, num_leapfrog_steps=20
    ... )
    >>> result = sampler.sample(nsamples=1000, burn=100)
    """

    def __init__(
        self,
        log_posterior_fn: Callable[[Array], Array],
        gradient_fn: Callable[[Array], Array],
        nvars: int,
        bkd: Backend[Array],
        step_size: float = 0.1,
        num_leapfrog_steps: int = 10,
        mass_matrix: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._log_posterior_fn = log_posterior_fn
        self._gradient_fn = gradient_fn
        self._nvars = nvars
        self._step_size = step_size
        self._num_leapfrog_steps = num_leapfrog_steps

        if mass_matrix is None:
            self._mass_matrix = bkd.asarray(
                np.eye(nvars, dtype=np.float64)
            )
            self._mass_matrix_inv = self._mass_matrix
        else:
            self._mass_matrix = mass_matrix
            self._mass_matrix_inv = bkd.inv(mass_matrix)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def _kinetic_energy(self, momentum: Array) -> float:
        """
        Compute kinetic energy: K(p) = 0.5 * p^T @ M^{-1} @ p

        Parameters
        ----------
        momentum : Array
            Momentum vector. Shape: (nvars, 1)

        Returns
        -------
        float
            Kinetic energy.
        """
        Minv_p = self._mass_matrix_inv @ momentum
        return 0.5 * float(momentum.T @ Minv_p)

    def _leapfrog(
        self,
        position: Array,
        momentum: Array,
    ) -> Tuple[Array, Array]:
        """
        Perform leapfrog integration.

        Parameters
        ----------
        position : Array
            Current position. Shape: (nvars, 1)
        momentum : Array
            Current momentum. Shape: (nvars, 1)

        Returns
        -------
        position : Array
            New position. Shape: (nvars, 1)
        momentum : Array
            New momentum. Shape: (nvars, 1)
        """
        eps = self._step_size
        q = position.copy() if hasattr(position, 'copy') else self._bkd.asarray(
            self._bkd.to_numpy(position).copy()
        )
        p = momentum.copy() if hasattr(momentum, 'copy') else self._bkd.asarray(
            self._bkd.to_numpy(momentum).copy()
        )

        # Half step for momentum
        grad = self._gradient_fn(q)
        p = p + 0.5 * eps * grad

        # Full steps for position and momentum
        for _ in range(self._num_leapfrog_steps - 1):
            q = q + eps * (self._mass_matrix_inv @ p)
            grad = self._gradient_fn(q)
            p = p + eps * grad

        # Final full step for position
        q = q + eps * (self._mass_matrix_inv @ p)

        # Final half step for momentum
        grad = self._gradient_fn(q)
        p = p + 0.5 * eps * grad

        return q, p

    def _hamiltonian(self, position: Array, momentum: Array) -> float:
        """
        Compute Hamiltonian: H(q, p) = -log_posterior(q) + K(p)

        Parameters
        ----------
        position : Array
            Position. Shape: (nvars, 1)
        momentum : Array
            Momentum. Shape: (nvars, 1)

        Returns
        -------
        float
            Hamiltonian value.
        """
        log_post = float(self._log_posterior_fn(position)[0])
        kinetic = self._kinetic_energy(momentum)
        return -log_post + kinetic

    def sample(
        self,
        nsamples: int,
        burn: int = 0,
        initial_state: Optional[Array] = None,
    ) -> MCMCResult[Array]:
        """
        Run HMC sampling.

        Parameters
        ----------
        nsamples : int
            Number of samples to return (after burn-in).
        burn : int, default=0
            Number of burn-in samples to discard.
        initial_state : Array, optional
            Initial position. Shape: (nvars, 1).
            If None, uses zeros.

        Returns
        -------
        MCMCResult
            MCMC results containing samples, acceptance rate, and log posteriors.
        """
        total_samples = nsamples + burn

        # Initialize position
        if initial_state is None:
            current_q = self._bkd.asarray(
                np.zeros((self._nvars, 1), dtype=np.float64)
            )
        else:
            if initial_state.shape != (self._nvars, 1):
                raise ValueError(
                    f"initial_state has wrong shape {initial_state.shape}, "
                    f"expected ({self._nvars}, 1)"
                )
            current_q = initial_state

        # Storage
        samples_np = np.zeros((self._nvars, total_samples))
        logposts_np = np.zeros(total_samples)
        n_accepted = 0

        for i in range(total_samples):
            # Sample momentum from N(0, M)
            p_np = np.random.randn(self._nvars, 1)
            current_p = self._bkd.asarray(p_np.astype(np.float64))
            if self._mass_matrix is not None:
                # p ~ N(0, M), so p = L @ z where z ~ N(0, I) and M = L @ L^T
                L = self._bkd.cholesky(self._mass_matrix)
                current_p = L @ current_p

            # Compute current Hamiltonian
            current_H = self._hamiltonian(current_q, current_p)

            # Leapfrog integration
            proposed_q, proposed_p = self._leapfrog(current_q, current_p)

            # Negate momentum for reversibility
            proposed_p = -proposed_p

            # Compute proposed Hamiltonian
            proposed_H = self._hamiltonian(proposed_q, proposed_p)

            # Metropolis accept/reject
            log_accept_prob = current_H - proposed_H

            if np.log(np.random.uniform()) < log_accept_prob:
                # Accept
                current_q = proposed_q
                if i >= burn:
                    n_accepted += 1

            # Store sample
            samples_np[:, i] = self._bkd.to_numpy(current_q)[:, 0]
            logposts_np[i] = float(self._log_posterior_fn(current_q)[0])

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
            f"HamiltonianMonteCarlo(nvars={self._nvars}, "
            f"step_size={self._step_size}, "
            f"num_leapfrog_steps={self._num_leapfrog_steps})"
        )
