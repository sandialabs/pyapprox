"""Morris screening sensitivity analysis.

This module provides Morris screening for sensitivity analysis,
computing elementary effects to rank variable importance.

References
----------
Morris, M.D. Factorial Sampling Plans for Preliminary Computational
Experiments. Technometrics 33(2):161-174, 1991.

Campolongo, F., Cariboni, J., Saltelli, A. An effective screening design
for sensitivity analysis of large models. Environmental Modelling &
Software 22(10):1509-1518, 2007.
"""

import math
from itertools import combinations
from typing import Generic, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.spatial.distance import cdist

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class MarginalWithPPF(Protocol, Generic[Array]):
    """Protocol for marginal distributions with percent point function."""

    def ppf(self, probs: Array) -> Array:
        """Inverse CDF (percent point function)."""
        ...


@runtime_checkable
class DistributionWithMarginals(Protocol, Generic[Array]):
    """Protocol for joint distributions with accessible marginals."""

    def nvars(self) -> int:
        """Number of random variables."""
        ...

    def marginals(self) -> Sequence[MarginalWithPPF[Array]]:
        """Return the marginal distributions."""
        ...


class MorrisSensitivityAnalysis(Generic[Array]):
    """Morris screening sensitivity analysis.

    Computes elementary effects along Morris trajectories to estimate
    the sensitivity of model outputs to inputs. Morris screening is a
    computationally efficient method for identifying important variables
    in high-dimensional problems.

    The method computes three statistics:
    - mu: Mean elementary effect (can cancel for non-monotonic functions)
    - mu_star: Mean absolute elementary effect (primary importance measure)
    - sigma: Standard deviation of elementary effects (indicates
    nonlinearity/interactions)

    Parameters
    ----------
    distribution : DistributionWithMarginals[Array]
        Joint distribution with marginals that have `ppf()` methods.
    nlevels : int
        Number of levels for the Morris grid. Must be even.
        Common choices: 4, 6, 8.
    bkd : Backend[Array]
        Backend for array operations.
    eps : float, optional
        Epsilon for grid bounds [eps, 1-eps]. Default is 0.
        Needed for unbounded distributions to avoid infinite values.

    Examples
    --------
    >>> from pyapprox.sensitivity.screening import MorrisSensitivityAnalysis
    >>> # Assuming `dist` has marginals with ppf() methods
    >>> morris = MorrisSensitivityAnalysis(dist, nlevels=4, bkd=bkd)
    >>> samples = morris.generate_samples(ntrajectories=10)
    >>> values = my_function(samples)  # Shape: (nqoi, nsamples)
    >>> morris.compute(values)
    >>> mu_star = morris.mu_star()  # Variable importance
    >>> sigma = morris.sigma()  # Nonlinearity/interaction indicator
    """

    def __init__(
        self,
        distribution: DistributionWithMarginals[Array],
        nlevels: int,
        bkd: Backend[Array],
        eps: float = 0.0,
    ) -> None:
        if not isinstance(distribution, DistributionWithMarginals):
            raise TypeError(
                "distribution must satisfy DistributionWithMarginals, "
                f"got {type(distribution).__name__}"
            )
        if nlevels % 2 != 0:
            raise ValueError("nlevels must be an even integer")
        self._distribution = distribution
        self._nvars = distribution.nvars()
        self._nlevels = nlevels
        self._eps = eps
        self._bkd = bkd
        self._samples: Optional[Array] = None
        self._ntrajectories: int = 0
        self._elem_effects: Optional[Array] = None
        self._mu: Optional[Array] = None
        self._mu_star: Optional[Array] = None
        self._sigma: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the backend used for array operations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def _get_trajectory(self) -> Array:
        """Compute a single Morris trajectory.

        Returns
        -------
        Array
            Shape (nvars, nvars+1) - the trajectory samples in [0, 1].
        """
        delta = self._nlevels / ((self._nlevels - 1) * 2)
        samples_1d = self._bkd.linspace(self._eps, 1 - self._eps, self._nlevels)

        initial_point = self._bkd.asarray(
            np.random.choice(self._bkd.to_numpy(samples_1d), self._nvars)
        )
        shifts = self._bkd.diag(
            self._bkd.asarray(np.random.choice([-delta, delta], self._nvars))
        )
        trajectory = self._bkd.zeros((self._nvars, self._nvars + 1))
        trajectory[:, 0] = initial_point
        for ii in range(self._nvars):
            trajectory[:, ii + 1] = self._bkd.copy(trajectory[:, ii])
            if (trajectory[ii, ii] - delta) >= 0 and (trajectory[ii, ii] + delta) <= 1:
                trajectory[ii, ii + 1] += shifts[ii, ii]
            elif (trajectory[ii, ii] - delta) >= 0:
                trajectory[ii, ii + 1] -= delta
            elif (trajectory[ii, ii] + delta) <= 1:
                trajectory[ii, ii + 1] += delta
            else:
                raise RuntimeError("Invalid trajectory state")
        return trajectory

    def _downselect_trajectories(
        self, candidate_samples: Array, ntrajectories: int
    ) -> Array:
        """Select trajectories that best fill the space.

        Uses a combinatorial optimization to find the set of trajectories
        with maximum sum of pairwise distances.

        Parameters
        ----------
        candidate_samples : Array
            Shape (ncandidates, nvars, nvars+1) - candidate trajectories.
        ntrajectories : int
            Number of trajectories to select.

        Returns
        -------
        Array
            Shape (nvars, ntrajectories * (nvars + 1)) - selected samples.
        """
        ncandidate_trajectories = candidate_samples.shape[0]
        distances = self._bkd.zeros((ncandidate_trajectories, ncandidate_trajectories))
        for ii in range(ncandidate_trajectories):
            for jj in range(ii + 1):
                distances[ii, jj] = float(
                    cdist(
                        self._bkd.to_numpy(candidate_samples[ii]).T,
                        self._bkd.to_numpy(candidate_samples[jj]).T,
                    ).sum()
                )
                distances[jj, ii] = distances[ii, jj]

        get_combinations = combinations(range(ncandidate_trajectories), ntrajectories)
        best_index: Optional[tuple[int, ...]] = None
        best_value = -np.inf
        for index in get_combinations:
            # Use math.sqrt since we're working with Python floats
            sum_sq = sum(
                float(distances[ix[0], ix[1]]) ** 2 for ix in combinations(index, 2)
            )
            value = math.sqrt(sum_sq)
            if value > best_value:
                best_value = value
                best_index = index

        assert best_index is not None
        samples = self._bkd.hstack([candidate_samples[ii, :, :] for ii in best_index])
        return samples

    def generate_samples(
        self, ntrajectories: int, ncandidate_trajectories: Optional[int] = None
    ) -> Array:
        """Generate Morris trajectories for sensitivity analysis.

        Parameters
        ----------
        ntrajectories : int
            Number of trajectories to generate.
        ncandidate_trajectories : int, optional
            Number of candidate trajectories for space-filling selection.
            If provided, must be > ntrajectories. The best ntrajectories
            are selected from the candidates.

        Returns
        -------
        Array
            Shape (nvars, ntrajectories * (nvars + 1)) - sample points.

        Notes
        -----
        The choice of nlevels should be linked to ntrajectories. For example,
        nlevels=4 with ntrajectories=10 is often considered reasonable.
        """
        if (
            ncandidate_trajectories is not None
            and ncandidate_trajectories <= ntrajectories
        ):
            raise ValueError("ncandidate_trajectories must be > ntrajectories")

        self._ntrajectories = ntrajectories
        n_to_generate = (
            ntrajectories
            if ncandidate_trajectories is None
            else ncandidate_trajectories
        )

        candidate_samples = self._bkd.stack(
            [self._get_trajectory() for _ in range(n_to_generate)],
            axis=0,
        )

        if ncandidate_trajectories is not None:
            self._samples = self._downselect_trajectories(
                candidate_samples, ntrajectories
            )
        else:
            self._samples = self._bkd.hstack(
                [candidate_samples[ii] for ii in range(n_to_generate)]
            )

        # Transform from [0, 1] to variable domain
        # Note: marginal.ppf expects shape (1, nsamples), returns (1, nsamples)
        marginals = self._distribution.marginals()
        for ii, marginal in enumerate(marginals):
            row_2d = self._bkd.reshape(self._samples[ii, :], (1, -1))
            transformed = marginal.ppf(row_2d)
            self._samples[ii, :] = self._bkd.reshape(transformed, (-1,))

        return self._samples

    def _compute_elementary_effects(self, values: Array) -> None:
        """Compute elementary effects from function values.

        Parameters
        ----------
        values : Array
            Shape (nqoi, nsamples) - function values at trajectory points.
        """
        assert self._samples is not None
        nqoi = values.shape[0]
        if self._samples.shape[1] != values.shape[1]:
            raise ValueError(
                f"Samples shape {self._samples.shape} inconsistent with "
                f"values shape {values.shape}"
            )
        ntrajectories = self._samples.shape[1] // (self._nvars + 1)
        self._elem_effects = self._bkd.zeros((self._nvars, ntrajectories, nqoi))
        abs_delta = self._nlevels / ((self._nlevels - 1) * 2)

        for ii in range(ntrajectories):
            start = ii * (self._nvars + 1)
            end = (ii + 1) * (self._nvars + 1)
            trajectory_samples = self._samples[:, start:end]
            # Compute signed delta for each variable
            delta = abs_delta * self._bkd.sign(
                self._bkd.diag(trajectory_samples[:, 1:] - trajectory_samples[:, :-1])
            )
            delta = self._bkd.reshape(delta, (self._nvars, 1))
            # Elementary effect = (f(x+delta) - f(x)) / delta
            traj_values = values[:, start:end]
            self._elem_effects[:, ii, :] = self._bkd.diff(traj_values, axis=1).T / delta

    def _compute_sensitivity_indices(self) -> None:
        """Compute mu, mu_star, and sigma from elementary effects."""
        assert self._elem_effects is not None
        self._mu = self._bkd.mean(self._elem_effects, axis=1)
        self._mu_star = self._bkd.mean(self._bkd.abs(self._elem_effects), axis=1)
        # Compute standard deviation using sqrt(variance)
        # Note: for ddof=1, use n-1 in denominator
        self._sigma = self._bkd.sqrt(
            self._bkd.var(self._elem_effects, axis=1)
            * self._elem_effects.shape[1]
            / (self._elem_effects.shape[1] - 1)
        )

    def compute(self, values: Array) -> None:
        """Compute Morris sensitivity indices from function values.

        Parameters
        ----------
        values : Array
            Shape (nqoi, nsamples) - function values at the samples
            returned by generate_samples(). Order must match.
        """
        self._compute_elementary_effects(values)
        self._compute_sensitivity_indices()

    def mu(self) -> Array:
        """Return the mean elementary effect.

        Can cancel for non-monotonic functions. Use mu_star for
        robust importance ranking.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - mean effect for each variable and QoI.
        """
        if self._mu is None:
            raise RuntimeError("Must call compute() first")
        return self._mu

    def mu_star(self) -> Array:
        """Return the mean absolute elementary effect.

        Primary importance measure. Larger values indicate more
        important variables.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - mean absolute effect for each variable and QoI.
        """
        if self._mu_star is None:
            raise RuntimeError("Must call compute() first")
        return self._mu_star

    def sigma(self) -> Array:
        """Return the standard deviation of elementary effects.

        Indicates nonlinearity and/or interaction effects. Low values
        suggest linear relationships; high values suggest nonlinearity
        or interactions with other variables.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - std dev for each variable and QoI.
        """
        if self._sigma is None:
            raise RuntimeError("Must call compute() first")
        return self._sigma

    def elementary_effects(self) -> Array:
        """Return the raw elementary effects.

        Returns
        -------
        Array
            Shape (nvars, ntrajectories, nqoi) - elementary effects.
        """
        if self._elem_effects is None:
            raise RuntimeError("Must call compute() first")
        return self._elem_effects

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        if self._mu is None:
            raise RuntimeError("Must call compute() first")
        return int(self._mu.shape[1])
