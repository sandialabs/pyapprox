"""Adaptive combination sparse grid.

This module provides adaptive sparse grid surrogates that refine
subspaces based on error indicators.
"""

from typing import Callable, Generic, List, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.affine.indices import (
    IterativeIndexGenerator,
    PriorityQueue,
    AdmissibilityCriteria,
    LinearGrowthRule,
)

from .smolyak import compute_smolyak_coefficients, _index_to_tuple
from .subspace import TensorProductSubspace
from .combination import CombinationSparseGrid
from .basis_factory import BasisFactoryProtocol


class AdaptiveCombinationSparseGrid(CombinationSparseGrid[Array], Generic[Array]):
    """Adaptive sparse grid with refinement based on error indicators.

    Implements the step_samples/step_values pattern for incremental
    construction of sparse grids. Subspaces are refined based on
    a user-specified refinement criteria.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis_factories : List[BasisFactoryProtocol[Array]]
        Factories for creating univariate bases for each dimension.
    growth_rule : IndexGrowthRuleProtocol
        Rule mapping level to number of points.
    admissibility : AdmissibilityCriteria[Array]
        Criteria for admissible subspace indices.
    refinement_priority : Callable[[Array, Array, "AdaptiveCombinationSparseGrid"], Tuple[float, float]], optional
        Function(subspace_index, subspace_values, grid) -> (priority, error)
        Higher priority indices are refined first. Default: L2 norm error.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> from pyapprox.typing.surrogates.affine.indices import (
    ...     LinearGrowthRule, MaxLevelCriteria
    ... )
    >>> from pyapprox.typing.surrogates.sparsegrids import PrebuiltBasisFactory
    >>> bkd = NumpyBkd()
    >>> bases = [LegendrePolynomial1D(bkd) for _ in range(2)]
    >>> factories = [PrebuiltBasisFactory(b) for b in bases]
    >>> growth = LinearGrowthRule(scale=2, shift=1)
    >>> admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=bkd)
    >>> grid = AdaptiveCombinationSparseGrid(bkd, factories, growth, admis)
    >>> samples = grid.step_samples()  # Get first samples
    >>> values = my_function(samples)
    >>> grid.step_values(values)
    >>> # Continue refining...
    >>> samples = grid.step_samples()
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis_factories: List[BasisFactoryProtocol[Array]],
        growth_rule: IndexGrowthRuleProtocol,
        admissibility: AdmissibilityCriteria[Array],
        refinement_priority: Optional[
            Callable[
                [Array, Array, "AdaptiveCombinationSparseGrid[Array]"],
                Tuple[float, float],
            ]
        ] = None,
    ):
        super().__init__(bkd, basis_factories, growth_rule)
        self._admissibility = admissibility
        self._refinement_priority = (
            refinement_priority or self._default_refinement_priority
        )

        # Index generator for tracking selected/candidate subspaces
        self._index_gen = IterativeIndexGenerator(self._nvars, bkd)
        self._index_gen.set_admissibility_criteria(admissibility)

        # Priority queue for candidate subspaces
        self._candidate_queue: Optional[PriorityQueue[Array]] = None

        # Tracking for step pattern
        self._first_step = True
        self._last_subspace_indices: Optional[Array] = None
        self._subspace_errors: List[float] = []

        # Unique samples tracking per subspace
        self._subspace_unique_sample_indices: List[List[int]] = []

        # Stored Smolyak coefficients for selected indices (incrementally updated)
        # Shape: (nselected + ncandidates,) with candidates having 0 coefficients
        self._selected_smolyak_coefs: Optional[Array] = None

    def _default_refinement_priority(
        self,
        subspace_index: Array,
        subspace_values: Array,
        grid: "AdaptiveCombinationSparseGrid[Array]",
    ) -> Tuple[float, float]:
        """Default refinement priority based on L2 norm error.

        Computes the interpolation error on the subspace samples.

        Parameters
        ----------
        subspace_index : Array
            Multi-index of the subspace.
        subspace_values : Array
            Function values at subspace samples.
        grid : AdaptiveCombinationSparseGrid
            The sparse grid (self).

        Returns
        -------
        Tuple[float, float]
            (priority, error) where higher priority = refine first.
        """
        key = _index_to_tuple(subspace_index)
        if key not in self._subspaces:
            return 0.0, float("inf")

        subspace = self._subspaces[key]
        samples = subspace.get_samples()

        # Evaluate current interpolant at subspace samples
        current_vals = self._evaluate_selected_only(samples)
        error = float(
            self._bkd.norm(subspace_values - current_vals)
            / max(1, subspace_values.shape[1])  # nsamples is in second dim
        )

        # Higher error = higher priority (return positive priority)
        return error, error

    def _evaluate_selected_only(self, samples: Array) -> Array:
        """Evaluate sparse grid using only selected subspaces."""
        if self._values is None or self._nqoi is None:
            return self._bkd.zeros((1, samples.shape[1]))

        # Get Smolyak coefficients for selected indices only
        selected_indices = self._index_gen.get_selected_indices()
        if selected_indices.shape[1] == 0:
            return self._bkd.zeros((self._nqoi, samples.shape[1]))

        smolyak_coefs = compute_smolyak_coefficients(selected_indices, self._bkd)

        npoints = samples.shape[1]
        result = self._bkd.zeros((self._nqoi, npoints))

        for j, index in enumerate(selected_indices.T):
            key = _index_to_tuple(index)
            if key in self._subspaces:
                coef = float(smolyak_coefs[j])
                if abs(coef) > 1e-14:
                    subspace = self._subspaces[key]
                    if subspace.get_values() is not None:
                        result = result + coef * subspace(samples)

        return result

    def step_samples(self) -> Optional[Array]:
        """Get samples for next refinement step.

        Returns
        -------
        Optional[Array]
            New samples of shape (nvars, nnew) or None if converged.
        """
        if self._first_step:
            return self._first_step_samples()
        return self._next_step_samples()

    def _first_step_samples(self) -> Array:
        """Get samples for the first step (initial subspaces)."""
        # Initialize with zero index
        zero_index = self._bkd.zeros(
            (self._nvars, 1), dtype=self._bkd.int64_dtype()
        )
        self._index_gen.set_selected_indices(zero_index)

        # Add selected subspaces (selected subspaces start with error 0)
        selected_indices = self._index_gen.get_selected_indices()
        for index in selected_indices.T:
            self._add_subspace(index)
            self._subspace_errors.append(0.0)

        # Initialize Smolyak coefficients for selected indices
        self._selected_smolyak_coefs = compute_smolyak_coefficients(
            selected_indices, self._bkd
        )

        # Add candidate subspaces
        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is not None:
            # Extend Smolyak coefficients with zeros for candidates
            ncandidates = cand_indices.shape[1]
            self._selected_smolyak_coefs = self._bkd.hstack((
                self._selected_smolyak_coefs,
                self._bkd.zeros((ncandidates,))
            ))
            for index in cand_indices.T:
                self._add_subspace(index)
                self._subspace_errors.append(float("inf"))

        self._last_subspace_indices = self._index_gen.get_indices()
        self._first_step = False

        # Collect and return unique samples
        self._collect_unique_samples()
        return self._bkd.copy(self._unique_samples)

    def _next_step_samples(self) -> Optional[Array]:
        """Get samples for subsequent steps (refinement)."""
        while self._candidate_queue is not None and not self._candidate_queue.empty():
            # Get best candidate subspace
            priority, error, best_idx = self._candidate_queue.get()
            best_index = self._index_gen._indices[:, best_idx]

            # Refine this subspace (moves from candidate to selected)
            new_cand_indices = self._index_gen.refine_index(best_index)

            # Extend Smolyak coefficients array for new candidates
            if self._selected_smolyak_coefs is not None:
                self._selected_smolyak_coefs = self._bkd.hstack((
                    self._selected_smolyak_coefs,
                    self._bkd.zeros((new_cand_indices.shape[1],))
                ))

            # Incrementally update Smolyak coefficients for the refined index
            # This is done after refine_index moves it from candidate to selected
            if self._selected_smolyak_coefs is not None:
                self._selected_smolyak_coefs = self._adjust_smolyak_coefficients(
                    self._selected_smolyak_coefs,
                    best_index,
                    self._index_gen.get_indices(),
                )

            # Reset error for the refined subspace
            self._subspace_errors[best_idx] = 0.0

            # Add new candidate subspaces WITHOUT invalidating unique_samples
            for index in new_cand_indices.T:
                self._add_subspace_without_invalidating(index)
                self._subspace_errors.append(float("inf"))

            if new_cand_indices.shape[1] == 0:
                # No new candidates from this one, try next best
                continue

            self._last_subspace_indices = new_cand_indices

            # Collect unique samples for new subspaces
            new_samples = self._get_new_unique_samples(new_cand_indices)

            if new_samples.shape[1] > 0:
                return new_samples
            # All samples already known, continue to next candidate

        return None

    def _add_subspace_without_invalidating(
        self, index: Array
    ) -> TensorProductSubspace[Array]:
        """Add a subspace without invalidating unique samples cache.

        This is needed for adaptive refinement where we want to track
        which samples are truly new vs already seen.
        """
        key = _index_to_tuple(index)
        if key in self._subspaces:
            return self._subspaces[key]

        subspace = TensorProductSubspace(
            self._bkd,
            index,
            self._basis_factories,
            self._growth_rule,
        )
        self._subspaces[key] = subspace
        self._subspace_list.append(subspace)

        # Invalidate Smolyak coefficients but NOT unique_samples
        self._smolyak_coefficients = None

        return subspace

    def _get_new_unique_samples(self, new_indices: Array) -> Array:
        """Get unique samples for newly added subspaces."""
        new_samples_list: List[Array] = []

        # Ensure unique_samples is initialized
        if self._unique_samples is None:
            self._collect_unique_samples()

        current_count = self._unique_samples.shape[1]

        for index in new_indices.T:
            key = _index_to_tuple(index)
            subspace = self._subspaces[key]
            subspace_samples = subspace.get_samples()

            # Find truly new samples
            for j in range(subspace_samples.shape[1]):
                sample = subspace_samples[:, j]
                sample_key = tuple(float(sample[i]) for i in range(self._nvars))

                if sample_key not in self._sample_to_idx:
                    self._sample_to_idx[sample_key] = (
                        current_count + len(new_samples_list)
                    )
                    new_samples_list.append(sample[:, None])

        if len(new_samples_list) == 0:
            return self._bkd.zeros((self._nvars, 0))

        new_samples = self._bkd.hstack(new_samples_list)
        self._unique_samples = self._bkd.hstack(
            (self._unique_samples, new_samples)
        )
        return new_samples

    def step_values(self, values: Array) -> None:
        """Provide function values for the samples from step_samples.

        Parameters
        ----------
        values : Array
            Values of shape (nqoi, nnew)
        """
        if self._values is None:
            self._values = values
            self._nqoi = values.shape[0]  # nqoi is first dim
        else:
            self._values = self._bkd.hstack((self._values, values))

        # Distribute values to subspaces
        self._distribute_values_to_subspaces()

        # Re-prioritize candidate subspaces
        self._prioritize_candidates()

    def _prioritize_candidates(self) -> None:
        """Compute priorities for all candidate subspaces."""
        self._candidate_queue = PriorityQueue(max_priority=True)

        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None:
            return

        for index in cand_indices.T:
            key = _index_to_tuple(index)
            if key not in self._subspaces:
                continue

            subspace = self._subspaces[key]
            subspace_values = subspace.get_values()
            if subspace_values is None:
                continue

            priority, error = self._refinement_priority(
                index, subspace_values, self
            )

            # Get index id
            idx = self._index_gen._cand_indices_dict[self._index_gen._hash_index(index)]
            self._candidate_queue.put(priority, error, idx)
            self._subspace_errors[idx] = error

    def error_estimate(self) -> float:
        """Return current error estimate.

        Returns
        -------
        float
            Sum of subspace errors.

        Raises
        ------
        RuntimeError
            If any subspace error is infinity after step_values() has been called.
            This indicates a bug in the error computation.
        """
        # Check for infinities - these should never appear after step_values()
        # has been called for all candidate subspaces
        inf_count = sum(1 for e in self._subspace_errors if e == float("inf"))
        if inf_count > 0:
            raise RuntimeError(
                f"Found {inf_count} subspace(s) with infinite error. "
                "This should not happen after step_values() is called. "
                "Check that values are provided for all candidate subspaces."
            )
        return sum(self._subspace_errors)

    def get_candidate_subspaces(self) -> List[Array]:
        """Return indices of candidate subspaces for refinement.

        Returns
        -------
        List[Array]
            Multi-indices of candidate subspaces.
        """
        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None:
            return []
        return [cand_indices[:, j] for j in range(cand_indices.shape[1])]

    def get_selected_subspace_indices(self) -> Array:
        """Return indices of selected subspaces.

        Returns
        -------
        Array
            Multi-indices of shape (nvars, nselected)
        """
        return self._index_gen.get_selected_indices()

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        if self._nqoi is None:
            raise ValueError("Values not set. Call step_values() first.")
        return self._nqoi

    def _adjust_smolyak_coefficients_with_candidate_index(
        self, candidate_index: Array
    ) -> Array:
        """Compute Smolyak coefficients as if candidate were selected.

        This is used by variance-based refinement criteria to compute what
        the mean and variance would be if a candidate subspace were added.

        Uses the stored _selected_smolyak_coefs and incrementally updates them
        rather than recomputing from scratch, following the legacy pattern in
        pyapprox.surrogates.sparsegrids.combination.AdaptiveCombinationSparseGrid._adjust_smolyak_coefficients_with_candidate_index().

        Parameters
        ----------
        candidate_index : Array
            Candidate index to temporarily include. Shape: (nvars,)

        Returns
        -------
        Array
            Adjusted Smolyak coefficients including the candidate.
        """
        if self._selected_smolyak_coefs is None:
            # Fallback: compute from scratch
            selected = self._index_gen.get_selected_indices()
            temp_indices = self._bkd.hstack((selected, candidate_index[:, None]))
            return compute_smolyak_coefficients(temp_indices, self._bkd)

        # Use stored coefficients and incrementally update
        # The stored coefficients already have zeros for candidates
        # We need to compute what they would be if this candidate were selected
        all_indices = self._index_gen.get_indices()
        return self._adjust_smolyak_coefficients(
            self._selected_smolyak_coefs,
            candidate_index,
            all_indices,
        )

    def mean(self) -> Array:
        """Compute mean of the interpolant using selected subspaces only.

        Uses the stored Smolyak coefficients for efficiency. Only selected
        subspaces (with non-zero coefficients) contribute to the mean.

        Returns
        -------
        Array
            Mean values of shape (nqoi,).
        """
        if self._selected_smolyak_coefs is not None:
            return self._compute_moment("integrate", self._selected_smolyak_coefs)
        # Fallback: compute from scratch
        smolyak_coefs = compute_smolyak_coefficients(
            self._index_gen.get_selected_indices(), self._bkd
        )
        return self._compute_moment("integrate", smolyak_coefs)

    def variance(self) -> Array:
        """Compute variance of the interpolant using selected subspaces only.

        Uses the stored Smolyak coefficients for efficiency. Only selected
        subspaces (with non-zero coefficients) contribute to the variance.

        Returns
        -------
        Array
            Variance values of shape (nqoi,).
        """
        if self._selected_smolyak_coefs is not None:
            return self._compute_moment("variance", self._selected_smolyak_coefs)
        # Fallback: compute from scratch
        smolyak_coefs = compute_smolyak_coefficients(
            self._index_gen.get_selected_indices(), self._bkd
        )
        return self._compute_moment("variance", smolyak_coefs)

    def __call__(self, samples: Array) -> Array:
        """Evaluate sparse grid interpolant using selected subspaces.

        Only selected subspaces (not candidates) are used for evaluation.

        Parameters
        ----------
        samples : Array
            Evaluation points of shape (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values of shape (nqoi, npoints)
        """
        if self._values is None or self._nqoi is None:
            raise ValueError("Values not set. Call step_values() first.")
        return self._evaluate_selected_only(samples)

    def __repr__(self) -> str:
        return (
            f"AdaptiveCombinationSparseGrid(nvars={self._nvars}, "
            f"nsubspaces={self.nsubspaces()}, "
            f"nselected={self._index_gen.nselected_indices()}, "
            f"ncandidates={self._index_gen.ncandidate_indices()})"
        )
