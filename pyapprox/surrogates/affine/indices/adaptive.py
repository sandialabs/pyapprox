"""Adaptive index refinement for sparse grids and PCE.

This module provides the AdaptiveIndexRefinement class that orchestrates
adaptive refinement using priority queues and refinement criteria.
"""

from typing import Generic, Optional

from pyapprox.surrogates.affine.indices.basis_generator import (
    BasisIndexGenerator,
)
from pyapprox.surrogates.affine.indices.generators import (
    IterativeIndexGenerator,
)
from pyapprox.surrogates.affine.indices.priority_queue import (
    PriorityQueue,
)
from pyapprox.surrogates.affine.indices.refinement import (
    LevelRefinementCriteria,
    RefinementCriteria,
)
from pyapprox.util.backends.protocols import Array, Backend


class AdaptiveIndexRefinement(Generic[Array]):
    """Orchestrates adaptive index refinement.

    Uses priority queues and refinement criteria to select which indices
    to refine. Follows the step_samples/step_values iteration pattern.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    index_generator : IterativeIndexGenerator[Array]
        Generator for multi-index sets.
    basis_generator : BasisIndexGenerator[Array]
        Generator mapping subspaces to basis indices.
    refinement_criteria : RefinementCriteria[Array], optional
        Criteria for computing refinement priorities.
    max_level : int
        Maximum refinement level. Default: 10.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.indices import (
    ...     IterativeIndexGenerator,
    ...     BasisIndexGenerator,
    ... )
    >>> bkd = NumpyBkd()
    >>> index_gen = IterativeIndexGenerator(nvars=2, bkd=bkd)
    >>> basis_gen = BasisIndexGenerator(bkd, nvars=2)
    >>> adaptive = AdaptiveIndexRefinement(
    ...     bkd, index_gen, basis_gen, max_level=5
    ... )
    """

    def __init__(
        self,
        bkd: Backend[Array],
        index_generator: IterativeIndexGenerator[Array],
        basis_generator: BasisIndexGenerator[Array],
        refinement_criteria: Optional[RefinementCriteria[Array]] = None,
        max_level: int = 10,
    ):
        self._bkd = bkd
        self._index_generator = index_generator
        self._basis_generator = basis_generator
        self._max_level = max_level

        # Default criteria if not provided
        if refinement_criteria is None:
            refinement_criteria = LevelRefinementCriteria(bkd, max_level)
        self._criteria = refinement_criteria

        # Priority queue for candidates
        self._queue = PriorityQueue[Array](max_priority=True)

        # Track refinement state
        self._total_error = 0.0
        self._step_count = 0
        self._pending_samples: Optional[Array] = None
        self._pending_index: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    @property
    def index_generator(self) -> IterativeIndexGenerator[Array]:
        """Return the index generator."""
        return self._index_generator

    @property
    def basis_generator(self) -> BasisIndexGenerator[Array]:
        """Return the basis generator."""
        return self._basis_generator

    def nselected(self) -> int:
        """Return number of selected indices."""
        return self._index_generator.nselected_indices()

    def ncandidates(self) -> int:
        """Return number of candidate indices."""
        return self._index_generator.ncandidate_indices()

    def error(self) -> float:
        """Return current error estimate.

        Returns
        -------
        float
            Cumulative error estimate.
        """
        return self._total_error

    def _add_candidates_to_queue(self, indices: Array) -> None:
        """Add candidate indices to the priority queue."""
        for col_idx in range(indices.shape[1]):
            index = indices[:, col_idx]
            error, priority = self._criteria(index)
            # Use column index as identifier
            index_id = self._index_generator.nindices() - indices.shape[1] + col_idx
            self._queue.put(priority, error, index_id)

    def initialize(self) -> None:
        """Initialize with candidate indices in queue.

        Call this after setting up the index generator with initial indices.
        """
        candidates = self._index_generator.get_candidate_indices()
        if candidates is not None:
            self._add_candidates_to_queue(candidates)

    def step_samples(self) -> Optional[Array]:
        """Get sample locations for the next refinement step.

        Returns
        -------
        Array or None
            Sample locations. Shape: (nvars, nsamples)
            Returns None if no more candidates or max iterations reached.
        """
        if self._queue.empty():
            return None

        # Get highest priority candidate
        priority, error, index_id = self._queue.get()

        # Get the actual index
        candidates = self._index_generator.get_candidate_indices()
        if candidates is None:
            return None

        # Find the index by ID - search through all indices
        all_indices = self._index_generator._get_indices()
        if index_id >= all_indices.shape[1]:
            return None

        index = all_indices[:, index_id]
        self._pending_index = index

        # Get basis indices for this subspace
        self._basis_generator.get_basis_indices(index)

        # For now, just return the index as sample (actual implementation
        # would compute quadrature/interpolation points)
        self._pending_samples = self._bkd.asarray(
            [[float(index[d]) for d in range(len(index))]],
            dtype=self._bkd.float64_dtype(),
        ).T

        return self._pending_samples

    def step_values(self, values: Array) -> None:
        """Incorporate function values and update refinement.

        Parameters
        ----------
        values : Array
            Function values at samples from step_samples().
            Shape: (nsamples, nqoi)
        """
        if self._pending_index is None:
            raise RuntimeError("Must call step_samples() first")

        # Update error estimate (simplified: use mean absolute value)
        error_contribution = self._bkd.to_float(self._bkd.mean(self._bkd.abs(values)))
        self._total_error += error_contribution

        # Refine the index
        new_candidates = self._index_generator.refine_index(self._pending_index)

        # Add new candidates to queue
        if new_candidates.shape[1] > 0:
            self._add_candidates_to_queue(new_candidates)

        # Clear pending state
        self._pending_index = None
        self._pending_samples = None
        self._step_count += 1

    def refine_to_tolerance(
        self,
        func: Callable[[Array], Array],
        tolerance: float,
        max_iterations: int = 100,
    ) -> None:
        """Refine until error estimate is below tolerance.

        Parameters
        ----------
        func : callable
            Function to evaluate. Takes samples (nvars, nsamples) and
            returns values (nsamples, nqoi).
        tolerance : float
            Target error tolerance.
        max_iterations : int
            Maximum number of refinement iterations.
        """
        for _ in range(max_iterations):
            samples = self.step_samples()
            if samples is None:
                break

            values = func(samples)
            self.step_values(values)

            if self.error() < tolerance:
                break

    def get_selected_indices(self) -> Array:
        """Return the selected multi-indices.

        Returns
        -------
        Array
            Selected indices. Shape: (nvars, nselected)
        """
        return self._index_generator.get_selected_indices()

    def __repr__(self) -> str:
        return (
            f"AdaptiveIndexRefinement("
            f"nsel={self.nselected()}, ncand={self.ncandidates()}, "
            f"error={self.error():.2e})"
        )
