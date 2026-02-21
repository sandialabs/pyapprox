"""Adaptive sparse grid fitter.

Iteratively refines a sparse grid using error indicators to select
the most important subspaces. Uses the step_samples/step_values pattern
for external control, or refine_to_tolerance for automatic refinement.
"""

from typing import Callable, Dict, Generic, List, Optional, Tuple, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.indices import (
    IterativeIndexGenerator,
    PriorityQueue,
    AdmissibilityCriteria,
)
from pyapprox.typing.surrogates.sparsegrids.smolyak import (
    compute_smolyak_coefficients,
    smolyak_coefs_with_candidate,
    _index_to_tuple,
)
from pyapprox.typing.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.typing.surrogates.sparsegrids.subspace_factory import (
    SubspaceFactoryProtocol,
)
from pyapprox.typing.surrogates.sparsegrids.sample_tracker import (
    SampleTracker,
)
from pyapprox.typing.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.typing.surrogates.sparsegrids.fit_result import (
    AdaptiveSparseGridFitResult,
)
from pyapprox.typing.surrogates.sparsegrids.candidate_info import (
    CandidateInfo,
    ConfigIdx,
)
from pyapprox.typing.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
    L2SurrogateDifferenceIndicator,
)


class AdaptiveSparseGridFitter(Generic[Array]):
    """Adaptive sparse grid fitter with step_samples/step_values pattern.

    Iteratively builds a sparse grid by selecting subspaces based on
    error indicators. Each step:
    1. step_samples() returns new sample locations
    2. step_values(values) provides function evaluations
    3. Internal reprioritization determines next subspace to refine

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    factory : SubspaceFactoryProtocol[Array]
        Factory for creating tensor product subspaces.
    admissibility : AdmissibilityCriteria[Array]
        Criteria for admissible subspace indices.
    error_indicator : ErrorIndicatorProtocol[Array], optional
        Error indicator for computing refinement priorities.
        Default: L2SurrogateDifferenceIndicator.
    nconfig_vars : int, optional
        Number of config/fidelity dimensions. Default: 0.
    verbosity : int, optional
        Verbosity level. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        factory: SubspaceFactoryProtocol[Array],
        admissibility: AdmissibilityCriteria[Array],
        error_indicator: Optional[ErrorIndicatorProtocol[Array]] = None,
        nconfig_vars: int = 0,
        verbosity: int = 0,
    ) -> None:
        self._bkd = bkd
        self._factory = factory
        self._admissibility = admissibility
        if error_indicator is None:
            error_indicator = L2SurrogateDifferenceIndicator(bkd)
        self._error_indicator = error_indicator
        self._nconfig_vars = nconfig_vars
        self._verbosity = verbosity
        self._nvars_physical = factory.nvars_physical()
        self._nvars_index = self._nvars_physical + nconfig_vars

        # Index generator for tracking selected/candidate indices
        self._index_gen: IterativeIndexGenerator[Array] = (
            IterativeIndexGenerator(self._nvars_index, bkd)
        )
        self._index_gen.set_admissibility_criteria(admissibility)

        # Subspace tracking (parallel to index_gen._indices columns)
        self._subspaces: List[TensorProductSubspace[Array]] = []
        self._subspace_keys: List[Tuple[int, ...]] = []
        self._subspace_errors: List[float] = []

        # Sample tracker (one per config group)
        self._trackers: Dict[ConfigIdx, SampleTracker[Array]] = {}

        # Mapping: subspace key -> tracker position for each config group
        self._tracker_positions: Dict[
            ConfigIdx, Dict[Tuple[int, ...], int]
        ] = {}

        # Priority queue for candidates
        self._candidate_queue: Optional[PriorityQueue[Array]] = None

        # State
        self._first_step = True
        self._nqoi: Optional[int] = None
        self._nsteps = 0

    def _get_config_idx(self, full_index: Array) -> ConfigIdx:
        """Extract config index from full multi-index."""
        if self._nconfig_vars == 0:
            return ()
        config_part = full_index[self._nvars_physical:]
        return tuple(
            int(config_part[i]) for i in range(self._nconfig_vars)
        )

    def _create_subspace(self, full_index: Array) -> None:
        """Create a subspace and register with the appropriate tracker."""
        config_idx = self._get_config_idx(full_index)

        if config_idx not in self._trackers:
            self._trackers[config_idx] = SampleTracker(
                self._bkd, self._factory
            )
            self._tracker_positions[config_idx] = {}

        tracker = self._trackers[config_idx]
        subspace = self._factory(full_index)
        pos = tracker.register(full_index, subspace)

        # Invariant: every subspace must contribute new samples
        unique_local = tracker.get_unique_local_indices(pos)
        if len(unique_local) == 0:
            raise ValueError(
                f"Subspace {_index_to_tuple(full_index)} contributes no "
                f"new samples."
            )

        key = _index_to_tuple(full_index)
        self._subspaces.append(subspace)
        self._subspace_keys.append(key)
        self._tracker_positions[config_idx][key] = pos

    def step_samples(self) -> Optional[Union[Array, Dict[ConfigIdx, Array]]]:
        """Get samples for next refinement step.

        Returns
        -------
        Optional[Union[Array, Dict[ConfigIdx, Array]]]
            New samples, or None if converged.
            For SF: Array of shape (nvars_physical, n_new).
            For MF: Dict mapping config_idx to sample arrays.
        """
        if self._first_step:
            return self._first_step_samples()
        return self._next_step_samples()

    def _first_step_samples(
        self,
    ) -> Union[Array, Dict[ConfigIdx, Array]]:
        """Get samples for the first step (zero index + candidates)."""
        # Initialize with zero index
        zero_index = self._bkd.zeros(
            (self._nvars_index, 1), dtype=self._bkd.int64_dtype()
        )
        self._index_gen.set_selected_indices(zero_index)

        # Add selected subspace (zero index)
        selected_indices = self._index_gen.get_selected_indices()
        for index in selected_indices.T:
            self._create_subspace(index)
            self._subspace_errors.append(0.0)

        # Add candidate subspaces
        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is not None:
            for index in cand_indices.T:
                self._create_subspace(index)
                self._subspace_errors.append(float("inf"))

        self._first_step = False

        # Return all unique samples
        return self._unwrap_samples()

    def _next_step_samples(
        self,
    ) -> Optional[Union[Array, Dict[ConfigIdx, Array]]]:
        """Get samples for subsequent steps (refinement)."""
        while (
            self._candidate_queue is not None
            and not self._candidate_queue.empty()
        ):
            priority, error, best_idx = self._candidate_queue.get()
            best_index = self._index_gen._indices[:, best_idx]

            if self._verbosity >= 1:
                idx_tuple = _index_to_tuple(best_index)
                print(
                    f"[Adaptive SG] Refining {idx_tuple}: "
                    f"priority={priority:.2e}, error={error:.2e}"
                )

            # Refine this subspace (move from candidate to selected)
            new_cand_indices = self._index_gen.refine_index(best_index)

            # Reset error for refined subspace
            self._subspace_errors[best_idx] = 0.0

            # Add new candidate subspaces
            for index in new_cand_indices.T:
                self._create_subspace(index)
                self._subspace_errors.append(float("inf"))

            if new_cand_indices.shape[1] == 0:
                continue

            # Return new unique samples from new candidates
            new_samples = self._collect_new_samples(new_cand_indices)
            if new_samples is not None:
                self._nsteps += 1
                return new_samples

        return None

    def _collect_new_samples(
        self, new_indices: Array
    ) -> Optional[Union[Array, Dict[ConfigIdx, Array]]]:
        """Collect new unique samples from newly added subspaces."""
        # For each new candidate, find its unique local indices
        # and collect those samples
        new_by_config: Dict[ConfigIdx, List[Array]] = {}

        for j in range(new_indices.shape[1]):
            full_index = new_indices[:, j]
            key = _index_to_tuple(full_index)
            config_idx = self._get_config_idx(full_index)
            tracker = self._trackers[config_idx]

            # Find subspace position via stored mapping
            pos_map = self._tracker_positions[config_idx]
            if key not in pos_map:
                continue
            pos = pos_map[key]

            unique_local = tracker.get_unique_local_indices(pos)
            if len(unique_local) > 0:
                subspace = self._subspaces[
                    self._subspace_keys.index(key)
                ]
                subspace_samples = subspace.get_samples()
                idx_arr = self._bkd.asarray(
                    unique_local, dtype=self._bkd.int64_dtype()
                )
                if config_idx not in new_by_config:
                    new_by_config[config_idx] = []
                new_by_config[config_idx].append(
                    subspace_samples[:, idx_arr]
                )

        if not new_by_config:
            return None

        result: Dict[ConfigIdx, Array] = {}
        for cfg, sample_list in new_by_config.items():
            result[cfg] = self._bkd.hstack(sample_list)

        if self._nconfig_vars == 0:
            return result[()]
        return result

    def _unwrap_samples(
        self,
    ) -> Union[Array, Dict[ConfigIdx, Array]]:
        """Return all unique samples from all trackers."""
        if self._nconfig_vars == 0:
            tracker = self._trackers[()]
            return tracker.collect_unique_samples()
        return {
            cfg: tracker.collect_unique_samples()
            for cfg, tracker in self._trackers.items()
        }

    def step_values(
        self, values: Union[Array, Dict[ConfigIdx, Array]]
    ) -> None:
        """Provide function values for the samples from step_samples.

        Parameters
        ----------
        values : Union[Array, Dict[ConfigIdx, Array]]
            For SF: Array of shape (nqoi, n_samples).
            For MF: Dict mapping config_idx to value arrays.
        """
        wrapped = self._wrap(values)
        for cfg, vals in wrapped.items():
            self._trackers[cfg].append_new_values(vals)

        # Distribute values to subspaces
        for tracker in self._trackers.values():
            tracker.distribute_values_to_subspaces()

        self._nqoi = next(iter(wrapped.values())).shape[0]

        # Re-prioritize candidates
        self._reprioritize_candidates()

    def _wrap(
        self, values: Union[Array, Dict[ConfigIdx, Array]]
    ) -> Dict[ConfigIdx, Array]:
        """Normalize values to dict form."""
        if isinstance(values, dict):
            return values
        return {(): values}

    def _reprioritize_candidates(self) -> None:
        """Compute priorities for all candidate subspaces."""
        assert self._nqoi is not None

        self._candidate_queue = PriorityQueue(max_priority=True)

        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None:
            return

        # Build selected surrogate ONCE
        selected_indices = self._index_gen.get_selected_indices()
        if selected_indices.shape[1] == 0:
            return

        selected_coefs = compute_smolyak_coefficients(
            selected_indices, self._bkd
        )
        selected_subspaces = self._get_subspaces_for_indices(
            selected_indices
        )
        selected_surrogate = CombinationSurrogate(
            self._bkd,
            self._nvars_physical,
            selected_subspaces,
            selected_coefs,
            self._nqoi,
            indices=selected_indices,
        )

        # For each candidate, build CandidateInfo and evaluate
        for j in range(cand_indices.shape[1]):
            cand_index = cand_indices[:, j]
            cand_key = _index_to_tuple(cand_index)

            # Find the subspace
            if cand_key not in self._subspace_keys:
                continue
            sub_pos = self._subspace_keys.index(cand_key)
            cand_subspace = self._subspaces[sub_pos]

            # Check if values are set
            if cand_subspace.get_values() is None:
                continue

            info = self._build_candidate_info(
                cand_index, cand_subspace,
                selected_indices, selected_coefs, selected_surrogate,
            )

            priority, error = self._error_indicator(info)

            # Get the position in index_gen._indices
            idx_id = self._index_gen._cand_indices_dict[
                self._index_gen._hash_index(cand_index)
            ]
            self._candidate_queue.put(priority, error, idx_id)
            self._subspace_errors[idx_id] = error

    def _build_candidate_info(
        self,
        candidate_index: Array,
        candidate_subspace: TensorProductSubspace[Array],
        selected_indices: Array,
        selected_coefs: Array,
        selected_surrogate: CombinationSurrogate[Array],
    ) -> CandidateInfo[Array]:
        """Build CandidateInfo for a candidate subspace."""
        config_idx = self._get_config_idx(candidate_index)
        tracker = self._trackers[config_idx]

        # Find position in tracker via stored mapping
        cand_key = _index_to_tuple(candidate_index)
        pos = self._tracker_positions[config_idx][cand_key]

        unique_local = tracker.get_unique_local_indices(pos)
        all_samples = tracker.collect_unique_samples()
        cand_samples = candidate_subspace.get_samples()
        new_samples = cand_samples[:, unique_local]

        # Build sel+candidate surrogate
        combined_coefs = smolyak_coefs_with_candidate(
            selected_indices, selected_coefs, candidate_index, self._bkd
        )
        all_subspaces = list(selected_surrogate.subspaces()) + [
            candidate_subspace
        ]
        combined_indices = self._bkd.hstack(
            (selected_indices, self._bkd.reshape(candidate_index, (-1, 1)))
        )

        sel_plus_surrogate = CombinationSurrogate(
            self._bkd,
            self._nvars_physical,
            all_subspaces,
            combined_coefs,
            self._nqoi,
            indices=combined_indices,
        )

        return CandidateInfo(
            candidate_index=candidate_index,
            candidate_subspace=candidate_subspace,
            all_samples=all_samples,
            new_samples=new_samples,
            new_sample_local_indices=unique_local,
            selected_surrogate=selected_surrogate,
            sel_plus_candidate_surrogate=sel_plus_surrogate,
            config_idx=config_idx if self._nconfig_vars > 0 else None,
        )

    def _get_subspaces_for_indices(
        self, indices: Array
    ) -> List[TensorProductSubspace[Array]]:
        """Get subspaces corresponding to given indices."""
        result = []
        for j in range(indices.shape[1]):
            key = _index_to_tuple(indices[:, j])
            idx = self._subspace_keys.index(key)
            result.append(self._subspaces[idx])
        return result

    def current_error(self) -> float:
        """Return sum of errors for candidate subspaces.

        Returns
        -------
        float
            Current total error estimate.
        """
        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None:
            return 0.0
        total = 0.0
        for j in range(cand_indices.shape[1]):
            idx = cand_indices[:, j]
            key = self._index_gen._hash_index(idx)
            pos = self._index_gen._cand_indices_dict[key]
            err = self._subspace_errors[pos]
            if err != float("inf"):
                total += err
        return total

    def result(
        self, converged: bool = False
    ) -> AdaptiveSparseGridFitResult[Array]:
        """Build result from current state.

        Parameters
        ----------
        converged : bool
            Whether the fitter converged to tolerance.

        Returns
        -------
        AdaptiveSparseGridFitResult[Array]
            The fit result containing the surrogate.
        """
        assert self._nqoi is not None

        selected_indices = self._index_gen.get_selected_indices()
        selected_coefs = compute_smolyak_coefficients(
            selected_indices, self._bkd
        )
        selected_subspaces = self._get_subspaces_for_indices(
            selected_indices
        )

        surrogate = CombinationSurrogate(
            self._bkd,
            self._nvars_physical,
            selected_subspaces,
            selected_coefs,
            self._nqoi,
            indices=selected_indices,
        )

        nsamples = sum(
            t.n_unique_samples() for t in self._trackers.values()
        )

        return AdaptiveSparseGridFitResult(
            surrogate=surrogate,
            indices=selected_indices,
            coefficients=selected_coefs,
            nsamples=nsamples,
            error=self.current_error(),
            nsteps=self._nsteps,
            converged=converged,
        )

    def refine_to_tolerance(
        self,
        target_fn: Callable[[Array], Array],
        tol: float = 1e-6,
        max_steps: int = 200,
    ) -> AdaptiveSparseGridFitResult[Array]:
        """Refine adaptively until error < tol or max_steps reached.

        Parameters
        ----------
        target_fn : callable
            Function mapping samples -> values, shape (nqoi, nsamples).
        tol : float
            Error tolerance.
        max_steps : int
            Maximum number of refinement steps.

        Returns
        -------
        AdaptiveSparseGridFitResult[Array]
            The fit result.
        """
        for _ in range(max_steps):
            samples = self.step_samples()
            if samples is None:
                return self.result(converged=True)

            if isinstance(samples, dict):
                values = {
                    cfg: target_fn(s) for cfg, s in samples.items()
                }
            else:
                values = target_fn(samples)

            self.step_values(values)

            if self.current_error() < tol:
                return self.result(converged=True)

        return self.result(converged=False)

    def nvars_physical(self) -> int:
        """Return number of physical variables."""
        return self._nvars_physical

    def __repr__(self) -> str:
        return (
            f"AdaptiveSparseGridFitter(nvars={self._nvars_physical}, "
            f"nsubspaces={len(self._subspaces)}, "
            f"nselected={self._index_gen.nselected_indices()}, "
            f"ncandidates={self._index_gen.ncandidate_indices()})"
        )
