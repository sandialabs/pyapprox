"""Adaptive sparse grid fitters.

Provides two classes:
- MultiFidelityAdaptiveSparseGridFitter: Core implementation that always
  operates on Dict[ConfigIdx, Array] for samples and values.
- SingleFidelityAdaptiveSparseGridFitter: Thin composition wrapper that
  adapts the Array <-> Dict boundary for single-fidelity use.
"""

from typing import Dict, Generic, List, Literal, Optional, Set, Tuple

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.surrogates.affine.indices import (
    AdmissibilityCriteria,
    IterativeIndexGenerator,
    PriorityQueue,
)
from pyapprox.surrogates.sparsegrids.candidate_info import (
    CandidateInfo,
    ConfigIdx,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.cost_model import (
    ConstantCostModel,
    CostModelProtocol,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
    L2SurplusIndicator,
)
from pyapprox.surrogates.sparsegrids.fit_result import (
    AdaptiveSparseGridFitResult,
)
from pyapprox.surrogates.sparsegrids.model_factory import (
    DictModelFactory,
    ModelFactoryProtocol,
)
from pyapprox.surrogates.sparsegrids.sample_tracker import (
    SampleTracker,
)
from pyapprox.surrogates.sparsegrids.smolyak import (
    IncrementalSmolyakCoefficients,
    SubspaceKey,
    _index_to_tuple,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    SubspaceFactoryProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

# Sentinel key for single-fidelity grids (nconfig_vars=0)
_SF_KEY: ConfigIdx = ()

SubsetType = Literal["selected", "candidate", "all"]


class MultiFidelityAdaptiveSparseGridFitter(Generic[Array]):
    """Adaptive sparse grid fitter for multi-fidelity models.

    Always operates on Dict[ConfigIdx, Array] for samples and values.
    Each config index identifies a model fidelity level.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    factory : SubspaceFactoryProtocol[Array]
        Factory for creating tensor product subspaces.
    admissibility : AdmissibilityCriteria[Array]
        Criteria for admissible subspace indices.
    nconfig_vars : int
        Number of config/fidelity dimensions.
    error_indicator : ErrorIndicatorProtocol[Array], optional
        Error indicator for computing refinement priorities.
        Default: L2SurplusPerCostIndicator --- RMS surplus on the
        candidate's new samples divided by its per-point cost, giving a
        surplus-per-unit-work priority that avoids the dilution
        separable refinements suffer under L2GlobalSurplusIndicator.
    cost_model : CostModelProtocol, optional
        Per-sample cost model. Default: ConstantCostModel() (unit cost).
    verbosity : int, optional
        Verbosity level. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        factory: SubspaceFactoryProtocol[Array],
        admissibility: AdmissibilityCriteria[Array],
        nconfig_vars: int,
        error_indicator: Optional[ErrorIndicatorProtocol[Array]] = None,
        cost_model: Optional[CostModelProtocol] = None,
        verbosity: int = 0,
    ) -> None:
        self._bkd = bkd
        self._factory = factory
        self._admissibility = admissibility
        if error_indicator is None:
            error_indicator = L2SurplusIndicator(bkd)
        if not isinstance(error_indicator, ErrorIndicatorProtocol):
            raise TypeError(
                "error_indicator must satisfy ErrorIndicatorProtocol, got "
                f"{type(error_indicator).__name__}"
            )
        self._error_indicator = error_indicator
        if cost_model is None:
            cost_model = ConstantCostModel()
        self._cost_model = cost_model
        self._nconfig_vars = nconfig_vars
        self._verbosity = verbosity
        self._nvars_physical = factory.nvars_physical()
        self._nvars_index = self._nvars_physical + nconfig_vars

        # Index generator for tracking selected/candidate indices
        self._index_gen: IterativeIndexGenerator[Array] = IterativeIndexGenerator(
            self._nvars_index, bkd
        )
        self._index_gen.set_admissibility_criteria(admissibility)

        # Smolyak coefficients of the selected set, updated one
        # promotion at a time.
        self._smolyak = IncrementalSmolyakCoefficients(self._nvars_index)

        # Subspace lookup by key, used inside per-candidate loops.
        self._subspace_by_key: Dict[SubspaceKey, TensorProductSubspace[Array]] = {}
        # Errors indexed by the index generator's column.
        self._subspace_errors: List[float] = []

        # Sample tracker (one per config group)
        self._trackers: Dict[ConfigIdx, SampleTracker[Array]] = {}

        # Mapping: subspace key -> tracker position for each config group
        self._tracker_positions: Dict[ConfigIdx, Dict[Tuple[int, ...], int]] = {}

        # Priority queue for candidates
        self._candidate_queue: Optional[PriorityQueue[Array]] = None

        # State
        self._first_step = True
        self._nqoi: Optional[int] = None
        self._nsteps = 0

    def _get_config_idx(self, full_index: Array) -> ConfigIdx:
        """Extract config index from full multi-index."""
        if self._nconfig_vars == 0:
            return _SF_KEY
        config_part = full_index[self._nvars_physical :]
        return tuple(
            self._bkd.to_int(config_part[i])
            for i in range(self._nconfig_vars)
        )

    def _create_subspace(self, full_index: Array) -> None:
        """Create a subspace and register with the appropriate tracker."""
        config_idx = self._get_config_idx(full_index)

        if config_idx not in self._trackers:
            self._trackers[config_idx] = SampleTracker(self._bkd, self._factory)
            self._tracker_positions[config_idx] = {}

        tracker = self._trackers[config_idx]
        subspace = self._factory(full_index)
        pos = tracker.register(full_index, subspace)

        # Invariant: every subspace must contribute new samples
        unique_local = tracker.get_unique_local_indices(pos)
        if len(unique_local) == 0:
            raise ValueError(
                f"Subspace {_index_to_tuple(full_index, self._bkd)} "
                f"contributes no new samples."
            )

        key = _index_to_tuple(full_index, self._bkd)
        self._subspace_by_key[key] = subspace
        self._tracker_positions[config_idx][key] = pos

    def _select(self, full_index: Array) -> None:
        """Record a promotion in the incremental coefficients.

        Called for the zero index and after each ``refine_index``, so the
        coefficients track the index generator's selected set one
        promotion at a time.
        """
        key = _index_to_tuple(full_index, self._bkd)
        self._smolyak.add(key)
        if not self._index_gen.is_selected(full_index):
            raise RuntimeError(
                f"subspace {key} was recorded as selected but the index "
                "generator does not consider it selected"
            )

    def step_samples(self) -> Optional[Dict[ConfigIdx, Array]]:
        """Get samples for next refinement step.

        Returns
        -------
        Optional[Dict[ConfigIdx, Array]]
            Dict mapping config_idx to sample arrays, or None if converged.
        """
        if self._first_step:
            return self._first_step_samples()
        return self._next_step_samples()

    def _first_step_samples(self) -> Dict[ConfigIdx, Array]:
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
            self._select(index)
            self._subspace_errors.append(0.0)

        # Add candidate subspaces
        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is not None:
            for index in cand_indices.T:
                self._create_subspace(index)
                self._subspace_errors.append(float("inf"))

        self._first_step = False

        # Return all unique samples per config
        return {
            cfg: tracker.collect_unique_samples()
            for cfg, tracker in self._trackers.items()
        }

    def _next_step_samples(
        self,
    ) -> Optional[Dict[ConfigIdx, Array]]:
        """Get samples for subsequent steps (refinement).

        Promotes the highest-priority candidate to selected and returns
        new samples from its admissible neighbor candidates.  When a
        promoted candidate has no admissible neighbors (e.g. due to
        downward-closed constraints), the next candidate is promoted
        immediately since no new evaluations are needed.
        """
        while self._candidate_queue is not None and not self._candidate_queue.empty():
            priority, error, best_idx = self._candidate_queue.get()
            best_index = self._index_gen._indices[:, best_idx]

            if self._verbosity >= 1:
                idx_tuple = _index_to_tuple(best_index, self._bkd)
                print(
                    f"[Adaptive SG] Refining {idx_tuple}: "
                    f"priority={priority:.2e}, error={error:.2e}"
                )

            # Refine this subspace (move from candidate to selected)
            new_cand_indices = self._index_gen.refine_index(best_index)
            self._select(best_index)

            # Reset error for refined subspace
            self._subspace_errors[best_idx] = 0.0

            # Add new candidate subspaces
            for index in new_cand_indices.T:
                self._create_subspace(index)
                self._subspace_errors.append(float("inf"))

            if new_cand_indices.shape[1] == 0:
                # No admissible neighbors (e.g. downward-closed blocks
                # children until siblings are selected). Promote next.
                continue

            # Return new unique samples from new candidates
            new_samples = self._collect_new_samples(new_cand_indices)
            if new_samples is not None:
                self._nsteps += 1
                return new_samples

        return None

    def _collect_new_samples(
        self, new_indices: Array
    ) -> Optional[Dict[ConfigIdx, Array]]:
        """Collect new unique samples from newly added subspaces."""
        new_by_config: Dict[ConfigIdx, List[Array]] = {}

        for j in range(new_indices.shape[1]):
            full_index = new_indices[:, j]
            key = _index_to_tuple(full_index, self._bkd)
            config_idx = self._get_config_idx(full_index)
            tracker = self._trackers[config_idx]

            pos_map = self._tracker_positions[config_idx]
            if key not in pos_map:
                continue
            pos = pos_map[key]

            unique_local = tracker.get_unique_local_indices(pos)
            if len(unique_local) > 0:
                subspace = self._subspace_by_key[key]
                subspace_samples = subspace.get_samples()
                idx_arr = self._bkd.asarray(unique_local, dtype=self._bkd.int64_dtype())
                if config_idx not in new_by_config:
                    new_by_config[config_idx] = []
                new_by_config[config_idx].append(subspace_samples[:, idx_arr])

        if not new_by_config:
            return None

        result: Dict[ConfigIdx, Array] = {}
        for cfg, sample_list in new_by_config.items():
            result[cfg] = self._bkd.hstack(sample_list)
        return result

    def step_values(self, values: Dict[ConfigIdx, Array]) -> None:
        """Provide function values for the samples from step_samples.

        Parameters
        ----------
        values : Dict[ConfigIdx, Array]
            Dict mapping config_idx to value arrays of shape
            (nqoi, n_samples).
        """
        for cfg, vals in values.items():
            self._trackers[cfg].append_new_values(vals)

        # Distribute values to subspaces. Every subspace registered by
        # the preceding step_samples must come out of this with values:
        # a config is absent from the batch exactly when it gained no
        # new subspaces, so no tracker is left holding unwritten ones.
        for cfg, tracker in self._trackers.items():
            tracker.distribute_values_to_subspaces()
            if tracker.npending() != 0:
                raise RuntimeError(
                    f"config {cfg} has {tracker.npending()} subspaces "
                    "without values after step_values; samples and "
                    "values are out of step"
                )

        self._nqoi = next(iter(values.values())).shape[0]

        # Re-prioritize candidates
        self._reprioritize_candidates()

    def _reprioritize_candidates(self) -> None:
        """Compute priorities for all candidate subspaces."""
        if self._nqoi is None:
            raise RuntimeError("nqoi not set; call step_values first")

        self._candidate_queue = PriorityQueue(max_priority=True)

        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None:
            return

        # Build selected surrogate ONCE
        selected_indices = self._index_gen.get_selected_indices()
        if selected_indices.shape[1] == 0:
            return

        # Build the selected surrogate from the incremental coefficients,
        # aligning subspaces with the keys they are stored against.
        selected_keys = self._smolyak.keys()
        selected_coefs = self._bkd.asarray(
            [float(c) for c in self._smolyak.coefficient_list(selected_keys)]
        )
        selected_subspaces = [
            self._subspace_by_key[key] for key in selected_keys
        ]
        selected_surrogate = CombinationSurrogate(
            self._bkd,
            self._nvars_physical,
            selected_subspaces,
            selected_coefs,
            self._nqoi,
            indices=self._keys_to_indices(selected_keys),
        )

        # For each candidate, build CandidateInfo and evaluate
        for j in range(cand_indices.shape[1]):
            cand_index = cand_indices[:, j]
            cand_key = _index_to_tuple(cand_index, self._bkd)

            # Find the subspace
            cand_subspace = self._subspace_by_key.get(cand_key)
            if cand_subspace is None:
                continue

            # Check if values are set
            if cand_subspace.get_values() is None:
                continue

            info = self._build_candidate_info(
                cand_index,
                cand_subspace,
                selected_indices,
                selected_coefs,
                selected_surrogate,
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
        if self._nqoi is None:
            raise RuntimeError("nqoi not set; call step_values first")
        config_idx = self._get_config_idx(candidate_index)
        tracker = self._trackers[config_idx]

        # Find position in tracker via stored mapping
        cand_key = _index_to_tuple(candidate_index, self._bkd)
        pos = self._tracker_positions[config_idx][cand_key]

        unique_local = tracker.get_unique_local_indices(pos)
        all_samples = tracker.collect_unique_samples()
        cand_samples = candidate_subspace.get_samples()
        new_samples = cand_samples[:, unique_local]

        # Build sel+candidate surrogate from the incremental update.
        combined = self._smolyak.with_added([cand_key])
        combined_keys = combined.keys()
        combined_coefs = self._bkd.asarray(
            [float(c) for c in combined.coefficient_list(combined_keys)]
        )
        all_subspaces = [
            self._subspace_by_key[key] for key in combined_keys
        ]

        sel_plus_surrogate = CombinationSurrogate(
            self._bkd,
            self._nvars_physical,
            all_subspaces,
            combined_coefs,
            self._nqoi,
            indices=self._keys_to_indices(combined_keys),
        )

        model_cost = self._cost_model(config_idx)
        subspace_cost = model_cost * len(unique_local)

        return CandidateInfo(
            candidate_index=candidate_index,
            candidate_subspace=candidate_subspace,
            all_samples=all_samples,
            new_samples=new_samples,
            new_sample_local_indices=unique_local,
            selected_surrogate=selected_surrogate,
            sel_plus_candidate_surrogate=sel_plus_surrogate,
            config_idx=config_idx if self._nconfig_vars > 0 else None,
            model_cost=model_cost,
            subspace_cost=subspace_cost,
        )

    def _keys_to_indices(self, keys: List[SubspaceKey]) -> Array:
        """Stack subspace keys into an index array, shape (nvars, nkeys)."""
        if len(keys) == 0:
            return self._bkd.zeros(
                (self._nvars_index, 0), dtype=self._bkd.int64_dtype()
            )
        return self._bkd.asarray(
            [[key[d] for key in keys] for d in range(self._nvars_index)],
            dtype=self._bkd.int64_dtype(),
        )

    def _get_subspaces_for_indices(
        self, indices: Array
    ) -> List[TensorProductSubspace[Array]]:
        """Get subspaces corresponding to given indices."""
        return [
            self._subspace_by_key[_index_to_tuple(indices[:, j], self._bkd)]
            for j in range(indices.shape[1])
        ]

    def current_error(self) -> float:
        """Return sum of errors for candidate subspaces."""
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
        self,
        converged: bool = False,
        include_candidates: bool = True,
    ) -> AdaptiveSparseGridFitResult[Array]:
        """Build result from current state.

        Parameters
        ----------
        converged : bool
            Whether the fitter converged to tolerance.
        include_candidates : bool
            If True (default), include candidate subspaces that have
            values in the surrogate so all evaluated data is used.

        Returns
        -------
        AdaptiveSparseGridFitResult[Array]
        """
        if self._nqoi is None:
            raise RuntimeError("nqoi not set; call step_values first")

        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is not None:
            for j in range(cand_indices.shape[1]):
                cand_key = _index_to_tuple(
                    cand_indices[:, j], self._bkd
                )
                cand_subspace = self._subspace_by_key.get(cand_key)
                if cand_subspace is not None:
                    if cand_subspace.get_values() is None:
                        raise RuntimeError(
                            "Cannot build result: candidate subspace "
                            f"{cand_key} has no values. "
                            "Call step_values before result."
                        )

        if include_candidates and cand_indices is not None:
            smolyak = self._smolyak.with_added(
                _index_to_tuple(cand_indices[:, j], self._bkd)
                for j in range(cand_indices.shape[1])
            )
        else:
            smolyak = self._smolyak

        all_keys = smolyak.keys()
        all_coefs = self._bkd.asarray(
            [float(c) for c in smolyak.coefficient_list(all_keys)]
        )
        all_subspaces = [self._subspace_by_key[key] for key in all_keys]
        all_indices = self._keys_to_indices(all_keys)

        surrogate = CombinationSurrogate(
            self._bkd,
            self._nvars_physical,
            all_subspaces,
            all_coefs,
            self._nqoi,
            indices=all_indices,
        )

        nsamples = sum(t.n_unique_samples() for t in self._trackers.values())

        return AdaptiveSparseGridFitResult(
            surrogate=surrogate,
            indices=all_indices,
            coefficients=all_coefs,
            nsamples=nsamples,
            error=self.current_error(),
            nsteps=self._nsteps,
            converged=converged,
        )

    def refine_to_tolerance(
        self,
        model_factory: ModelFactoryProtocol[Array],
        tol: float = 1e-6,
        max_steps: int = 200,
    ) -> AdaptiveSparseGridFitResult[Array]:
        """Refine adaptively until error < tol or max_steps reached.

        Parameters
        ----------
        model_factory : ModelFactoryProtocol[Array]
            Factory mapping config indices to FunctionProtocol models.
        tol : float
            Error tolerance.
        max_steps : int
            Maximum number of refinement steps.

        Returns
        -------
        AdaptiveSparseGridFitResult[Array]
        """
        for _ in range(max_steps):
            samples = self.step_samples()
            if samples is None:
                return self.result(converged=True)

            values = {
                cfg: model_factory.get_model(cfg)(s) for cfg, s in samples.items()
            }
            self.step_values(values)

            if self.current_error() < tol:
                return self.result(converged=True)

        return self.result(converged=False)

    def nvars_physical(self) -> int:
        """Return number of physical variables."""
        return self._nvars_physical

    def _get_tracker_positions(
        self, config_idx: ConfigIdx, subset: SubsetType
    ) -> Set[int]:
        """Get tracker positions for the specified subset.

        Parameters
        ----------
        config_idx : ConfigIdx
            Configuration index.
        subset : SubsetType
            "selected", "candidate", or "all".

        Returns
        -------
        Set[int]
            Tracker positions matching the subset filter.
        """
        pos_map = self._tracker_positions.get(config_idx, {})

        if subset == "all":
            return set(pos_map.values())

        if subset == "selected":
            index_set: Optional[Array] = self._index_gen.get_selected_indices()
        else:
            index_set = self._index_gen.get_candidate_indices()

        if index_set is None:
            return set()

        result: Set[int] = set()
        for j in range(index_set.shape[1]):
            idx = index_set[:, j]
            idx_config = self._get_config_idx(idx)
            if idx_config != config_idx:
                continue
            key = _index_to_tuple(idx, self._bkd)
            if key in pos_map:
                result.add(pos_map[key])
        return result

    def get_samples(self, subset: SubsetType = "all") -> Dict[ConfigIdx, Array]:
        """Return unique samples per config, filtered by subset.

        Parameters
        ----------
        subset : SubsetType
            "selected", "candidate", or "all".

        Returns
        -------
        Dict[ConfigIdx, Array]
            Maps config_idx to sample arrays of shape
            (nvars_physical, n_unique).
        """
        result: Dict[ConfigIdx, Array] = {}
        for cfg, tracker in self._trackers.items():
            if subset == "all":
                result[cfg] = tracker.collect_filtered_unique_samples(None)
            else:
                positions = self._get_tracker_positions(cfg, subset)
                result[cfg] = tracker.collect_filtered_unique_samples(positions)
        return result

    def get_values(
        self, subset: SubsetType = "all"
    ) -> Dict[ConfigIdx, Optional[Array]]:
        """Return unique values per config, filtered by subset.

        Parameters
        ----------
        subset : SubsetType
            "selected", "candidate", or "all".

        Returns
        -------
        Dict[ConfigIdx, Optional[Array]]
            Maps config_idx to value arrays of shape (nqoi, n_unique),
            or None if no values have been set for that config.
        """
        result: Dict[ConfigIdx, Optional[Array]] = {}
        for cfg, tracker in self._trackers.items():
            if subset == "all":
                result[cfg] = tracker.collect_filtered_unique_values(None)
            else:
                positions = self._get_tracker_positions(cfg, subset)
                result[cfg] = tracker.collect_filtered_unique_values(positions)
        return result

    def get_selected_indices(self) -> Array:
        """Return indices of selected subspaces.

        Returns
        -------
        Array
            Selected indices, shape (nvars_index, nselected).
        """
        return self._index_gen.get_selected_indices()

    def get_candidate_indices(self) -> Optional[Array]:
        """Return indices of candidate subspaces.

        Returns
        -------
        Optional[Array]
            Candidate indices, shape (nvars_index, ncandidates),
            or None if there are no candidates.
        """
        return self._index_gen.get_candidate_indices()

    def cumulative_cost(self, cost_model: Optional[CostModelProtocol] = None) -> float:
        """Return total cumulative cost of all evaluations.

        Parameters
        ----------
        cost_model : Optional[CostModelProtocol]
            Cost model to use. If None, uses the fitter's cost model.

        Returns
        -------
        float
            Sum of n_unique_samples * cost_per_sample across all configs.
        """
        if cost_model is None:
            cost_model = self._cost_model
        total = 0.0
        for cfg, tracker in self._trackers.items():
            total += tracker.n_unique_samples() * cost_model(cfg)
        return total

    def nselected(self) -> int:
        """Return number of selected subspaces."""
        return self._index_gen.nselected_indices()

    def ncandidates(self) -> int:
        """Return number of candidate subspaces."""
        return self._index_gen.ncandidate_indices()

    def __repr__(self) -> str:
        return (
            f"MultiFidelityAdaptiveSparseGridFitter("
            f"nvars={self._nvars_physical}, "
            f"nconfig_vars={self._nconfig_vars}, "
            f"nsubspaces={len(self._subspace_by_key)}, "
            f"nselected={self._index_gen.nselected_indices()}, "
            f"ncandidates={self._index_gen.ncandidate_indices()})"
        )


class SingleFidelityAdaptiveSparseGridFitter(Generic[Array]):
    """Adaptive sparse grid fitter for single-fidelity models.

    Thin composition wrapper around MultiFidelityAdaptiveSparseGridFitter
    that converts between Array and Dict[ConfigIdx, Array] at the boundary.

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
        Default: L2SurplusIndicator.
    verbosity : int, optional
        Verbosity level. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        factory: SubspaceFactoryProtocol[Array],
        admissibility: AdmissibilityCriteria[Array],
        error_indicator: Optional[ErrorIndicatorProtocol[Array]] = None,
        verbosity: int = 0,
    ) -> None:
        self._fitter = MultiFidelityAdaptiveSparseGridFitter(
            bkd,
            factory,
            admissibility,
            nconfig_vars=0,
            error_indicator=error_indicator,
            verbosity=verbosity,
        )

    def step_samples(self) -> Optional[Array]:
        """Get samples for next refinement step.

        Returns
        -------
        Optional[Array]
            New samples of shape (nvars, n_new), or None if converged.
        """
        result = self._fitter.step_samples()
        if result is None:
            return None
        return result[_SF_KEY]

    def step_values(self, values: Array) -> None:
        """Provide function values for the samples from step_samples.

        Parameters
        ----------
        values : Array
            Values of shape (nqoi, n_samples).
        """
        self._fitter.step_values({_SF_KEY: values})

    def current_error(self) -> float:
        """Return sum of errors for candidate subspaces."""
        return self._fitter.current_error()

    def result(
        self,
        converged: bool = False,
        include_candidates: bool = True,
    ) -> AdaptiveSparseGridFitResult[Array]:
        """Build result from current state."""
        return self._fitter.result(
            converged=converged,
            include_candidates=include_candidates,
        )

    def refine_to_tolerance(
        self,
        target_fn: FunctionProtocol[Array],
        tol: float = 1e-6,
        max_steps: int = 200,
    ) -> AdaptiveSparseGridFitResult[Array]:
        """Refine adaptively until error < tol or max_steps reached.

        Parameters
        ----------
        target_fn : FunctionProtocol[Array]
            Function satisfying FunctionProtocol.
        tol : float
            Error tolerance.
        max_steps : int
            Maximum number of refinement steps.

        Returns
        -------
        AdaptiveSparseGridFitResult[Array]
        """
        if not isinstance(target_fn, FunctionProtocol):
            raise TypeError(
                "target_fn must satisfy FunctionProtocol, "
                f"got {type(target_fn).__name__}"
            )
        factory: DictModelFactory[Array] = DictModelFactory(
            {_SF_KEY: target_fn}
        )
        return self._fitter.refine_to_tolerance(factory, tol, max_steps)

    def nvars_physical(self) -> int:
        """Return number of physical variables."""
        return self._fitter.nvars_physical()

    def get_samples(self, subset: SubsetType = "all") -> Array:
        """Return unique samples, filtered by subset.

        Parameters
        ----------
        subset : SubsetType
            "selected", "candidate", or "all".

        Returns
        -------
        Array
            Samples of shape (nvars, n_unique).
        """
        return self._fitter.get_samples(subset)[_SF_KEY]

    def get_values(self, subset: SubsetType = "all") -> Optional[Array]:
        """Return unique values, filtered by subset.

        Parameters
        ----------
        subset : SubsetType
            "selected", "candidate", or "all".

        Returns
        -------
        Optional[Array]
            Values of shape (nqoi, n_unique), or None if no values set.
        """
        return self._fitter.get_values(subset)[_SF_KEY]

    def get_selected_indices(self) -> Array:
        """Return indices of selected subspaces.

        Returns
        -------
        Array
            Selected indices, shape (nvars, nselected).
        """
        return self._fitter.get_selected_indices()

    def get_candidate_indices(self) -> Optional[Array]:
        """Return indices of candidate subspaces.

        Returns
        -------
        Optional[Array]
            Candidate indices, or None if there are no candidates.
        """
        return self._fitter.get_candidate_indices()

    def cumulative_cost(self, cost_model: Optional[CostModelProtocol] = None) -> float:
        """Return total cumulative cost of all evaluations.

        Parameters
        ----------
        cost_model : Optional[CostModelProtocol]
            Cost model to use. If None, uses unit cost.

        Returns
        -------
        float
            Total cost.
        """
        return self._fitter.cumulative_cost(cost_model)

    def nselected(self) -> int:
        """Return number of selected subspaces."""
        return self._fitter.nselected()

    def ncandidates(self) -> int:
        """Return number of candidate subspaces."""
        return self._fitter.ncandidates()

    def __repr__(self) -> str:
        return (
            f"SingleFidelityAdaptiveSparseGridFitter("
            f"nvars={self._fitter.nvars_physical()}, "
            f"nsubspaces={len(self._fitter._subspace_by_key)}, "
            f"nselected={self._fitter.nselected()}, "
            f"ncandidates={self._fitter.ncandidates()})"
        )
