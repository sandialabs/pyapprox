"""Multi-index adaptive sparse grid for multi-fidelity models.

This module provides sparse grid construction for multi-fidelity models
where configuration/fidelity indices determine which model is evaluated,
but subspaces remain physical-dimension interpolants only.

Key insight: Subspaces are interpolants over PHYSICAL dimensions only.
Config indices are metadata that determine which model provides values.
"""

from typing import Dict, Generic, List, Optional, Sequence, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.affine.indices import (
    AdmissibilityCriteria,
    Max1DLevelsCriteria,
    LinearGrowthRule,
)
from pyapprox.typing.surrogates.affine.indices.basis_generator import (
    BasisIndexGenerator,
)

from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.protocols import (
    SparseGridRefinementCriteriaProtocol,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.cost import (
    ConfigIndexCostFunction,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.l2norm import (
    L2NormRefinementCriteria,
)
from pyapprox.typing.surrogates.sparsegrids.smolyak import (
    _index_to_tuple,
    compute_smolyak_coefficients,
)
from pyapprox.typing.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.typing.surrogates.sparsegrids.adaptive import (
    AdaptiveCombinationSparseGrid,
)
from .protocols import MultiFidelityModelFactoryProtocol


class MultiIndexAdaptiveCombinationSparseGrid(
    AdaptiveCombinationSparseGrid[Array], Generic[Array]
):
    """Multi-fidelity adaptive sparse grid with physical-only subspaces.

    Inherits from AdaptiveCombinationSparseGrid to reuse:
    - Priority queue for O(log n) candidate selection
    - Incremental Smolyak coefficient updates
    - Error tracking and ``error_estimate()``
    - ``_evaluate_with_selected_indices()`` for refinement criteria
    - ``_prioritize_candidates()`` for error computation on ALL candidates
    - ``mean()`` and ``variance()`` moment computation

    Overrides subspace construction and step API to handle the multi-index
    (physical + config) structure where subspaces are physical-only
    interpolants but the Smolyak combination operates over the full index
    space.

    Two usage modes:
        - **Manual mode**: Use step_samples() / step_values() with dict API
        - **Auto mode**: Use step() with model factory

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    physical_basis_factories : Sequence[BasisFactoryProtocol[Array]]
        Factories for creating univariate bases for physical input variables.
    nconfig_vars : int
        Number of configuration/fidelity variables.
    config_bounds : Array
        Maximum level for each config variable. Shape: (nconfig_vars,).
    nqoi : int
        Number of quantities of interest.
    growth_rule : IndexGrowthRuleProtocol, optional
        Rule mapping level to number of points for physical dimensions.
        Default: LinearGrowthRule(scale=2, shift=1).
    admissibility : AdmissibilityCriteria[Array], optional
        Criteria for admissible subspace indices (over full index space).
        If None, uses Max1DLevelsCriteria with config_bounds.
    refinement_priority : SparseGridRefinementCriteriaProtocol[Array], optional
        Criteria for computing refinement priorities.
        Default: L2NormRefinementCriteria with ConfigIndexCostFunction.
    model_factory : MultiFidelityModelFactoryProtocol[Array], optional
        Factory for creating models per config index. Required for auto mode.
    verbosity : int, optional
        Verbosity level (0-2).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        physical_basis_factories: Sequence[BasisFactoryProtocol[Array]],
        nconfig_vars: int,
        config_bounds: Array,
        nqoi: int,
        growth_rule: Optional[IndexGrowthRuleProtocol] = None,
        admissibility: Optional[AdmissibilityCriteria[Array]] = None,
        refinement_priority: Optional[
            SparseGridRefinementCriteriaProtocol[Array]
        ] = None,
        model_factory: Optional[MultiFidelityModelFactoryProtocol[Array]] = None,
        verbosity: int = 0,
    ) -> None:
        # Validate model factory if provided
        if model_factory is not None:
            if not isinstance(model_factory, MultiFidelityModelFactoryProtocol):
                raise TypeError(
                    f"model_factory must satisfy MultiFidelityModelFactoryProtocol, "
                    f"got {type(model_factory).__name__}"
                )

        self._nconfig_vars = nconfig_vars
        self._config_bounds = bkd.copy(config_bounds)
        self._nqoi_expected = nqoi
        self._model_factory = model_factory

        if config_bounds.shape[0] != nconfig_vars:
            raise ValueError(
                f"config_bounds must have length {nconfig_vars}, "
                f"got {config_bounds.shape[0]}"
            )

        # Default growth rule
        if growth_rule is None:
            growth_rule = LinearGrowthRule(scale=2, shift=1)

        self._physical_basis_factories = list(physical_basis_factories)
        self._growth_rule = growth_rule
        nvars_physical = len(physical_basis_factories)
        nvars_full = nvars_physical + nconfig_vars

        # Default admissibility with config bounds
        if admissibility is None:
            max_levels = bkd.hstack(
                [
                    bkd.full((nvars_physical,), 1000.0),
                    bkd.flatten(config_bounds),
                ]
            )
            admissibility = Max1DLevelsCriteria(max_levels, bkd)

        # Default refinement with config-aware cost
        if refinement_priority is None:
            cost_fn = ConfigIndexCostFunction(bkd, nvars_physical)
            refinement_priority = L2NormRefinementCriteria(bkd, cost_fn)

        # Call parent __init__ with physical basis factories and
        # nvars_index=nvars_full so the IterativeIndexGenerator operates
        # over the full (physical + config) index space.
        super().__init__(
            bkd,
            physical_basis_factories,
            growth_rule,
            admissibility,
            refinement_priority,
            verbosity,
            nvars_index=nvars_full,
        )

        # Store nvars_physical explicitly (parent's self._nvars is
        # len(physical_basis_factories) which is correct)
        self._nvars_physical = nvars_physical

        # Track pending subspace keys for step_samples/step_values
        self._pending_subspace_keys: List[Tuple[int, ...]] = []

        # Per-config BasisIndexGenerator for sample deduplication.
        # Within a config group, nested subspaces share sample points;
        # the BasisIndexGenerator tracks which are unique via integer
        # basis indices (no float hashing needed).
        all_nested = all(
            getattr(f, "is_nested", lambda: False)()
            for f in physical_basis_factories
        )
        self._all_nested = all_nested
        self._config_basis_gens: Dict[
            Tuple[int, ...], BasisIndexGenerator[Array]
        ] = {}
        # Maps full_index_key -> subspace_idx within its config's BasisIndexGenerator
        self._subspace_config_idx: Dict[Tuple[int, ...], int] = {}

        # Per-config accumulated values: config_idx -> (nqoi, n_unique)
        self._config_values: Dict[Tuple[int, ...], Array] = {}

    def _create_subspace(self, index: Array) -> TensorProductSubspace[Array]:
        """Create a subspace over physical dimensions only.

        The index may be a full (physical + config) index; only the
        physical part is used to build the tensor product subspace.

        Parameters
        ----------
        index : Array
            Full index. Shape: (nvars_physical + nconfig_vars,) or
            (nvars_physical,).

        Returns
        -------
        TensorProductSubspace[Array]
            Subspace with samples over physical dimensions only.
        """
        physical_index = index[: self._nvars_physical]
        return TensorProductSubspace(
            self._bkd,
            physical_index,
            self._physical_basis_factories,
            self._growth_rule,
        )

    def _get_or_create_config_basis_gen(
        self, config_idx: Tuple[int, ...]
    ) -> BasisIndexGenerator[Array]:
        """Get or create a BasisIndexGenerator for a config group."""
        if config_idx not in self._config_basis_gens:
            self._config_basis_gens[config_idx] = BasisIndexGenerator(
                self._bkd,
                self._nvars_physical,
                self._growth_rule,
                all_nested=self._all_nested,
            )
        return self._config_basis_gens[config_idx]

    def _register_subspace_with_config(
        self, full_key: Tuple[int, ...], index: Array,
        subspace: TensorProductSubspace[Array],
    ) -> None:
        """Register a subspace with its config group's BasisIndexGenerator."""
        config_idx = self._extract_config_index(index)
        basis_gen = self._get_or_create_config_basis_gen(config_idx)
        physical_index = index[: self._nvars_physical]
        config_subspace_idx = basis_gen.n_unique_samples()  # use as counter
        # The subspace_idx for the BasisIndexGenerator is the count of
        # subspaces already registered in this config group.
        config_sub_idx = len(basis_gen._unique_subspace_basis_idx)
        basis_gen._set_unique_subspace_basis_indices(
            physical_index, config_sub_idx,
            subspace_samples=subspace.get_samples(),
        )
        self._subspace_config_idx[full_key] = config_sub_idx

    def _add_subspace(self, index: Array) -> TensorProductSubspace[Array]:
        """Add a subspace with per-config BasisIndexGenerator dedup."""
        key = _index_to_tuple(index)
        if key in self._subspaces:
            return self._subspaces[key]

        subspace = self._create_subspace(index)
        self._subspaces[key] = subspace
        self._subspace_list.append(subspace)

        self._register_subspace_with_config(key, index, subspace)

        self._smolyak_coefficients = None
        self._unique_samples = None

        return subspace

    def _add_subspace_without_invalidating(
        self, index: Array
    ) -> TensorProductSubspace[Array]:
        """Add a subspace without invalidating unique samples cache."""
        key = _index_to_tuple(index)
        if key in self._subspaces:
            return self._subspaces[key]

        subspace = self._create_subspace(index)
        self._subspaces[key] = subspace
        self._subspace_list.append(subspace)

        self._register_subspace_with_config(key, index, subspace)

        self._smolyak_coefficients = None

        return subspace

    def _extract_config_index(self, full_index: Array) -> Tuple[int, ...]:
        """Extract config index tuple from full index."""
        config_part = full_index[self._nvars_physical:]
        return tuple(int(x) for x in self._bkd.to_numpy(config_part))

    # -- Step API (dict-based) --

    def step_samples(self) -> Optional[Dict[Tuple[int, ...], Array]]:
        """Get physical samples grouped by config index for next step.

        Delegates to the parent's step machinery (priority queue,
        incremental Smolyak, index generator) but returns samples
        grouped by config index instead of a flat array.

        Returns
        -------
        Optional[Dict[Tuple[int, ...], Array]]
            None if converged (no more candidates).
            Otherwise, maps config_index -> physical_samples.
            Each physical_samples has shape (nvars_physical, nsamples).
        """
        if self._first_step:
            return self._first_step_samples_dict()
        return self._next_step_samples_dict()

    def _first_step_samples_dict(
        self,
    ) -> Optional[Dict[Tuple[int, ...], Array]]:
        """First step: initialize index gen and return samples by config.

        Replicates the parent's _first_step_samples() logic but skips
        _collect_unique_samples() since multi-fidelity grids don't use
        BasisIndexGenerator for cross-config sample deduplication.
        """
        # Initialize with zero index over the full index space
        zero_index = self._bkd.zeros(
            (self._nvars_index, 1), dtype=self._bkd.int64_dtype()
        )
        self._index_gen.set_selected_indices(zero_index)

        # Add selected subspaces (start with error 0)
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
            ncandidates = cand_indices.shape[1]
            self._selected_smolyak_coefs = self._bkd.hstack(
                (self._selected_smolyak_coefs, self._bkd.zeros((ncandidates,)))
            )
            for index in cand_indices.T:
                self._add_subspace(index)
                self._subspace_errors.append(float("inf"))

        self._first_step = False

        # Gather samples for ALL subspaces (selected + candidates).
        # The selected subspace (0,...,0) also needs values on first step.
        all_indices = self._index_gen.get_indices()
        self._pending_subspace_keys = [
            _index_to_tuple(all_indices[:, j])
            for j in range(all_indices.shape[1])
        ]
        return self._group_samples_by_config(all_indices)

    def _next_step_samples_dict(
        self,
    ) -> Optional[Dict[Tuple[int, ...], Array]]:
        """Subsequent steps: pop best candidate, refine, return new samples.

        Uses parent's priority queue and index generator.
        """
        while (
            self._candidate_queue is not None
            and not self._candidate_queue.empty()
        ):
            priority, error, best_idx = self._candidate_queue.get()
            best_index = self._index_gen._indices[:, best_idx]
            best_index_tuple = _index_to_tuple(best_index)

            if self._verbosity >= 1:
                total_error = sum(
                    e for e in self._subspace_errors if e != float("inf")
                )
                print(
                    f"[MF SG] Refining {best_index_tuple}: "
                    f"priority={priority:.2e}, error={error:.2e}, "
                    f"total_error={total_error:.2e}"
                )

            # Refine (moves from candidate to selected in index gen)
            new_cand_indices = self._index_gen.refine_index(best_index)

            # Extend Smolyak coefficients for new candidates
            if self._selected_smolyak_coefs is not None:
                self._selected_smolyak_coefs = self._bkd.hstack(
                    (
                        self._selected_smolyak_coefs,
                        self._bkd.zeros((new_cand_indices.shape[1],)),
                    )
                )

            # Incrementally update Smolyak coefficients.
            # Only iterate over selected positions (not candidates).
            if self._selected_smolyak_coefs is not None:
                selected_positions = self._index_gen._get_selected_idx()
                self._selected_smolyak_coefs = (
                    self._adjust_smolyak_coefficients(
                        self._selected_smolyak_coefs,
                        best_index,
                        selected_positions,
                    )
                )

            # Reset error for the refined subspace
            self._subspace_errors[best_idx] = 0.0

            # Add new candidate subspaces
            for index in new_cand_indices.T:
                self._add_subspace_without_invalidating(index)
                self._subspace_errors.append(float("inf"))

            if new_cand_indices.shape[1] == 0:
                if self._verbosity >= 2:
                    print(
                        f"[MF SG] No new candidates from {best_index_tuple}, "
                        "trying next best"
                    )
                continue

            # Group new candidate samples by config
            result = self._group_samples_by_config(new_cand_indices)
            if result:
                self._pending_subspace_keys = [
                    _index_to_tuple(new_cand_indices[:, j])
                    for j in range(new_cand_indices.shape[1])
                ]
                return result

        return None

    def _gather_candidate_samples(
        self,
    ) -> Optional[Dict[Tuple[int, ...], Array]]:
        """Gather samples from all candidate subspaces, grouped by config."""
        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None or cand_indices.shape[1] == 0:
            return None

        self._pending_subspace_keys = [
            _index_to_tuple(cand_indices[:, j])
            for j in range(cand_indices.shape[1])
        ]
        return self._group_samples_by_config(cand_indices)

    def _group_samples_by_config(
        self, indices: Array
    ) -> Dict[Tuple[int, ...], Array]:
        """Group NEW unique subspace samples by config index.

        Uses per-config BasisIndexGenerator to identify which samples
        in each subspace are new (not seen in a previously-added
        subspace with the same config). Only those new samples are
        returned, avoiding duplicate model evaluations.

        Parameters
        ----------
        indices : Array
            Full indices. Shape: (nvars_full, nindices)

        Returns
        -------
        Dict[Tuple[int, ...], Array]
            Maps config_index -> new unique physical samples
            (nvars_physical, n_new).
        """
        config_samples: Dict[Tuple[int, ...], List[Array]] = {}

        for j in range(indices.shape[1]):
            full_index = indices[:, j]
            key = _index_to_tuple(full_index)
            config_idx = self._extract_config_index(full_index)

            if key not in self._subspaces:
                continue

            subspace = self._subspaces[key]
            config_sub_idx = self._subspace_config_idx[key]
            basis_gen = self._config_basis_gens[config_idx]

            unique_local = basis_gen.get_unique_local_indices(config_sub_idx)

            if len(unique_local) > 0:
                samples = subspace.get_samples()
                idx_array = self._bkd.asarray(
                    unique_local, dtype=self._bkd.int64_dtype()
                )
                if config_idx not in config_samples:
                    config_samples[config_idx] = []
                config_samples[config_idx].append(samples[:, idx_array])

        result: Dict[Tuple[int, ...], Array] = {}
        for config_idx, samples_list in config_samples.items():
            result[config_idx] = self._bkd.hstack(samples_list)

        return result

    def step_values(self, values_dict: Dict[Tuple[int, ...], Array]) -> None:
        """Provide function values for the NEW unique samples per config.

        The user provides values only for the new unique samples returned
        by ``step_samples()``. This method uses the per-config
        ``BasisIndexGenerator`` to distribute values to all pending
        subspaces (including samples shared with earlier subspaces via
        nesting).

        Parameters
        ----------
        values_dict : Dict[Tuple[int, ...], Array]
            Maps config_index -> values (nqoi, n_new_unique_for_config).
        """
        if not self._pending_subspace_keys:
            raise ValueError("Call step_samples() before step_values()")

        # 1. Append new unique values to per-config value stores.
        for config_idx, new_vals in values_dict.items():
            if config_idx not in self._config_values:
                self._config_values[config_idx] = new_vals
            else:
                self._config_values[config_idx] = self._bkd.hstack(
                    (self._config_values[config_idx], new_vals)
                )

        # 2. Distribute values to each pending subspace using the
        #    BasisIndexGenerator's global index mapping.
        all_new_values: List[Array] = []
        for full_key in self._pending_subspace_keys:
            full_index = self._bkd.asarray(
                full_key, dtype=self._bkd.int64_dtype()
            )
            config_idx = self._extract_config_index(full_index)

            if config_idx not in values_dict:
                raise ValueError(f"Missing values for config {config_idx}")

            subspace = self._subspaces[full_key]
            basis_gen = self._config_basis_gens[config_idx]
            config_sub_idx = self._subspace_config_idx[full_key]
            global_indices = basis_gen.get_subspace_value_indices(
                config_sub_idx
            )

            subspace_values = self._config_values[config_idx][:, global_indices]
            subspace.set_values(subspace_values)

            # Collect only the NEW unique values for _values
            unique_local = basis_gen.get_unique_local_indices(config_sub_idx)
            if len(unique_local) > 0:
                idx_array = self._bkd.asarray(
                    unique_local, dtype=self._bkd.int64_dtype()
                )
                all_new_values.append(subspace_values[:, idx_array])

        # 3. Accumulate global _values array (mirrors parent pattern).
        if all_new_values:
            new_values = self._bkd.hstack(all_new_values)
            if self._values is None:
                self._values = new_values
                self._nqoi = new_values.shape[0]
            else:
                self._values = self._bkd.hstack((self._values, new_values))
                self._nqoi = new_values.shape[0]
        elif self._nqoi is None:
            # Edge case: all samples were duplicates but nqoi not set yet
            first_vals = next(iter(values_dict.values()))
            self._nqoi = first_vals.shape[0]

        # 4. Clear pending state and re-prioritize.
        self._pending_subspace_keys = []
        self._prioritize_candidates()

    # -- Auto mode --

    def step(self) -> bool:
        """Perform one refinement step with automatic model evaluation.

        Returns
        -------
        bool
            False if converged (no more samples), True otherwise.
        """
        if self._model_factory is None:
            raise ValueError(
                "model_factory required for auto mode. "
                "Use step_samples()/step_values() for manual mode."
            )

        config_requests = self.step_samples()
        if config_requests is None:
            return False

        values_dict: Dict[Tuple[int, ...], Array] = {}
        for config_idx, phys_samples in config_requests.items():
            model = self._model_factory.get_model(config_idx)
            values_dict[config_idx] = model(phys_samples)

        self.step_values(values_dict)
        return True

    def refine_to_tolerance(self, tol: float, max_steps: int = 100) -> int:
        """Refine until error estimate < tol or max_steps reached.

        Parameters
        ----------
        tol : float
            Target error tolerance.
        max_steps : int
            Maximum refinement steps.

        Returns
        -------
        int
            Number of steps taken.
        """
        for step_count in range(max_steps):
            if not self.step():
                return step_count
            if self.error_estimate() < tol:
                return step_count + 1
        return max_steps

    # -- Evaluation --

    def __call__(self, samples: Array) -> Array:
        """Evaluate surrogate at physical samples.

        Parameters
        ----------
        samples : Array
            Physical samples of shape (nvars_physical, nsamples).

        Returns
        -------
        Array
            Interpolant values of shape (nqoi, nsamples).
        """
        if samples.shape[0] != self._nvars_physical:
            raise ValueError(
                f"Expected {self._nvars_physical} physical variables, "
                f"got {samples.shape[0]}"
            )

        if self._nqoi is None:
            return self._bkd.zeros((self._nqoi_expected, samples.shape[1]))

        return self._evaluate_with_all_indices(samples)

    # -- Accessors --

    def nvars(self) -> int:
        """Return number of input variables (physical only)."""
        return self._nvars_physical

    def nvars_physical(self) -> int:
        """Return number of physical input variables."""
        return self._nvars_physical

    def nconfig_vars(self) -> int:
        """Return number of configuration variables."""
        return self._nconfig_vars

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        if self._nqoi is not None:
            return self._nqoi
        return self._nqoi_expected

    def get_config_bounds(self) -> Array:
        """Return configuration variable bounds."""
        return self._bkd.copy(self._config_bounds)

    def get_selected_subspace_indices(self) -> Array:
        """Return indices of selected subspaces.

        Returns
        -------
        Array
            Multi-indices of shape (nvars_physical + nconfig_vars, nselected)
        """
        return self._index_gen.get_selected_indices()

    def set_verbosity(self, level: int) -> None:
        """Set the verbosity level."""
        self._verbosity = level
        self._index_gen.set_verbosity(level)

    def __repr__(self) -> str:
        return (
            f"MultiIndexAdaptiveCombinationSparseGrid("
            f"nvars_physical={self._nvars_physical}, "
            f"nconfig_vars={self._nconfig_vars}, "
            f"nsubspaces={self.nsubspaces()}, "
            f"nselected={self._index_gen.nselected_indices()}, "
            f"ncandidates={self._index_gen.ncandidate_indices()})"
        )
