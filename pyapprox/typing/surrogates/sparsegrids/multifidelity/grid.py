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
from pyapprox.typing.surrogates.affine.indices.generators import (
    IterativeIndexGenerator,
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
    compute_smolyak_coefficients,
)
from pyapprox.typing.surrogates.sparsegrids.subspace import TensorProductSubspace
from .protocols import MultiFidelityModelFactoryProtocol


def _index_to_tuple(index: Array) -> Tuple[int, ...]:
    """Convert an index array to a tuple for dict keys."""
    return tuple(int(x) for x in index)


class MultiIndexAdaptiveCombinationSparseGrid(Generic[Array]):
    """Multi-fidelity adaptive sparse grid with physical-only subspaces.

    This sparse grid supports multi-fidelity models by tracking subspaces
    with full indices (physical + config) but building PHYSICAL-ONLY
    interpolants. The config part of each subspace's index determines which
    model was called to get values, but does NOT add interpolation dimensions.

    Key Design:
        - Subspaces are tensor product interpolants over PHYSICAL dimensions
        - Config index is metadata: determines which model provides values
        - Samples returned are physical-only: shape (nvars_physical, nsamples)
        - Evaluation takes physical samples: shape (nvars_physical, npoints)

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

        self._bkd = bkd
        self._nvars_physical = len(physical_basis_factories)
        self._nconfig_vars = nconfig_vars
        self._config_bounds = bkd.copy(config_bounds)
        self._nqoi_expected = nqoi
        self._nqoi: Optional[int] = None
        self._model_factory = model_factory
        self._verbosity = verbosity

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

        # Default admissibility with config bounds
        if admissibility is None:
            # No limit on physical vars, bounded on config vars
            max_levels = bkd.hstack(
                [
                    bkd.full((self._nvars_physical,), 1000.0),
                    bkd.flatten(config_bounds),
                ]
            )
            admissibility = Max1DLevelsCriteria(max_levels, bkd)

        # Default refinement with config-aware cost
        if refinement_priority is None:
            cost_fn = ConfigIndexCostFunction(bkd, self._nvars_physical)
            refinement_priority = L2NormRefinementCriteria(bkd, cost_fn)

        self._refinement_priority = refinement_priority

        # Index generator over full (physical + config) space
        nvars_full = self._nvars_physical + nconfig_vars
        self._index_gen = IterativeIndexGenerator(nvars_full, bkd)
        self._index_gen.set_admissibility_criteria(admissibility)
        self._index_gen.set_verbosity(verbosity)

        # Initialize with zero index
        zero_index = bkd.zeros((nvars_full, 1), dtype=bkd.int64_dtype())
        self._index_gen.set_selected_indices(zero_index)

        # Physical-only subspaces keyed by FULL index tuple
        self._subspaces: Dict[Tuple[int, ...], TensorProductSubspace[Array]] = {}

        # Priority queue for refinement: (priority, full_index_tuple)
        self._priorities: Dict[Tuple[int, ...], float] = {}
        self._errors: Dict[Tuple[int, ...], float] = {}

        # Total error estimate
        self._total_error = 0.0

        # Track current step state for step_samples/step_values
        self._pending_subspace_keys: List[Tuple[int, ...]] = []

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

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

    def nsubspaces(self) -> int:
        """Return total number of subspaces."""
        return len(self._subspaces)

    def get_config_bounds(self) -> Array:
        """Return configuration variable bounds."""
        return self._bkd.copy(self._config_bounds)

    def verbosity(self) -> int:
        """Return the current verbosity level."""
        return self._verbosity

    def set_verbosity(self, level: int) -> None:
        """Set the verbosity level."""
        self._verbosity = level
        self._index_gen.set_verbosity(level)

    def error_estimate(self) -> float:
        """Return current total error estimate."""
        return self._total_error

    def _evaluate_selected_only(self, samples: Array) -> Array:
        """Evaluate only selected subspaces at given samples.

        Used by refinement criteria to compute hierarchical surpluses.

        Parameters
        ----------
        samples : Array
            Physical samples of shape (nvars_physical, nsamples).

        Returns
        -------
        Array
            Interpolant values from selected subspaces only.
        """
        if self._nqoi is None:
            return self._bkd.zeros((self._nqoi_expected, samples.shape[1]))

        # Get Smolyak coefficients for selected indices
        selected_indices = self._index_gen.get_selected_indices()
        if selected_indices.shape[1] == 0:
            return self._bkd.zeros((self._nqoi, samples.shape[1]))

        smolyak_coefs = compute_smolyak_coefficients(selected_indices, self._bkd)

        npoints = samples.shape[1]
        result = self._bkd.zeros((self._nqoi, npoints))

        for j in range(selected_indices.shape[1]):
            index = selected_indices[:, j]
            key = _index_to_tuple(index)

            coef = float(smolyak_coefs[j])
            if abs(coef) < 1e-14:
                continue

            if key not in self._subspaces:
                continue

            subspace = self._subspaces[key]
            if subspace.get_values() is None:
                continue

            result = result + coef * subspace(samples)

        return result

    def get_selected_subspace_indices(self) -> Array:
        """Return indices of selected subspaces.

        Returns
        -------
        Array
            Multi-indices of shape (nvars_physical + nconfig_vars, nselected)
        """
        return self._index_gen.get_selected_indices()

    def _extract_config_index(self, full_index: Array) -> Tuple[int, ...]:
        """Extract config index tuple from full index."""
        config_part = full_index[self._nvars_physical:]
        return tuple(int(x) for x in self._bkd.to_numpy(config_part))

    def _extract_physical_index(self, full_index: Array) -> Array:
        """Extract physical index from full index."""
        return full_index[: self._nvars_physical]

    def _create_physical_subspace(
        self, full_index: Array
    ) -> TensorProductSubspace[Array]:
        """Create a subspace over physical dimensions only.

        Parameters
        ----------
        full_index : Array
            Full index including physical and config parts.

        Returns
        -------
        TensorProductSubspace
            Subspace with samples over physical dimensions only.
        """
        physical_index = self._extract_physical_index(full_index)

        subspace = TensorProductSubspace(
            self._bkd,
            physical_index,
            self._physical_basis_factories,
            self._growth_rule,
        )
        return subspace

    def step_samples(self) -> Optional[Dict[Tuple[int, ...], Array]]:
        """Get physical samples grouped by config index for next step.

        Returns
        -------
        Optional[Dict[Tuple[int, ...], Array]]
            None if converged (no more candidates).
            Otherwise, maps config_index -> physical_samples for that config.
            Each physical_samples has shape (nvars_physical, n_samples_for_config).
        """
        # Get candidate indices
        candidate_indices = self._index_gen.get_candidate_indices()
        if candidate_indices is None or candidate_indices.shape[1] == 0:
            return None

        # Find highest priority candidate to refine
        if not self._priorities:
            # First step: refine all candidates (typically just [0,0,...,0])
            indices_to_refine = [
                _index_to_tuple(candidate_indices[:, j])
                for j in range(candidate_indices.shape[1])
            ]
        else:
            # Find candidate with highest priority
            best_priority = float("-inf")
            best_key = None
            for j in range(candidate_indices.shape[1]):
                key = _index_to_tuple(candidate_indices[:, j])
                # New candidates don't have priorities yet - give them high priority
                priority = self._priorities.get(key, float("inf"))
                if priority > best_priority:
                    best_priority = priority
                    best_key = key
            if best_key is None:
                return None
            indices_to_refine = [best_key]

        # Create subspaces for indices to refine and group samples by config
        config_samples: Dict[Tuple[int, ...], List[Array]] = {}
        self._pending_subspace_keys = []

        for full_key in indices_to_refine:
            full_index = self._bkd.asarray(full_key, dtype=self._bkd.int64_dtype())
            config_idx = self._extract_config_index(full_index)

            # Create physical-only subspace
            subspace = self._create_physical_subspace(full_index)
            self._subspaces[full_key] = subspace
            self._pending_subspace_keys.append(full_key)

            # Get physical samples from subspace
            physical_samples = subspace.get_samples()

            if config_idx not in config_samples:
                config_samples[config_idx] = []
            config_samples[config_idx].append(physical_samples)

        # Merge samples for each config
        result: Dict[Tuple[int, ...], Array] = {}
        for config_idx, samples_list in config_samples.items():
            # Concatenate all samples for this config
            # Note: Some samples may be duplicates across subspaces
            all_samples = self._bkd.hstack(samples_list)
            result[config_idx] = all_samples

        return result

    def step_values(self, values_dict: Dict[Tuple[int, ...], Array]) -> None:
        """Provide function values grouped by config index.

        Parameters
        ----------
        values_dict : Dict[Tuple[int, ...], Array]
            Maps config_index -> values array (nqoi, nsamples_for_config).

        Raises
        ------
        ValueError
            If values shape is wrong or step_samples() wasn't called.
        """
        if not self._pending_subspace_keys:
            raise ValueError("Call step_samples() before step_values()")

        # Track value offset for each config
        config_offsets: Dict[Tuple[int, ...], int] = {
            cfg: 0 for cfg in values_dict.keys()
        }

        # Distribute values to subspaces
        for full_key in self._pending_subspace_keys:
            full_index = self._bkd.asarray(full_key, dtype=self._bkd.int64_dtype())
            config_idx = self._extract_config_index(full_index)
            subspace = self._subspaces[full_key]

            if config_idx not in values_dict:
                raise ValueError(f"Missing values for config {config_idx}")

            config_values = values_dict[config_idx]
            nsamples = subspace.nsamples()
            offset = config_offsets[config_idx]

            # Extract values for this subspace
            subspace_values = config_values[:, offset : offset + nsamples]
            subspace.set_values(subspace_values)
            config_offsets[config_idx] = offset + nsamples

            # Compute refinement priority for this subspace
            priority, error = self._refinement_priority(
                full_index, subspace_values, self  # type: ignore
            )
            self._priorities[full_key] = priority
            self._errors[full_key] = error

            # Update total error
            self._total_error += error

            # Refine this index in the index generator
            self._index_gen.refine_index(full_index)

        # Update nqoi from first values
        first_values = next(iter(values_dict.values()))
        self._nqoi = first_values.shape[0]

        # Clear pending state
        self._pending_subspace_keys = []

    def step(self) -> bool:
        """Perform one refinement step with automatic model evaluation.

        Uses model_factory from constructor.

        Returns
        -------
        bool
            False if converged (no more samples), True otherwise.

        Raises
        ------
        ValueError
            If model_factory was not provided in constructor.
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

    def __call__(self, samples: Array) -> Array:
        """Evaluate surrogate at physical samples.

        Uses full Smolyak combination over physical-only subspaces.

        Parameters
        ----------
        samples : Array
            Physical samples of shape (nvars_physical, nsamples).

        Returns
        -------
        Array
            Interpolant values of shape (nqoi, nsamples).

        Raises
        ------
        ValueError
            If samples has wrong number of dimensions.
        """
        if samples.shape[0] != self._nvars_physical:
            raise ValueError(
                f"Expected {self._nvars_physical} physical variables, "
                f"got {samples.shape[0]}"
            )

        if self._nqoi is None:
            return self._bkd.zeros((self._nqoi_expected, samples.shape[1]))

        # Get Smolyak coefficients for selected indices
        selected_indices = self._index_gen.get_selected_indices()
        if selected_indices.shape[1] == 0:
            return self._bkd.zeros((self._nqoi, samples.shape[1]))

        smolyak_coefs = compute_smolyak_coefficients(selected_indices, self._bkd)

        npoints = samples.shape[1]
        result = self._bkd.zeros((self._nqoi, npoints))

        for j in range(selected_indices.shape[1]):
            index = selected_indices[:, j]
            key = _index_to_tuple(index)

            coef = float(smolyak_coefs[j])
            if abs(coef) < 1e-14:
                continue

            if key not in self._subspaces:
                continue

            subspace = self._subspaces[key]
            if subspace.get_values() is None:
                continue

            # Evaluate physical-only subspace at physical samples
            result = result + coef * subspace(samples)

        return result

    def __repr__(self) -> str:
        return (
            f"MultiIndexAdaptiveCombinationSparseGrid("
            f"nvars_physical={self._nvars_physical}, "
            f"nconfig_vars={self._nconfig_vars}, "
            f"nsubspaces={self.nsubspaces()}, "
            f"nselected={self._index_gen.nselected_indices()}, "
            f"ncandidates={self._index_gen.ncandidate_indices()})"
        )
