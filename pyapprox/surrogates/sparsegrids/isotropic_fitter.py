"""Isotropic sparse grid fitter.

Builds a fixed-level sparse grid using HyperbolicIndexGenerator,
collects unique samples, and fits a CombinationSurrogate.
"""

from typing import Dict, Generic, List, Optional, Tuple, Union

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.indices import HyperbolicIndexGenerator
from pyapprox.surrogates.sparsegrids.smolyak import (
    compute_smolyak_coefficients,
    _index_to_tuple,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    SubspaceFactoryProtocol,
)
from pyapprox.surrogates.sparsegrids.sample_tracker import (
    SampleTracker,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.fit_result import (
    IsotropicSparseGridFitResult,
)
from pyapprox.surrogates.sparsegrids.candidate_info import ConfigIdx


class IsotropicSparseGridFitter(Generic[Array]):
    """Fixed-level isotropic sparse grid fitter.

    Builds all subspaces at construction time, provides sample locations
    via get_samples(), and fits a CombinationSurrogate from values via fit().

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    factory : SubspaceFactoryProtocol[Array]
        Factory for creating tensor product subspaces.
    level : int
        Maximum level (L1 norm bound for subspace indices).
    pnorm : float, optional
        p-norm for the hyperbolic cross index set. Default: 1.0.
    nconfig_vars : int, optional
        Number of config/fidelity dimensions. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        factory: SubspaceFactoryProtocol[Array],
        level: int,
        pnorm: float = 1.0,
        nconfig_vars: int = 0,
    ) -> None:
        self._bkd = bkd
        self._factory = factory
        self._level = level
        self._pnorm = pnorm
        self._nconfig_vars = nconfig_vars
        self._nvars_physical = factory.nvars_physical()
        self._nvars_index = self._nvars_physical + nconfig_vars

        # Build index set
        self._index_gen = HyperbolicIndexGenerator(
            nvars=self._nvars_index,
            max_level=level,
            pnorm=pnorm,
            bkd=bkd,
        )
        indices = self._index_gen.get_selected_indices()

        # Create subspaces and register with sample trackers
        # One tracker per config group
        self._trackers: Dict[ConfigIdx, SampleTracker[Array]] = {}
        self._subspaces: List[TensorProductSubspace[Array]] = []
        self._subspace_keys: List[Tuple[int, ...]] = []
        # Maps (config_idx, subspace_pos_in_tracker) for each subspace
        self._subspace_config: List[Tuple[ConfigIdx, int]] = []

        for j in range(indices.shape[1]):
            full_index = indices[:, j]
            self._create_subspace(full_index)

        # Compute Smolyak coefficients
        physical_indices = self._bkd.zeros(
            (self._nvars_index, len(self._subspaces)),
            dtype=self._bkd.int64_dtype(),
        )
        for j, subspace in enumerate(self._subspaces):
            physical_indices[: self._nvars_physical, j] = (
                subspace.get_index()
            )
            # Fill config dims from full index
            full_idx_arr = indices[:, j]
            physical_indices[self._nvars_physical :, j] = (
                full_idx_arr[self._nvars_physical :]
            )
        self._indices = indices
        self._smolyak_coefs = compute_smolyak_coefficients(indices, bkd)

    def _create_subspace(self, full_index: Array) -> None:
        """Create a subspace and register with the appropriate tracker."""
        config_idx = self._get_config_idx(full_index)

        if config_idx not in self._trackers:
            self._trackers[config_idx] = SampleTracker(
                self._bkd, self._factory
            )

        tracker = self._trackers[config_idx]
        subspace = self._factory(full_index)
        pos = tracker.register(full_index, subspace)

        # Invariant: every subspace must contribute new samples
        unique_local = tracker.get_unique_local_indices(pos)
        if len(unique_local) == 0:
            raise ValueError(
                f"Subspace {_index_to_tuple(full_index)} contributes no "
                f"new samples. This indicates a problem with the growth "
                f"rule or basis factory configuration."
            )

        key = _index_to_tuple(full_index)
        self._subspaces.append(subspace)
        self._subspace_keys.append(key)
        self._subspace_config.append((config_idx, pos))

    def _get_config_idx(self, full_index: Array) -> ConfigIdx:
        """Extract config index from full multi-index."""
        if self._nconfig_vars == 0:
            return ()
        config_part = full_index[self._nvars_physical :]
        return tuple(int(config_part[i]) for i in range(self._nconfig_vars))

    def get_samples(self) -> Union[Array, Dict[ConfigIdx, Array]]:
        """Return unique sample locations.

        For single-fidelity (nconfig_vars=0), returns Array of shape
        (nvars_physical, n_unique). For multi-fidelity, returns a dict
        mapping config_idx to sample arrays.

        Returns
        -------
        Union[Array, Dict[ConfigIdx, Array]]
            Sample locations.
        """
        if self._nconfig_vars == 0:
            tracker = self._trackers[()]
            return tracker.collect_unique_samples()
        else:
            return {
                cfg: tracker.collect_unique_samples()
                for cfg, tracker in self._trackers.items()
            }

    def get_values(
        self,
    ) -> Optional[Union[Array, Dict[ConfigIdx, Array]]]:
        """Return unique values aligned with get_samples().

        Must be called after fit(). Returns None if fit() has not been
        called yet.

        For single-fidelity (nconfig_vars=0), returns Array of shape
        (nqoi, n_unique). For multi-fidelity, returns a dict mapping
        config_idx to value arrays.

        Returns
        -------
        Optional[Union[Array, Dict[ConfigIdx, Array]]]
            Unique values, or None if fit() has not been called.
        """
        # Check if any tracker has values
        has_values = any(
            t.nqoi() is not None for t in self._trackers.values()
        )
        if not has_values:
            return None

        if self._nconfig_vars == 0:
            tracker = self._trackers[()]
            vals = tracker.collect_filtered_unique_values(None)
            return vals
        else:
            dict_result: Dict[ConfigIdx, Array] = {}
            for cfg, tracker in self._trackers.items():
                vals = tracker.collect_filtered_unique_values(None)
                if vals is not None:
                    dict_result[cfg] = vals
            return dict_result

    def fit(
        self, values: Union[Array, Dict[ConfigIdx, Array]]
    ) -> IsotropicSparseGridFitResult[Array]:
        """Fit the sparse grid from function values.

        Parameters
        ----------
        values : Union[Array, Dict[ConfigIdx, Array]]
            For SF: Array of shape (nqoi, n_unique_samples).
            For MF: dict mapping config_idx to value arrays.

        Returns
        -------
        IsotropicSparseGridFitResult[Array]
            The fit result containing the surrogate.
        """
        wrapped = self._wrap(values)
        for cfg, vals in wrapped.items():
            self._trackers[cfg].append_new_values(vals)

        # Distribute values to subspaces
        for tracker in self._trackers.values():
            tracker.distribute_values_to_subspaces()

        nqoi = next(iter(wrapped.values())).shape[0]

        # Build surrogate
        surrogate = CombinationSurrogate(
            self._bkd,
            self._nvars_physical,
            self._subspaces,
            self._smolyak_coefs,
            nqoi,
            indices=self._indices,
        )

        nsamples = sum(
            t.n_unique_samples() for t in self._trackers.values()
        )

        return IsotropicSparseGridFitResult(
            surrogate=surrogate,
            indices=self._indices,
            coefficients=self._smolyak_coefs,
            nsamples=nsamples,
        )

    def _wrap(
        self, values: Union[Array, Dict[ConfigIdx, Array]]
    ) -> Dict[ConfigIdx, Array]:
        """Normalize values to dict form."""
        if isinstance(values, dict):
            return values
        return {(): values}

    def level(self) -> int:
        """Return the sparse grid level."""
        return self._level

    def nvars_physical(self) -> int:
        """Return number of physical variables."""
        return self._nvars_physical

    def __repr__(self) -> str:
        return (
            f"IsotropicSparseGridFitter(nvars={self._nvars_physical}, "
            f"level={self._level}, nsubspaces={len(self._subspaces)})"
        )
