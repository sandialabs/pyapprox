"""Sample tracking and deduplication for sparse grids.

Wraps BasisIndexGenerator to provide a clean interface for registering
subspaces, tracking unique samples, and distributing values.
"""

from typing import Generic, List, Optional, Set

from pyapprox.surrogates.affine.indices.basis_generator import (
    BasisIndexGenerator,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    SubspaceFactoryProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class SampleTracker(Generic[Array]):
    """Track unique samples across subspaces for one config group.

    For single-fidelity grids, there is one SampleTracker. For
    multi-fidelity grids, there is one per config_idx.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    factory : SubspaceFactoryProtocol[Array]
        Factory used to determine nestedness and growth rules.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        factory: SubspaceFactoryProtocol[Array],
    ) -> None:
        self._bkd = bkd
        nvars = factory.nvars_physical()
        self._basis_gen = BasisIndexGenerator(
            bkd,
            nvars,
            factory.growth_rules(),
            all_nested=factory.is_nested(),
        )
        # Values array: (nqoi, n_unique) — grows as new samples arrive
        self._values: Optional[Array] = None
        self._nqoi: Optional[int] = None
        # Map from subspace position to registered subspace
        self._registered: List[TensorProductSubspace[Array]] = []
        # Positions whose values have not been written to their subspace
        # yet, and the largest global value index each one needs.
        self._pending: List[int] = []
        self._max_global_index: List[int] = []

    def register(
        self,
        approx_index: Array,
        subspace: TensorProductSubspace[Array],
    ) -> int:
        """Register a subspace and build deduplication mappings.

        Parameters
        ----------
        approx_index : Array
            Full multi-index (may include config dims). Only the first
            nvars_physical dims are used for basis index generation.
        subspace : TensorProductSubspace[Array]
            The subspace to register.

        Returns
        -------
        int
            Position of this subspace (0-indexed).
        """
        nvars_phys = self._basis_gen.nvars()
        physical_index = approx_index[:nvars_phys]
        pos = len(self._registered)
        self._registered.append(subspace)
        self._basis_gen._set_unique_subspace_basis_indices(
            physical_index,
            pos,
            subspace_samples=subspace.get_samples(),
        )
        # Record the highest global value index this subspace reads, so
        # distribute_values_to_subspaces can tell whether enough values
        # have arrived to write it. Global indices are assigned in
        # registration order and values are appended in that same order,
        # so this is a cheap high-water mark rather than a set test.
        global_indices = self._basis_gen.get_subspace_value_indices(pos)
        self._max_global_index.append(
            int(self._bkd.to_int(self._bkd.max(global_indices)))
        )
        self._pending.append(pos)
        return pos

    def get_unique_local_indices(self, pos: int) -> List[int]:
        """Return local sample indices that are new (first occurrence).

        Parameters
        ----------
        pos : int
            Subspace position from register().

        Returns
        -------
        List[int]
            Local indices of unique samples in this subspace.
        """
        return self._basis_gen.get_unique_local_indices(pos)

    def get_new_samples(
        self, pos: int, subspace: TensorProductSubspace[Array]
    ) -> Optional[Array]:
        """Return samples that are new for this subspace.

        Parameters
        ----------
        pos : int
            Subspace position from register().
        subspace : TensorProductSubspace[Array]
            The subspace.

        Returns
        -------
        Optional[Array]
            New samples, shape (nvars, n_new), or None if no new samples.
        """
        unique_local = self.get_unique_local_indices(pos)
        if len(unique_local) == 0:
            return None
        samples = subspace.get_samples()
        idx_array = self._bkd.asarray(unique_local, dtype=self._bkd.int64_dtype())
        return samples[:, idx_array]

    def append_new_values(self, values: Array) -> None:
        """Append values for every sample registered but not yet valued.

        One call must supply every sample registered since the last
        append, in the order the samples were handed out. That order is
        the batch returned by the fitter's ``step_samples``, which is
        exactly these samples, so evaluating that batch and passing the
        result back is aligned by construction.

        Global value indices are assigned in registration order and
        index this array directly, so a short batch would leave later
        indices pointing past the end and a long one would misalign
        every subsequent subspace. Neither is detectable once the values
        are in, so the count is checked here.

        The count cannot catch values that are correctly sized but come
        from the wrong samples. Build the array from the samples the
        tracker handed out rather than reconstructing it independently.

        Parameters
        ----------
        values : Array
            New values, shape (nqoi, n_outstanding).

        Raises
        ------
        ValueError
            If the column count is not the number of outstanding samples.
        """
        held = 0 if self._values is None else self._values.shape[1]
        expected = self.n_unique_samples() - held
        if values.shape[1] != expected:
            raise ValueError(
                f"expected values for all {expected} outstanding samples "
                f"({self.n_unique_samples()} registered, {held} already "
                f"held), got {values.shape[1]}"
            )
        if self._values is None:
            self._values = self._bkd.copy(values)
            self._nqoi = values.shape[0]
        else:
            self._values = self._bkd.hstack((self._values, values))

    def get_subspace_value_indices(self, pos: int) -> Array:
        """Return global value indices for all samples in a subspace.

        Parameters
        ----------
        pos : int
            Subspace position from register().

        Returns
        -------
        Array
            Array mapping local sample index to global value index.
            Shape: (nsamples_in_subspace,)
        """
        return self._basis_gen.get_subspace_value_indices(pos)

    def get_subspace_values(self, pos: int) -> Array:
        """Return values for a specific subspace using dedup mappings.

        Parameters
        ----------
        pos : int
            Subspace position from register().

        Returns
        -------
        Array
            Values for this subspace, shape (nqoi, n_subspace_samples).
        """
        if self._values is None:
            raise ValueError("No values set. Call append_new_values() first.")
        global_indices = self._basis_gen.get_subspace_value_indices(pos)
        return self._values[:, global_indices]

    def distribute_values_to_subspaces(self) -> None:
        """Write values to subspaces that have not received them yet.

        A subspace is never partially filled: ``get_subspace_values``
        gathers all of its samples from the global array in one indexing
        operation, so it is either unwritten or complete. With nested
        rules most of those values were computed by ancestor subspaces
        and are already held, so a subspace typically becomes writable in
        the same ``append_new_values`` that supplies its one new point.

        Only pending positions are visited, and each is written once.
        Rewriting every registered subspace on every call costs O(total
        samples) per step, and once subspace values are write-once it
        would raise on the second visit.

        Under the adaptive fitter every registered subspace becomes
        writable in the append that follows its registration, so nothing
        is left pending once values arrive. The condition is kept
        because this is a public class: a caller that registers
        subspaces without supplying their values yet should be skipped
        rather than gathering past the end of the array.
        """
        if self._values is None:
            return
        nvalues = self._values.shape[1]
        still_pending: List[int] = []
        for pos in self._pending:
            # Global indices are assigned in registration order, and
            # append_new_values rejects a batch that does not cover every
            # outstanding sample. So the values array is a dense prefix
            # of the assigned indices: once it reaches this subspace's
            # highest index, all of its lower indices are present too.
            if self._max_global_index[pos] < nvalues:
                self._registered[pos].set_values(self.get_subspace_values(pos))
            else:
                still_pending.append(pos)
        self._pending = still_pending

    def npending(self) -> int:
        """Return how many registered subspaces still lack values.

        Zero after every ``step_values`` under the adaptive fitter,
        including for a config that supplied no samples in that step: a
        config is absent from the batch exactly when it gained no new
        subspaces, so it has nothing outstanding.
        """
        return len(self._pending)

    def n_unique_samples(self) -> int:
        """Return total number of unique samples tracked."""
        return self._basis_gen.n_unique_samples()

    def collect_unique_samples(self) -> Array:
        """Collect all unique samples into a single array.

        Returns
        -------
        Array
            All unique samples, shape (nvars, n_unique).
        """
        nvars = self._basis_gen.nvars()
        n_unique = self.n_unique_samples()
        if n_unique == 0:
            return self._bkd.zeros((nvars, 0))

        result = self._bkd.zeros((nvars, n_unique))
        global_idx = 0
        for pos, subspace in enumerate(self._registered):
            unique_local = self.get_unique_local_indices(pos)
            if len(unique_local) > 0:
                samples = subspace.get_samples()
                idx_array = self._bkd.asarray(
                    unique_local, dtype=self._bkd.int64_dtype()
                )
                n_new = len(unique_local)
                result[:, global_idx : global_idx + n_new] = samples[:, idx_array]
                global_idx += n_new
        return result

    def n_filtered_unique_samples(self, positions: Optional[Set[int]] = None) -> int:
        """Count unique samples for a filtered set of positions.

        Parameters
        ----------
        positions : Optional[Set[int]]
            Set of subspace positions to include. None means all.

        Returns
        -------
        int
            Number of unique samples owned by the filtered positions.
        """
        if positions is not None and len(positions) == 0:
            return 0
        count = 0
        for pos in range(len(self._registered)):
            if positions is not None and pos not in positions:
                continue
            count += len(self.get_unique_local_indices(pos))
        return count

    def collect_filtered_unique_samples(
        self, positions: Optional[Set[int]] = None
    ) -> Array:
        """Collect unique samples from a filtered set of subspace positions.

        A sample is included iff the subspace that first contributed it
        (where it was "new") is in the positions set.
        positions=None is equivalent to collect_unique_samples().

        Parameters
        ----------
        positions : Optional[Set[int]]
            Set of subspace positions to include. None means all.

        Returns
        -------
        Array
            Unique samples, shape (nvars, n_filtered).
        """
        nvars = self._basis_gen.nvars()
        n_filtered = self.n_filtered_unique_samples(positions)
        if n_filtered == 0:
            return self._bkd.zeros((nvars, 0))

        result = self._bkd.zeros((nvars, n_filtered))
        out_idx = 0
        for pos, subspace in enumerate(self._registered):
            if positions is not None and pos not in positions:
                continue
            unique_local = self.get_unique_local_indices(pos)
            if len(unique_local) > 0:
                samples = subspace.get_samples()
                idx_array = self._bkd.asarray(
                    unique_local, dtype=self._bkd.int64_dtype()
                )
                n_new = len(unique_local)
                result[:, out_idx : out_idx + n_new] = samples[:, idx_array]
                out_idx += n_new
        return result

    def collect_filtered_unique_values(
        self, positions: Optional[Set[int]] = None
    ) -> Optional[Array]:
        """Collect unique values aligned with collect_filtered_unique_samples.

        Parameters
        ----------
        positions : Optional[Set[int]]
            Set of subspace positions to include. None means all.

        Returns
        -------
        Optional[Array]
            Unique values, shape (nqoi, n_filtered), or None if no
            values have been set.
        """
        if self._values is None:
            return None

        n_filtered = self.n_filtered_unique_samples(positions)
        if n_filtered == 0:
            assert self._nqoi is not None
            return self._bkd.zeros((self._nqoi, 0))

        assert self._nqoi is not None
        result = self._bkd.zeros((self._nqoi, n_filtered))
        out_idx = 0
        global_idx = 0
        for pos in range(len(self._registered)):
            unique_local = self.get_unique_local_indices(pos)
            n_new = len(unique_local)
            if positions is not None and pos not in positions:
                global_idx += n_new
                continue
            if n_new > 0:
                result[:, out_idx : out_idx + n_new] = self._values[
                    :, global_idx : global_idx + n_new
                ]
                out_idx += n_new
                global_idx += n_new
        return result

    def nqoi(self) -> Optional[int]:
        """Return nqoi if values have been set, else None."""
        return self._nqoi
