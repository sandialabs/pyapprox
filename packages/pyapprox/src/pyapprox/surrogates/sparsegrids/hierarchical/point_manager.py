"""Central bookkeeping for hierarchical sparse grid points."""

from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_nd import (
    HierarchicalBasisND,
)
from pyapprox.surrogates.sparsegrids.candidate_info import ConfigIdx
from pyapprox.util.backends.protocols import Array, Backend

PointKey = Tuple[Tuple[int, ...], Tuple[int, ...]]


class PointManager(Generic[Array]):
    """Manages hierarchical grid points, their values, surpluses, and status.

    Lifecycle of a point:
      1. registered (pending) — awaiting function evaluation
      2. evaluated — has values and surpluses
      3. active OR redundant — classified by the fitter

    Both active and redundant points contribute surpluses to the
    surrogate. Only active points spawn children during refinement.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis_nd : HierarchicalBasisND[Array]
        ND basis for computing node coordinates.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis_nd: HierarchicalBasisND[Array],
    ) -> None:
        self._bkd = bkd
        self._basis_nd = basis_nd

        self._key_to_id: Dict[PointKey, int] = {}
        self._id_to_key: List[PointKey] = []
        self._id_to_config: List[ConfigIdx] = []
        self._subspace_to_ids: Dict[Tuple[int, ...], Set[int]] = {}

        self._values: Dict[int, Array] = {}
        self._surpluses: Dict[int, Array] = {}
        self._evaluated: Set[int] = set()
        self._pending: Set[int] = set()

        self._active: Set[int] = set()
        self._redundant: Set[int] = set()
        self._refined: Dict[int, Set[int]] = {}

    # -- Point registration --

    def register_point(
        self, key: PointKey, config_idx: ConfigIdx = ()
    ) -> int:
        """Register a point. Idempotent: returns existing ID if known."""
        if key in self._key_to_id:
            return self._key_to_id[key]
        point_id = len(self._id_to_key)
        self._key_to_id[key] = point_id
        self._id_to_key.append(key)
        self._id_to_config.append(config_idx)
        self._pending.add(point_id)
        self._refined[point_id] = set()

        subspace = key[0]
        if subspace not in self._subspace_to_ids:
            self._subspace_to_ids[subspace] = set()
        self._subspace_to_ids[subspace].add(point_id)

        return point_id

    def get_key(self, point_id: int) -> PointKey:
        return self._id_to_key[point_id]

    def get_config_idx(self, point_id: int) -> ConfigIdx:
        return self._id_to_config[point_id]

    def get_pending_samples(
        self, config_idx: ConfigIdx = ()
    ) -> Tuple[List[int], Optional[Array]]:
        """Return (point_ids, coordinates) for pending points with this config.

        Returns ([], None) if no pending points match.
        """
        ids = [
            pid
            for pid in sorted(self._pending)
            if self._id_to_config[pid] == config_idx
        ]
        if not ids:
            return ([], None)
        bkd = self._bkd
        coords = bkd.hstack(
            [self._basis_nd.node(*self._id_to_key[pid]) for pid in ids]
        )
        return (ids, coords)

    # -- Values and surpluses --

    def set_values_and_surpluses(
        self,
        point_ids: List[int],
        values: Array,
        surrogate_predictions: Array,
    ) -> None:
        """Store function values and compute surpluses.

        Parameters
        ----------
        point_ids : list of int
            IDs of points being evaluated (must be pending).
        values : Array
            Function values, shape (nqoi, npts).
        surrogate_predictions : Array
            Current surrogate predictions at these points, shape (nqoi, npts).
        """
        for i, pid in enumerate(point_ids):
            val = values[:, i]
            pred = surrogate_predictions[:, i]
            self._values[pid] = val
            self._surpluses[pid] = val - pred
            self._evaluated.add(pid)
            self._pending.discard(pid)

    def set_surplus(self, point_id: int, surplus: Array) -> None:
        self._surpluses[point_id] = surplus

    def get_surplus(self, point_id: int) -> Array:
        return self._surpluses[point_id]

    def get_value(self, point_id: int) -> Array:
        return self._values[point_id]

    def is_evaluated(self, point_id: int) -> bool:
        return point_id in self._evaluated

    # -- A_i / R_i tracking --

    def mark_active(self, point_id: int) -> None:
        self._active.add(point_id)

    def mark_redundant(self, point_id: int) -> None:
        self._redundant.add(point_id)

    def is_active(self, point_id: int) -> bool:
        return point_id in self._active

    def is_redundant(self, point_id: int) -> bool:
        return point_id in self._redundant

    def get_subspace_active_ids(
        self, subspace: Tuple[int, ...]
    ) -> Set[int]:
        return self._subspace_to_ids.get(subspace, set()) & self._active

    def get_subspace_redundant_ids(
        self, subspace: Tuple[int, ...]
    ) -> Set[int]:
        return self._subspace_to_ids.get(subspace, set()) & self._redundant

    def get_subspace_point_ids(
        self, subspace: Tuple[int, ...]
    ) -> Set[int]:
        return set(self._subspace_to_ids.get(subspace, set()))

    # -- Refinement ledger --

    def mark_refined(self, point_id: int, direction: int) -> None:
        self._refined[point_id].add(direction)

    def is_refined(self, point_id: int, direction: int) -> bool:
        return direction in self._refined.get(point_id, set())

    def is_point_resolved(self, point_id: int, nvars: int) -> bool:
        """A point is resolved if redundant, or active and refined in all directions."""
        if point_id in self._redundant:
            return True
        if point_id in self._active:
            return all(
                d in self._refined.get(point_id, set())
                for d in range(nvars)
            )
        return False

    # -- Subspace completion --

    def is_subspace_complete(
        self, subspace: Tuple[int, ...], nvars: int
    ) -> bool:
        """A subspace is complete when all points in every backward neighbor are resolved."""
        for d in range(nvars):
            if subspace[d] > 0:
                backward = list(subspace)
                backward[d] -= 1
                backward_sub = tuple(backward)
                for pid in self._subspace_to_ids.get(backward_sub, set()):
                    if not self.is_point_resolved(pid, nvars):
                        return False
        return True

    def subspaces_affected_by(
        self, point_id: int, nvars: int
    ) -> Set[Tuple[int, ...]]:
        """Forward-neighbor subspaces whose completion status may change."""
        key = self._id_to_key[point_id]
        subspace = key[0]
        affected: Set[Tuple[int, ...]] = set()
        for d in range(nvars):
            fwd = list(subspace)
            fwd[d] += 1
            affected.add(tuple(fwd))
        return affected

    # -- Iteration and queries --

    def n_points(self) -> int:
        return len(self._id_to_key)

    def n_evaluated(self) -> int:
        return len(self._evaluated)

    def n_pending(self) -> int:
        return len(self._pending)

    def iter_evaluated(
        self,
    ) -> Iterator[Tuple[int, PointKey, Array, Array]]:
        """Iterate over evaluated points: (id, key, surplus, value)."""
        for pid in sorted(self._evaluated):
            yield (
                pid,
                self._id_to_key[pid],
                self._surpluses[pid],
                self._values[pid],
            )

    def collect_coordinates(self, subset: str = "all") -> Optional[Array]:
        """Collect coordinates of points matching the subset filter.

        Parameters
        ----------
        subset : str
            "all", "evaluated", "pending", "active", or "redundant".

        Returns None if no points match.
        """
        id_set: Set[int]
        if subset == "all":
            id_set = set(range(len(self._id_to_key)))
        elif subset == "evaluated":
            id_set = self._evaluated
        elif subset == "pending":
            id_set = self._pending
        elif subset == "active":
            id_set = self._active
        elif subset == "redundant":
            id_set = self._redundant
        else:
            raise ValueError(f"Unknown subset: {subset}")

        if not id_set:
            return None
        ids = sorted(id_set)
        return self._bkd.hstack(
            [self._basis_nd.node(*self._id_to_key[pid]) for pid in ids]
        )

    def points_by_config(self) -> Dict[ConfigIdx, List[int]]:
        """Group all registered point IDs by their ConfigIdx."""
        result: Dict[ConfigIdx, List[int]] = {}
        for pid, cfg in enumerate(self._id_to_config):
            result.setdefault(cfg, []).append(pid)
        return result
