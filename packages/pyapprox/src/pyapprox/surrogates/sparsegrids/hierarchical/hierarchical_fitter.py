"""Hierarchical sparse grid fitters (multi- and single-fidelity)."""

from typing import Dict, Generic, List, Optional, Set, Tuple

from pyapprox.surrogates.affine.indices.admissibility import (
    AdmissibilityCriteria,
    AlwaysAdmissible,
)
from pyapprox.surrogates.affine.indices.priority_queue import PriorityQueue
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_nd import (
    HierarchicalBasisND,
)
from pyapprox.surrogates.sparsegrids.candidate_info import ConfigIdx
from pyapprox.surrogates.sparsegrids.cost_model import (
    ConstantCostModel,
    CostModelProtocol,
)
from pyapprox.surrogates.sparsegrids.fit_result import (
    AdaptiveSparseGridFitResult,
)
from pyapprox.surrogates.sparsegrids.hierarchical.deferred_registry import (
    DeferredRefinementRegistry,
)
from pyapprox.surrogates.sparsegrids.hierarchical.error_indicators import (
    GammaIndicator,
    HierarchicalErrorIndicator,
)
from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_surrogate import (
    HierarchicalSurrogate,
)
from pyapprox.surrogates.sparsegrids.hierarchical.point_manager import (
    PointManager,
)
from pyapprox.surrogates.sparsegrids.model_factory import (
    DictModelFactory,
    ModelFactoryProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

_SF_KEY: ConfigIdx = ()


class MultiFidelityHierarchicalFitter(Generic[Array]):
    """Adaptive hierarchical sparse grid fitter (multi-fidelity).

    Uses point-greedy refinement with hierarchical surpluses. Supports
    multi-fidelity models via ConfigIdx and downward-closed or
    always-admissible subspace selection.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    bases_1d : list of HierarchicalBasis1D
        One 1D basis per physical dimension.
    admissibility : AdmissibilityCriteria
        Subspace admissibility criteria.
    nconfig_vars : int
        Number of config variables (0 for single-fidelity).
    error_indicator : HierarchicalErrorIndicator, optional
        Per-point error indicator. Default: GammaIndicator.
    cost_model : CostModelProtocol, optional
        Per-sample cost model. Default: ConstantCostModel.
    batch_size : int
        Number of active points to pop per step. Default: 1.
    verbosity : int
        Verbosity level (0=silent).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        bases_1d: List[HierarchicalBasis1D[Array]],
        admissibility: AdmissibilityCriteria[Array],
        nconfig_vars: int = 0,
        error_indicator: Optional[HierarchicalErrorIndicator[Array]] = None,
        cost_model: Optional[CostModelProtocol] = None,
        batch_size: int = 1,
        verbosity: int = 0,
    ) -> None:
        self._bkd = bkd
        self._bases_1d = bases_1d
        self._basis_nd = HierarchicalBasisND(bkd, bases_1d)
        self._nvars_physical = len(bases_1d)
        self._nconfig_vars = nconfig_vars

        self._admissibility = admissibility
        self._bypass_downward_closure = isinstance(
            admissibility, AlwaysAdmissible
        )

        self._error_indicator: HierarchicalErrorIndicator[Array] = (
            error_indicator if error_indicator is not None
            else GammaIndicator(bkd)
        )
        self._cost_model: CostModelProtocol = (
            cost_model if cost_model is not None else ConstantCostModel()
        )
        self._batch_size = batch_size
        self._verbosity = verbosity

        self._point_mgr = PointManager(bkd, self._basis_nd)
        self._queue: PriorityQueue[Array] = PriorityQueue(max_priority=True)
        self._deferred = DeferredRefinementRegistry()

        self._first_step = True
        self._nsteps = 0
        self._nqoi = 0
        self._newly_evaluated_ids: List[int] = []

        self._promoted_subspaces: Set[Tuple[int, ...]] = set()
        self._cached_surrogate: Optional[HierarchicalSurrogate[Array]] = None
        self._use_ancestor_surplus = True

    # -- Public API --

    def step_samples(self) -> Optional[Dict[ConfigIdx, Array]]:
        """Return pending sample coordinates, or None if terminated."""
        if self._first_step:
            return self._first_step_samples()
        return self._next_step_samples()

    def step_values(self, values: Dict[ConfigIdx, Array]) -> None:
        """Supply function values for pending samples.

        Pending points are sorted by subspace level sum and processed
        in increasing order. Each point's surplus is computed against a
        running surrogate that includes all previously evaluated points
        plus the points already processed in this batch. This guarantees
        that every basis function nonzero at a node is already in the
        surrogate when that node's surplus is computed.

        Parameters
        ----------
        values : dict
            {ConfigIdx: Array of shape (nqoi, npts)}.
        """
        self._newly_evaluated_ids.clear()

        all_pending: List[Tuple[int, ConfigIdx]] = []
        values_by_pid: dict[int, Array] = {}

        for config_idx, vals in values.items():
            pending_ids, _ = self._point_mgr.get_pending_samples(config_idx)
            if len(pending_ids) == 0:
                continue
            for i, pid in enumerate(pending_ids):
                all_pending.append((pid, config_idx))
                values_by_pid[pid] = vals[:, i]

        if not all_pending:
            return

        self._nqoi = next(iter(values.values())).shape[0]

        def _level_sum(pid: int) -> int:
            return sum(self._point_mgr.get_key(pid)[0])

        all_pending.sort(key=lambda x: (_level_sum(x[0]), x[0]))

        if self._cached_surrogate is not None:
            surrogate = self._cached_surrogate
        else:
            surrogate = self._build_surrogate_snapshot()
        bkd = self._bkd

        ii = 0
        while ii < len(all_pending):
            cur_level = _level_sum(all_pending[ii][0])
            jj = ii
            while jj < len(all_pending) and _level_sum(all_pending[jj][0]) == cur_level:
                jj += 1
            group = all_pending[ii:jj]
            ii = jj

            keys = [self._point_mgr.get_key(pid) for pid, _ in group]
            f_vals = bkd.stack(
                [values_by_pid[pid] for pid, _ in group], axis=1
            ) if len(group) > 1 else values_by_pid[group[0][0]].reshape(-1, 1)

            if surrogate.n_points() > 0:
                if self._use_ancestor_surplus:
                    group_surplus_matrix = (
                        surrogate.compute_surpluses_at_grid_points(
                            keys, f_vals
                        )
                    )
                else:
                    nodes = bkd.hstack(
                        [self._basis_nd.node(*k) for k in keys]
                    )
                    group_surplus_matrix = f_vals - surrogate(nodes)
            else:
                group_surplus_matrix = bkd.copy(f_vals)

            group_surpluses: List[Array] = []
            group_weights: List[float] = []
            for idx_in_group, (pid, config_idx) in enumerate(group):
                key = keys[idx_in_group]
                val = values_by_pid[pid]
                surplus = group_surplus_matrix[:, idx_in_group]
                pred = val - surplus
                self._point_mgr.set_values_and_surpluses(
                    [pid], val.reshape(-1, 1), pred.reshape(-1, 1)
                )
                self._newly_evaluated_ids.append(pid)
                group_surpluses.append(surplus)
                group_weights.append(
                    self._basis_nd.quadrature_weight(*key)
                )
            surrogate.add_points(keys, group_surpluses, group_weights)

        self._cached_surrogate = surrogate

        for pid in self._newly_evaluated_ids:
            surplus = self._point_mgr.get_surplus(pid)
            weight = self._basis_nd.quadrature_weight(
                *self._point_mgr.get_key(pid)
            )
            config_idx = self._point_mgr.get_config_idx(pid)
            cost = self._cost_model(config_idx)
            priority, error = self._error_indicator(surplus, weight, cost)
            self._queue.put(priority, error, pid)

    def current_error(self) -> float:
        """Sum of error contributions from all queued (unrefined) points."""
        total = 0.0
        for _, _, _, error, _ in self._queue._heap:
            total += error
        return total

    def result(
        self, converged: bool = False
    ) -> AdaptiveSparseGridFitResult[Array]:
        if self._cached_surrogate is not None:
            surrogate = self._cached_surrogate
        else:
            surrogate = self._build_surrogate_snapshot()
        sel_indices = self._get_promoted_indices()
        n_sel = sel_indices.shape[1] if sel_indices is not None else 0
        coeffs = self._bkd.ones(
            (n_sel,), dtype=self._bkd.double_dtype()
        )
        return AdaptiveSparseGridFitResult(
            surrogate=surrogate,
            indices=sel_indices,
            coefficients=coeffs,
            nsamples=self._point_mgr.n_evaluated(),
            error=self.current_error(),
            nsteps=self._nsteps,
            converged=converged,
        )

    def refine_to_tolerance(
        self,
        model_factory: ModelFactoryProtocol,
        tol: float = 1e-6,
        max_steps: int = 200,
    ) -> AdaptiveSparseGridFitResult[Array]:
        """Run the full adaptive loop until tolerance or max steps."""
        for iteration in range(max_steps):
            samples = self.step_samples()
            if samples is None:
                return self.result(converged=True)
            values: Dict[ConfigIdx, Array] = {}
            for config_idx, coords in samples.items():
                model = model_factory.get_model(config_idx)
                values[config_idx] = model(coords)
            self.step_values(values)
            if iteration > 0 and self.current_error() < tol:
                return self.result(converged=True)
        return self.result(converged=False)

    def get_samples(
        self, subset: str = "all"
    ) -> Dict[ConfigIdx, Array]:
        result: Dict[ConfigIdx, Array] = {}
        by_config = self._point_mgr.points_by_config()
        for cfg, ids in by_config.items():
            filtered = self._filter_ids(ids, subset)
            if filtered:
                coords = self._bkd.hstack(
                    [
                        self._basis_nd.node(
                            *self._point_mgr.get_key(pid)
                        )
                        for pid in filtered
                    ]
                )
                result[cfg] = coords
        return result

    def get_values(
        self, subset: str = "all"
    ) -> Dict[ConfigIdx, Optional[Array]]:
        result: Dict[ConfigIdx, Optional[Array]] = {}
        by_config = self._point_mgr.points_by_config()
        for cfg, ids in by_config.items():
            filtered = self._filter_ids(ids, subset)
            if not filtered:
                result[cfg] = None
                continue
            eval_ids = [
                pid for pid in filtered
                if self._point_mgr.is_evaluated(pid)
            ]
            if not eval_ids:
                result[cfg] = None
            else:
                result[cfg] = self._bkd.stack(
                    [self._point_mgr.get_value(pid) for pid in eval_ids],
                    axis=1,
                )
        return result

    def get_selected_indices(self) -> Optional[Array]:
        return self._get_promoted_indices()

    def get_candidate_indices(self) -> Optional[Array]:
        return None

    def cumulative_cost(
        self, cost_model: Optional[CostModelProtocol] = None
    ) -> float:
        cm = cost_model if cost_model is not None else self._cost_model
        total = 0.0
        for pid in range(self._point_mgr.n_points()):
            if self._point_mgr.is_evaluated(pid):
                cfg = self._point_mgr.get_config_idx(pid)
                total += cm(cfg)
        return total

    def nvars_physical(self) -> int:
        return self._nvars_physical

    def nselected(self) -> int:
        return len(self._promoted_subspaces)

    def ncandidates(self) -> int:
        return 0

    # -- Internal algorithm --

    def _first_step_samples(self) -> Dict[ConfigIdx, Array]:
        root_sub = (0,) * self._nvars_physical
        self._promoted_subspaces.add(root_sub)
        root_config = (0,) * self._nconfig_vars if self._nconfig_vars > 0 else _SF_KEY
        for pt in self._basis_nd.points_in_subspace(root_sub):
            self._point_mgr.register_point((root_sub, pt), root_config)

        self._first_step = False
        return self._collect_pending_by_config()

    def _next_step_samples(self) -> Optional[Dict[ConfigIdx, Array]]:
        while True:
            n_productive = 0

            while not self._queue.empty() and n_productive < self._batch_size:
                priority, error, point_id = self._queue.get()

                self._point_mgr.mark_active(point_id)
                point_sub = self._point_mgr.get_key(point_id)[0]
                newly_promoted = point_sub not in self._promoted_subspaces
                self._promoted_subspaces.add(point_sub)
                spawned = self._refine_point(point_id)
                if not self._bypass_downward_closure and newly_promoted:
                    self._notify_promotion(point_sub)

                new_pending = [
                    pid for pid in spawned
                    if pid in self._point_mgr._pending
                ]
                if new_pending:
                    n_productive += 1

            pending = self._collect_pending_by_config()

            if (
                not pending
                and self._queue.empty()
                and self._deferred.empty()
            ):
                return None

            if not pending:
                continue

            self._nsteps += 1
            return pending

    def _refine_point(self, point_id: int) -> List[int]:
        if self._bypass_downward_closure:
            return self._refine_point_local(point_id)
        return self._refine_point_downward_closed(point_id)

    def _refine_point_local(self, point_id: int) -> List[int]:
        key = self._point_mgr.get_key(point_id)
        subspace_level, point_index = key
        config_idx = self._point_mgr.get_config_idx(point_id)
        new_point_ids: List[int] = []

        for direction in range(self._nvars_physical):
            self._point_mgr.mark_refined(point_id, direction)
            children = self._basis_nd.children_of_point(
                subspace_level, point_index, direction
            )
            for child_sub, child_idx in children:
                new_point_ids.extend(
                    self._register_with_ancestors(
                        child_sub, child_idx, config_idx
                    )
                )

        return new_point_ids

    def _refine_point_downward_closed(self, point_id: int) -> List[int]:
        key = self._point_mgr.get_key(point_id)
        subspace_level, point_index = key
        config_idx = self._point_mgr.get_config_idx(point_id)
        new_point_ids: List[int] = []

        for direction in range(self._nvars_physical):
            self._point_mgr.mark_refined(point_id, direction)

            children = self._basis_nd.children_of_point(
                subspace_level, point_index, direction
            )
            if not children:
                continue

            target_sub_t = children[0][0]

            if self._is_target_admissible(target_sub_t):
                for child_sub, child_idx in children:
                    pid = self._point_mgr.register_point(
                        (child_sub, child_idx), config_idx
                    )
                    new_point_ids.append(pid)
            else:
                blockers: set[Tuple[int, ...]] = set()
                for d2 in range(self._nvars_physical):
                    if target_sub_t[d2] > 0:
                        backward = list(target_sub_t)
                        backward[d2] -= 1
                        backward_sub = tuple(backward)
                        if backward_sub not in self._promoted_subspaces:
                            blockers.add(backward_sub)
                if blockers:
                    self._deferred.defer(
                        point_id, direction, target_sub_t, blockers
                    )

        return new_point_ids

    def _is_target_admissible(
        self, target_sub: Tuple[int, ...]
    ) -> bool:
        index_arr = self._bkd.asarray(
            target_sub, dtype=self._bkd.int64_dtype()
        )
        if not self._admissibility(index_arr):
            return False
        for d in range(self._nvars_physical):
            if target_sub[d] > 0:
                backward = list(target_sub)
                backward[d] -= 1
                backward_sub = tuple(backward)
                if backward_sub not in self._promoted_subspaces:
                    if self._is_empty_subspace(backward_sub):
                        self._auto_promote(backward_sub)
                    else:
                        return False
        return True

    def _is_empty_subspace(self, sub: Tuple[int, ...]) -> bool:
        return len(self._basis_nd.points_in_subspace(sub)) == 0

    def _auto_promote(self, sub: Tuple[int, ...]) -> None:
        """Promote an empty subspace and recursively promote its empty backward neighbors."""
        if sub in self._promoted_subspaces:
            return
        for d in range(self._nvars_physical):
            if sub[d] > 0:
                backward = list(sub)
                backward[d] -= 1
                backward_sub = tuple(backward)
                if backward_sub not in self._promoted_subspaces:
                    if self._is_empty_subspace(backward_sub):
                        self._auto_promote(backward_sub)
        self._promoted_subspaces.add(sub)
        self._notify_promotion(sub)

    def _notify_promotion(self, promoted_sub: Tuple[int, ...]) -> None:
        released = self._deferred.notify_complete(promoted_sub)
        for task in released:
            cfg = self._point_mgr.get_config_idx(task.point_id)
            if self._is_target_admissible(task.target_subspace):
                task_key = self._point_mgr.get_key(task.point_id)
                children = self._basis_nd.children_of_point(
                    task_key[0], task_key[1], task.direction
                )
                for child_sub, child_idx in children:
                    self._point_mgr.register_point(
                        (child_sub, child_idx), cfg
                    )

    def _register_with_ancestors(
        self,
        child_sub: Tuple[int, ...],
        child_idx: Tuple[int, ...],
        config_idx: ConfigIdx,
    ) -> List[int]:
        """Register a child point and all its unevaluated ancestors.

        Returns list of newly registered point IDs (child + ancestors).
        """
        new_ids: List[int] = []
        pid = self._point_mgr.register_point(
            (child_sub, child_idx), config_idx
        )
        new_ids.append(pid)

        for anc_sub, anc_idx in self._basis_nd.ancestors_of_point(
            child_sub, child_idx
        ):
            if (anc_sub, anc_idx) not in self._point_mgr._key_to_id:
                anc_pid = self._point_mgr.register_point(
                    (anc_sub, anc_idx), config_idx
                )
                new_ids.append(anc_pid)

        return new_ids

    # -- Helpers --

    def _get_promoted_indices(self) -> Optional[Array]:
        if not self._promoted_subspaces:
            return None
        subs = sorted(self._promoted_subspaces)
        cols = []
        for sub in subs:
            full = list(sub) + [0] * self._nconfig_vars
            cols.append(
                self._bkd.asarray(full, dtype=self._bkd.int64_dtype())
                .reshape(-1, 1)
            )
        return self._bkd.hstack(cols)

    def _build_surrogate_snapshot(self) -> HierarchicalSurrogate[Array]:
        bkd = self._bkd
        point_keys = []
        surplus_list = []
        weight_list = []
        for pid, key, surplus, _ in self._point_mgr.iter_evaluated():
            point_keys.append(key)
            surplus_list.append(surplus)
            weight_list.append(
                self._basis_nd.quadrature_weight(*key)
            )
        if not point_keys:
            nqoi = max(self._nqoi, 1)
            empty_surpluses = bkd.zeros(
                (nqoi, 0), dtype=bkd.double_dtype()
            )
            empty_weights = bkd.zeros((0,), dtype=bkd.double_dtype())
            return HierarchicalSurrogate(
                bkd, self._basis_nd, [], empty_surpluses, empty_weights
            )
        surpluses = bkd.stack(surplus_list, axis=1) if len(surplus_list) > 1 else surplus_list[0].reshape(-1, 1)
        weights = bkd.asarray(weight_list, dtype=bkd.double_dtype())
        return HierarchicalSurrogate(
            bkd, self._basis_nd, point_keys, surpluses, weights
        )

    def _collect_pending_by_config(self) -> Dict[ConfigIdx, Array]:
        result: Dict[ConfigIdx, Array] = {}
        configs = set(self._point_mgr._id_to_config)
        for cfg in configs:
            ids, coords = self._point_mgr.get_pending_samples(cfg)
            if ids and coords is not None:
                result[cfg] = coords
        return result

    def _filter_ids(self, ids: List[int], subset: str) -> List[int]:
        if subset == "all":
            return ids
        if subset == "evaluated":
            return [pid for pid in ids if self._point_mgr.is_evaluated(pid)]
        if subset == "pending":
            return [pid for pid in ids if not self._point_mgr.is_evaluated(pid)]
        if subset == "active":
            return [pid for pid in ids if self._point_mgr.is_active(pid)]
        if subset == "redundant":
            return [pid for pid in ids if self._point_mgr.is_redundant(pid)]
        raise ValueError(f"Unknown subset: {subset}")


class SingleFidelityHierarchicalFitter(Generic[Array]):
    """Thin wrapper converting Array <-> Dict[ConfigIdx, Array].

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    bases_1d : list of HierarchicalBasis1D
        One 1D basis per physical dimension.
    admissibility : AdmissibilityCriteria
        Subspace admissibility criteria.
    error_indicator : HierarchicalErrorIndicator, optional
        Per-point error indicator. Default: GammaIndicator.
    cost_model : CostModelProtocol, optional
        Per-sample cost model. Default: ConstantCostModel.
    batch_size : int
        Number of active points to pop per step.
    verbosity : int
        Verbosity level.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        bases_1d: List[HierarchicalBasis1D[Array]],
        admissibility: AdmissibilityCriteria[Array],
        error_indicator: Optional[HierarchicalErrorIndicator[Array]] = None,
        cost_model: Optional[CostModelProtocol] = None,
        batch_size: int = 1,
        verbosity: int = 0,
    ) -> None:
        self._fitter = MultiFidelityHierarchicalFitter(
            bkd=bkd,
            bases_1d=bases_1d,
            admissibility=admissibility,
            nconfig_vars=0,
            error_indicator=error_indicator,
            cost_model=cost_model,
            batch_size=batch_size,
            verbosity=verbosity,
        )
        self._bkd = bkd

    def step_samples(self) -> Optional[Array]:
        result = self._fitter.step_samples()
        if result is None:
            return None
        return result[_SF_KEY]

    def step_values(self, values: Array) -> None:
        self._fitter.step_values({_SF_KEY: values})

    def current_error(self) -> float:
        return self._fitter.current_error()

    def result(
        self, converged: bool = False
    ) -> AdaptiveSparseGridFitResult[Array]:
        return self._fitter.result(converged)

    def refine_to_tolerance(
        self,
        target_fn: object,
        tol: float = 1e-6,
        max_steps: int = 200,
    ) -> AdaptiveSparseGridFitResult[Array]:
        factory = DictModelFactory({_SF_KEY: target_fn})
        return self._fitter.refine_to_tolerance(factory, tol, max_steps)

    def get_samples(self, subset: str = "all") -> Array:
        return self._fitter.get_samples(subset)[_SF_KEY]

    def get_values(self, subset: str = "all") -> Optional[Array]:
        return self._fitter.get_values(subset)[_SF_KEY]

    def get_selected_indices(self) -> Array:
        return self._fitter.get_selected_indices()

    def get_candidate_indices(self) -> Optional[Array]:
        return self._fitter.get_candidate_indices()

    def nvars_physical(self) -> int:
        return self._fitter.nvars_physical()

    def nselected(self) -> int:
        return self._fitter.nselected()

    def ncandidates(self) -> int:
        return self._fitter.ncandidates()
