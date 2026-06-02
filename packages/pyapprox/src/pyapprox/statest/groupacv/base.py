"""Base GroupACV estimator implementation.

This module provides the BaseGroupACVEstimator abstract class for Group
Approximate Control Variate estimation using the Template Method pattern.

Concrete implementations are in variants.py:
    - GroupACVEstimatorIS (independent sampling)
    - GroupACVEstimatorNested (nested sampling)
"""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
)

from pyapprox.statest.groupacv.utils import (
    _grouped_acv_sigma,
    _grouped_acv_sigma_block,
    get_model_subsets,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.statest.groupacv.result import (
        GroupACVAllocationResult,
    )
    from pyapprox.statest.statistics import (
        MultiOutputStatistic,
    )


class BaseGroupACVEstimator(ABC, Generic[Array]):
    """Abstract base class for Group Approximate Control Variate estimators.

    This class uses the Template Method pattern to allow subclasses to
    customize subset preprocessing and allocation matrix generation.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic object containing covariance information

    costs : Array
        The computational costs of each model

    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.

    model_subsets : List[Array], optional
        List of model subsets. If None, all subsets are generated.

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def __init__(
        self,
        stat: "MultiOutputStatistic[Array]",
        costs: Array,
        reg_blue: float = 0,
        model_subsets: Optional[List[Array]] = None,
        asketch: Optional[Array] = None,
        use_pseudo_inv: bool = True,
        known_quantities: Optional[Dict[Tuple[int, str], Array]] = None,
    ):
        from pyapprox.statest.statistics import (
            MultiOutputMean,
            MultiOutputMeanAndVariance,
            MultiOutputVariance,
        )

        self._bkd = stat.bkd()
        self._use_pseudo_inv = use_pseudo_inv
        self._costs = self._bkd.array(costs)
        self._nmodels = len(costs)
        self._reg_blue = reg_blue
        if not isinstance(
            stat,
            (MultiOutputMean, MultiOutputVariance, MultiOutputMeanAndVariance),
        ):
            raise ValueError("GroupACV only supports estimation of mean or variance")
        self._stat = stat

        if model_subsets is None:
            model_subsets = get_model_subsets(self._nmodels, self._bkd)

        self._model_subsets, self._subsets, self._allocation_mat = self._set_subsets(
            model_subsets
        )
        self._npartitions = self._allocation_mat.shape[1]
        self._partitions_per_model = self._get_partitions_per_model()
        self._partitions_intersect = self._get_subset_intersecting_partitions()
        self._restriction_matrices = [
            self._restriction_matrix(subset).T
            for ii, subset in enumerate(self._subsets)
        ]
        self._R = self._bkd.hstack(self._restriction_matrices)
        # set npatition_samples above small constant,
        # otherwise gradient will not be defined.
        self._npartition_samples_lb = 0  # 1e-5
        self._asketch = self._validate_asketch(asketch)

        self._setup_known_quantities(known_quantities)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _setup_known_quantities(
        self,
        known_quantities: Optional[Dict[Tuple[int, str], Array]],
    ) -> None:
        from pyapprox.statest.statistics import MultiOutputMeanAndVariance

        nstats = self._stat.nstats()

        if not known_quantities:
            self._has_known_quantities = False
            self._nT_stats = self._nmodels * nstats
            self._known_values = None
            self._T_stat_indices: List[int] = list(
                range(self._nmodels * nstats)
            )
            self._K_stat_indices: List[int] = []
            return

        for (model_idx, stat_name), values in known_quantities.items():
            if model_idx == 0:
                raise ValueError(
                    "Model 0 (high-fidelity) cannot have known quantities"
                )
            if model_idx < 1 or model_idx >= self._nmodels:
                raise ValueError(
                    f"Model index {model_idx} out of range "
                    f"[1, {self._nmodels})"
                )
            slots = self._stat.stat_slot_indices(stat_name)
            values_arr = self._bkd.asarray(values)
            if values_arr.shape != (len(slots),):
                raise ValueError(
                    f"known_quantities[({model_idx}, '{stat_name}')] "
                    f"shape {values_arr.shape} must be ({len(slots)},)"
                )
            if not self._bkd.all_bool(self._bkd.isfinite(values_arr)):
                raise ValueError(
                    f"known_quantities[({model_idx}, '{stat_name}')] "
                    f"contains non-finite values"
                )

        if isinstance(self._stat, MultiOutputMeanAndVariance):
            models_with_keys: Dict[int, List[str]] = {}
            for model_idx, stat_name in known_quantities:
                models_with_keys.setdefault(model_idx, []).append(stat_name)
            for model_idx, keys in models_with_keys.items():
                has_mean = "mean" in keys
                has_var = "variance" in keys
                if has_mean != has_var:
                    present = "mean" if has_mean else "variance"
                    missing = "variance" if has_mean else "mean"
                    raise ValueError(
                        f"MultiOutputMeanAndVariance requires per-model "
                        f"all-or-nothing: model {model_idx} has known "
                        f"{present} but not {missing}"
                    )

        self._has_known_quantities = True
        nan_val = float("nan")
        self._known_values = self._bkd.full(
            (self._nmodels, nstats), nan_val
        )
        for (model_idx, stat_name), values in known_quantities.items():
            slots = self._stat.stat_slot_indices(stat_name)
            values_arr = self._bkd.asarray(values)
            for q_idx, slot in enumerate(slots):
                self._known_values[model_idx, slot] = values_arr[q_idx]

        self._T_stat_indices = []
        self._K_stat_indices = []
        for m in range(self._nmodels):
            for s in range(nstats):
                flat = m * nstats + s
                if not bool(
                    self._bkd.all_bool(
                        self._bkd.isfinite(self._known_values[m:m+1, s:s+1])
                    )
                ):
                    self._T_stat_indices.append(flat)
                else:
                    self._K_stat_indices.append(flat)
        self._nT_stats = len(self._T_stat_indices)

        self._R_full = self._R
        self._restriction_matrices_full = list(self._restriction_matrices)
        self._asketch_full = self._asketch

        T = self._T_stat_indices
        self._restriction_matrices = [
            Rk[T, :] for Rk in self._restriction_matrices_full
        ]
        self._R = self._bkd.hstack(self._restriction_matrices)
        self._asketch = self._asketch_full[:, T]

    def nsubsets(self) -> int:
        """Return the number of subsets."""
        return len(self._subsets)

    def npartitions(self) -> int:
        """Return the number of partitions."""
        return self._npartitions

    def nmodels(self) -> int:
        """Return the number of models."""
        return self._nmodels

    def _restriction_matrix(self, subset: Array) -> Array:
        # TODO Consider replacing _restriction_matrix.T.dot(A) with
        # special indexing applied to A
        nsubset = len(subset)
        mat = self._bkd.zeros((nsubset, self.nmodels() * self._stat.nstats()))
        for ii in range(nsubset):
            mat[ii, subset[ii]] = 1.0
        return mat

    def _check_cov(self, cov: Array, costs: Array) -> tuple[Array, Array]:
        if cov.shape[0] != len(costs):
            raise ValueError("cov and costs are inconsistent")
        return cov, self._bkd.asarray(costs)

    def _preprocess_model_subsets(self, model_subsets: List[Array]) -> List[Array]:
        """Hook for preprocessing model subsets. Default returns input unchanged.

        Subclasses can override this method to customize subset preprocessing.

        Parameters
        ----------
        model_subsets : List[Array]
            List of model subsets

        Returns
        -------
        List[Array]
            Preprocessed model subsets
        """
        return model_subsets

    @abstractmethod
    def _get_allocation_matrix(self, subsets: List[Array]) -> Array:
        """Abstract method to get allocation matrix. Must be overridden.

        Parameters
        ----------
        subsets : List[Array]
            List of subsets expanded to stat indices

        Returns
        -------
        Array
            Allocation matrix of shape (nsubsets, npartitions)
        """
        pass

    def _set_subsets(
        self, model_subsets: List[Array],
    ) -> tuple[List[Array], List[Array], Array]:
        """Template method for subset setup - DO NOT override in subclasses.

        This method orchestrates the subset setup process by calling hook
        methods that subclasses can customize.

        Parameters
        ----------
        model_subsets : List[Array]
            List of model subsets, or None to generate all subsets

        Returns
        -------
        Tuple[List[Array], List[Array], Array]
            model_subsets, subsets (expanded to stat indices), allocation_mat
        """
        if model_subsets is None:
            model_subsets = get_model_subsets(self.nmodels(), self._bkd)

        # Hook 1: preprocessing (subclasses override)
        model_subsets = self._preprocess_model_subsets(model_subsets)

        # Common: expand to stat indices
        # stats ordered by all stats model 0, all stats model 1 and so on
        # ordering of statistics for a given model is determined by
        # the stat class
        model_stat_ids = self._bkd.reshape(
            self._bkd.arange(self._nmodels * self._stat.nstats(), dtype=int),
            (self._nmodels, self._stat.nstats()),
        )
        subsets = []
        for ii in range(len(model_subsets)):
            subsets.append(
                self._bkd.hstack(
                    [model_stat_ids[model_id] for model_id in model_subsets[ii]]
                )
            )

        # Hook 2: allocation matrix (subclasses override)
        allocation_mat = self._get_allocation_matrix(subsets)

        return model_subsets, subsets, allocation_mat

    def _get_partitions_per_model(self) -> Array:
        # assume npartitions = nsubsets
        npartitions = self._allocation_mat.shape[1]
        partitions_per_model = self._bkd.full((self.nmodels(), npartitions), 0.0)
        for ii, model_subset in enumerate(self._model_subsets):
            # Use bkd.asarray for boolean mask to avoid mypy comparison-overlap
            mask = self._bkd.equal(self._allocation_mat[ii], 1.0)
            for m_id in model_subset:
                partitions_per_model[m_id, mask] = 1.0
        return partitions_per_model

    def _compute_nsamples_per_model(self, npartition_samples: Array) -> Array:
        if self._bkd.is_integer_dtype(npartition_samples):
            raise TypeError(
                "_compute_nsamples_per_model requires float-typed "
                f"npartition_samples, got dtype={npartition_samples.dtype}"
            )
        nsamples_per_model = self._bkd.einsum(
            "ji,i->j", self._partitions_per_model, npartition_samples
        )
        return nsamples_per_model

    def _estimator_cost(self, npartition_samples: Array) -> Array:
        return self._bkd.sum(
            self._costs * self._compute_nsamples_per_model(npartition_samples)
        )

    def _get_subset_intersecting_partitions(self) -> Array:
        amat = self._allocation_mat
        npartitions = self._allocation_mat.shape[1]
        partition_intersect = self._bkd.full(
            (self.nsubsets(), self.nsubsets(), npartitions), 0.0
        )
        for ii, subset_ii in enumerate(self._subsets):
            for jj, subset_jj in enumerate(self._subsets):
                # partitions are shared when sum of allocation entry is 2
                mask = self._bkd.equal(amat[ii] + amat[jj], 2.0)
                partition_intersect[ii, jj, mask] = 1.0
        return partition_intersect

    def _nintersect_samples(self, npartition_samples: Array) -> Array:
        """
        Get the number of samples in the intersection of two subsets.

        Note the number of samples per subset is simply the diagonal of this
        matrix
        """
        if self._bkd.is_integer_dtype(npartition_samples):
            raise TypeError(
                "_nintersect_samples requires float-typed "
                f"npartition_samples, got dtype={npartition_samples.dtype}"
            )
        return self._bkd.einsum(
            "ijk,k->ij", self._partitions_intersect, npartition_samples
        )

    def _sigma(self, npartition_samples: Array) -> Array:
        sigma_blocks = _grouped_acv_sigma(
            self.nmodels(),
            self._nintersect_samples(npartition_samples),
            self._subsets,
            self._stat,
        )
        return self._bkd.vstack(
            [self._bkd.hstack(row) for row in sigma_blocks]
        )

    def _inv(self, mat: Array) -> Array:
        if self._use_pseudo_inv:
            return self._bkd.pinv(mat)
        return self._bkd.inv(mat)

    def _psi_matrix_from_sigma(self, Sigma: Array) -> Array:
        # TODO instead of applying R matrices just collect correct rows
        # and columns
        psi_reg_mat = (
            self._bkd.eye(self._nT_stats) * self._reg_blue
        )
        return (
            self._bkd.multidot(
                [self._R, self._inv(Sigma), self._R.T]
            )
            + psi_reg_mat
        )

    def _block_precision_contribution(self, k: int, n_k: Array) -> Array:
        """Compute R_k @ inv(Sigma_k(n_k)) @ R_k^T for partition k."""
        subset = self._subsets[k]
        sigma_k = _grouped_acv_sigma_block(
            subset, subset, n_k, n_k, n_k, self._stat
        )
        bkd = self._bkd
        if bkd.all_bool(sigma_k == 0):
            return bkd.zeros((self._nT_stats, self._nT_stats))
        sigma_k_inv = self._inv(sigma_k)
        R_k = self._restriction_matrices[k]
        return bkd.multidot([R_k, sigma_k_inv, R_k.T])

    def _block_weight_contribution(
        self, k: int, n_k: Array, psi_inv: Array
    ) -> Array:
        """Compute Sigma_k^{-1} R_k^T Psi^{-1} for partition k."""
        subset = self._subsets[k]
        sigma_k = _grouped_acv_sigma_block(
            subset, subset, n_k, n_k, n_k, self._stat
        )
        bkd = self._bkd
        if bkd.all_bool(sigma_k == 0):
            block_size = sigma_k.shape[0]
            return bkd.zeros((block_size, self._nT_stats))
        sigma_k_inv = self._inv(sigma_k)
        R_k = self._restriction_matrices[k]
        return sigma_k_inv @ R_k.T @ psi_inv

    def _psi_matrix(self, npartition_samples: Array) -> Array:
        Sigma = self._sigma(npartition_samples)
        return self._psi_matrix_from_sigma(Sigma)

    def _psi_inv_from_npartition_samples(self, npartition_samples: Array) -> Array:
        psi = self._psi_matrix(npartition_samples)
        psi_inv = self._inv(psi)
        return psi_inv

    def _covariance_from_npartition_samples(self, npartition_samples: Array) -> Array:
        psi_inv = self._psi_inv_from_npartition_samples(npartition_samples)
        return self._bkd.multidot([self._asketch, psi_inv, self._asketch.T])

    def _get_model_subset_costs(self, subsets: List[Array], costs: Array) -> Array:
        subset_costs = self._bkd.array(
            [self._bkd.sum(costs[subset]) for subset in subsets])
        return subset_costs

    def _nelder_mead_min_nlf_samples_constraint(
            self, x: Array, min_nlf_samples: int, ii: int) -> Array:
        return self._bkd.sum(
            self._bkd.asarray(self._partitions_per_model[ii]) * x
        ) - min_nlf_samples

    def _get_nelder_mead_constraints(
        self,
        target_cost: float,
        min_nhf_samples: int,
        min_nlf_samples: Optional[List[int]],
        constraint_reg: float = 0,
    ) -> List[dict[str, object]]:
        cons = [
            {
                "type": "ineq",
                "fun": self._cost_constraint,
                "args": (target_cost,),
            }
        ]
        cons += [
            {
                "type": "ineq",
                "fun": self._nelder_mead_min_nlf_samples_constraint,
                "args": [min_nhf_samples, 0],
            }
        ]
        if min_nlf_samples is not None:
            if len(min_nlf_samples) != self.nmodels() - 1:
                raise ValueError(
                    f"len(min_nlf_samples) {len(min_nlf_samples)} "
                    f"!= nmodels - 1 ({self.nmodels() - 1})"
                )
            for ii in range(1, self.nmodels()):
                cons += [
                    {
                        "type": "ineq",
                        "fun": self._nelder_mead_min_nlf_samples_constraint,
                        "args": [
                            min_nlf_samples,
                            ii,
                        ],
                    }
                ]
        return cons

    def _cost_constraint(self, npartition_samples: Array, target_cost: float) -> Array:
        return target_cost - self._estimator_cost(npartition_samples)

    def _init_guess(self, target_cost: float) -> Array:
        # start with the same number of samples per partition

        # get the number of samples per model when 1 sample is in each
        # partition
        nsamples_per_model = self._compute_nsamples_per_model(
            self._bkd.full((self.npartitions(),), 1.0)
        )
        # nsamples_per_model[0] = max(0, min_nhf_samples)
        cost = self._bkd.sum(nsamples_per_model * self._costs)

        # the total number of samples per partition is then target_cost/cost
        # we take the floor to make sure we do not exceed the target cost
        return self._bkd.full(
            (self.npartitions(),),
            self._bkd.to_float(self._bkd.floor(target_cost / cost)),
            dtype=self._bkd.double_dtype(),
        )[:, None]

    def _validate_asketch(self, asketch: Optional[Array]) -> Array:
        if asketch is None:
            asketch = self._bkd.full(
                (self._stat.nstats(), self._stat.nstats() * self.nmodels()),
                0.0,
            )
            for nn in range(self._stat.nstats()):
                asketch[nn, nn] = 1.0
        asketch = self._bkd.asarray(asketch)
        if asketch.shape != (
            self._stat.nstats(),
            self._stat.nstats() * self.nmodels(),
        ):
            raise ValueError(
                "aksetch shape {0} must be {1}".format(
                    asketch.shape,
                    (
                        self._stat.nstats(),
                        self._stat.nstats() * self.nmodels(),
                    ),
                )
            )
        return asketch

    def _get_partition_splits(self, npartition_samples: Array) -> Array:
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        cumsum_vals = self._bkd.cumsum(npartition_samples)
        # Convert to int64 using backend methods
        cumsum_int = self._bkd.asarray(
            self._bkd.round(cumsum_vals), dtype=self._bkd.int64_dtype()
        )
        splits = self._bkd.hstack(
            (
                self._bkd.zeros((1,), dtype=self._bkd.int64_dtype()),
                cumsum_int,
            )
        )
        return splits

    def _grouped_acv_beta(self, npartition_samples: Array) -> Array:
        sigma = self._sigma(npartition_samples)
        psi_matrix = self._psi_matrix_from_sigma(sigma)
        beta = self._bkd.stack(
            [
                self._bkd.multidot(
                    [
                        self._inv(sigma),
                        self._R.T,
                        self._bkd.solve(psi_matrix, asketch),
                    ]
                )
                for asketch in self._asketch
            ],
            axis=0,
        )
        return beta

    def _compute_correction(self, beta: Array) -> Array:
        """Deterministic correction from known-mean models.

        Computes sum_{(l,s) in K_stat} s_{(l,s)} * q_{(l,s)}
        where s_full = R_full @ beta.T and q are the known values.
        """
        nstats = self._stat.nstats()
        nstats_output = self._asketch.shape[0]
        s_full = self._R_full @ beta.T
        correction = self._bkd.zeros((nstats_output,))
        if self._known_values is None:
            raise RuntimeError("_known_values must be set for correction")
        for flat_idx in self._K_stat_indices:
            model = flat_idx // nstats
            slot = flat_idx % nstats
            correction = (
                correction
                + s_full[flat_idx, :] * self._known_values[model, slot]
            )
        return correction

    def __repr__(self) -> str:
        return "{0}(nmodels={1}, nsubsets={2})".format(
            self.__class__.__name__, self.nmodels(), self.nsubsets()
        )


class FittedGroupACVEstimator(Generic[Array]):
    """Frozen GroupACV estimator with a fixed allocation.

    Composes (template: BaseGroupACVEstimator, allocation: GroupACVAllocationResult).
    Eagerly computes covariance and sigma from discrete (post-rounding) counts.
    """

    def __init__(
        self,
        template: BaseGroupACVEstimator[Array],
        allocation: "GroupACVAllocationResult[Array]",
    ) -> None:
        if not allocation.success:
            raise ValueError(
                f"Cannot create fitted estimator from failed allocation: "
                f"{allocation.message}"
            )
        bkd = template._bkd
        if not bkd.is_integer_dtype(allocation.npartition_samples):
            raise TypeError(
                f"allocation.npartition_samples must be integer-typed, "
                f"got dtype={allocation.npartition_samples.dtype}"
            )
        if not bkd.is_integer_dtype(allocation.nsamples_per_model):
            raise TypeError(
                f"allocation.nsamples_per_model must be integer-typed, "
                f"got dtype={allocation.nsamples_per_model.dtype}"
            )

        self._template = template
        self._allocation = allocation
        self._bkd = bkd
        self._stat = template._stat

        self._nps_float = bkd.asarray(
            allocation.npartition_samples, dtype=bkd.double_dtype()
        )
        self._covariance_val = template._covariance_from_npartition_samples(
            self._nps_float
        )
        self._sample_splits_val = self._sample_splits_per_model(
            allocation.npartition_samples
        )

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def covariance(self) -> Array:
        """Return the estimator covariance at the allocated sample count."""
        return self._covariance_val

    def npartition_samples(self) -> Array:
        """Return the number of samples per partition."""
        return self._allocation.npartition_samples

    def nsamples_per_model(self) -> Array:
        """Return the number of samples allocated to each model."""
        return self._allocation.nsamples_per_model

    def objective_value(self) -> Array:
        """Return the objective value from the allocation."""
        return self._allocation.objective_value

    def actual_cost(self) -> float:
        """Return the actual cost of the allocation."""
        return self._allocation.actual_cost

    def sample_splits(self) -> List[Array]:
        """Return the eagerly-computed sample splits per model."""
        return self._sample_splits_val

    def _sample_splits_per_model(
        self, npartition_samples: Array,
    ) -> List[Array]:
        t = self._template
        partition_splits = t._get_partition_splits(npartition_samples)
        splits_per_model = []
        for ii in range(t.nmodels()):
            active_partitions = self._bkd.where(
                t._partitions_per_model[ii]
            )[0]
            splits = self._bkd.full(
                (t.npartitions(), 2), -1, dtype=int
            )
            lb = self._bkd.zeros((), dtype=self._bkd.int64_dtype())
            ub = self._bkd.zeros((), dtype=self._bkd.int64_dtype())
            for idx in active_partitions:
                ub += partition_splits[idx + 1] - partition_splits[idx]
                splits[idx] = self._bkd.array([lb, ub])
                lb = self._bkd.copy(ub)
            splits_per_model.append(splits)
        return splits_per_model

    def _separate_values_per_model(
        self, values_per_model: List[Array],
    ) -> List[Array]:
        t = self._template
        nsamples_per_model = self._allocation.nsamples_per_model
        if len(values_per_model) != t.nmodels():
            raise ValueError(
                f"len(values_per_model) {len(values_per_model)} "
                f"!= nmodels {t.nmodels()}"
            )
        for ii in range(t.nmodels()):
            nsamples_ii = self._bkd.to_int(nsamples_per_model[ii])
            if values_per_model[ii].shape[1] != nsamples_ii:
                raise ValueError(
                    f"values_per_model[{ii}].shape[1]: "
                    f"{values_per_model[ii].shape[1]} != "
                    f"nsamples_per_model[{ii}]: {nsamples_per_model[ii]}"
                )

        splits_per_model = self._sample_splits_val
        values_per_subset = []
        for ii, model_subset in enumerate(t._model_subsets):
            values = []
            active_partitions = self._bkd.where(t._allocation_mat[ii])[0]
            for model_id in model_subset:
                splits = splits_per_model[model_id]
                values.append(
                    self._bkd.hstack(
                        [
                            values_per_model[model_id][
                                :, splits[idx, 0] : splits[idx, 1]
                            ]
                            for idx in active_partitions
                        ]
                    )
                )
            values_per_subset.append(self._bkd.vstack(values))
        return values_per_subset

    def _estimate(self, values_per_subset: List[Array]) -> Array:
        t = self._template
        beta = t._grouped_acv_beta(self._nps_float)
        ll, mm = 0, 0
        acv_stat: Array = self._bkd.zeros((beta.shape[0],))
        for kk in range(t.nsubsets()):
            mm += len(t._subsets[kk])
            if values_per_subset[kk].shape[1] > 0:
                subset_stat = self._stat.sample_estimate(
                    values_per_subset[kk]
                )
                acv_stat += (beta[:, ll:mm]) @ subset_stat
            ll = mm
        return acv_stat

    def _traditional_acv_weights(self) -> Array:
        t = self._template
        beta = t._grouped_acv_beta(self._nps_float)
        if not self._bkd.allclose(
            self._bkd.sum(beta, axis=1),
            self._bkd.ones((beta.shape[0],)),
        ):
            raise ValueError(
                f"beta rows do not sum to 1: "
                f"{self._bkd.sum(beta, axis=1)}"
            )
        alpha = self._bkd.zeros(
            (beta.shape[0], (t._nmodels - 1) * self._stat.nstats())
        )
        zeros = self._bkd.zeros((beta.shape[0],))
        kk = 0
        for subset in t._subsets:
            for jj in subset:
                if jj < self._stat.nstats():
                    kk += 1
                    continue
                alpha[:, jj - self._stat.nstats()] += self._bkd.maximum(
                    beta[:, kk], zeros
                )
                kk += 1
        return alpha

    def _extract_from_flattened_subset_matrix(
        self, mat: Array, subset_idx: int,
    ) -> Array:
        t = self._template
        nprev_stats = sum(
            [subset.shape[0] for subset in t._subsets[:subset_idx]]
        )
        subset = t._subsets[subset_idx]
        subset_mat = mat[:, nprev_stats : nprev_stats + subset.shape[0]]
        R = t._restriction_matrix(subset)
        expanded_mat = (R.T @ subset_mat.T).T
        return expanded_mat

    def _group_to_traditional_estimators_from_alpha(
        self, subset_ests: List[Array], alpha: Array,
    ) -> Tuple[Array, Array, Array]:
        t = self._template
        beta = t._grouped_acv_beta(self._nps_float)
        Q0 = self._bkd.zeros((self._stat.nstats(),))
        Qe = self._bkd.zeros((self._stat.nstats(), (t._nmodels - 1)))
        Qu = self._bkd.zeros((self._stat.nstats(), (t._nmodels - 1)))
        for ii, subset in enumerate(t._subsets):
            beta_tilde = self._extract_from_flattened_subset_matrix(beta, ii)
            B0_tilde = beta_tilde[:, : self._stat.nstats()]
            BL_tilde = beta_tilde[:, self._stat.nstats() :]
            R = t._restriction_matrix(subset)
            Q_tilde = R.T @ subset_ests[ii]
            Q0_tilde = Q_tilde[: self._stat.nstats()]
            QL_tilde = Q_tilde[self._stat.nstats() :]
            Q0 += B0_tilde @ Q0_tilde
            nstats = self._stat.nstats()
            for kk in range(beta.shape[0]):
                for ll in range(1, t._nmodels):
                    zero = self._bkd.zeros(())
                    bl_kk = BL_tilde[kk, (ll - 1) * nstats + kk]
                    wu = (
                        (1 / alpha[kk, (ll - 1) * nstats + kk])
                        * self._bkd.maximum(bl_kk, zero)
                    )
                    we = (
                        -(1 / alpha[kk, (ll - 1) * nstats + kk])
                        * self._bkd.minimum(bl_kk, zero)
                    )
                    Qe[kk, ll - 1] += QL_tilde[(ll - 1) * nstats + kk] * we
                    Qu[kk, ll - 1] += QL_tilde[(ll - 1) * nstats + kk] * wu
        return Q0, Qe, Qu

    def _group_to_traditional_estimators(
        self, subset_ests: List[Array],
    ) -> Tuple[Array, Array, Array]:
        alpha = self._traditional_acv_weights()
        return self._group_to_traditional_estimators_from_alpha(
            subset_ests, alpha
        )

    def generate_samples_per_model(
        self,
        rvs: Callable[[int], Array],
        npilot_samples: int = 0,
    ) -> List[Array]:
        """Generate samples for each model based on the allocation."""
        t = self._template
        npartition_samples = self._allocation.npartition_samples
        npart_sum = self._bkd.sum(npartition_samples)
        ntotal_independent_samples = self._bkd.to_int(npart_sum)
        partition_splits = t._get_partition_splits(npartition_samples)
        samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        for ii in range(t.nmodels()):
            active_partitions = self._bkd.where(
                t._partitions_per_model[ii]
            )[0]
            samples_per_model.append(
                self._bkd.hstack(
                    [
                        samples[
                            :,
                            self._bkd.to_int(
                                partition_splits[idx]
                            ) : self._bkd.to_int(
                                partition_splits[idx + 1]
                            ),
                        ]
                        for idx in active_partitions
                    ]
                )
            )
        if npilot_samples == 0:
            return samples_per_model

        if self._bkd.max(
            t._partitions_per_model[0] * npartition_samples
        ) < npilot_samples:
            raise ValueError(
                "Insert pilot samples currently only supported when only"
                " the largest subset of those containing the "
                "high-fidelity model can fit all pilot samples. "
                f"npilot = {npilot_samples} != "
                f"{self._bkd.max(t._partitions_per_model[0] * npartition_samples)}"
            )
        return self._remove_pilot_samples(
            npilot_samples, samples_per_model
        )[0]

    def _reduce_model_sample_splits(
        self,
        model_id: int,
        partition_id: int,
        nsamples_to_reduce: int,
    ) -> tuple[Array, tuple[Array, Array]]:
        splits_per_model = self._sample_splits_val
        lb, ub = splits_per_model[model_id][partition_id]
        sample_splits = self._bkd.copy(splits_per_model[model_id])
        sample_splits[partition_id][0] = lb + nsamples_to_reduce
        removed_split = lb, lb + nsamples_to_reduce
        return sample_splits, removed_split

    def _remove_pilot_samples(
        self,
        npilot_samples: int,
        samples_per_model: List[Array],
    ) -> tuple[List[Array], Optional[Array]]:
        t = self._template
        npartition_samples = self._allocation.npartition_samples
        nsamples_per_model = self._allocation.nsamples_per_model
        active_hf_subsets = self._bkd.where(
            self._bkd.equal(t._partitions_per_model[0], 1)
        )[0]
        part_id = self._bkd.to_int(active_hf_subsets[
            self._bkd.argmax(npartition_samples[active_hf_subsets])
        ])
        removed_samples = None
        for model_id_arr in t._subsets[part_id]:
            mid = self._bkd.to_int(model_id_arr)
            if npilot_samples > npartition_samples[part_id]:
                raise ValueError(
                    f"Too many pilot values {npilot_samples}+>"
                    f"{npartition_samples[part_id]}"
                )
            if (
                samples_per_model[mid].shape[1]
                != self._bkd.to_int(nsamples_per_model[mid])
            ):
                raise ValueError("samples per model has the wrong size")
            splits, removed_split = self._reduce_model_sample_splits(
                mid, part_id, npilot_samples
            )
            rs0 = self._bkd.to_int(removed_split[0])
            rs1 = self._bkd.to_int(removed_split[1])
            if removed_samples is None:
                removed_samples = samples_per_model[mid][:, rs0:rs1]
            else:
                if not self._bkd.allclose(
                    removed_samples,
                    samples_per_model[mid][:, rs0:rs1],
                ):
                    raise ValueError(
                        "Removed samples differ across models"
                    )
            samples_per_model[mid] = self._bkd.hstack(
                [
                    samples_per_model[mid][
                        :,
                        self._bkd.to_int(splits[idx, 0]) : self._bkd.to_int(
                            splits[idx, 1]
                        ),
                    ]
                    for idx in self._bkd.where(
                        self._bkd.equal(t._partitions_per_model[mid], 1)
                    )[0]
                ]
            )
        return samples_per_model, removed_samples

    def insert_pilot_values(
        self,
        pilot_values: List[Array],
        values_per_model: List[Array],
    ) -> List[Array]:
        """Insert pilot values into the model evaluation arrays."""
        t = self._template
        npartition_samples = self._allocation.npartition_samples
        npilot_values = pilot_values[0].shape[1]
        if self._bkd.to_float(self._bkd.max(
            t._partitions_per_model[0] * npartition_samples
        )) < npilot_values:
            raise ValueError(
                "Insert pilot samples currently only supported when only"
                " the largest subset of those containing the "
                "high-fidelity model can fit all pilot samples"
            )

        new_values_per_model = [self._bkd.copy(v) for v in values_per_model]
        active_hf_subsets = self._bkd.where(
            self._bkd.equal(t._partitions_per_model[0], 1)
        )[0]
        part_id = self._bkd.to_int(active_hf_subsets[
            self._bkd.argmax(npartition_samples[active_hf_subsets])
        ])
        splits_per_model = self._sample_splits_val
        for model_id_arr in t._subsets[part_id]:
            mid = self._bkd.to_int(model_id_arr)
            npilot_values = pilot_values[mid].shape[1]
            if npilot_values != pilot_values[0].shape[1]:
                raise ValueError(
                    "Must have the same number of pilot values "
                    "for each model"
                )
            if npilot_values > npartition_samples[part_id]:
                raise ValueError(
                    f"Too many pilot values {npilot_values}>"
                    f"{npartition_samples[part_id]}"
                )
            lb = self._bkd.to_int(splits_per_model[mid][part_id][0])
            new_values_per_model[mid] = self._bkd.hstack(
                (
                    values_per_model[mid][:, :lb],
                    pilot_values[mid],
                    values_per_model[mid][:, lb:],
                )
            )
        return new_values_per_model

    def __call__(self, values_per_model: List[Array]) -> Array:
        """Compute the GroupACV estimate from model values."""
        t = self._template
        values_per_subset = self._separate_values_per_model(values_per_model)
        stochastic_est = self._estimate(values_per_subset)
        if not t._has_known_quantities:
            return stochastic_est
        beta = t._grouped_acv_beta(self._nps_float)
        return stochastic_est - t._compute_correction(beta)

    def __repr__(self) -> str:
        obj = self._bkd.to_float(
            self._bkd.flatten(self._allocation.objective_value)[0]
        )
        cost = self._allocation.actual_cost
        nsamp = self._allocation.nsamples_per_model
        return (
            f"{type(self).__name__}(criteria={obj:.3g} "
            f"cost={cost:.5g} nsamples={nsamp})"
        )
