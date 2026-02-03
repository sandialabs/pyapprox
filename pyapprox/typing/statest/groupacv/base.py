"""Base GroupACV estimator implementation.

This module provides the GroupACVEstimator class for Group Approximate
Control Variate estimation.
"""

from typing import Generic, List, TYPE_CHECKING

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend

from pyapprox.typing.statest.groupacv.utils import (
    get_model_subsets,
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _nest_subsets,
    _grouped_acv_sigma,
)

from pyapprox.typing.statest.groupacv.optimization import (
    GroupACVObjective,
    GroupACVTraceObjective,
    GroupACVLogDetObjective,
    GroupACVCostConstraint,
)

if TYPE_CHECKING:
    from pyapprox.typing.statest.statistics import (
        MultiOutputStatistic,
        MultiOutputMean,
        MultiOutputVariance,
        MultiOutputMeanAndVariance,
    )


class GroupACVEstimator(Generic[Array]):
    """Group Approximate Control Variate estimator.

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

    est_type : str, optional
        Estimation type: "is" for independent sampling or "nested".
        Default is "is".

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def __init__(
        self,
        stat: "MultiOutputStatistic",
        costs: Array,
        reg_blue: float = 0,
        model_subsets: List[Array] = None,
        est_type: str = "is",
        asketch: Array = None,
        use_pseudo_inv: bool = True,
    ):
        from pyapprox.typing.statest.statistics import (
            MultiOutputMean,
            MultiOutputVariance,
            MultiOutputMeanAndVariance,
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
            raise ValueError(
                "GroupACV only supports estimation of mean or variance"
            )
        self._stat = stat

        self._model_subsets, self._subsets, self._allocation_mat = (
            self._set_subsets(model_subsets, est_type)
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
        self._optimized_criteria = None
        self._asketch = self._validate_asketch(asketch)

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

    def _check_cov(self, cov, costs):
        if cov.shape[0] != len(costs):
            print(cov.shape, costs.shape)
            raise ValueError("cov and costs are inconsistent")
        return cov, self._bkd.asarray(costs)

    def _set_subsets(self, model_subsets: Array, est_type: str):
        if model_subsets is None:
            model_subsets = get_model_subsets(self.nmodels(), self._bkd)
        if est_type == "is":
            get_allocation_mat = _get_allocation_matrix_is
        elif est_type == "nested":
            zero = self._bkd.zeros((1,), dtype=int)
            for ii, subset in enumerate(model_subsets):
                if not isinstance(subset, self._bkd.array_type()):
                    raise ValueError(
                        "subset must be an instance of {0}".format(
                            self._bkd.array_type()
                        )
                    )
                if self._bkd.allclose(subset, zero):
                    del model_subsets[ii]
                    break
            model_subsets = _nest_subsets(
                model_subsets, self.nmodels(), self._bkd
            )[0]
            get_allocation_mat = _get_allocation_matrix_nested
        else:
            raise ValueError(
                "incorrect est_type {0} specified".format(est_type)
            )
        # amend subsets to include indices into each statistic
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
                    [
                        model_stat_ids[model_id]
                        for model_id in model_subsets[ii]
                    ]
                )
            )
        return model_subsets, subsets, get_allocation_mat(subsets, self._bkd)

    def _get_partitions_per_model(self):
        # assume npartitions = nsubsets
        npartitions = self._allocation_mat.shape[1]
        partitions_per_model = self._bkd.full(
            (self.nmodels(), npartitions), 0.0
        )
        for ii, model_subset in enumerate(self._model_subsets):
            partitions_per_model[
                np.ix_(model_subset, self._allocation_mat[ii] == 1)
            ] = 1
        return partitions_per_model

    def _compute_nsamples_per_model(self, npartition_samples):
        nsamples_per_model = self._bkd.einsum(
            "ji,i->j", self._partitions_per_model, npartition_samples
        )
        return nsamples_per_model

    def _estimator_cost(self, npartition_samples):
        return sum(
            self._costs * self._compute_nsamples_per_model(npartition_samples)
        )

    def _get_subset_intersecting_partitions(self):
        amat = self._allocation_mat
        npartitions = self._allocation_mat.shape[1]
        partition_intersect = self._bkd.full(
            (self.nsubsets(), self.nsubsets(), npartitions), 0.0
        )
        for ii, subset_ii in enumerate(self._subsets):
            for jj, subset_jj in enumerate(self._subsets):
                # partitions are shared when sum of allocation entry is 2
                partition_intersect[ii, jj, amat[ii] + amat[jj] == 2] = 1.0
        return partition_intersect

    def _nintersect_samples(self, npartition_samples):
        """
        Get the number of samples in the intersection of two subsets.

        Note the number of samples per subset is simply the diagonal of this
        matrix
        """
        return self._bkd.einsum(
            "ijk,k->ij", self._partitions_intersect, npartition_samples
        )

    def _sigma(self, npartition_samples):
        Sigma = _grouped_acv_sigma(
            self.nmodels(),
            self._nintersect_samples(npartition_samples),
            self._subsets,
            self._stat,
        )
        Sigma = self._bkd.vstack([self._bkd.hstack(row) for row in Sigma])
        return Sigma

    def _inv(self, mat):
        if self._use_pseudo_inv:
            return self._bkd.pinv(mat)
        return self._bkd.inv(mat)

    def _psi_matrix_from_sigma(self, Sigma):
        # TODO instead of applying R matrices just collect correct rows
        # and columns
        psi_reg_mat = (
            self._bkd.eye(self.nmodels() * self._stat.nstats())
            * self._reg_blue
        )
        # sigma_reg_mat = self._bkd.eye(Sigma.shape[0]) * self._reg_blue
        # print(Sigma)
        return (
            self._bkd.multidot(
                # (self._R, self._inv(Sigma + sigma_reg_mat), self._R.T)
                (self._R, self._inv(Sigma), self._R.T)
            )
            + psi_reg_mat
        )

    def _psi_matrix(self, npartition_samples):
        Sigma = self._sigma(npartition_samples)
        return self._psi_matrix_from_sigma(Sigma)

    def _psi_inv_from_npartition_samples(self, npartition_samples):
        psi = self._psi_matrix(npartition_samples)
        # print(self._bkd.cond(psi), "COND")
        psi_inv = self._inv(psi)
        return psi_inv

    def _covariance_from_npartition_samples(self, npartition_samples):
        psi_inv = self._psi_inv_from_npartition_samples(npartition_samples)
        return self._bkd.multidot((self._asketch, psi_inv, self._asketch.T))

    def _get_model_subset_costs(self, subsets, costs):
        subset_costs = self._bkd.array(
            [costs[subset].sum() for subset in subsets]
        )
        return subset_costs

    def _nelder_mead_min_nlf_samples_constraint(self, x, min_nlf_samples, ii):
        return (
            self._partitions_per_model[ii].numpy() * x
        ).sum() - min_nlf_samples

    def _get_nelder_mead_constraints(
        self, target_cost, min_nhf_samples, min_nlf_samples, constraint_reg=0
    ):
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
            assert len(min_nlf_samples) == self.nmodels() - 1
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

    def _init_guess(self, target_cost):
        # start with the same number of samples per partition

        # get the number of samples per model when 1 sample is in each
        # partition
        nsamples_per_model = self._compute_nsamples_per_model(
            self._bkd.full((self.npartitions(),), 1.0)
        )
        # nsamples_per_model[0] = max(0, min_nhf_samples)
        cost = (nsamples_per_model * self._costs).sum()

        # the total number of samples per partition is then target_cost/cost
        # we take the floor to make sure we do not exceed the target cost
        return self._bkd.full(
            (self.npartitions(),), self._bkd.floor(target_cost / cost)
        )[:, None]

    def _set_optimized_params_base(
        self,
        rounded_npartition_samples,
        rounded_nsamples_per_model,
        rounded_target_cost,
    ):
        self._rounded_npartition_samples = rounded_npartition_samples
        self._rounded_nsamples_per_model = rounded_nsamples_per_model
        self._rounded_target_cost = rounded_target_cost
        self._opt_sample_splits = self._sample_splits_per_model()
        self._optimized_sigma = self._sigma(self._rounded_npartition_samples)
        self._optimized_covariance = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples
        )
        # Compute optimized criteria if objective is available
        if hasattr(self, "_objective") and self._objective is not None:
            self._optimized_criteria = self._objective(
                self._rounded_npartition_samples[:, None]
            )
        else:
            # Use default objective for criteria computation
            obj = self.default_objective()
            obj.set_estimator(self)
            self._optimized_criteria = obj(
                self._rounded_npartition_samples[:, None]
            )

    def optimized_covariance(self) -> Array:
        """Return the optimized covariance matrix."""
        return self._optimized_covariance

    def set_npartition_samples(self, npartition_samples: Array) -> None:
        """Set the sample allocation for estimation.

        Parameters
        ----------
        npartition_samples : Array
            Number of samples in each partition. Shape (npartitions,).
        """
        self._set_optimized_params(npartition_samples)

    def npartition_samples(self) -> Array:
        """Get current allocation.

        Returns
        -------
        Array
            Number of samples in each partition. Shape (npartitions,).

        Raises
        ------
        RuntimeError
            If allocation has not been set.
        """
        if (
            not hasattr(self, "_rounded_npartition_samples")
            or self._rounded_npartition_samples is None
        ):
            raise RuntimeError(
                "Allocation not set. Call set_npartition_samples() or use "
                "GroupACVAllocationOptimizer."
            )
        return self._rounded_npartition_samples

    def covariance(self) -> Array:
        """Compute covariance using stored allocation.

        Returns
        -------
        Array
            Covariance matrix of the estimator.
        """
        return self._covariance_from_npartition_samples(self.npartition_samples())

    def _set_optimized_params(self, npartition_samples, round_nsamples=True):
        # expected scalar type Double but found Float error can occur
        # with torch if npartition samples is not torch.double need to
        # think of a check that works for all backends
        if round_nsamples:
            # add 1e-4 to avoid rounding down value that is at the constraint
            # boundary but has numerical noise. Best value depends on
            # constraint satisfaction tolerance
            rounded_npartition_samples = self._bkd.floor(
                npartition_samples + 1e-4
            )
        else:
            rounded_npartition_samples = npartition_samples
        self._set_optimized_params_base(
            rounded_npartition_samples,
            self._compute_nsamples_per_model(rounded_npartition_samples),
            self._estimator_cost(rounded_npartition_samples),
        )

    def _validate_asketch(self, asketch):
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

    def default_objective(self) -> GroupACVObjective:
        """Return the default objective function."""
        return GroupACVLogDetObjective(self._bkd)

    def set_objective(self, objective: GroupACVObjective):
        """Set the objective function."""
        self._objective = objective

    def _get_partition_splits(self, npartition_samples):
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        cumsum_vals = self._bkd.cumsum(npartition_samples)
        # Convert to int64 - use backend array to handle type conversion
        cumsum_int = self._bkd.array(
            self._bkd.to_numpy(cumsum_vals).astype(int),
            dtype=self._bkd.int64_dtype(),
        )
        splits = self._bkd.hstack(
            (
                self._bkd.zeros((1,), dtype=self._bkd.int64_dtype()),
                cumsum_int,
            )
        )
        return splits

    def generate_samples_per_model(self, rvs, npilot_samples=0):
        """Generate samples for each model based on optimized allocation.

        Parameters
        ----------
        rvs : callable
            Function that generates random samples: rvs(nsamples) -> Array

        npilot_samples : int, optional
            Number of pilot samples to remove. Default is 0.

        Returns
        -------
        samples_per_model : List[Array]
            List of samples for each model
        """
        # Convert to int for rvs call - this is at a boundary where we're
        # generating samples, not computing gradients through rvs
        ntotal_independent_samples = int(self._rounded_npartition_samples.sum())
        partition_splits = self._get_partition_splits(
            self._rounded_npartition_samples
        )
        samples = rvs(ntotal_independent_samples)
        samples_per_model = []
        for ii in range(self.nmodels()):
            active_partitions = self._bkd.where(
                self._partitions_per_model[ii]
            )[0]
            samples_per_model.append(
                self._bkd.hstack(
                    [
                        samples[
                            :,
                            partition_splits[idx] : partition_splits[idx + 1],
                        ]
                        for idx in active_partitions
                    ]
                )
            )
        if npilot_samples == 0:
            return samples_per_model

        if (
            self._partitions_per_model[0] * self._rounded_npartition_samples
        ).max() < npilot_samples:
            msg = "Insert pilot samples currently only supported when only"
            msg += " the largest subset of those containing the "
            msg += "high-fidelity model can fit all pilot samples. "
            msg += "npilot = {0} != {1}".format(
                npilot_samples,
                (
                    self._partitions_per_model[0]
                    * self._rounded_npartition_samples
                ).max(),
            )
            raise ValueError(msg)
        return self._remove_pilot_samples(npilot_samples, samples_per_model)[0]

    def _sample_splits_per_model(self):
        # for each model get the sample splits in values_per_model
        # that correspond to each partition used in values_per_model.
        # If the model is not evaluated for a partition, then
        # the splits will be [-1, -1]
        partition_splits = self._get_partition_splits(
            self._rounded_npartition_samples
        )
        splits_per_model = []
        for ii in range(self.nmodels()):
            active_partitions = self._bkd.where(
                self._partitions_per_model[ii]
            )[0]
            splits = self._bkd.full((self.npartitions(), 2), -1, dtype=int)
            lb, ub = 0, 0
            for ii, idx in enumerate(active_partitions):
                ub += partition_splits[idx + 1] - partition_splits[idx]
                splits[idx] = self._bkd.array([lb, ub])
                lb = self._bkd.copy(ub)
            splits_per_model.append(splits)
        return splits_per_model

    def _separate_values_per_model(self, values_per_model):
        """Separate values per model into values per subset.

        Parameters
        ----------
        values_per_model : List[Array]
            Values for each model. Each array has shape (nqoi, nsamples_for_model).

        Returns
        -------
        List[Array]
            Values for each subset. Each array has shape (nqoi*nmodels_in_subset, nsamples_in_subset).
        """
        if len(values_per_model) != self.nmodels():
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self.nmodels()
            )
            raise ValueError(msg)
        for ii in range(self.nmodels()):
            # values shape is (nqoi, nsamples), so nsamples is shape[1]
            if (
                values_per_model[ii].shape[1]
                != self._rounded_nsamples_per_model[ii]
            ):
                msg = "{0} != {1}".format(
                    "values_per_model[{0}].shape[1]: {1}".format(
                        ii, values_per_model[ii].shape[1]
                    ),
                    "nsamples_per_model[{0}]: {1}".format(
                        ii, self._rounded_nsamples_per_model[ii]
                    ),
                )
                raise ValueError(msg)

        values_per_subset = []
        for ii, model_subset in enumerate(self._model_subsets):
            values = []
            active_partitions = self._bkd.where(self._allocation_mat[ii])[0]
            for model_id in model_subset:
                splits = self._opt_sample_splits[model_id]
                # values shape is (nqoi, nsamples), slice along columns
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
            # Stack models vertically: (nqoi*nmodels_in_subset, nsamples)
            values_per_subset.append(self._bkd.vstack(values))
        return values_per_subset

    def _grouped_acv_beta(self, sigma: Array) -> Array:
        psi_matrix = self._psi_matrix_from_sigma(sigma)
        beta = self._bkd.stack(
            [
                self._bkd.multidot(
                    (
                        self._inv(sigma),
                        self._R.T,
                        self._bkd.solve(psi_matrix, asketch),
                    )
                )
                for asketch in self._asketch
            ],
            axis=0,
        )
        return beta

    def _estimate(self, values_per_subset: List[Array]) -> Array:
        beta = self._grouped_acv_beta(self._optimized_sigma)
        ll, mm = 0, 0
        acv_stat = 0
        for kk in range(self.nsubsets()):
            mm += len(self._subsets[kk])
            # values shape is (nqoi*nmodels_in_subset, nsamples), check nsamples > 0
            if values_per_subset[kk].shape[1] > 0:
                subset_stat = self._stat.sample_estimate(values_per_subset[kk])
                acv_stat += (beta[:, ll:mm]) @ subset_stat
            ll = mm
        return acv_stat

    def _traditional_acv_weights(self) -> Array:
        beta = self._grouped_acv_beta(self._optimized_sigma)
        assert self._bkd.allclose(
            beta.sum(axis=1), self._bkd.ones(beta.shape[0])
        ), beta.sum(axis=1)
        alpha = self._bkd.zeros(
            (beta.shape[0], (self._nmodels - 1) * self._stat.nstats())
        )
        zeros = self._bkd.zeros((beta.shape[0],))
        kk = 0
        for subset in self._subsets:
            for jj in subset:
                if jj < self._stat.nstats():
                    kk += 1
                    continue
                alpha[:, jj - self._stat.nstats()] += self._bkd.maximum(
                    beta[:, kk], zeros
                )
                # alpha[:, jj - self._stat.nstats()] -= self._bkd.maximum(
                #     beta[:, kk], zeros
                # )
                kk += 1
        return alpha

    def _extract_from_flattened_subset_matrix(
        self, mat: Array, subset_idx: int
    ) -> Array:
        nprev_stats = sum(
            [subset.shape[0] for subset in self._subsets[:subset_idx]]
        )
        subset = self._subsets[subset_idx]
        subset_mat = mat[:, nprev_stats : nprev_stats + subset.shape[0]]
        R = self._restriction_matrix(subset)
        expanded_mat = (R.T @ subset_mat.T).T
        return expanded_mat

    def _group_to_traditional_estimators_from_alpha(
        self, subset_ests: List[Array], alpha: Array
    ) -> Array:
        beta = self._grouped_acv_beta(self._optimized_sigma)
        # beta shape (nstats, sum_s\insubsets nstats * nmodels_in_subset(s)
        Q0 = self._bkd.zeros((self._stat.nstats(),))
        Qe = self._bkd.zeros((self._stat.nstats(), (self._nmodels - 1)))
        Qu = self._bkd.zeros((self._stat.nstats(), (self._nmodels - 1)))
        # zeros = self._bkd.zeros((self._stat.nstats(), (self._nmodels - 1)))
        for ii, subset in enumerate(self._subsets):
            beta_tilde = self._extract_from_flattened_subset_matrix(beta, ii)
            # print(subset)
            # print(beta)
            # print(beta_tilde)
            B0_tilde = beta_tilde[:, : self._stat.nstats()]
            BL_tilde = beta_tilde[:, self._stat.nstats() :]
            R = self._restriction_matrix(subset)
            Q_tilde = R.T @ subset_ests[ii]
            Q0_tilde = Q_tilde[: self._stat.nstats()]
            QL_tilde = Q_tilde[self._stat.nstats() :]
            # print("\n", subset)
            # print(beta)
            # print(subset_ests[ii])
            # print(Q_tilde, "QT")
            # print(Q0_tilde, "Q0")
            # print(B0_tilde, "B0")
            # print(QL_tilde, "QL")
            # print(BL_tilde, "BL")
            # print(subset_ests[ii])
            Q0 += B0_tilde @ Q0_tilde
            # print(BL_tilde.shape, zeros.shape, alpha.shape)
            # we = self._bkd.maximum(BL_tilde, zeros) / alpha
            # wu = -self._bkd.minimum(BL_tilde, zeros) / alpha
            nstats = self._stat.nstats()
            # print(alpha.shape, beta.shape, Qe.shape)
            for kk in range(beta.shape[0]):
                for ll in range(1, self._nmodels):
                    print(
                        kk,
                        ll,
                        (ll - 1) * nstats + kk,
                        BL_tilde.shape,
                        alpha.shape,
                    )
                    wu = (1 / alpha[kk, (ll - 1) * nstats + kk]) * max(
                        BL_tilde[kk, (ll - 1) * nstats + kk], 0.0
                    )
                    we = -(1 / alpha[kk, (ll - 1) * nstats + kk]) * min(
                        BL_tilde[kk, (ll - 1) * nstats + kk], 0.0
                    )
                    Qe[kk, ll - 1] += QL_tilde[(ll - 1) * nstats + kk] * we
                    Qu[kk, ll - 1] += QL_tilde[(ll - 1) * nstats + kk] * wu
            # print(Qe)
        return Q0, Qe, Qu

    def _group_to_traditional_estimators(
        self, subset_ests: List[Array]
    ) -> Array:
        # Implement equations (15) in arxiv paper
        # wu = w_l^{k,u} and  we = w_l^{k,e} from arxiv paper
        alpha = self._traditional_acv_weights()
        return self._group_to_traditional_estimators_from_alpha(
            subset_ests, alpha
        )

    def __call__(self, values_per_model: List[Array]) -> Array:
        """Compute the GroupACV estimate from model values.

        Parameters
        ----------
        values_per_model : List[Array]
            List of model evaluation values, one array per model.
            Each array has shape (nsamples, nqoi).

        Returns
        -------
        Array
            The GroupACV estimate
        """
        values_per_subset = self._separate_values_per_model(values_per_model)
        return self._estimate(values_per_subset)

    def _reduce_model_sample_splits(
        self, model_id, partition_id, nsamples_to_reduce
    ):
        """return splits that occur when removing N samples of
        a partition of a given model"""
        lb, ub = self._opt_sample_splits[model_id][partition_id]
        sample_splits = self._bkd.copy(self._opt_sample_splits[model_id])
        sample_splits[partition_id][0] = lb + nsamples_to_reduce
        removed_split = lb, lb + nsamples_to_reduce
        return sample_splits, removed_split

    def _remove_pilot_samples(self, npilot_samples, samples_per_model):
        active_hf_subsets = self._bkd.where(
            self._partitions_per_model[0] == 1
        )[0]
        partition_id = active_hf_subsets[
            self._bkd.argmax(
                self._rounded_npartition_samples[active_hf_subsets]
            )
        ]
        removed_samples = None
        for model_id in self._subsets[partition_id]:
            if npilot_samples > self._rounded_npartition_samples[partition_id]:
                msg = "Too many pilot values {0}+>{1}".format(
                    npilot_samples,
                    self._rounded_npartition_samples[partition_id],
                )
                raise ValueError(msg)
            if (
                samples_per_model[model_id].shape[1]
                != self._rounded_nsamples_per_model[model_id]
            ):
                raise ValueError("samples per model has the wrong size")
            splits, removed_split = self._reduce_model_sample_splits(
                model_id, partition_id, npilot_samples
            )
            # removed samples must be computed before samples_per_model is
            # redefined below
            if removed_samples is None:
                removed_samples = samples_per_model[model_id][
                    :, removed_split[0] : removed_split[1]
                ]
            else:
                assert self._bkd.allclose(
                    removed_samples,
                    samples_per_model[model_id][
                        :, removed_split[0] : removed_split[1]
                    ],
                )
            samples_per_model[model_id] = self._bkd.hstack(
                [
                    samples_per_model[model_id][
                        :, splits[idx, 0] : splits[idx, 1]
                    ]
                    for idx in self._bkd.where(
                        self._partitions_per_model[model_id] == 1
                    )[0]
                ]
            )
        return samples_per_model, removed_samples

    def insert_pilot_values(self, pilot_values, values_per_model):
        """Insert pilot values into the model evaluation arrays.

        Parameters
        ----------
        pilot_values : List[Array]
            Pilot values for each model. Shape: (nqoi, npilot_samples)

        values_per_model : List[Array]
            Model evaluation values. Shape: (nqoi, nsamples_for_model)

        Returns
        -------
        List[Array]
            Updated values_per_model with pilot values inserted.
            Shape: (nqoi, nsamples_for_model + npilot_samples)
        """
        # nsamples is shape[1] for (nqoi, nsamples) convention
        npilot_values = pilot_values[0].shape[1]
        if (
            self._partitions_per_model[0] * self._rounded_npartition_samples
        ).max() < npilot_values:
            msg = "Insert pilot samples currently only supported when only"
            msg += " the largest subset of those containing the "
            msg += "high-fidelity model can fit all pilot samples"
            raise ValueError(msg)

        new_values_per_model = [self._bkd.copy(v) for v in values_per_model]
        active_hf_subsets = self._bkd.where(
            self._partitions_per_model[0] == 1
        )[0]
        partition_id = active_hf_subsets[
            self._bkd.argmax(
                self._rounded_npartition_samples[active_hf_subsets]
            )
        ]
        for model_id in self._subsets[partition_id]:
            npilot_values = pilot_values[model_id].shape[1]
            if npilot_values != pilot_values[0].shape[1]:
                msg = "Must have the same number of pilot values "
                msg += "for each model"
                raise ValueError(msg)
            if npilot_values > self._rounded_npartition_samples[partition_id]:
                raise ValueError(
                    "Too many pilot values {0}>{1}".format(
                        npilot_values + values_per_model[model_id].shape[1],
                        self._rounded_npartition_samples[partition_id],
                    )
                )
            lb, ub = self._opt_sample_splits[model_id][partition_id]
            # Pilot samples become first samples of the chosen partition
            # For (nqoi, nsamples) convention, concatenate along columns
            new_values_per_model[model_id] = self._bkd.hstack(
                (
                    values_per_model[model_id][:, :lb],
                    pilot_values[model_id],
                    values_per_model[model_id][:, lb:],
                )
            )
        return new_values_per_model

    def __repr__(self):
        if self._optimized_criteria is None:
            return "{0}()".format(self.__class__.__name__)
        rep = "{0}(criteria={1:.3g}".format(
            self.__class__.__name__, self._optimized_criteria
        )
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._rounded_target_cost, self._rounded_nsamples_per_model
        )
        return rep
