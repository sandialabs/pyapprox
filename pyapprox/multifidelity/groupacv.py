from abc import abstractmethod
from itertools import combinations
from typing import List, Union

import numpy as np

from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.multifidelity.stats import (
    MultiOutputMean,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
    MultiOutputStatistic,
)
from pyapprox.interface.model import Model
from pyapprox.optimization.scipy import (
    ScipyConstrainedOptimizer,
    ScipyConstrainedDifferentialEvolutionOptimizer,
)
from pyapprox.optimization.minimize import (
    Constraint,
    ConstrainedOptimizer,
    OptimizationResult,
    ChainedOptimizer,
    Optimizer,
)


def get_model_subsets(
    nmodels: int, bkd: BackendMixin, max_subset_nmodels: int = None
):
    """
    Parameters
    ----------
    nmodels : integer
        The number of models

    max_subset_nmodels : integer
        The maximum number of in a subset.
    """
    if max_subset_nmodels is None:
        max_subset_nmodels = nmodels
    assert max_subset_nmodels > 0
    assert max_subset_nmodels <= nmodels
    subsets = []
    model_indices = bkd.arange(nmodels)
    for nsubset_lfmodels in range(1, max_subset_nmodels + 1):
        for subset_indices in combinations(model_indices, nsubset_lfmodels):
            idx = bkd.asarray(subset_indices, dtype=int)
            subsets.append(idx)
    return subsets


def _get_allocation_matrix_is(subsets: Array, bkd: BackendMixin):
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = bkd.full(
        (nsubsets, npartitions), 0.0, dtype=bkd.double_type()
    )
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, ii] = 1.0
    return allocation_mat


def _get_allocation_matrix_nested(subsets: Array, bkd: BackendMixin):
    # nest partitions according to order of subsets
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = bkd.full(
        (nsubsets, npartitions), 0.0, dtype=bkd.double_type()
    )
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, : ii + 1] = 1.0
    return allocation_mat


def _nest_subsets(subsets: Array, nmodels: int, bkd: BackendMixin):
    for subset in subsets:
        if np.allclose(subset, [0]):
            raise ValueError("Cannot use subset [0]")
    idx = sorted(
        list(range(len(subsets))),
        key=lambda ii: (len(subsets[ii]), tuple(nmodels - subsets[ii])),
        reverse=True,
    )
    return [subsets[ii] for ii in idx], bkd.array(idx)


def _grouped_acv_sigma_block(
    subset0: Array,
    subset1: Array,
    nsamples_intersect: int,
    nsamples_subset0: int,
    nsamples_subset1: int,
    stat,
):
    nsubset0 = len(subset0)
    nsubset1 = len(subset1)
    zero_block = stat._bkd.full((nsubset0, nsubset1), 0.0)
    if (nsamples_subset0 * nsamples_subset1) == 0:
        return zero_block
    if (
        nsamples_subset0 < stat.min_nsamples()
        or nsamples_subset1 < stat.min_nsamples()
    ):
        return zero_block
    block = stat._group_acv_sigma_block(
        subset0,
        subset1,
        nsamples_intersect,
        nsamples_subset0,
        nsamples_subset1,
    )
    return block


def _grouped_acv_sigma(
    nmodels: int, nsamples_intersect: int, subsets: Array, stat
):
    nsubsets = len(subsets)
    Sigma = [[None for jj in range(nsubsets)] for ii in range(nsubsets)]
    for ii, subset0 in enumerate(subsets):
        N_ii = nsamples_intersect[ii, ii]
        Sigma[ii][ii] = _grouped_acv_sigma_block(
            subset0, subset0, N_ii, N_ii, N_ii, stat
        )
        for jj, subset1 in enumerate(subsets[:ii]):
            N_jj = nsamples_intersect[jj, jj]
            Sigma[ii][jj] = _grouped_acv_sigma_block(
                subset0, subset1, nsamples_intersect[ii, jj], N_ii, N_jj, stat
            )
            Sigma[jj][ii] = Sigma[ii][jj].T
    return Sigma


class GroupACVObjective(Model):
    def __init__(self):
        super().__init__()
        self._est = None
        self._bkd = None

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._est.npartitions()

    def set_estimator(self, estimator):
        self._est = estimator
        self._bkd = self._est._bkd

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def hessian_implemented(self) -> bool:
        return self._bkd.hessian_implemented()

    @abstractmethod
    def _objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        raise NotImplementedError

    def _values(self, npartition_samples: Array) -> Array:
        return self._objective_wrapper(npartition_samples[:, 0])

    def _jacobian(self, npartition_samples: Array) -> Array:
        return self._bkd.grad(
            self._objective_wrapper, npartition_samples[:, 0]
        )[1][None, ...]

    def _hessian(self, npartition_samples: Array) -> Array:
        return self._bkd.hessian(
            self._objective_wrapper, npartition_samples[:, 0]
        )[None, ...]


class GroupACVTraceObjective(GroupACVObjective):
    def _objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        trace = self._bkd.trace(
            self._est._covariance_from_npartition_samples(
                npartition_samples_1d
            )
        )
        # conversion below is necessary for torch
        return self._bkd.hstack((trace,))[:, None]


class GroupACVLogDetObjective(GroupACVObjective):
    def _objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        cov = self._est._covariance_from_npartition_samples(
            npartition_samples_1d
        )
        sign, logdet = self._bkd.slogdet(cov)
        if logdet < -1e16:
            # when cov is singular logdet returns np.inf
            # make sure to return positive value to indicate
            # to minimizer this is a bad point. Only really is
            # an issue if starting from poor initial guess or using
            # global optimizer
            return self._bkd.asarray([[np.inf]])
        # conversion below is necessary for torch
        return self._bkd.hstack((logdet,))[:, None]


class MLBLUEObjective(GroupACVTraceObjective):
    def _jacobian(self, npartition_samples: Array):
        # apply derivative of inverse matrix
        # d_m X^{-1} = X^{-1} (d_mX) X^{-1}
        # where X = psi_matrix and d_mX is RC_mR.T (not multiplied by nsamples)
        # Objective is e^T X e so
        # grad is e^T X^{-1} d_mX X^{-1} e = \gamma^T(d_mX)\gamma
        # compute sigma blocks with npartition_samples = 1
        Sigma_blocks = _grouped_acv_sigma(
            self._est.nmodels(),
            self._bkd.eye(npartition_samples.shape[0]),
            # self._est._nintersect_samples(npartition_samples[:, 0]),
            self._est._subsets,
            self._est._stat,
        )
        # compute psi matrix with partition sizes
        # todo cache psi_matrix when it is computed when evaluatin objective
        psi_matrix = self._est._psi_matrix(npartition_samples[:, 0])
        psi_inv = self._est._inv(psi_matrix)
        Rmats = self._est._restriction_matrices
        jacobian = 0
        for kk in range(self._est._stat.nstats()):
            gamma = psi_inv @ self._est._asketch[kk : kk + 1].T
            jacobian += self._bkd.hstack(
                [
                    self._bkd.multidot(
                        (
                            -gamma.T,
                            Rmats[ii],
                            self._est._inv(Sigma_blocks[ii][ii]),
                            Rmats[ii].T,
                            gamma,
                        )
                    )
                    for ii in range(len(Sigma_blocks))
                ]
            )
        return jacobian

    def _hessian(self, npartition_samples: Array):
        # apply derivative of inverse matrix twice
        # d_m X^{-1} = X^{-1} (d_mX) X^{-1}
        # where X = psi_matrix and d_mX is RC_mR.T (not multiplied by nsamples)
        # Applying twice we have
        # d_mn X^{-1} = d_n(X^{-1} (d_mX) X^{-1})
        # = X^{-1}(d_nX)X^{-1}d_mX^{-1} + X^{-1}(d_mX)X^{-1}d_nX^{-1}
        # So hessian is
        # = \gamma^T(d_nX)X^{-1}d_m\gamma + \gamma^T(d_mX)X^{-1}d_nX^{-1}/gamma
        # = \gamma^T(d_nX)\xi + \xi^Td_nX^{-1}/gamma
        # = \eta^T + \eta, \eta = \xi^Td_nX^{-1}/gamma

        Sigma_blocks = _grouped_acv_sigma(
            self._est.nmodels(),
            self._bkd.eye(npartition_samples.shape[0]),
            self._est._subsets,
            self._est._stat,
        )
        psi_matrix = self._est._psi_matrix(npartition_samples[:, 0])
        psi_inv = self._est._inv(psi_matrix)
        Rmats = self._est._restriction_matrices
        hess = [
            [0 for jj in range(len(Sigma_blocks))]
            for ii in range(len(Sigma_blocks))
        ]
        sigma_invs = [
            self._est._inv(Sigma_blocks[ii][ii])
            for ii in range(len(Sigma_blocks))
        ]
        psi_derivs = [
            self._bkd.multidot((Rmats[ii], sigma_invs[ii], Rmats[ii].T))
            for ii in range(len(Sigma_blocks))
        ]
        for kk in range(self._est._stat.nstats()):
            gamma = psi_inv @ self._est._asketch[kk : kk + 1].T
            for ii in range(len(Sigma_blocks)):
                xi = self._bkd.multidot((psi_inv, psi_derivs[ii], gamma))
                for jj in range(ii, len(Sigma_blocks)):
                    eta = self._bkd.multidot((xi.T, psi_derivs[jj], gamma))
                    hess[ii][jj] += eta.T + eta
                    hess[jj][ii] = hess[ii][jj]
        hess = self._bkd.vstack([self._bkd.hstack(row) for row in hess])[
            None, ...
        ]
        return hess


class GroupACVConstraint(Constraint):
    def __init__(self, bounds, keep_feasible=True):
        super().__init__(bounds, keep_feasible)
        self._est = None

    def set_estimator(self, estimator):
        self._est = estimator
        self._bkd = self._est._bkd


class GroupACVCostContstraint(GroupACVConstraint):
    def __init__(self, bounds: Array, keep_feasible: bool = True):
        if bounds.shape[0] != self.nqoi():
            # the number of columns is checked with call to super().__init__
            raise ValueError("Bounds must have shape (2, 2)")
        super().__init__(bounds, keep_feasible)
        self._target_cost = None
        self._min_nhf_samples = None

    def jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def nvars(self) -> int:
        return self._est.npartitions()

    def set_budget(self, target_cost: float, min_nhf_samples: int):
        self._target_cost = target_cost
        self._min_nhf_samples = min_nhf_samples
        self._validate_target_cost_min_nhf_samples()

    def _validate_target_cost_min_nhf_samples(self):
        lb = self._min_nhf_samples * self._est._costs[0]
        ub = self._target_cost
        if ub < lb:
            msg = "target_cost {0} & cost of min_nhf_samples {1} ".format(
                self._target_cost, lb
            )
            msg += "are inconsistent"
            raise ValueError(msg)

    def nqoi(self) -> int:
        return 2

    def _values(self, npartition_samples: Array) -> Array:
        return self._bkd.array(
            [
                self._target_cost
                - self._est._estimator_cost(npartition_samples[:, 0]),
                self._bkd.sum(
                    self._est._partitions_per_model[0]
                    * npartition_samples[:, 0]
                )
                - self._min_nhf_samples,
            ]
        )[None, :]

    def _jacobian(self, npartition_samples: Array) -> Array:
        return self._bkd.vstack(
            (
                -(self._est._costs[None, :] @ self._est._partitions_per_model),
                self._est._partitions_per_model[0][None, :],
            )
        )

    def _hessian(self, npartition_samples: Array) -> Array:
        return self._bkd.zeros(
            (
                self.nqoi(),
                npartition_samples.shape[0],
                npartition_samples.shape[0],
            )
        )


class GroupACVOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self._target_cost = None
        self._min_nhf_samples = None
        self._est = None

    def set_budget(self, target_cost: float, min_nhf_samples: int = 1):
        self._target_cost = target_cost
        self._min_nhf_samples = min_nhf_samples

    def set_estimator(self, est: "GroupACVEstimator"):
        self._est = est
        self._bkd = self._est._bkd

    @abstractmethod
    def _minimize(self, iterate: Array) -> OptimizationResult:
        raise NotImplementedError

    def minimize(self, iterate: Array) -> OptimizationResult:
        result = self._minimize(iterate)
        if not isinstance(result, OptimizationResult):
            raise RuntimeError(
                "{0}.minimize did not return OptimizationResult".format(self)
            )
        return result

    def set_verbosity(self, verbosity: int):
        super().set_verbosity(verbosity)
        self._optimizer._verbosity = verbosity


class GroupACVGradientOptimizer(GroupACVOptimizer):
    def __init__(self, optimizer: ConstrainedOptimizer):
        super().__init__()
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                "optimizer must be an instance of ConstrainedOptimizer "
                f"but was {type(optimizer)}"
            )
        self._optimizer = optimizer

    def _minimize(self, iterate: Array) -> OptimizationResult:
        return self._optimizer.minimize(iterate)

    def set_budget(self, target_cost: float, min_nhf_samples: int = 1):
        if not hasattr(self, "_constraint"):
            raise RuntimeError(
                "must call GroupACVGradientOptimizer.set_estimator()"
            )
        super().set_budget(target_cost, min_nhf_samples)
        self._constraint.set_budget(target_cost, min_nhf_samples)
        max_npartition_samples = (
            self._target_cost // self._est._costs.min() + 1
        )
        for ii in range(self._optimizer._bounds.shape[0]):
            self._optimizer._bounds[ii, 1] = max_npartition_samples

    def get_objective(self) -> GroupACVObjective:
        #    return GroupACVTraceObjective()
        return GroupACVLogDetObjective()

    def set_estimator(self, est: "GroupACVEstimator"):
        super().set_estimator(est)
        objective = self.get_objective()
        objective.set_estimator(self._est)
        self._optimizer.set_objective_function(objective)
        self._constraint = GroupACVCostContstraint(
            bounds=self._bkd.array([[0, np.inf], [0, np.inf]])
        )
        self._constraint.set_estimator(self._est)
        self._optimizer.set_constraints([self._constraint])
        self._optimizer.set_bounds(
            self._bkd.reshape(
                self._bkd.tile(
                    self._bkd.array([0, np.inf]), (self._est.npartitions(),)
                ),
                (self._est.npartitions(), 2),
            )
        )
        self._objective = self._optimizer._objective


class MLBLUEGradientOptimizer(GroupACVGradientOptimizer):
    def get_objective(self):
        return MLBLUEObjective()


class MLBLUESPDOptimizer(GroupACVOptimizer):
    def __init__(self, solver_name: str = "CVXOPT"):
        try:
            import cvxpy
        except ImportError:
            raise ValueError(
                "MLBLUESPDOptimizer can only be used when optional dependency"
                " cvxpy is installed"
            )
        self._cvxpy = cvxpy

        super().__init__()
        self._min_nlf_samples = None
        self._solver_name = solver_name

    def _cvxpy_psi(self, nsps_cvxpy):
        Psi = self._est._psi_blocks_flat @ nsps_cvxpy
        Psi = self._cvxpy.reshape(
            Psi, (self._est.nmodels(), self._est.nmodels())
        )
        return Psi

    def _cvxpy_spd_constraint(self, nsps_cvxpy, t_cvxpy):
        Psi = self._cvxpy_psi(nsps_cvxpy)
        mat = self._cvxpy.bmat(
            [
                [Psi, self._est._asketch.T],
                [self._est._asketch, self._cvxpy.reshape(t_cvxpy, (1, 1))],
            ]
        )
        return mat

    def _init_guess(self):
        return None

    def _minimize(self, iterate):
        if self._est._stat.nstats() != 1:
            raise RuntimeError("SPD solver only works for single outputs")
        # if iterate is not None:
        #     raise ValueError("iterate must be None")
        t_cvxpy = self._cvxpy.Variable(nonneg=True)
        nsps_cvxpy = self._cvxpy.Variable(self._est.nsubsets(), nonneg=True)
        obj = self._cvxpy.Minimize(t_cvxpy)
        subset_costs = self._est._get_model_subset_costs(
            self._est._subsets, self._est._costs
        )
        constraints = [subset_costs @ nsps_cvxpy <= self._target_cost]
        constraints += [
            self._est._partitions_per_model[0] @ nsps_cvxpy
            >= self._min_nhf_samples
        ]
        if self._min_nlf_samples is not None:
            constraints += [
                self._est._partitions_per_model[ii + 1] @ nsps_cvxpy
                >= self._min_nlf_samples[ii]
                for ii in range(self.nmodels() - 1)
            ]
        constraints += [self._cvxpy_spd_constraint(nsps_cvxpy, t_cvxpy) >> 0]
        prob = self._cvxpy.Problem(obj, constraints)
        # prob.solve(verbose=0, solver=self._solver_name)
        prob.solve(solver=self._solver_name)
        if t_cvxpy.value is None:
            raise RuntimeError("solver did not converge")
        result = OptimizationResult(
            {
                "x": self._bkd.array(nsps_cvxpy.value)[:, None],
                "fun": t_cvxpy.value,
                "success": True,
            }
        )
        return result


class ChainedACVOptimizer(ChainedOptimizer):
    def __init__(self, optimizer1, optimizer2):
        if not isinstance(optimizer1, GroupACVOptimizer):
            raise ValueError(
                "optimizer1 must be an instance of GroupACVOptimizer"
            )
        if not isinstance(optimizer2, GroupACVOptimizer):
            raise ValueError(
                "optimizer2 must be an instance of GroupACVOptimizer"
            )
        super().__init__(optimizer1, optimizer2)

    def set_budget(self, target_cost, min_nhf_samples):
        self._optimizer1.set_budget(target_cost, min_nhf_samples)
        self._optimizer2.set_budget(target_cost, min_nhf_samples)

    def set_estimator(self, est: "GroupACVEstimator"):
        self._optimizer1.set_estimator(est)
        self._optimizer2.set_estimator(est)
        self._objective = self._optimizer2._objective


# TODO to enable multioutput I changed to require asketch has a row for each output
# this messesd up etc code. Consider requiring asketch to have a column for each
# output. Then need to change transpose on asketch in this file.
class GroupACVEstimator:
    def __init__(
        self,
        stat: MultiOutputStatistic,
        costs: Array,
        reg_blue: float = 0,
        model_subsets: List[Array] = None,
        est_type: str = "is",
        asketch: Array = None,
        use_pseudo_inv: bool = True,
    ):
        self._bkd = stat._bkd
        self._use_pseudo_inv = use_pseudo_inv
        # self._cov, self._costs = self._check_cov(stat._cov, costs)
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
        self._objective = None

    def nsubsets(self) -> int:
        return len(self._subsets)

    def npartitions(self) -> int:
        return self._npartitions

    def nmodels(self) -> int:
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
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.get_default_optimizer())
            # raise RuntimeError("must call est.set_optimizer()")
        self._optimized_criteria = self._optimizer._objective(
            self._rounded_npartition_samples[:, None]
        )

    def optimized_covariance(self) -> Array:
        return self._optimized_covariance

    def _set_optimized_params(self, npartition_samples, round_nsamples=True):
        # expected scalar type Double but found Float error can occur
        # with torch if npartition samples is not torch.double need to
        # think of a check that works for all backends
        if round_nsamples:
            # add 1e-15 to avoid rounding down value that is at the constraint
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
                (self._stat.nstats(), self._stat.nstats() * self.nmodels()), 0
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

    def get_default_optimizer(self) -> ChainedACVOptimizer:
        opt1 = GroupACVGradientOptimizer(
            ScipyConstrainedDifferentialEvolutionOptimizer(
                opts={"maxiter": 20}
            )
        )
        local_opt = ScipyConstrainedOptimizer()
        local_opt._opts["gtol"] = 1e-8
        opt2 = GroupACVGradientOptimizer(local_opt)

        return ChainedACVOptimizer(opt1, opt2)

    def set_optimizer(
        self, optimizer: Union[GroupACVOptimizer, ChainedACVOptimizer]
    ):
        if not isinstance(optimizer, GroupACVOptimizer) and not isinstance(
            optimizer, ChainedACVOptimizer
        ):
            raise ValueError(
                "optimizer must be instance of GroupACVOptimizer"
                "or ChainedACVOptimizer"
            )
        self._optimizer = optimizer
        self._optimizer.set_estimator(self)

    def allocate_samples(
        self,
        target_cost: float,
        min_nhf_samples: int = 1,
        round_nsamples: bool = True,
        iterate: Array = None,
    ):
        """
        Optimize the sample allocation.

        Parameters
        ----------
        target_cost : float
            The computational budget used to compute the estimator.

        min_nhf_samples : float
            The minimum number of high-fidelity samples before rounding.
            Unforunately, there is no way to enforce that the min_nhf_samples
            is met after rounding. As the differentiable constraint
            enforces that the sum of the nsamples in each partition involving
            the high-fidelity model is zero. But when each partition nsample
            is rounded the rounded nhf_samples may be less than desired. It
            will be close though

        round_nsamples: bool
            True - round the optimal number of samples allocated to each model
                   down to the nearest integer
            False - do not round

        iterate: Array
            The initial guess of the optimal sample allocation passed to the
            optimizer. If None a default initial iterate is used.
        """
        if not hasattr(self, "_optimizer"):
            # raise RuntimeError("must call set_optimizer")
            self.set_optimizer(self.get_default_optimizer())
        self._optimizer.set_budget(
            target_cost, max(self._stat.min_nsamples(), min_nhf_samples)
        )
        if iterate is None:
            iterate = self._init_guess(target_cost)
        result = self._optimizer.minimize(iterate)

        if not result.success or self._bkd.any(result.x < 0):
            print(result)
            print(result.message)
            raise RuntimeError("optimization not successful")

        self._set_optimized_params(result.x[:, 0], round_nsamples)

    def _get_partition_splits(self, npartition_samples):
        """
        Get the indices, into the flattened array of all samples/values,
        of each indpendent sample partition
        """
        splits = self._bkd.hstack(
            (
                self._bkd.zeros((1,), dtype=int),
                self._bkd.cumsum(npartition_samples, dtype=int),
            )
        )
        return splits

    def generate_samples_per_model(self, rvs, npilot_samples=0):
        ntotal_independent_samples = self._rounded_npartition_samples.sum()
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
        if len(values_per_model) != self.nmodels():
            msg = "len(values_per_model) {0} != nmodels {1}".format(
                len(values_per_model), self.nmodels()
            )
            raise ValueError(msg)
        for ii in range(self.nmodels()):
            if (
                values_per_model[ii].shape[0]
                != self._rounded_nsamples_per_model[ii]
            ):
                msg = "{0} != {1}".format(
                    "len(values_per_model[{0}]): {1}".format(
                        ii, values_per_model[ii].shape[0]
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
                values.append(
                    self._bkd.vstack(
                        [
                            values_per_model[model_id][
                                splits[idx, 0] : splits[idx, 1], :
                            ]
                            for idx in active_partitions
                        ]
                    )
                )
            values_per_subset.append(self._bkd.hstack(values))
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
            if values_per_subset[kk].shape[0] > 0:
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
            print("\n", subset)
            # print(beta)
            print(subset_ests[ii])
            print(Q_tilde, "QT")
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
        print(subset_ests, "SE")
        # Implement equations (15) in arxiv paper
        # wu = w_l^{k,u} and  we = w_l^{k,e} from arxiv paper
        alpha = self._traditional_acv_weights()
        return self._group_to_traditional_estimators_from_alpha(
            subset_ests, alpha
        )

    def __call__(self, values_per_model: List[Array]) -> Array:
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
        npilot_values = pilot_values[0].shape[0]
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
            npilot_values = pilot_values[model_id].shape[0]
            if npilot_values != pilot_values[0].shape[0]:
                msg = "Must have the same number of pilot values "
                msg += "for each model"
                raise ValueError(msg)
            if npilot_values > self._rounded_npartition_samples[partition_id]:
                raise ValueError(
                    "Too many pilot values {0}>{1}".format(
                        npilot_values + values_per_model[model_id].shape[0],
                        self._rounded_npartition_samples[partition_id],
                    )
                )
            lb, ub = self._opt_sample_splits[model_id][partition_id]
            # Pilot samples become first samples of the chosen partition
            new_values_per_model[model_id] = self._bkd.vstack(
                (
                    values_per_model[model_id][:lb],
                    pilot_values[model_id],
                    values_per_model[model_id][lb:],
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


class MLBLUEEstimator(GroupACVEstimator):
    def __init__(
        self,
        stat: MultiOutputStatistic,
        costs: Array,
        reg_blue: float = 0,
        subsets: List[Array] = None,
        asketch: Array = None,
    ):
        # Currently stats is ignored.
        super().__init__(
            stat,
            costs,
            reg_blue,
            subsets,
            est_type="is",
            asketch=asketch,
        )
        self._best_model_indices = self._bkd.arange(len(costs))

        # compute psi blocks once and store because they are independent
        # of the number of samples per partition/subset
        self._psi_blocks = self._compute_psi_blocks()
        self._psi_blocks_flat = self._bkd.hstack(
            [b.flatten()[:, None] for b in self._psi_blocks]
        )

        self._obj_jac = True

    def _compute_psi_blocks(self):
        submats = []
        for ii, subset in enumerate(self._subsets):
            R = self._restriction_matrix(subset)
            submat = self._bkd.multidot(
                (
                    R.T,
                    self._inv(self._stat._cov[np.ix_(subset, subset)]),
                    R,
                )
            )
            submats.append(submat)
        return submats

    def _psi_matrix(self, npartition_samples):
        psi = (
            self._bkd.eye(self.nmodels() * self._stat.nstats())
            * self._reg_blue
        )
        psi += (self._psi_blocks_flat @ npartition_samples).reshape(
            (
                self.nmodels() * self._stat.nstats(),
                self.nmodels() * self._stat.nstats(),
            )
        )
        return psi

    def estimate_all_means(self, values_per_subset):
        asketch = self._bkd.copy(self._asketch)
        means = self._bkd.empty(self.nmodels())
        if self._stat.nstats() > 1:
            raise NotImplementedError(
                "Must adjust this function to work for multiple outputs"
            )
        for ii in range(self.nmodels()):
            self._asketch = self._bkd.full((self.nmodels()), 0.0)
            self._asketch[ii] = 1.0
            means[ii] = self._estimate(values_per_subset)
        self._asketch = asketch
        return means


# cvxpy requires cmake
# on osx with M1 chip install via
# arch -arm64 brew install cmake
# must also install cvxopt via
# pip install cvxopt
# pip install cvxpy
