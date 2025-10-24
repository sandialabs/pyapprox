import unittest

import torch
import numpy as np

from pyapprox.multifidelity.stats import (
    _nqoisq_nqoisq_subproblem,
    _nqoi_nqoisq_subproblem,
)
from pyapprox.multifidelity._optim import (
    _allocate_samples_mfmc,
    _allocate_samples_mlmc,
)
from pyapprox.multifidelity.acv import (
    MFMCEstimator,
    MLMCEstimator,
    ACVLogDeterminantObjective,
    ACVPartitionConstraint,
)
from pyapprox.multifidelity.factory import (
    get_estimator,
    BestEstimator,
    numerically_compute_estimator_variance,
    multioutput_stats,
)
from pyapprox.multifidelity.tests.test_stats import (
    _setup_multioutput_model_subproblem,
)
from pyapprox.multifidelity.stats import (
    _get_nsamples_intersect,
    _get_nsamples_subset,
)
from pyapprox.benchmarks.multifidelity_benchmarks import (
    PolynomialModelEnsembleBenchmark,
)
from pyapprox.util.backends.torch import TorchMixin

# from pyapprox.util.print_wrapper import *


def _log_single_qoi_criteria(qoi_idx, stat_type, criteria_type, variance):
    # use if stat == Variance and target is variance
    # return torch.log(variance[qoi_idx[0], qoi_idx[0]])
    # use if stat == MeanAndVariance and target is variance
    if stat_type == "mean_var" and criteria_type == "var":
        return torch.log(variance[3 + qoi_idx[0], 3 + qoi_idx[0]])
    if stat_type == "mean_var" and criteria_type == "mean":
        return torch.log(variance[qoi_idx[0], qoi_idx[0]])
    raise ValueError


class TestMOMC:
    def setUp(self):
        np.random.seed(1)

    def test_generalized_recursive_difference_allocation_matrices(self):
        bkd = self.get_backend()
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[2, 0])
        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[0, 1])
        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[0, 0])
        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        cov = bkd.array(np.random.normal(0, 1, (4, 4)))
        costs = bkd.ones(4)
        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[0, 1, 2])
        npartition_samples = torch.as_tensor([2, 2, 4, 4], dtype=torch.double)
        nsamples_intersect = _get_nsamples_intersect(
            est._allocation_mat, npartition_samples, bkd
        )

        nsamples_interesect_true = bkd.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
            ]
        )
        assert bkd.allclose(nsamples_intersect, nsamples_interesect_true)
        nsamples_subset = _get_nsamples_subset(
            est._allocation_mat, npartition_samples, bkd
        )
        assert bkd.allclose(
            nsamples_subset, bkd.array([0.0, 2, 2, 2, 2, 4, 4, 4])
        )

    def test_generalized_multifidelity_allocation_matrices(self):
        bkd = self.get_backend()
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )
        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("gmf", stat, costs, recursion_index=[2, 0])

        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("gmf", stat, costs, recursion_index=[0, 1])
        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("gmf", stat, costs, recursion_index=[0, 0])
        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_generalized_independent_samples_allocation_matrices(self):
        bkd = self.get_backend()
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("gis", stat, costs, recursion_index=[2, 0])
        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ]
            ),
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("gis", stat, costs, recursion_index=[0, 1])
        assert np.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator("gis", stat, costs, recursion_index=[0, 0])
        assert bkd.allclose(
            est._allocation_mat,
            bkd.array(
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    def _check_estimator_variances(
        self,
        model_idx,
        qoi_idx,
        recursion_index,
        est_type,
        stat_type,
        tree_depth=None,
        max_nmodels=None,
        ntrials=int(1e4),
        target_cost=50,
    ):
        bkd = self.get_backend()
        ntrials = int(ntrials)
        rtol, atol = 4.6e-2, 1.01e-3
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )

        # change costs so less samples are used in the estimator
        # to speed up test
        costs = bkd.array([2, 1.5, 1])[model_idx]

        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        pilot_args = [cov]
        if est_type == "gmf" or est_type == "grd" or est_type == "gis":
            if tree_depth is not None:
                kwargs = {"tree_depth": tree_depth}
            else:
                kwargs = {"recursion_index": bkd.asarray(recursion_index)}
        else:
            kwargs = {}

        if stat_type == "mean":
            # idx = nqoi
            if est_type == "cv":
                kwargs["lowfi_stats"] = bkd.stack(
                    [m for m in means[1:]], axis=0
                )
        if "variance" in stat_type:
            if est_type == "cv":
                lfcovs = []
                lb, ub = 0, 0
                for ii in range(1, nmodels):
                    ub += nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                    lb = ub
                tril_idx = bkd.tril_indices(lfcovs[0].shape[0])
                kwargs["lowfi_stats"] = bkd.stack(
                    [cov[tril_idx[0], tril_idx[1]].flatten() for cov in lfcovs]
                )
            W = benchmark.covariance_of_centered_values_kronker_product()
            W = _nqoisq_nqoisq_subproblem(
                W,
                benchmark.nmodels(),
                benchmark.nqoi(),
                model_idx,
                qoi_idx,
                bkd,
            )
            # npilot_samples = int(1e6)
            # pilot_samples = benchmark.prior().rvs(npilot_samples)
            # pilot_values = bkd.hstack([f(pilot_samples) for f in funs])
            # W = get_W_from_pilot(pilot_values, nmodels)
            pilot_args.append(W)
            # idx = nqoi**2
        if stat_type == "mean_variance":
            if est_type == "cv":
                lfcovs = []
                lb, ub = 0, 0
                for ii in range(1, nmodels):
                    ub += nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                    lb = ub
                tril_idx = bkd.tril_indices(lfcovs[0].shape[0])
                kwargs["lowfi_stats"] = bkd.stack(
                    [
                        bkd.hstack(
                            (m, cov[tril_idx[0], tril_idx[1]].flatten())
                        )
                        for m, cov in zip(means[1:], lfcovs)
                    ],
                    axis=0,
                )
            B = benchmark.covariance_of_mean_and_variance_estimators()
            B = _nqoi_nqoisq_subproblem(
                B,
                benchmark.nmodels(),
                benchmark.nqoi(),
                model_idx,
                qoi_idx,
                bkd,
            )
            pilot_args.append(B)
            # idx = nqoi + nqoi**2

        # check covariance matrix is positive definite
        # bkd.linalg.cholesky(cov)
        stat = multioutput_stats[stat_type](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(*pilot_args)
        idx = stat.nstats()
        est = get_estimator(
            est_type, stat, costs, max_nmodels=max_nmodels, **kwargs
        )
        if hasattr(est, "get_default_optimizer"):
            optimizer = est.get_default_optimizer()
            optimizer.set_verbosity(0)
            optimizer._optimizer1._opts["maxiter"] = 500
            optimizer._optimizer2._opts["method"] = "slsqp"
            est.set_optimizer(optimizer)

        est.allocate_samples(target_cost)
        print(est._optimized_covariance)

        max_eval_concurrency = 1
        if isinstance(est, BestEstimator):
            funs_subset = [funs[idx] for idx in est._best_model_indices]
        else:
            funs_subset = funs
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs_subset,
                benchmark.prior(),
                est,
                ntrials,
                max_eval_concurrency,
                True,
            )
        )
        hfcovar = hfcovar.numpy()

        # np.set_printoptions(linewidth=1000)
        # print(hfcovar_mc)
        # print(hfcovar)
        # print((np.abs(hfcovar_mc-hfcovar)-(atol+rtol*np.abs(hfcovar))).min())
        # assert np.allclose(hfcovar_mc, hfcovar, atol=atol, rtol=rtol)
        if est_type != "mc":
            CF_mc = bkd.cov(delta.T, ddof=1)
            cf_mc = bkd.cov(bkd.vstack((Q.T, delta.T)), ddof=1)[:idx, idx:]
            CF, cf = est._get_discrepancy_covariances(
                est._rounded_npartition_samples
            )
            print(CF.shape, CF_mc.shape)
            print(cf.shape, cf_mc.shape)
            # print(CF)
            # print(CF_mc)
            # print((np.abs(CF_mc-CF)-(atol + rtol*np.abs(CF))).max())
            # torch.set_printoptions(linewidth=1000)
            # print(cf - cf_mc)
            # print((np.abs(cf_mc - cf) - (atol + rtol * np.abs(cf_mc))).max())
            assert bkd.allclose(CF, CF_mc, atol=atol, rtol=rtol)
            assert bkd.allclose(cf, cf_mc, atol=atol, rtol=rtol)

        # print(covar_mc, 'v_mc')
        # print(covar, 'v')
        # print((covar_mc-covar)/covar)
        assert bkd.allclose(covar_mc, covar, atol=atol, rtol=rtol)

    def test_estimator_variances(self):
        test_cases = [
            [[0], [0, 1, 2], None, "mc", "mean"],
            [[0], [0, 1, 2], None, "mc", "variance"],
            [[0], [0, 1], None, "mc", "mean_variance"],
            [[0, 1, 2], [0, 1], None, "cv", "mean"],
            [[0, 1, 2], [0, 1], None, "cv", "variance"],
            [[0, 1, 2], [0, 2], None, "cv", "mean_variance", None, None, 5e4],
            [[0, 1, 2], [0], [0, 1], "grd", "mean"],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "mean"],
            [[0, 1, 2], [0, 1], [0, 0], "grd", "mean"],
            [[0, 1], [0, 1, 2], [0], "grd", "variance", None, None, 2e4],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "variance"],
            [
                [0, 1, 2],
                [0, 1],
                [0, 1],
                "grd",
                "mean_variance",
                None,
                None,
                5e4,
            ],
            [[0, 1, 2], [0], [0, 1], "gis", "mean"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean", None, None, 1e4, 10],
            [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", 2, None, 5e4, 100],
            [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", None, 3, 1e4, 100],
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1],
                ["gmf", "grd", "gis", "mlmc"],
                "mean",
                None,
                3,
                1e4,
                100,
            ],
            [[0, 1, 2], [0, 1, 2], [0, 1], "grd", "mean", None, 3],
            [[0, 1, 2], [1], [0, 1], "grd", "variance", None, 3],
            [[0, 1], [0, 2], [0], "gmf", "mean"],
            [[0, 1], [0], [0], "gmf", "variance"],
            [[0, 1], [0, 2], [0], "gmf", "variance"],
            [[0, 1, 2], [0], [0, 0], "gmf", "variance"],
            [
                [0, 1, 2],
                [0, 2],
                [0, 0],
                "gmf",
                "variance",
                None,
                None,
                1e4,
                100,
            ],
            [[0, 1], [0], [0], "gmf", "mean_variance"],
            [[0, 1, 2], [0], None, "mfmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "mean", None, None, 5e4],
            [[0, 1, 2], [0, 1], None, "mlmc", "mean", None, None, 5e4],
            [[0, 1, 2], [0], None, "mlmc", "variance"],
            [
                [0, 1, 2],
                [0],
                None,
                "mlmc",
                "mean_variance",
                None,
                None,
                1e4,
                100,
            ],
            [[0], [0, 1, 2], None, "mc", "variance"],
            # [[0, 1, 2], [0, 2], None, "gmf", "variance", 2, None, int(7e4)],
            [
                [0, 1, 2],
                [0, 1],
                [0, 1],
                "gmf",
                "variance",
                None,
                None,
                int(7e4),
            ],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_estimator_variances(*test_case)

    def test_numerical_mfmc_sample_optimization(self):
        # check for scalar output case we require MFMC analytical solution
        bkd = self.get_backend()
        model_idx, qoi_idx = [0, 1, 2], [0]
        recursion_index = [0, 1]
        target_cost = 10.0
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )
        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            "gmf", stat, costs, recursion_index=bkd.asarray(recursion_index)
        )
        mfmc_model_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            cov, costs, target_cost, bkd
        )
        mfmc_est = MFMCEstimator(stat, costs)
        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)
        # print(mfmc_log_variance)

        partition_ratios = mfmc_est._native_ratios_to_npartition_ratios(
            mfmc_model_ratios
        )
        assert bkd.allclose(
            bkd.exp(objective(partition_ratios[:, None])),
            bkd.exp(mfmc_log_variance),
        )
        assert bkd.allclose(
            objective.jacobian(partition_ratios[:, None]),
            bkd.zeros((1, mfmc_model_ratios.shape[0])),
        )
        errors = objective.check_apply_jacobian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 1, 14)),
        )
        assert errors.min() / errors.max() < 3.0e-6
        objective.apply_hessian_implemented = lambda: True
        errors = objective.check_apply_hessian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 1, 14)),
        )
        assert errors.min() / errors.max() < 3.0e-6

        constraint = ACVPartitionConstraint(est, target_cost)
        errors = constraint.check_apply_jacobian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 1, 14)),
        )
        assert errors.min() / errors.max() < 1.0e-6
        constraint.weighted_hessian_implemented = lambda: True
        errors = constraint.check_apply_hessian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 1, 14)),
            weights=bkd.ones((partition_ratios.shape[0] + 1, 1)),
        )
        assert errors.min() / errors.max() < 1.0e-6

        # test mapping from partition ratios to model ratios
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples
        )
        est_cost = (nsamples_per_model * est._costs.numpy()).sum()
        assert bkd.allclose(
            nsamples_per_model,
            bkd.hstack(
                (nsamples_per_model[0], model_ratios * npartition_samples[0])
            ),
        )
        assert bkd.allclose(model_ratios, mfmc_model_ratios)
        assert bkd.allclose(
            model_ratios * npartition_samples[0],
            bkd.cumsum(npartition_samples)[1:],
        )
        assert bkd.allclose(bkd.asarray(target_cost), est_cost)
        # get nsample ratios before rounding
        # avoid using est._allocate_samples so we do not start
        # from mfmc exact solution
        partition_ratios, obj_val = est._allocate_samples(target_cost)
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples
        )
        est_cost = (nsamples_per_model * est._costs.numpy()).sum()
        assert bkd.allclose(
            bkd.exp(bkd.asarray(obj_val)), bkd.exp(mfmc_log_variance)
        )
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        torch.set_printoptions(precision=16)
        # print(model_ratios, mfmc_model_ratios)
        assert bkd.allclose(model_ratios, mfmc_model_ratios)

    def test_numerical_mlmc_sample_optimization(self):
        bkd = self.get_backend()
        # check for scalar output case we require MFMC analytical solution
        model_idx, qoi_idx = [0, 1, 2], [0]
        recursion_index = [0, 1]
        target_cost = 10
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )

        # The following will give mlmc with unit variance
        # and level variances var[f_i-f_{i+1}] = [1, 4, 4]
        target_cost = 81.0
        costs = bkd.array([6.0, 3.0, 1.0])
        cov = bkd.asarray(
            [[1.00, 0.50, 0.25], [0.50, 1.00, 0.50], [0.25, 0.50, 4.00]]
        )

        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            "grd", stat, costs, recursion_index=bkd.asarray(recursion_index)
        )
        mlmc_model_ratios, mlmc_log_variance = _allocate_samples_mlmc(
            cov, costs, target_cost, bkd
        )

        # We are trying to test numerical GRD optimization, but to recover
        # MLMC solution we need to use suboptimal weights of MLMC so adjust
        # the necessary functions here
        mlmc_est = MLMCEstimator(stat, costs)
        est._weights = mlmc_est._weights

        def mlmc_cov(npartition_samples):
            CF, cf = est._get_discrepancy_covariances(npartition_samples)
            weights = est._weights(CF, cf)
            return est._covariance_non_optimal_weights(
                est._stat.high_fidelity_estimator_covariance(
                    npartition_samples[0]
                ),
                weights,
                CF,
                cf,
            )

        est._covariance_from_npartition_samples = mlmc_cov

        # test mapping from partition ratios to model ratios
        partition_ratios = mlmc_est._native_ratios_to_npartition_ratios(
            mlmc_model_ratios
        )
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples
        )
        est_cost = (nsamples_per_model * est._costs).sum()
        assert bkd.allclose(
            nsamples_per_model,
            bkd.hstack(
                (nsamples_per_model[0], model_ratios * npartition_samples[0])
            ),
        )
        assert bkd.allclose(model_ratios, mlmc_model_ratios)
        assert bkd.allclose(bkd.asarray(target_cost), est_cost)

        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)
        assert bkd.allclose(
            bkd.exp(objective(partition_ratios[:, None])),
            bkd.exp(mlmc_log_variance),
        )
        assert bkd.allclose(
            objective.jacobian(partition_ratios[:, None]),
            bkd.zeros((1, mlmc_model_ratios.shape[0])),
        )

        errors = objective.check_apply_jacobian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 1, 14)),
        )
        assert errors.min() / errors.max() < 3.0e-6
        objective.apply_hessian_implemented = lambda: True
        errors = objective.check_apply_hessian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 0, 13)),
        )
        assert errors.min() / errors.max() < 3.0e-6

        constraint = ACVPartitionConstraint(est, target_cost)
        errors = constraint.check_apply_jacobian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 1, 14)),
        )
        assert errors.min() / errors.max() < 1.0e-6
        constraint.weighted_hessian_implemented = lambda: True
        errors = constraint.check_apply_hessian(
            partition_ratios[:, None] + 1,
            fd_eps=bkd.flip(bkd.logspace(-12, 1, 14)),
            weights=bkd.ones((partition_ratios.shape[0] + 1, 1)),
        )
        assert errors.min() / errors.max() < 1.0e-6

        # get nsample ratios before rounding
        # avoid using est._allocate_samples so we do not start
        # from mlmc exact solution
        partition_ratios, obj_val = est._allocate_samples(target_cost)
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples
        )
        est_cost = (nsamples_per_model * est._costs).sum()
        assert bkd.allclose(
            bkd.exp(bkd.asarray(obj_val)), bkd.exp(mlmc_log_variance)
        )
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        assert bkd.allclose(model_ratios, mlmc_model_ratios)

    def test_best_model_subset_estimator(self):
        bkd = self.get_backend()
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem([0, 1, 2], [0, 1, 2], bkd)
        )
        stat = multioutput_stats["mean"](3, backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            ["gmf", "mfmc", "gis"],
            stat,
            costs,
            max_nmodels=3,
            nprocs=1,
            verbosity=1,
        )
        target_cost = 10
        est._save_candidate_estimators = True
        est.allocate_samples(target_cost)
        criteria = bkd.array(
            [e[0]._optimized_criteria for e in est._candidate_estimators]
        )
        assert bkd.allclose(criteria.min(), est._optimized_criteria)

        ntrials, max_eval_concurrency = int(1e3), 1
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs,
                benchmark.prior(),
                est,
                ntrials,
                max_eval_concurrency,
                True,
            )
        )

        rtol, atol = 2e-2, 1e-3
        assert bkd.allclose(covar_mc, covar, atol=atol, rtol=rtol)

        ntrials, max_eval_concurrency = int(1e4), 4
        qoi_idx = [0, 1]
        target_cost = 50
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem([0, 1, 2], qoi_idx, bkd)
        )
        W = benchmark.covariance_of_centered_values_kronker_product()
        W = _nqoisq_nqoisq_subproblem(
            W, benchmark.nmodels(), benchmark.nqoi(), [0, 1, 2], qoi_idx, bkd
        )
        B = benchmark.covariance_of_mean_and_variance_estimators()
        B = _nqoi_nqoisq_subproblem(
            B, benchmark.nmodels(), benchmark.nqoi(), [0, 1, 2], qoi_idx, bkd
        )
        stat = multioutput_stats["mean_variance"](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(cov, W, B)
        est = get_estimator("gmf", stat, costs)
        est.allocate_samples(target_cost)
        # {
        #     "init_guess": {
        #         "disp": True,
        #         "maxiter": 100,
        #         "lower_bound": 1e-3,
        #     }
        # },
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs,
                benchmark.prior(),
                est,
                ntrials,
                max_eval_concurrency,
                True,
            )
        )
        rtol, atol = 2e-2, 1e-4
        assert bkd.allclose(covar_mc, covar, atol=atol, rtol=rtol)

    def test_insert_pilot_samples(self):
        bkd = self.get_backend()
        # This test is specific to ACV sampling strategies (not yet MLBLUE)
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem([0, 1, 2], [0, 1, 2], bkd)
        )
        nqoi = 3

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = bkd.array([0.1, 0.05])
        stat = multioutput_stats["mean"](nqoi, backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            "grd", stat, costs, max_nmodels=3, recursion_index=(0, 1)
        )
        target_cost = 100
        est.allocate_samples(target_cost)

        np.random.seed(1)
        samples_per_model = est.generate_samples_per_model(
            benchmark.prior().rvs
        )
        values_per_model = [
            fun(samples) for fun, samples in zip(funs, samples_per_model)
        ]
        est_val = est(values_per_model)

        # start from same seed so samples will be generated in the same order
        # as above
        # variable.rvs() does not create nested samples when starting from
        # the same randomstate and nvars() > 1, e.g.
        # partial(variable.rvs, random_state=random_state)(3) !=
        # partial(variable.rvs, random_state=random_state)(4)[:, :3]
        np.random.seed(1)
        npilot_samples = 5
        pilot_samples = benchmark.prior().rvs(npilot_samples)
        pilot_values = [f(pilot_samples) for f in benchmark.models()]
        assert bkd.allclose(
            pilot_values[0], values_per_model[0][:npilot_samples]
        )

        samples_per_model_wo_pilot = est.generate_samples_per_model(
            benchmark.prior().rvs, npilot_samples
        )
        values_per_model_wo_pilot = [
            fun(samples)
            for fun, samples in zip(funs, samples_per_model_wo_pilot)
        ]
        nvalues_per_model_wo_pilot = [
            v.shape[0] for v in values_per_model_wo_pilot
        ]
        modified_values_per_model = est.insert_pilot_values(
            pilot_values, values_per_model_wo_pilot
        )
        for ii in range(len(values_per_model)):
            assert bkd.allclose(
                modified_values_per_model[ii], values_per_model[ii]
            )
            # make sure that values_per_model_wo_pilot is not being modified
            # by insert_pilot_values
            assert bkd.allclose(
                bkd.asarray(values_per_model_wo_pilot[ii].shape[0]),
                bkd.asarray(nvalues_per_model_wo_pilot[ii]),
            )
        est_stats = est(modified_values_per_model)
        assert bkd.allclose(est_stats, est_val)

    def _check_bootstrap_estimator(self, est_name, target_cost):
        bkd = self.get_backend()
        qoi_idx = [0, 1]
        model_idx = [0, 1, 2]
        ntrials = 1e3
        # model_idx = [0, 1]
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx, bkd)
        )
        nqoi = len(qoi_idx)

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = bkd.array([0.1, 0.05])[: len(model_idx) - 1]

        stat = multioutput_stats["mean"](nqoi, backend=bkd)
        stat.set_pilot_quantities(cov)
        if est_name == "cv":
            est = get_estimator("cv", stat, costs, lowfi_stats=means[1:])
        else:
            est = get_estimator(est_name, stat, costs)

        if est_name in ["mc", "cv"]:
            est.allocate_samples(target_cost)
        else:
            optimizer = est.get_default_optimizer()
            # optimizer._optimizer1._opts["maxiter"] = 100
            # optimizer._optimizer2._opts["maxiter"] = 2000
            optimizer._optimizer2.set_verbosity(0)
            est.set_optimizer(optimizer)
            est.allocate_samples(target_cost)
            # {
            #     "scaling": 1.0,
            #     "maxiter": 200,
            #     "init_guess": {
            #         "disp": True,
            #         "maxiter": 100,
            #         "lower_bound": 1e-3,
            #     },
            # },

        samples_per_model = est.generate_samples_per_model(
            benchmark.prior().rvs
        )
        values_per_model = [
            f(samples) for f, samples in zip(funs, samples_per_model)
        ]

        bootstrap_values_mean, bootstrap_values_cov = est.bootstrap(
            values_per_model, ntrials
        )
        # print(bootstrap_values_mean, bootstrap_values_cov, "I")
        # print(bootstrap_values_mean-means[0])
        # print(bootstrap_values_cov, "CB")
        # print(est._optimized_covariance, "C")
        assert bkd.allclose(
            bootstrap_values_mean, means[0], atol=1e-3, rtol=1e-2
        )
        assert bkd.allclose(
            bootstrap_values_cov,
            est._optimized_covariance,
            atol=1e-3,
            rtol=1e-2,
        )

        if est_name == "mc":
            return

        # just test that these functions pass
        # stat_name = "mean"
        stat_name = "mean_variance"
        stat = multioutput_stats[stat_name](nqoi, backend=bkd)
        npilot_samples = 20
        pilot_samples = benchmark.prior().rvs(npilot_samples)
        pilot_values_per_model = [fun(pilot_samples) for fun in funs]
        stat.set_pilot_quantities(
            *stat.compute_pilot_quantities(pilot_values_per_model)
        )

        if est_name == "cv":
            lfcovs = []
            lb, ub = 0, 0
            for ii in range(1, len(cov)):
                ub += nqoi
                lfcovs.append(cov[lb:ub, lb:ub])
                lb = ub
            if stat_name == "mean":
                lowfi_stats = means[1:]
            elif stat_name == "variance":
                tril_idx = bkd.tril_indices(lfcovs[0].shape[0])
                lowfi_stats = bkd.stack(
                    [cov[tril_idx[0], tril_idx[1]].flatten() for cov in lfcovs]
                )

            else:
                tril_idx = bkd.tril_indices(lfcovs[0].shape[0])
                lowfi_stats = bkd.stack(
                    [
                        bkd.hstack(
                            (m, cov[tril_idx[0], tril_idx[1]].flatten())
                        )
                        for m, cov in zip(means[1:], lfcovs)
                    ],
                    axis=0,
                )
            est = get_estimator("cv", stat, costs, lowfi_stats=lowfi_stats)
        else:
            est = get_estimator(est_name, stat, costs)
        if est_name in ["mc", "cv"]:
            est.allocate_samples(target_cost)
        else:
            est.allocate_samples(target_cost)
            # {
            #     "scaling": 1.0,
            #     "maxiter": 200,
            #     "init_guess": {
            #         "disp": True,
            #         "maxiter": 100,
            #         "lower_bound": 1e-3,
            #     },
            # },
        # print(est)
        samples_per_model = est.generate_samples_per_model(
            benchmark.prior().rvs
        )
        values_per_model = [
            f(samples) for f, samples in zip(funs, samples_per_model)
        ]

        (
            bootstrap_values_mean,
            bootstrap_values_cov,
            bootstrap_weights_mean,
            bootstrap_weights_cov,
        ) = est.bootstrap(
            values_per_model,
            ntrials,
            mode="values_weights",
            pilot_values=pilot_values_per_model,
        )
        # print(bootstrap_values_mean, bootstrap_values_cov, "J")
        # print(bootstrap_weights_mean, bootstrap_weights_cov)

        (
            bootstrap_values_mean,
            bootstrap_values_cov,
            bootstrap_weights_mean,
            bootstrap_weights_cov,
        ) = est.bootstrap(
            values_per_model,
            ntrials,
            mode="weights",
            pilot_values=pilot_values_per_model,
        )
        # print(bootstrap_values_mean, bootstrap_values_cov, "K")
        # print(bootstrap_weights_mean, bootstrap_weights_cov)

    def test_bootstrap_estimator(self):
        test_cases = [
            ["mc", 1000],
            ["cv", 42],
            ["mfmc", 10000],
            ["gmf", 10000],
            ["grd", 10000],
            ["gis", 10000],
        ]
        for test_case in test_cases:
            print(test_case)
            np.random.seed(1)
            self._check_bootstrap_estimator(*test_case)

    def test_polynomial_ensemble(self):
        bkd = self.get_backend()
        benchmark = PolynomialModelEnsembleBenchmark(backend=bkd)
        cov = benchmark.covariance()
        nmodels = cov.shape[0]
        costs = bkd.asarray([10**-ii for ii in range(nmodels)])
        print(costs)

        stat = multioutput_stats["mean"](benchmark.nqoi(), backend=bkd)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            "gmf",
            stat,
            costs,
            recursion_index=bkd.zeros(nmodels - 1, dtype=int),
        )
        optimizer = est.get_default_optimizer()
        optimizer.set_verbosity(0)
        optimizer._optimizer1._opts["maxiter"] = 500
        # using trust-constr
        # causes scipy optimization  to violate bounds which should be strictly enforced
        optimizer._optimizer2._opts["method"] = "slsqp"
        est.set_optimizer(optimizer)
        # increasing target_cost to say causes test to not pass
        # not sure if this is because estimator variance becomes very small
        # causes issues with the optimization or with the MC estimation of
        # variance
        est.allocate_samples(30)

        covar = est._covariance_from_npartition_samples(
            est._rounded_npartition_samples
        )
        print(covar)

        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                benchmark.models(), benchmark.prior(), est, 10000, 1, True
            )
        )
        print(covar_mc, covar)
        assert bkd.allclose(covar_mc, covar, rtol=1e-2)


class TestTorchMOMC(TestMOMC, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
