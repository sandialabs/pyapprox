import unittest

import torch
import numpy as np

from pyapprox.util.utilities import check_gradients
from pyapprox.multifidelity.stats import (
    _nqoisq_nqoisq_subproblem, _nqoi_nqoisq_subproblem)
from pyapprox.multifidelity._optim import (
    _allocate_samples_mfmc, _allocate_samples_mlmc)
from pyapprox.multifidelity.acv import (
    MFMCEstimator, MLMCEstimator)
from pyapprox.multifidelity.factory import (
    get_estimator, BestEstimator, numerically_compute_estimator_variance,
    multioutput_stats)
from pyapprox.multifidelity.tests.test_stats import (
    _setup_multioutput_model_subproblem)
from pyapprox.multifidelity.stats import (
    _get_nsamples_intersect, _get_nsamples_subset)


def _log_single_qoi_criteria(
        qoi_idx, stat_type, criteria_type, variance):
    # use if stat == Variance and target is variance
    # return torch.log(variance[qoi_idx[0], qoi_idx[0]])
    # use if stat == MeanAndVariance and target is variance
    if stat_type == "mean_var" and criteria_type == "var":
        return torch.log(variance[3+qoi_idx[0], 3+qoi_idx[0]])
    if stat_type == "mean_var" and criteria_type == "mean":
        return torch.log(variance[qoi_idx[0], qoi_idx[0]])
    raise ValueError


class TestMOMC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_generalized_recursive_difference_allocation_matrices(self):
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[2, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  0.,  0.,  1.,  0.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  1.,  0.,  0.,  1.]])
        )

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[0, 1])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  0.,  0.,  0.],
                      [0.,  0.,  0.,  1.,  1.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[0, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  0.,  1.,  0.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        cov = np.random.normal(0, 1, (4, 4))
        costs = np.ones(4)
        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs, recursion_index=[0, 1, 2])
        npartition_samples = torch.as_tensor([2, 2, 4, 4], dtype=torch.double)
        nsamples_intersect = _get_nsamples_intersect(
            est._allocation_mat, npartition_samples)

        nsamples_interesect_true = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 2., 2., 0., 0., 0., 0., 0.],
             [0., 2., 2., 0., 0., 0., 0., 0.],
             [0., 0., 0., 2., 2., 0., 0., 0.],
             [0., 0., 0., 2., 2., 0., 0., 0.],
             [0., 0., 0., 0., 0., 4., 4., 0.],
             [0., 0., 0., 0., 0., 4., 4., 0.],
             [0., 0., 0., 0., 0., 0., 0., 4.]])
        assert np.allclose(nsamples_intersect, nsamples_interesect_true)
        nsamples_subset = _get_nsamples_subset(
            est._allocation_mat, npartition_samples)
        assert np.allclose(nsamples_subset, [0, 2, 2, 2, 2, 4, 4, 4])

    def test_generalized_multifidelity_allocation_matrices(self):
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("gmf", stat, costs, recursion_index=[2, 0])

        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  1.,  1.],
                      [0.,  0.,  1.,  1.,  0.,  1.],
                      [0.,  0.,  1.,  0.,  0.,  1.]])
        )

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("gmf", stat, costs, recursion_index=[0, 1])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("gmf", stat, costs, recursion_index=[0, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  1.,  0.,  1.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

    def test_generalized_independent_samples_allocation_matrices(self):
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("gis", stat, costs, recursion_index=[2, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  0.,  0.,  1.,  1.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  1.,  1.,  0.,  1.]])
        )

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("gis", stat, costs, recursion_index=[0, 1])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  0.,  0.],
                      [0.,  0.,  0.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("gis", stat, costs, recursion_index=[0, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

    def _check_estimator_variances(self, model_idx, qoi_idx, recursion_index,
                                   est_type, stat_type, tree_depth=None,
                                   max_nmodels=None, ntrials=int(1e4),
                                   target_cost=50):
        ntrials = int(ntrials)
        rtol, atol = 4.6e-2, 1.01e-3
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)

        # change costs so less samples are used in the estimator
        # to speed up test
        costs = np.array([2, 1.5, 1])[model_idx]

        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        pilot_args = [cov]
        if est_type == "gmf" or est_type == "grd" or est_type == "gis":
            if tree_depth is not None:
                kwargs = {"tree_depth": tree_depth}
            else:
                kwargs = {"recursion_index": np.asarray(recursion_index)}
        else:
            kwargs = {}

        if stat_type == "mean":
            idx = nqoi
            if est_type == "cv":
                kwargs["lowfi_stats"] = [m for m in means[1:]]
        if "variance" in stat_type:
            if est_type == "cv":
                lfcovs = []
                lb, ub = 0, 0
                for ii in range(1, nmodels):
                    ub += nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                    lb = ub
                kwargs["lowfi_stats"] = [cov.flatten() for cov in lfcovs]
            W = model.covariance_of_centered_values_kronker_product()
            W = _nqoisq_nqoisq_subproblem(
                W, model.nmodels, model.nqoi, model_idx, qoi_idx)
            # npilot_samples = int(1e6)
            # pilot_samples = model.variable.rvs(npilot_samples)
            # pilot_values = np.hstack([f(pilot_samples) for f in funs])
            # W = get_W_from_pilot(pilot_values, nmodels)
            pilot_args.append(W)
            idx = nqoi**2
        if stat_type == "mean_variance":
            if est_type == "cv":
                lfcovs = []
                lb, ub = 0, 0
                for ii in range(1, nmodels):
                    ub += nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                    lb = ub
                kwargs["lowfi_stats"] = [
                    np.hstack((m, cov.flatten()))
                    for m, cov in zip(means[1:], lfcovs)]
            B = model.covariance_of_mean_and_variance_estimators()
            B = _nqoi_nqoisq_subproblem(
                B, model.nmodels, model.nqoi, model_idx, qoi_idx)
            pilot_args.append(B)
            idx = nqoi+nqoi**2

        # check covariance matrix is positive definite
        # np.linalg.cholesky(cov)
        stat = multioutput_stats[stat_type](len(qoi_idx))
        stat.set_pilot_quantities(*pilot_args)
        est = get_estimator(
            est_type, stat, costs, max_nmodels=max_nmodels, **kwargs)

        est.allocate_samples(target_cost)

        max_eval_concurrency = 1
        if isinstance(est, BestEstimator):
            funs_subset = [funs[idx] for idx in est._best_model_indices]
        else:
            funs_subset = funs
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs_subset,
                model.variable, est, ntrials, max_eval_concurrency, True)
        )
        hfcovar = hfcovar.numpy()

        # np.set_printoptions(linewidth=1000)
        # print(hfcovar_mc)
        # print(hfcovar)
        # print((np.abs(hfcovar_mc-hfcovar)-(atol+rtol*np.abs(hfcovar))).min())
        # assert np.allclose(hfcovar_mc, hfcovar, atol=atol, rtol=rtol)
        if est_type != "mc":
            CF_mc = np.cov(delta.T, ddof=1)
            cf_mc = np.cov(Q.T, delta.T, ddof=1)[:idx, idx:]
            CF, cf = est._get_discrepancy_covariances(
                est._rounded_npartition_samples)
            CF, cf = CF.numpy(), cf.numpy()
            # print(CF)
            # print(CF_mc)
            # print((np.abs(CF_mc-CF)-(atol + rtol*np.abs(CF))).min())
            # print(cf)
            # print(cf_mc)
            # print((np.abs(cf_mc-cf)-(atol + rtol*np.abs(cf))).min())
            assert np.allclose(CF_mc, CF, atol=atol, rtol=rtol)
            assert np.allclose(cf_mc, cf, atol=atol, rtol=rtol)

        # print(covar_mc, 'v_mc')
        # print(covar, 'v')
        # print((covar_mc-covar)/covar)
        assert np.allclose(covar_mc, covar, atol=atol, rtol=rtol)

    def test_estimator_variances(self):
        test_cases = [
            [[0], [0, 1, 2], None, "mc", "mean"],
            [[0], [0, 1, 2], None, "mc", "variance"],
            [[0], [0, 1], None, "mc", "mean_variance"],
            [[0, 1, 2], [0, 1], None, "cv", "mean"],
            [[0, 1, 2], [0, 1], None, "cv", "variance"],
            [[0, 1, 2], [0, 2], None, "cv", "mean_variance",
             None, None, 5e4],
            [[0, 1, 2], [0], [0, 1], "grd", "mean"],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "mean"],
            [[0, 1, 2], [0, 1], [0, 0], "grd", "mean"],
            [[0, 1], [0, 1, 2], [0], "grd", "variance"],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "variance"],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "mean_variance",
             None, None, 5e4],
            [[0, 1, 2], [0], [0, 1], "gis", "mean"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean"],
            # test now fixed bug that occured when number of samples in
            # an acv_subset was 1
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean",
             None, None, 1e4, 10],
            # [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", 2, None, 5e4, 100], # fails for some reason when using truncated eigvals to compute log determinant, but works if using torch.logdet even though no eigenvalues are truncated
            [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", None, 3, 1e4, 100],
            [[0, 1, 2], [0, 1, 2], [0, 1],
             ["gmf", "grd", "gis", "mlmc"], "mean", None, 3, 1e4, 100],
            [[0, 1, 2], [0, 1, 2], [0, 1], "grd", "mean", None, 3],
            [[0, 1, 2], [1], [0, 1], "grd", "variance", None, 3],
            [[0, 1], [0, 2], [0], "gmf", "mean"],
            [[0, 1], [0], [0], "gmf", "variance"],
            [[0, 1], [0, 2], [0], "gmf", "variance"],
            [[0, 1, 2], [0], [0, 0], "gmf", "variance"],
            [[0, 1, 2], [0, 2], [0, 0], "gmf", "variance",
             None, None, 1e4, 100],
            [[0, 1], [0], [0], "gmf", "mean_variance"],
            [[0, 1, 2], [0], None, "mfmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "mean", None, None, 5e4],
            [[0, 1, 2], [0, 1], None, "mlmc", "mean", None, None, 5e4],
            [[0, 1, 2], [0], None, "mlmc", "variance"],
            [[0, 1, 2], [0], None, "mlmc", "mean_variance",
             None, None, 1e4, 100],
            [[0], [0, 1, 2], None, "mc", "variance"],
            [[0, 1, 2], [0, 2], None, "gmf", "variance", 2, None, int(5e4)],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_estimator_variances(*test_case)

    def test_numerical_mfmc_sample_optimization(self):
        # check for scalar output case we require MFMC analytical solution
        model_idx, qoi_idx = [0, 1, 2], [0]
        recursion_index = [0, 1]
        target_cost = 10
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("gmf", stat, costs,
                            recursion_index=np.asarray(recursion_index))
        mfmc_model_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            cov, costs, target_cost)
        assert np.allclose(
            np.exp(est._objective(
                target_cost, MFMCEstimator._native_ratios_to_npartition_ratios(
                    mfmc_model_ratios))[0]),
            np.exp(mfmc_log_variance))
        assert np.allclose(
            est._objective(
                target_cost, MFMCEstimator._native_ratios_to_npartition_ratios(
                    mfmc_model_ratios))[1], 0)

        partition_ratios = torch.as_tensor(
            MFMCEstimator._native_ratios_to_npartition_ratios(
                mfmc_model_ratios), dtype=torch.double)
        errors = check_gradients(
            lambda z: est._objective(target_cost, z[:, 0]), True,
            partition_ratios[:, None].numpy()+1,
            fd_eps=np.logspace(-12, 1, 14)[::-1])
        assert errors.min()/errors.max() < 3.e-6

        cons = est._get_constraints(target_cost)
        for con in cons:
            errors = check_gradients(
                lambda z: con["fun"](z[:, 0], *con["args"]),
                lambda z: con["jac"](z[:, 0], *con["args"]),
                partition_ratios[:, None].numpy()+1,
                fd_eps=np.logspace(-12, 1, 14)[::-1], disp=False)
        assert errors.min()/errors.max() < 1e-6

        # test mapping from partition ratios to model ratios
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples)
        est_cost = (nsamples_per_model*est._costs.numpy()).sum()
        assert np.allclose(
            nsamples_per_model, np.hstack(
                (nsamples_per_model[0], model_ratios*npartition_samples[0])))
        assert np.allclose(model_ratios, mfmc_model_ratios)
        assert np.allclose(model_ratios*npartition_samples[0],
                           np.cumsum(npartition_samples)[1:])
        assert np.allclose(target_cost, est_cost)
        # get nsample ratios before rounding
        # avoid using est._allocate_samples so we do not start
        # from mfmc exact solution
        partition_ratios, obj_val = est._allocate_samples(
            target_cost, {"scaling": 1.})
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples)
        est_cost = (nsamples_per_model*est._costs.numpy()).sum()
        assert np.allclose(np.exp(obj_val), np.exp(mfmc_log_variance))
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        assert np.allclose(model_ratios, mfmc_model_ratios)

    def test_numerical_mlmc_sample_optimization(self):
        # check for scalar output case we require MFMC analytical solution
        model_idx, qoi_idx = [0, 1, 2], [0]
        recursion_index = [0, 1]
        target_cost = 10
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)

        # The following will give mlmc with unit variance
        # and level variances var[f_i-f_{i+1}] = [1, 4, 4]
        target_cost = 81
        costs = [6, 3, 1]
        cov = np.asarray([[1.00, 0.50, 0.25],
                          [0.50, 1.00, 0.50],
                          [0.25, 0.50, 4.00]])

        stat = multioutput_stats["mean"](len(qoi_idx))
        stat.set_pilot_quantities(cov)
        est = get_estimator("grd", stat, costs,
                            recursion_index=np.asarray(recursion_index))
        mlmc_model_ratios, mlmc_log_variance = _allocate_samples_mlmc(
            cov, costs, target_cost)

        # We are trying to test numerical GRD optimization, but to recover
        # MLMC solution we need to use suboptimal weights of MLMC so adjust
        # the necessary functions here
        est._weights = MLMCEstimator._weights

        def mlmc_cov(npartition_samples):
            CF, cf = est._get_discrepancy_covariances(npartition_samples)
            weights = est._weights(CF, cf)
            return est._covariance_non_optimal_weights(
                est._stat.high_fidelity_estimator_covariance(
                    npartition_samples[0]), weights, CF, cf)
        est._covariance_from_npartition_samples = mlmc_cov

        # test mapping from partition ratios to model ratios
        partition_ratios = torch.as_tensor(
            MLMCEstimator._native_ratios_to_npartition_ratios(
                mlmc_model_ratios), dtype=torch.double)
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples)
        est_cost = (nsamples_per_model*est._costs.numpy()).sum()
        assert np.allclose(
            nsamples_per_model, np.hstack(
                (nsamples_per_model[0], model_ratios*npartition_samples[0])))
        assert np.allclose(model_ratios, mlmc_model_ratios)
        assert np.allclose(target_cost, est_cost)

        assert np.allclose(
            np.exp(est._objective(
                target_cost, MLMCEstimator._native_ratios_to_npartition_ratios(
                    mlmc_model_ratios))[0]),
            np.exp(mlmc_log_variance))
        assert np.allclose(
            est._objective(
                target_cost, MLMCEstimator._native_ratios_to_npartition_ratios(
                    mlmc_model_ratios))[1], 0)

        errors = check_gradients(
            lambda z: est._objective(target_cost, z[:, 0]), True,
            partition_ratios[:, None].numpy()+1,
            fd_eps=np.logspace(-12, 1, 14)[::-1])
        assert errors.min()/errors.max() < 1e-6

        cons = est._get_constraints(target_cost)
        for con in cons:
            errors = check_gradients(
                lambda z: con["fun"](z[:, 0], *con["args"]),
                lambda z: con["jac"](z[:, 0], *con["args"]),
                partition_ratios[:, None].numpy()+1,
                fd_eps=np.logspace(-12, 1, 14)[::-1], disp=False)
        assert errors.min()/errors.max() < 1e-6

        # get nsample ratios before rounding
        # avoid using est._allocate_samples so we do not start
        # from mlmc exact solution
        partition_ratios, obj_val = est._allocate_samples(
            target_cost, {"scaling": 1.})
        npartition_samples = est._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios)
        nsamples_per_model = est._compute_nsamples_per_model(
            npartition_samples)
        est_cost = (nsamples_per_model*est._costs.numpy()).sum()
        assert np.allclose(np.exp(obj_val), np.exp(mlmc_log_variance))
        model_ratios = est._partition_ratios_to_model_ratios(partition_ratios)
        assert np.allclose(model_ratios, mlmc_model_ratios)

    def test_best_model_subset_estimator(self):
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        stat = multioutput_stats["mean"](3)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            ["gmf", "mfmc", "gis"], stat, costs, max_nmodels=3)
        target_cost = 10
        est._save_candidate_estimators = True
        np.set_printoptions(linewidth=1000)
        est.allocate_samples(target_cost, {"verbosity": 1, "nprocs": 1})

        criteria = np.array(
            [e[0]._optimized_criteria for e in est._candidate_estimators])
        assert np.allclose(criteria.min(), est._optimized_criteria)

        ntrials, max_eval_concurrency = int(1e3), 1
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs, model.variable, est, ntrials, max_eval_concurrency, True)
        )

        rtol, atol = 2e-2, 1e-3
        assert np.allclose(covar_mc, covar, atol=atol, rtol=rtol)

        ntrials, max_eval_concurrency = int(1e4), 4
        qoi_idx = [0, 1]
        target_cost = 50
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            [0, 1, 2], qoi_idx)
        W = model.covariance_of_centered_values_kronker_product()
        W = _nqoisq_nqoisq_subproblem(
            W, model.nmodels, model.nqoi, [0, 1, 2], qoi_idx)
        B = model.covariance_of_mean_and_variance_estimators()
        B = _nqoi_nqoisq_subproblem(
            B, model.nmodels, model.nqoi, [0, 1, 2], qoi_idx)
        stat = multioutput_stats["mean_variance"](len(qoi_idx))
        stat.set_pilot_quantities(cov, W, B)
        est = get_estimator("gmf", stat, costs)
        est.allocate_samples(target_cost)
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs, model.variable, est, ntrials, max_eval_concurrency, True)
        )
        rtol, atol = 2e-2, 1e-4
        assert np.allclose(covar_mc, covar, atol=atol, rtol=rtol)

    def test_insert_pilot_samples(self):
        # This test is specific to ACV sampling strategies (not yet MLBLUE)
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        nqoi = 3

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = 0.1, 0.05
        stat = multioutput_stats["mean"](nqoi)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            "grd", stat, costs, max_nmodels=3, recursion_index=(0, 1))
        target_cost = 100
        est.allocate_samples(target_cost, {"verbosity": 0, "nprocs": 1})

        np.random.seed(1)
        samples_per_model = est.generate_samples_per_model(model.variable.rvs)
        values_per_model = [
            fun(samples) for fun, samples in zip(funs, samples_per_model)]
        est_val = est(values_per_model)

        # start from same seed so samples will be generated in the same order
        # as above
        # variable.rvs() does not create nested samples when starting from
        # the same randomstate and num_vars() > 1, e.g.
        # partial(variable.rvs, random_state=random_state)(3) !=
        # partial(variable.rvs, random_state=random_state)(4)[:, :3]
        np.random.seed(1)
        npilot_samples = 5
        pilot_samples = model.variable.rvs(npilot_samples)
        pilot_values = [f(pilot_samples) for f in model.funs]
        assert np.allclose(
            pilot_values[0], values_per_model[0][:npilot_samples])

        samples_per_model_wo_pilot = est.generate_samples_per_model(
            model.variable.rvs, npilot_samples)
        values_per_model_wo_pilot = [
            fun(samples) for fun, samples in
            zip(funs, samples_per_model_wo_pilot)]
        nvalues_per_model_wo_pilot = [
            v.shape[0] for v in values_per_model_wo_pilot]
        modified_values_per_model = est.insert_pilot_values(
            pilot_values, values_per_model_wo_pilot)
        for ii in range(len(values_per_model)):
            assert np.allclose(modified_values_per_model[ii],
                               values_per_model[ii])
            # make sure that values_per_model_wo_pilot is not being modified
            # by insert_pilot_values
            assert np.allclose(values_per_model_wo_pilot[ii].shape[0],
                               nvalues_per_model_wo_pilot[ii])
        est_stats = est(modified_values_per_model)
        assert np.allclose(est_stats, est_val)

    def _check_bootstrap_estimator(self, est_name, target_cost):
        qoi_idx = [0, 1]
        model_idx = [0, 1, 2]
        # model_idx = [0, 1]
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        nqoi = len(qoi_idx)

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = [0.1, 0.05][:len(model_idx)-1]

        stat = multioutput_stats["mean"](nqoi)
        stat.set_pilot_quantities(cov)
        if est_name == "cv":
            est = get_estimator("cv", stat, costs, lowfi_stats=means[1:])
        else:
            est = get_estimator(est_name, stat, costs)
        est.allocate_samples(target_cost)
        print(est)

        samples_per_model = est.generate_samples_per_model(model.variable.rvs)
        values_per_model = [
            f(samples) for f, samples in zip(funs, samples_per_model)]

        bootstrap_stats, bootstrap_cov = est.bootstrap(
            values_per_model, 1e3)
        # print(bootstrap_stats, means[0])
        # print(bootstrap_stats-means[0])
        # print(bootstrap_cov, "CB")
        # print(est._optimized_covariance, "C")
        assert np.allclose(bootstrap_stats, means[0], atol=1e-3, rtol=1e-2)
        assert np.allclose(bootstrap_cov, est._optimized_covariance,
                           atol=1e-3, rtol=1e-2)

    def test_bootstrap_estimator(self):
        test_cases = [
            ["mc", 1000], ["cv", 42], ["mfmc", 100], ["gmf", 100],
            ["grd", 10000], ["gis", 10000]]
        for test_case in test_cases[-1:]:
            print(test_case)
            self._check_bootstrap_estimator(*test_case)

    def test_polynomial_ensemeble(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        benchmark = setup_benchmark("polynomial_ensemble")
        cov = benchmark.covariance
        nmodels = cov.shape[0]
        costs = np.asarray([10**-ii for ii in range(nmodels)])

        stat = multioutput_stats["mean"](benchmark.nqoi)
        stat.set_pilot_quantities(cov)
        est = get_estimator(
            "gmf", stat, costs, recursion_index=np.zeros(nmodels-1, dtype=int))
        est.allocate_samples(100)

        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                benchmark.funs, benchmark.variable, est, 1000, 1, True)
        )
        assert np.allclose(covar_mc, covar, rtol=1e-2)


if __name__ == "__main__":
    momc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMOMC)
    unittest.TextTestRunner(verbosity=2).run(momc_test_suite)
