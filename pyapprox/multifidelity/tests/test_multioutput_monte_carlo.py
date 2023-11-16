import unittest
from functools import partial

import torch
import numpy as np

from pyapprox.util.utilities import check_gradients
from pyapprox.multifidelity.stats import (
    _nqoisq_nqoisq_subproblem, _nqoi_nqoisq_subproblem)
from pyapprox.multifidelity._optim import (
    _allocate_samples_mfmc, _allocate_samples_mlmc)
from pyapprox.multifidelity.multioutput_monte_carlo import (
    get_estimator, ACVEstimator, MFMCEstimator, MLMCEstimator, CVEstimator,
    numerically_compute_estimator_variance)
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

        est = get_estimator("grd", "mean", len(qoi_idx),
                            costs, cov, recursion_index=[2, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  0.,  0.,  1.,  0.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  1.,  0.,  0.,  1.]])
        )

        est = get_estimator("grd", "mean", len(qoi_idx),
                            costs, cov, recursion_index=[0, 1])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  0.,  0.,  0.],
                      [0.,  0.,  0.,  1.,  1.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        est = get_estimator(
            "grd", "mean", len(qoi_idx), costs, cov, recursion_index=[0, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  0.,  1.,  0.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        cov = np.random.normal(0, 1, (4, 4))
        costs = np.ones(4)
        est = get_estimator(
            "grd", "mean", len(qoi_idx), costs, cov, recursion_index=[0, 1, 2])
        npartition_samples = torch.as_tensor([2, 2, 4, 4], dtype=torch.double)
        nsamples_intersect = _get_nsamples_intersect(
            est._allocation_mat, npartition_samples)
        print(nsamples_intersect)
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
        est = get_estimator(
            "gmf", "mean", len(qoi_idx), costs, cov, recursion_index=[2, 0])

        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  1.,  1.],
                      [0.,  0.,  1.,  1.,  0.,  1.],
                      [0.,  0.,  1.,  0.,  0.,  1.]])
        )

        est = get_estimator(
            "gmf", "mean", len(qoi_idx), costs, cov, recursion_index=[0, 1])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        est = get_estimator(
            "gmf", "mean", len(qoi_idx), costs, cov, recursion_index=[0, 0])
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
        est = get_estimator(
            "gis", "mean", len(qoi_idx), costs, cov, recursion_index=[2, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  0.,  0.,  1.,  1.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  1.,  1.,  0.,  1.]])
        )

        est = get_estimator(
            "gis", "mean", len(qoi_idx), costs, cov, recursion_index=[0, 1])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  0.,  0.],
                      [0.,  0.,  0.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

        est = get_estimator(
            "gis", "mean", len(qoi_idx), costs, cov, recursion_index=[0, 0])
        assert np.allclose(
            est._allocation_mat,
            np.array([[0.,  1.,  1.,  1.,  1.,  1.],
                      [0.,  0.,  0.,  1.,  0.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  1.]])
        )

    def _check_estimator_variances(self, model_idx, qoi_idx, recursion_index,
                                   est_type, stat_type, tree_depth=None,
                                   max_nmodels=None, ntrials=int(1e4)):
        rtol, atol = 4.6e-2, 1.01e-3
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)

        # change costs so less samples are used in the estimator
        # to speed up test
        costs = [2, 1.5, 1][:len(costs)]

        nqoi = len(qoi_idx)
        nmodels = len(model_idx)

        args = []
        if est_type != "mc" and est_type != "cv":
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
                    print(lfcovs[-1], lb, ub, cov[lb:ub])
                    lb = ub
                kwargs["lowfi_stats"] = [cov.flatten() for cov in lfcovs]
            W = model.covariance_of_centered_values_kronker_product()
            W = _nqoisq_nqoisq_subproblem(
                W, model.nmodels, model.nqoi, model_idx, qoi_idx)
            # npilot_samples = int(1e6)
            # pilot_samples = model.variable.rvs(npilot_samples)
            # pilot_values = np.hstack([f(pilot_samples) for f in funs])
            # W = get_W_from_pilot(pilot_values, nmodels)
            args.append(W)
            idx = nqoi**2
        if stat_type == "mean_variance":
            if est_type == "cv":
                lfcovs = []
                lb, ub = 0, 0
                for ii in range(1, nmodels):
                    ub += nqoi
                    lfcovs.append(cov[lb:ub, lb:ub])
                    print(lfcovs[-1], lb, ub, cov[lb:ub])
                    lb = ub
                kwargs["lowfi_stats"] = [
                    np.hstack((m, cov.flatten()))
                    for m, cov in zip(means[1:], lfcovs)]
            B = model.covariance_of_mean_and_variance_estimators()
            B = _nqoi_nqoisq_subproblem(
                B, model.nmodels, model.nqoi, model_idx, qoi_idx)
            args.append(B)
            idx = nqoi+nqoi**2
        est = get_estimator(
            est_type, stat_type, nqoi, costs, cov, *args,
            max_nmodels=max_nmodels, **kwargs)

        # must call opt otherwise best_est will not be set for
        # best model subset acv
        est.allocate_samples(100)
        # est._nsamples_per_model = torch.as_tensor(
        #     [10, 20, 30], dtype=torch.double)
        # est._rounded_npartition_samples = torch.as_tensor(
        #     [10, 10, 20], dtype=torch.double)
        print(est)

        max_eval_concurrency = 1
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs, model.variable, est, ntrials, max_eval_concurrency, True)
        )

        assert np.allclose(hfcovar_mc, hfcovar, atol=atol, rtol=rtol)
        if est_type != "mc":
            print(delta.shape)
            CF_mc = torch.as_tensor(
                np.cov(delta.T, ddof=1), dtype=torch.double)
            cf_mc = torch.as_tensor(
                np.cov(Q.T, delta.T, ddof=1)[:idx, idx:], dtype=torch.double)
            CF, cf = est._get_discrepancy_covariances(
                est._rounded_npartition_samples)
            CF, cf = CF.numpy(), cf.numpy()
            assert np.allclose(CF_mc, CF, atol=atol, rtol=rtol)
            assert np.allclose(cf_mc, cf, atol=atol, rtol=rtol)

        np.set_printoptions(linewidth=1000)
        print(covar_mc, 'v_mc')
        print(covar, 'v')
        print((covar_mc-covar)/covar)
        assert np.allclose(covar_mc, covar, atol=atol, rtol=rtol)

    def test_estimator_variances(self):
        test_cases = [
            [[0], [0, 1, 2], None, "mc", "mean"],
            [[0], [0, 1, 2], None, "mc", "variance"],
            [[0], [0, 1], None, "mc", "mean_variance"],
            [[0, 1, 2], [0, 1], None, "cv", "mean"],
            [[0, 1, 2], [0, 1], None, "cv", "variance"],
            [[0, 1, 2], [0, 1], None, "cv", "mean_variance"],
            [[0, 1, 2], [0], [0, 1], "grd", "mean"],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "mean"],
            [[0, 1, 2], [0, 1], [0, 0], "grd", "mean"],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "variance"],
            [[0, 1, 2], [0, 1], [0, 1], "grd", "mean_variance"],
            [[0, 1, 2], [0], [0, 1], "gis", "mean"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean"],
            [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", 2],
            [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", None, 3],
            [[0, 1], [0, 2], [0], "gmf", "mean"],
            [[0, 1], [0], [0], "gmf", "variance"],
            [[0, 1], [0, 2], [0], "gmf", "variance"],
            [[0, 1, 2], [0], [0, 0], "gmf", "variance"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "variance"],
            [[0, 1], [0], [0], "gmf", "mean_variance"],
            # following is slow test remove for speed
            # [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean_variance", None,
            #   None, int(1e5)],
            [[0, 1, 2], [0], None, "mfmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "mean"],
            [[0, 1, 2], [0, 1], None, "mlmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "variance"],
            [[0, 1, 2], [0], None, "mlmc", "mean_variance"],
            [[0, 1, 2], [0, 1, 2], None, "gmf", "variance", 2],
            [[0], [0, 1, 2], None, "mc", "variance"],
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
        est = get_estimator("gmf", "mean", len(qoi_idx), costs, cov,
                            recursion_index=np.asarray(recursion_index))
        mfmc_model_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            cov, costs, target_cost)
        assert np.allclose(
            np.exp(est._objective(
                target_cost, MFMCEstimator._mfmc_ratios_to_npartition_ratios(
                    mfmc_model_ratios))[0]),
            np.exp(mfmc_log_variance))
        assert np.allclose(
            est._objective(
                target_cost, MFMCEstimator._mfmc_ratios_to_npartition_ratios(
                    mfmc_model_ratios))[1], 0)

        partition_ratios = torch.as_tensor(
            MFMCEstimator._mfmc_ratios_to_npartition_ratios(
                mfmc_model_ratios), dtype=torch.double)
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
            target_cost)
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

        est = get_estimator("grd", "mean", len(qoi_idx), costs, cov,
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
            MLMCEstimator._mlmc_ratios_to_npartition_ratios(
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
                target_cost, MLMCEstimator._mlmc_ratios_to_npartition_ratios(
                    mlmc_model_ratios))[0]),
            np.exp(mlmc_log_variance))
        assert np.allclose(
            est._objective(
                target_cost, MLMCEstimator._mlmc_ratios_to_npartition_ratios(
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
            target_cost)
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
        est = get_estimator("gmf", "mean", 3, costs, cov, max_nmodels=3)
        target_cost = 10
        est.allocate_samples(target_cost, verbosity=1, nprocs=1)

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
        est = get_estimator(
            "gmf", "mean_variance", 3, costs, cov, W, B)
        est.allocate_samples(target_cost)
        hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
            numerically_compute_estimator_variance(
                funs, model.variable, est, ntrials, max_eval_concurrency, True)
        )
        rtol, atol = 2e-2, 1e-4
        assert np.allclose(covar_mc, covar, atol=atol, rtol=rtol)

    def test_insert_pilot_samples(self):
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        nqoi = 3

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = 0.1, 0.05
        est = get_estimator(
            "gmf", "mean", nqoi, costs, cov, max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)

        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(funs)
        est_val = est(acv_values)

        # This test is specific to ACVMF sampling strategy
        npilot_samples = 5
        pilot_samples = acv_samples[0][1][:, :npilot_samples]
        pilot_values = [f(pilot_samples) for f in model.funs]
        assert np.allclose(pilot_values[0], acv_values[0][1][:npilot_samples])

        values_per_model = est.combine_acv_values(acv_values)
        values_per_model_wo_pilot = [
            vals[npilot_samples:] for vals in values_per_model]
        values_per_model = [
            np.vstack((pilot_values[ii], vals))
            for ii, vals in enumerate(values_per_model_wo_pilot)]
        acv_values = est.separate_model_values(values_per_model)
        est_stats = est(acv_values)
        assert np.allclose(est_stats, est_val)

        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(
            funs, [pilot_samples, pilot_values])
        est_val_pilot = est(acv_values)
        assert np.allclose(est_val, est_val_pilot)
        for ii in range(1, 3):
            assert np.allclose(
                acv_samples[ii][0][:, :npilot_samples],
                acv_samples[0][1][:, :npilot_samples])
            assert np.allclose(
                acv_samples[ii][1][:, :npilot_samples],
                acv_samples[0][1][:, :npilot_samples])

        npilot_samples = 8
        pilot_samples = est.best_est.generate_samples(npilot_samples)
        self.assertRaises(ValueError, est.generate_data,
                          funs, [pilot_samples, pilot_values])

        # modify costs so more hf samples are used
        costs[1:] = 0.5, 0.05
        est = get_estimator("gmf", "mean", nqoi, costs, cov, max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)
        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(funs)
        est_val = est(acv_values)

        npilot_samples = 5
        samples_per_model = est.combine_acv_samples(acv_samples)
        samples_per_model_wo_pilot = [
            s[:, npilot_samples:] for s in samples_per_model]
        values_per_model = est.combine_acv_values(acv_values)
        values_per_model_wo_pilot = [
            vals[npilot_samples:] for vals in values_per_model]

        pilot_samples = acv_samples[0][1][:, :npilot_samples]
        pilot_values = [f(pilot_samples) for f in model.funs]
        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples1, acv_values1 = est.generate_data(
            funs, [pilot_samples, pilot_values])
        est_val_pilot = est(acv_values1)
        assert np.allclose(est_val, est_val_pilot)

        pilot_data = pilot_samples, pilot_values
        acv_values2 = est.combine_pilot_data(
            samples_per_model_wo_pilot, values_per_model_wo_pilot, pilot_data)
        assert np.allclose(acv_values1[0][1], acv_values[0][1])
        for ii in range(1, len(acv_values2)):
            assert np.allclose(acv_values2[ii][0], acv_values[ii][0])
            assert np.allclose(acv_values2[ii][1], acv_values[ii][1])
        est_val_pilot = est(acv_values2)
        assert np.allclose(est_val, est_val_pilot)

    def test_bootstrap_estimator(self):
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        nqoi = 3

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = 0.1, 0.05
        est = get_estimator("gmf", "mean", nqoi, costs, cov, max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)

        samples_per_model = est._generate_estimator_samples(None)[0]
        print(samples_per_model)
        values_per_model = [
            f(samples) for f, samples in zip(funs, samples_per_model)]

        bootstrap_mean, bootstrap_variance = est.bootstrap(
            values_per_model, 100) # 10000


if __name__ == "__main__":
    momc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMOMC)
    unittest.TextTestRunner(verbosity=2).run(momc_test_suite)
